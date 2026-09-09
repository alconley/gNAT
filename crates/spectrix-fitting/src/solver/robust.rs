//! Robust optimization deliberately kept separate from the numerical compatibility profile.

use std::cell::{Cell, RefCell};

use super::{
    Bounds, CovarianceStatus, DMatrix, DVector, Dyn, FitDiagnostics, FitError, FitOptions,
    FitProblem, FitResult, LeastSquaresProblem, LevenbergMarquardt, Model, ObjectiveKind, Owned,
    POISSON_FLOOR, ParameterKind, ParameterLayout, ParameterValues, TerminationStatus,
    base_estimates, check_allocation, checked_zeros, confidence_bands, covariance_public,
    derived_estimates, objective_residuals, objective_value, poisson_deviance_residual_gradient,
    statistics, validate_problem,
};

const STATIONARITY_TOLERANCE: f64 = 1.0e-6;

struct Budget {
    used: Cell<usize>,
    limit: usize,
    best: RefCell<(Vec<f64>, f64)>,
}

struct Target<'a> {
    problem: &'a FitProblem,
    layout: &'a ParameterLayout,
    parameters: DVector<f64>,
    objective: ObjectiveKind,
    budget: &'a Budget,
    stop_at: usize,
}

impl Target<'_> {
    fn external(&self) -> Result<ParameterValues, FitError> {
        self.layout
            .values_with_transform(self.parameters.as_slice(), from_unclipped)
    }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for Target<'_> {
    type ParameterStorage = Owned<f64, Dyn>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;

    fn set_params(&mut self, parameters: &DVector<f64>) {
        // The transform itself enforces physical bounds. Clipping these internal
        // coordinates invalidates LM's step and can make a boundary absorbing.
        self.parameters.clone_from(parameters);
    }

    fn params(&self) -> DVector<f64> {
        self.parameters.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        if self.budget.used.get() >= self.stop_at.min(self.budget.limit) {
            return None;
        }
        self.budget.used.set(self.budget.used.get() + 1);
        let external = self.external().ok()?;
        let prediction = evaluate(self.problem.model(), &self.problem.x, &external).ok()?;
        let residuals = objective_residuals(
            &self.problem.y,
            &prediction,
            self.problem.weights.as_deref(),
            self.objective,
        );
        let cost = objective_value(&residuals);
        if !cost.is_finite() {
            return None;
        }
        if cost < self.budget.best.borrow().1 {
            *self.budget.best.borrow_mut() = (self.parameters.as_slice().to_vec(), cost);
        }
        Some(DVector::from_vec(residuals))
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        let external = self.external().ok()?;
        let prediction = evaluate(self.problem.model(), &self.problem.x, &external).ok()?;
        let mut jacobian = physical_jacobian(self.problem, self.layout, &external).ok()?;
        residual_jacobian(self.problem, self.objective, &prediction, &mut jacobian);
        for (column, entry) in self.layout.free.iter().enumerate() {
            jacobian.column_mut(column).scale_mut(transform_gradient(
                self.parameters[column],
                self.layout.entries[*entry].definition.bounds,
            ));
        }
        jacobian
            .iter()
            .all(|value| value.is_finite())
            .then_some(jacobian)
    }
}

fn evaluate(model: &dyn Model, x: &[f64], values: &ParameterValues) -> Result<Vec<f64>, FitError> {
    let mut output = checked_zeros(x.len())?;
    model.evaluate(x, values, &mut output)?;
    if output.iter().any(|value| !value.is_finite()) {
        return Err(FitError::NonFinite {
            context: "robust model evaluation".to_owned(),
        });
    }
    Ok(output)
}

fn from_unclipped(internal: f64, bounds: Bounds) -> f64 {
    let lower = bounds.lower.lower_value();
    let upper = bounds.upper.upper_value();
    match (lower.is_finite(), upper.is_finite()) {
        (false, false) => internal,
        (true, false) => lower + internal * (internal / (internal.hypot(1.0) + 1.0)),
        (false, true) => upper - internal * (internal / (internal.hypot(1.0) + 1.0)),
        (true, true) => lower + ((internal.sin() + 1.0) / 2.0) * (upper - lower),
    }
}

fn to_unclipped(value: f64, bounds: Bounds) -> f64 {
    let lower = bounds.lower.lower_value();
    let upper = bounds.upper.upper_value();
    match (lower.is_finite(), upper.is_finite()) {
        (false, false) => value,
        (true, false) => (value - lower).sqrt() * (value - lower + 2.0).sqrt(),
        (false, true) => (upper - value).sqrt() * (upper - value + 2.0).sqrt(),
        (true, true) => (2.0 * ((value - lower) / (upper - lower)) - 1.0)
            .clamp(-1.0, 1.0)
            .asin(),
    }
}

fn transform_gradient(internal: f64, bounds: Bounds) -> f64 {
    let lower = bounds.lower.lower_value();
    let upper = bounds.upper.upper_value();
    match (lower.is_finite(), upper.is_finite()) {
        (false, false) => 1.0,
        (true, false) => internal / internal.hypot(1.0),
        (false, true) => -internal / internal.hypot(1.0),
        (true, true) => internal.cos() * (upper - lower) / 2.0,
    }
}

fn parameter_scale(value: f64, bounds: Bounds) -> f64 {
    let span = bounds.upper.upper_value() - bounds.lower.lower_value();
    if span.is_finite() {
        span.min(value.abs().max(1.0))
    } else {
        value.abs().max(1.0)
    }
}

fn interior(value: f64, bounds: Bounds) -> f64 {
    let lower = bounds.lower.lower_value();
    let upper = bounds.upper.upper_value();
    let margin = 1.0e-5 * parameter_scale(value, bounds);
    let minimum = (lower + margin).max(lower.next_up());
    let maximum = (upper - margin).min(upper.next_down());
    if minimum <= maximum {
        value.max(minimum).min(maximum)
    } else {
        value
    }
}

fn internal_seed(layout: &ParameterLayout, values: &ParameterValues) -> Result<Vec<f64>, FitError> {
    layout
        .free
        .iter()
        .map(|entry| {
            let definition = &layout.entries[*entry].definition;
            let value = values.require(&definition.name)?;
            definition.bounds.validate(value, &definition.name)?;
            Ok(to_unclipped(
                interior(value, definition.bounds),
                definition.bounds,
            ))
        })
        .collect()
}

fn physical_jacobian(
    problem: &FitProblem,
    layout: &ParameterLayout,
    values: &ParameterValues,
) -> Result<DMatrix<f64>, FitError> {
    let rows = problem.x.len();
    let columns = layout.free.len();
    let names = layout
        .free
        .iter()
        .map(|entry| layout.entries[*entry].definition.name.clone())
        .collect::<Vec<_>>();
    let mut analytic = checked_zeros(rows.saturating_mul(columns))?;
    if problem
        .model
        .robust_jacobian(&problem.x, values, &names, &mut analytic)?
        && analytic.iter().all(|value| value.is_finite())
    {
        return Ok(DMatrix::from_row_slice(rows, columns, &analytic));
    }
    let mut jacobian = DMatrix::zeros(rows, columns);
    for (column, entry) in layout.free.iter().enumerate() {
        let definition = &layout.entries[*entry].definition;
        let value = values.require(&definition.name)?;
        let step = (f64::EPSILON.cbrt() * parameter_scale(value, definition.bounds))
            .max(8.0 * f64::EPSILON * value.abs());
        let plus = (value + step).min(definition.bounds.upper.upper_value());
        let minus = (value - step).max(definition.bounds.lower.lower_value());
        if plus <= minus {
            return Err(FitError::Solver {
                message: format!("cannot resolve derivative of {}", definition.name),
            });
        }
        let plus_curve = evaluate(
            problem.model(),
            &problem.x,
            &layout.set_free_external(values, column, plus),
        )?;
        let minus_curve = evaluate(
            problem.model(),
            &problem.x,
            &layout.set_free_external(values, column, minus),
        )?;
        for row in 0..rows {
            jacobian[(row, column)] = (plus_curve[row] - minus_curve[row]) / (plus - minus);
        }
    }
    if jacobian.iter().any(|value| !value.is_finite()) {
        return Err(FitError::NonFinite {
            context: "physical model derivatives".to_owned(),
        });
    }
    Ok(jacobian)
}

fn residual_jacobian(
    problem: &FitProblem,
    objective: ObjectiveKind,
    prediction: &[f64],
    jacobian: &mut DMatrix<f64>,
) {
    for row in 0..jacobian.nrows() {
        let weight = problem.weights.as_ref().map_or(1.0, |weights| weights[row]);
        let gradient = match objective {
            ObjectiveKind::LeastSquares => -weight,
            ObjectiveKind::PoissonDeviance => {
                weight * poisson_deviance_residual_gradient(problem.y[row], prediction[row])
            }
        };
        jacobian.row_mut(row).scale_mut(gradient);
    }
}

fn near_bounds(value: f64, bounds: Bounds) -> (bool, bool) {
    let tolerance =
        f64::EPSILON.sqrt() * parameter_scale(value, bounds) + 8.0 * f64::EPSILON * value.abs();
    (
        value - bounds.lower.lower_value() <= tolerance,
        bounds.upper.upper_value() - value <= tolerance,
    )
}

struct Inspection {
    values: ParameterValues,
    prediction: Vec<f64>,
    residuals: Vec<f64>,
    diagnostics: FitDiagnostics,
    covariance: Option<DMatrix<f64>>,
}

impl Inspection {
    fn stationary(&self) -> bool {
        self.diagnostics
            .optimality
            .is_some_and(|value| value <= STATIONARITY_TOLERANCE)
    }

    fn needs_recovery(&self, options: &FitOptions) -> bool {
        !self.stationary()
            || (options.calculate_covariance
                && matches!(
                    self.diagnostics.covariance_status,
                    CovarianceStatus::RankDeficient
                        | CovarianceStatus::NumericalFailure
                        | CovarianceStatus::ActiveBounds
                ))
    }
}

fn projected_optimality(
    layout: &ParameterLayout,
    values: &ParameterValues,
    objective_jacobian: &DMatrix<f64>,
    residuals: &[f64],
    diagnostics: &mut FitDiagnostics,
) -> Result<f64, FitError> {
    let residual = DVector::from_column_slice(residuals);
    let gradient = objective_jacobian.transpose() * &residual;
    let mut optimality = 0.0_f64;
    for (column, entry) in layout.free.iter().enumerate() {
        let definition = &layout.entries[*entry].definition;
        let (lower, upper) = near_bounds(values.require(&definition.name)?, definition.bounds);
        if lower || upper {
            diagnostics
                .affected_parameters
                .push(definition.name.clone());
        }
        let scale = objective_jacobian.column(column).norm() * residual.norm().max(1.0);
        if !scale.is_finite() || !gradient[column].is_finite() {
            optimality = f64::INFINITY;
            if !(lower || upper) {
                diagnostics
                    .affected_parameters
                    .push(definition.name.clone());
            }
            continue;
        }
        if (lower && gradient[column] >= 0.0) || (upper && gradient[column] <= 0.0) {
            continue;
        }
        let projected = if scale > 0.0 {
            gradient[column].abs() / scale
        } else {
            0.0
        };
        optimality = optimality.max(projected);
        if projected > STATIONARITY_TOLERANCE && !(lower || upper) {
            diagnostics
                .affected_parameters
                .push(definition.name.clone());
        }
    }
    Ok(optimality)
}

fn inspect(
    problem: &FitProblem,
    options: &FitOptions,
    layout: &ParameterLayout,
    internal: &[f64],
) -> Result<Inspection, FitError> {
    let values = layout.values_with_transform(internal, from_unclipped)?;
    let prediction = evaluate(problem.model(), &problem.x, &values)?;
    let residuals = objective_residuals(
        &problem.y,
        &prediction,
        problem.weights.as_deref(),
        options.objective,
    );
    let mut diagnostics = FitDiagnostics::default();
    let mut covariance = None;
    if layout.free.is_empty() {
        diagnostics.optimality = Some(0.0);
        diagnostics.covariance_status = CovarianceStatus::NoFreeParameters;
    } else if let Ok(mut jacobian) = physical_jacobian(problem, layout, &values) {
        let mut objective_jacobian = jacobian.clone();
        residual_jacobian(
            problem,
            options.objective,
            &prediction,
            &mut objective_jacobian,
        );
        let optimality = projected_optimality(
            layout,
            &values,
            &objective_jacobian,
            &residuals,
            &mut diagnostics,
        )?;
        diagnostics.optimality = optimality.is_finite().then_some(optimality);
        if !options.calculate_covariance {
            diagnostics.covariance_status = CovarianceStatus::Disabled;
        } else if optimality > STATIONARITY_TOLERANCE || !optimality.is_finite() {
            diagnostics.covariance_status = CovarianceStatus::NotConverged;
        } else if !diagnostics.affected_parameters.is_empty() {
            diagnostics.covariance_status = CovarianceStatus::ActiveBounds;
        } else if options.objective == ObjectiveKind::LeastSquares
            && problem.x.len() == layout.free.len()
        {
            diagnostics.covariance_status = CovarianceStatus::InsufficientInformation;
        } else {
            for row in 0..jacobian.nrows() {
                let weight = problem.weights.as_ref().map_or(1.0, |weights| weights[row]);
                let factor = match options.objective {
                    ObjectiveKind::LeastSquares => weight,
                    ObjectiveKind::PoissonDeviance => {
                        weight / prediction[row].max(POISSON_FLOOR).sqrt()
                    }
                };
                jacobian.row_mut(row).scale_mut(factor);
            }
            let scale = match options.objective {
                ObjectiveKind::LeastSquares => {
                    objective_value(&residuals) / (problem.x.len() - layout.free.len()) as f64
                }
                ObjectiveKind::PoissonDeviance => 1.0,
            };
            covariance = information_inverse(jacobian, scale, layout, &mut diagnostics);
        }
    } else {
        diagnostics.covariance_status = CovarianceStatus::NumericalFailure;
        diagnostics.affected_parameters = layout
            .free
            .iter()
            .map(|entry| layout.entries[*entry].definition.name.clone())
            .collect();
    }
    // Poisson information is meaningful only for positive model means.
    if options.objective == ObjectiveKind::PoissonDeviance
        && prediction
            .iter()
            .zip(&problem.y)
            .any(|(mean, observed)| *mean < 0.0 || (*mean == 0.0 && *observed > 0.0))
    {
        diagnostics.covariance_status = CovarianceStatus::NumericalFailure;
        diagnostics.optimality = None;
        covariance = None;
    }
    Ok(Inspection {
        values,
        prediction,
        residuals,
        diagnostics,
        covariance,
    })
}

fn information_inverse(
    mut jacobian: DMatrix<f64>,
    scale: f64,
    layout: &ParameterLayout,
    diagnostics: &mut FitDiagnostics,
) -> Option<DMatrix<f64>> {
    diagnostics.covariance_status = CovarianceStatus::NumericalFailure;
    if !scale.is_finite() || jacobian.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let scales = (0..jacobian.ncols())
        .map(|column| jacobian.column(column).norm())
        .collect::<Vec<_>>();
    if scales.iter().any(|scale| !scale.is_finite()) {
        return None;
    }
    for (column, norm) in scales.iter().enumerate() {
        if *norm > 0.0 && norm.is_finite() {
            jacobian.column_mut(column).scale_mut(1.0 / norm);
        }
    }
    let threshold_factor = f64::EPSILON * jacobian.nrows().max(jacobian.ncols()) as f64;
    let svd = jacobian.svd(false, true);
    let largest = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let threshold = threshold_factor * largest;
    let rank = svd
        .singular_values
        .iter()
        .filter(|value| value.is_finite() && **value > threshold)
        .count();
    diagnostics.rank = Some(rank);
    let vectors = svd.v_t?;
    if rank < layout.free.len() || scales.iter().any(|scale| *scale <= 0.0) {
        diagnostics.covariance_status = CovarianceStatus::RankDeficient;
        for (column, entry) in layout.free.iter().enumerate() {
            if scales[column] == 0.0
                || svd
                    .singular_values
                    .iter()
                    .enumerate()
                    .any(|(row, singular)| {
                        *singular <= threshold && vectors[(row, column)].abs() > 0.1
                    })
            {
                diagnostics
                    .affected_parameters
                    .push(layout.entries[*entry].definition.name.clone());
            }
        }
        return None;
    }
    let inverse_squares =
        DMatrix::from_diagonal(&svd.singular_values.map(|value| scale / value.powi(2)));
    let mut covariance = vectors.transpose() * inverse_squares * vectors;
    for row in 0..covariance.nrows() {
        for column in 0..covariance.ncols() {
            covariance[(row, column)] /= scales[row] * scales[column];
        }
    }
    if covariance.iter().any(|value| !value.is_finite()) {
        return None;
    }
    diagnostics.covariance_status = CovarianceStatus::Available;
    Some(covariance)
}

struct Search<'a> {
    problem: &'a FitProblem,
    options: &'a FitOptions,
    layout: &'a ParameterLayout,
    budget: Budget,
    attempts: usize,
}

impl Search<'_> {
    fn run(&mut self, seed: Vec<f64>, allowance: usize) {
        if self.budget.used.get() >= self.budget.limit || allowance == 0 {
            return;
        }
        let target = Target {
            problem: self.problem,
            layout: self.layout,
            parameters: DVector::from_vec(seed),
            objective: self.options.objective,
            budget: &self.budget,
            stop_at: self.budget.used.get().saturating_add(allowance),
        };
        let solver = LevenbergMarquardt::new()
            .with_ftol(self.options.ftol)
            .with_xtol(self.options.xtol)
            .with_gtol(self.options.gtol)
            .with_stepbound(self.options.step_bound)
            .with_patience(self.options.evaluation_patience)
            .with_scale_diag(true);
        let (_returned, _report) = solver.minimize(target);
        self.attempts += 1;
    }

    fn remaining(&self) -> usize {
        self.budget.limit.saturating_sub(self.budget.used.get())
    }

    fn best(&self) -> Result<Inspection, FitError> {
        inspect(
            self.problem,
            self.options,
            self.layout,
            &self.budget.best.borrow().0,
        )
    }

    fn polish(&mut self, allowance: usize) -> Result<(), FitError> {
        let values = self
            .layout
            .values_with_transform(&self.budget.best.borrow().0, from_unclipped)?;
        self.run(internal_seed(self.layout, &values)?, allowance);
        Ok(())
    }
}

pub(super) fn fit(
    problem: &FitProblem,
    options: &FitOptions,
    starts: &[ParameterValues],
) -> Result<FitResult, FitError> {
    validate_problem(problem, options)?;
    let layout = ParameterLayout::new(problem.model.parameter_definitions())?;
    if problem.x.len() < layout.free.len() {
        return Err(FitError::InsufficientDegreesOfFreedom {
            observations: problem.x.len(),
            variables: layout.free.len(),
        });
    }
    let initial_coordinates = layout
        .free
        .iter()
        .map(|entry| {
            let definition = &layout.entries[*entry].definition;
            to_unclipped(definition.initial, definition.bounds)
        })
        .collect::<Vec<_>>();
    let initial = layout.values_with_transform(&initial_coordinates, from_unclipped)?;
    let prediction = evaluate(problem.model(), &problem.x, &initial)?;
    let initial_objective = objective_value(&objective_residuals(
        &problem.y,
        &prediction,
        problem.weights.as_deref(),
        options.objective,
    ));
    if !initial_objective.is_finite() {
        return Err(FitError::NonFinite {
            context: "initial fit objective".to_owned(),
        });
    }
    let mut search = Search {
        problem,
        options,
        layout: &layout,
        budget: Budget {
            used: Cell::new(0),
            limit: options
                .evaluation_patience
                .saturating_mul(layout.free.len() + 1),
            best: RefCell::new((initial_coordinates, initial_objective)),
        },
        attempts: 0,
    };
    if !layout.free.is_empty() {
        search.run(
            internal_seed(&layout, &initial)?,
            (search.remaining() / 2).max(1),
        );
        search.polish((search.remaining() / 4).max(1))?;
        if search.best()?.needs_recovery(options) {
            for (index, values) in starts.iter().enumerate() {
                let allowance = search.remaining() / (starts.len() - index + 1);
                search.run(internal_seed(&layout, values)?, allowance);
            }
        }
        // Polishing preserves every fitted background coefficient along with the peaks.
        search.polish(search.remaining())?;
    }
    let selected = search.best()?;
    finish(
        problem,
        options,
        &layout,
        selected,
        initial_objective,
        &search,
    )
}

fn finish(
    problem: &FitProblem,
    options: &FitOptions,
    layout: &ParameterLayout,
    mut selected: Inspection,
    initial_objective: f64,
    search: &Search<'_>,
) -> Result<FitResult, FitError> {
    let stationary = selected.stationary();
    selected.diagnostics.attempts = search.attempts;
    selected.diagnostics.budget_exhausted = search.remaining() == 0;
    let statistics = statistics(
        &problem.y,
        &selected.prediction,
        &selected.residuals,
        layout.free.len(),
        search.budget.used.get(),
        options.objective,
        initial_objective,
    );
    let mut parameters = base_estimates(layout, &selected.values, selected.covariance.as_ref());
    for parameter in &mut parameters {
        if parameter.kind == ParameterKind::Free {
            let (lower, upper) = near_bounds(parameter.value, parameter.bounds);
            parameter.active_bound = lower || upper;
        }
    }
    parameters.extend(derived_estimates(
        layout,
        &problem.model.derived_parameters(&selected.values)?,
        selected.covariance.as_ref(),
    ));
    let evaluation_x = options.evaluation_x.as_deref().unwrap_or(&problem.x);
    check_allocation(evaluation_x.len(), layout.free.len().max(1))?;
    let best_fit = evaluate(problem.model(), evaluation_x, &selected.values)?;
    let components = problem.model.components(evaluation_x, &selected.values)?;
    let (confidence_band, component_bands) = if let Some(matrix) = &selected.covariance {
        match confidence_bands(
            problem.model(),
            evaluation_x,
            &selected.values,
            layout,
            matrix,
            &statistics,
            options.objective,
            options.confidence_sigma,
            &best_fit,
            &components,
            super::SolverProfile::Robust,
        ) {
            Ok((total, components)) => (Some(total), components),
            Err(error) => {
                selected.diagnostics.confidence_band_error = Some(error.to_string());
                (None, Vec::new())
            }
        }
    } else {
        (None, Vec::new())
    };
    Ok(FitResult {
        termination: TerminationStatus {
            success: stationary,
            reason: if stationary {
                "stationary"
            } else if search.remaining() == 0 {
                "max_evaluations"
            } else {
                "not_stationary"
            }
            .to_owned(),
            message: if stationary {
                "physical projected gradient is stationary"
            } else {
                "best finite result retained; physical stationarity was not reached"
            }
            .to_owned(),
        },
        parameters,
        statistics,
        raw_residuals: problem
            .y
            .iter()
            .zip(&selected.prediction)
            .map(|(y, model)| y - model)
            .collect(),
        residuals: selected.residuals,
        observation_x: problem.x.clone(),
        diagnostics: selected.diagnostics,
        evaluation_x: evaluation_x.to_vec(),
        best_fit,
        components,
        covariance: selected
            .covariance
            .as_ref()
            .map(|matrix| covariance_public(layout, matrix)),
        confidence_band,
        component_bands,
    })
}
