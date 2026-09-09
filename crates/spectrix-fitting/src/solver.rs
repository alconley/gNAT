use std::collections::{BTreeMap, BTreeSet};

use levenberg_marquardt::{
    LeastSquaresProblem, LevenbergMarquardt, MinimizationReport, TerminationReason,
};
use nalgebra::{DMatrix, DVector, Dyn, storage::Owned};
use statrs::{
    distribution::{ChiSquared, ContinuousCDF as _, StudentsT},
    function::erf::erf,
};

use crate::{
    Bounds, ComponentCurve, DerivedParameter, FitError, Model, ParameterBinding,
    ParameterDefinition, ParameterEstimate, ParameterKind, ParameterValues,
};

const MAX_VALUES: usize = 16_777_216;

mod robust;

/// Numerical compatibility profiles offered by the fitting engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolverProfile {
    /// lmfit 1.3.4 `Model.fit(method="leastsq")` compatible settings.
    #[default]
    Lmfit134,
    /// Physical-coordinate diagnostics, stable bounds, and automatic polishing.
    Robust,
}

/// Residual objective minimized by the nonlinear solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum ObjectiveKind {
    /// Minimize the weighted squared difference between observations and the model.
    #[default]
    LeastSquares,
    /// Minimize the Poisson deviance for non-negative count observations.
    PoissonDeviance,
}

/// Configuration for a nonlinear least-squares or Poisson-deviance fit.
#[derive(Debug, Clone, PartialEq)]
pub struct FitOptions {
    /// Numerical compatibility profile.
    pub profile: SolverProfile,
    /// Residual objective used by the solver.
    pub objective: ObjectiveKind,
    /// Maximum evaluation factor; the effective limit is this value times `nvarys + 1`.
    pub evaluation_patience: usize,
    /// Relative objective-function tolerance.
    pub ftol: f64,
    /// Relative parameter tolerance.
    pub xtol: f64,
    /// Residual/Jacobian orthogonality tolerance.
    pub gtol: f64,
    /// Initial trust-region step factor.
    pub step_bound: f64,
    /// MINPACK forward-difference control.
    pub epsfcn: f64,
    /// Gaussian-equivalent sigma used for confidence bands.
    pub confidence_sigma: f64,
    /// Whether to calculate covariance, propagated uncertainties, and confidence bands.
    pub calculate_covariance: bool,
    /// Optional grid for reported curves and confidence bands.
    pub evaluation_x: Option<Vec<f64>>,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            profile: SolverProfile::Lmfit134,
            objective: ObjectiveKind::LeastSquares,
            evaluation_patience: 2_000,
            ftol: 1.5e-8,
            xtol: 1.5e-8,
            gtol: 0.0,
            step_bound: 100.0,
            epsfcn: 1.0e-10,
            confidence_sigma: 1.0,
            calculate_covariance: true,
            evaluation_x: None,
        }
    }
}

impl FitOptions {
    /// Recommended settings for new fits; compatibility defaults remain unchanged.
    #[must_use]
    pub fn robust() -> Self {
        Self {
            profile: SolverProfile::Robust,
            ftol: 1.0e-13,
            xtol: 1.0e-13,
            gtol: 1.0e-8,
            ..Self::default()
        }
    }
}

/// A model, observations, and optional residual weights.
pub struct FitProblem {
    model: Box<dyn Model>,
    x: Vec<f64>,
    y: Vec<f64>,
    weights: Option<Vec<f64>>,
}

impl FitProblem {
    /// Creates an unweighted fit problem.
    #[must_use]
    pub fn new(model: Box<dyn Model>, x: Vec<f64>, y: Vec<f64>) -> Self {
        Self {
            model,
            x,
            y,
            weights: None,
        }
    }

    /// Adds multiplicative residual weights.
    #[must_use]
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Returns the independent data.
    #[must_use]
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    /// Returns the dependent data.
    #[must_use]
    pub fn y(&self) -> &[f64] {
        &self.y
    }

    /// Returns the model.
    #[must_use]
    pub fn model(&self) -> &dyn Model {
        self.model.as_ref()
    }
}

/// Scalar fit statistics for the selected objective.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct FitStatistics {
    /// Objective used to produce this fit.
    #[cfg_attr(feature = "serde", serde(default))]
    pub objective: ObjectiveKind,
    /// Number of fitted observations.
    pub observations: usize,
    /// Number of independently varied parameters.
    pub variables: usize,
    /// Remaining degrees of freedom.
    pub degrees_of_freedom: usize,
    /// Weighted sum of squared residuals.
    pub chi_square: f64,
    /// Chi-square divided by `max(1, degrees_of_freedom)`. With zero degrees
    /// of freedom this compatibility field is not a residual variance estimate.
    pub reduced_chi_square: f64,
    /// Selected objective evaluated at the exact user-supplied starting values.
    #[cfg_attr(feature = "serde", serde(default))]
    pub initial_objective: Option<f64>,
    /// Selected objective evaluated at the returned parameters.
    #[cfg_attr(feature = "serde", serde(default))]
    pub final_objective: f64,
    /// Fractional objective reduction `(initial - final) / initial`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub objective_improvement: Option<f64>,
    /// Root-mean-square unweighted difference between observations and the model.
    #[cfg_attr(feature = "serde", serde(default))]
    pub rmse: f64,
    /// Poisson deviance when [`ObjectiveKind::PoissonDeviance`] was used.
    #[cfg_attr(feature = "serde", serde(default))]
    pub deviance: Option<f64>,
    /// Poisson deviance divided by the degrees of freedom.
    #[cfg_attr(feature = "serde", serde(default))]
    pub reduced_deviance: Option<f64>,
    /// Pearson chi-square for a Poisson fit.
    #[cfg_attr(feature = "serde", serde(default))]
    pub pearson_chi_square: Option<f64>,
    /// Pearson chi-square divided by the degrees of freedom for a Poisson fit.
    #[cfg_attr(feature = "serde", serde(default))]
    pub reduced_pearson_chi_square: Option<f64>,
    /// Chi-square survival probability for the final Poisson deviance.
    #[cfg_attr(feature = "serde", serde(default))]
    pub goodness_of_fit_p_value: Option<f64>,
    /// Akaike information criterion.
    pub aic: Option<f64>,
    /// Bayesian information criterion.
    pub bic: Option<f64>,
    /// Coefficient of determination on unweighted observations.
    pub r_squared: Option<f64>,
    /// Number of residual evaluations performed by the optimizer.
    pub evaluations: usize,
}

/// Dense covariance and correlation matrices in row-major public form.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Covariance {
    /// Parameter ordering for both matrices.
    pub parameter_names: Vec<String>,
    /// Physical parameter covariance: reduced-chi-square scaled for least squares,
    /// and likelihood information based for Poisson fits.
    pub matrix: Vec<Vec<f64>>,
    /// Correlation matrix.
    pub correlations: Vec<Vec<f64>>,
}

/// Why symmetric parameter uncertainties are available or withheld.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum CovarianceStatus {
    /// A saved result predates structured covariance diagnostics.
    #[default]
    Unknown,
    /// The converged, interior model has full numerical rank.
    Available,
    /// Covariance was not requested.
    Disabled,
    /// There are no independently varying parameters.
    NoFreeParameters,
    /// The parameters can be fitted, but no residual degrees of freedom remain
    /// to estimate least-squares noise variance.
    InsufficientInformation,
    /// The returned solution is not stationary.
    NotConverged,
    /// One or more parameters are limited by their bounds.
    ActiveBounds,
    /// Observations cannot distinguish all the varied parameters.
    RankDeficient,
    /// A derivative or matrix calculation could not be evaluated reliably.
    NumericalFailure,
}

/// Additional diagnostics for robust fits, defaulted when reading older results.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct FitDiagnostics {
    /// Number of optimizer runs, including polishing and recovery.
    pub attempts: usize,
    /// Maximum scaled projected gradient at the selected solution.
    pub optimality: Option<f64>,
    /// Numerical rank of the scaled information Jacobian, when available.
    pub rank: Option<usize>,
    /// Explanation of covariance availability.
    pub covariance_status: CovarianceStatus,
    /// Parameter names involved in boundary or rank failures.
    pub affected_parameters: Vec<String>,
    /// Whether the shared residual-evaluation budget was exhausted.
    pub budget_exhausted: bool,
    /// Reason confidence bands could not be propagated, without discarding the fit.
    pub confidence_band_error: Option<String>,
}

/// Why the solver stopped and whether the estimate is considered successful.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct TerminationStatus {
    /// Stable machine-readable reason.
    pub reason: String,
    /// Human-readable detail.
    pub message: String,
    /// Whether the termination is classified as successful.
    pub success: bool,
}

/// A covariance-based confidence band on an explicit grid.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ConfidenceBand {
    /// Evaluation grid.
    pub x: Vec<f64>,
    /// Best-fit curve.
    pub best_fit: Vec<f64>,
    /// Student-t-scaled least-squares or normal-scaled Poisson uncertainty.
    pub uncertainty: Vec<f64>,
    /// Lower confidence curve.
    pub lower: Vec<f64>,
    /// Upper confidence curve.
    pub upper: Vec<f64>,
}

/// Complete structured result of a nonlinear fit.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct FitResult {
    /// Solver termination information.
    pub termination: TerminationStatus,
    /// Fitted, fixed, shared, and derived parameter estimates.
    pub parameters: Vec<ParameterEstimate>,
    /// Fit statistics.
    pub statistics: FitStatistics,
    /// Weighted residuals on the observation grid (`data - model`).
    pub residuals: Vec<f64>,
    /// Untransformed residuals on the observation grid (`data - model`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub raw_residuals: Vec<f64>,
    /// Coordinates of the observations, including any external background windows.
    #[cfg_attr(feature = "serde", serde(default))]
    pub observation_x: Vec<f64>,
    /// Structured numerical diagnostics; absent from older saved fits.
    #[cfg_attr(feature = "serde", serde(default))]
    pub diagnostics: FitDiagnostics,
    /// Grid used for `best_fit`, components, and confidence bands.
    pub evaluation_x: Vec<f64>,
    /// Best-fit total curve on the evaluation grid.
    pub best_fit: Vec<f64>,
    /// Named component curves on the evaluation grid.
    pub components: Vec<ComponentCurve>,
    /// Covariance and correlations, absent when the normal matrix is singular.
    pub covariance: Option<Covariance>,
    /// Total-model confidence band, absent when covariance is unavailable.
    pub confidence_band: Option<ConfidenceBand>,
    /// Per-component confidence bands, absent when covariance is unavailable.
    pub component_bands: Vec<(String, ConfidenceBand)>,
}

/// Fits a validated problem and returns all estimates, diagnostics, and bands.
#[expect(
    clippy::too_many_lines,
    reason = "the top-level fit pipeline is intentionally linear and auditable"
)]
pub fn fit(problem: &FitProblem, options: &FitOptions) -> Result<FitResult, FitError> {
    if options.profile == SolverProfile::Robust {
        return robust::fit(problem, options, &[]);
    }
    validate_problem(problem, options)?;
    let definitions = problem.model.parameter_definitions();
    let layout = ParameterLayout::new(definitions)?;
    if problem.x.len() <= layout.free.len() {
        return Err(FitError::InsufficientDegreesOfFreedom {
            observations: problem.x.len(),
            variables: layout.free.len(),
        });
    }

    let evaluation_x = options
        .evaluation_x
        .as_deref()
        .unwrap_or(problem.x.as_slice());
    check_allocation(evaluation_x.len(), layout.free.len().max(1))?;

    let initial_internal = layout.initial_internal();
    let mut target = InternalProblem {
        parameters: DVector::from_vec(initial_internal),
        model: problem.model.as_ref(),
        x: &problem.x,
        y: &problem.y,
        weights: problem.weights.as_deref(),
        objective: options.objective,
        layout: &layout,
        epsfcn: options.epsfcn,
    };

    let initial_values = layout.external_values(target.parameters.as_slice())?;
    let mut initial_curve = checked_zeros(problem.x.len())?;
    problem
        .model
        .evaluate(&problem.x, &initial_values, &mut initial_curve)?;
    let initial_residuals = objective_residuals(
        &problem.y,
        &initial_curve,
        problem.weights.as_deref(),
        options.objective,
    );
    let initial_objective = objective_value(&initial_residuals);

    let (report, evaluations) = if layout.free.is_empty() {
        (None, 1)
    } else {
        let solver = LevenbergMarquardt::new()
            .with_ftol(options.ftol)
            .with_xtol(options.xtol)
            .with_gtol(options.gtol)
            .with_stepbound(options.step_bound)
            .with_patience(options.evaluation_patience)
            .with_scale_diag(true);
        let (returned, report) = solver.minimize(target);
        target = returned;
        let evaluations = report.number_of_evaluations;
        (Some(report), evaluations)
    };

    let external = layout.external_values(target.parameters.as_slice())?;
    let mut fitted_observations = checked_zeros(problem.x.len())?;
    problem
        .model
        .evaluate(&problem.x, &external, &mut fitted_observations)?;
    let raw_residuals = problem
        .y
        .iter()
        .zip(&fitted_observations)
        .map(|(data, model)| data - model)
        .collect::<Vec<_>>();
    let residuals = objective_residuals(
        &problem.y,
        &fitted_observations,
        problem.weights.as_deref(),
        options.objective,
    );
    let statistics = statistics(
        &problem.y,
        &fitted_observations,
        &residuals,
        layout.free.len(),
        evaluations,
        options.objective,
        initial_objective,
    );

    let covariance_internal = if !options.calculate_covariance || layout.free.is_empty() {
        None
    } else {
        target.jacobian().and_then(|jacobian| {
            let scale = if options.objective == ObjectiveKind::PoissonDeviance {
                1.0
            } else {
                statistics.reduced_chi_square
            };
            invert_normal_matrix(&jacobian, scale)
        })
    };
    let covariance_external = covariance_internal
        .as_ref()
        .map(|internal| layout.external_covariance(target.parameters.as_slice(), internal));
    let covariance = covariance_external
        .as_ref()
        .map(|matrix| covariance_public(&layout, matrix));

    let mut parameters = base_estimates(&layout, &external, covariance_external.as_ref());
    let derived = problem.model.derived_parameters(&external)?;
    parameters.extend(derived_estimates(
        &layout,
        &derived,
        covariance_external.as_ref(),
    ));

    let mut best_fit = checked_zeros(evaluation_x.len())?;
    problem
        .model
        .evaluate(evaluation_x, &external, &mut best_fit)?;
    let components = problem.model.components(evaluation_x, &external)?;
    let (confidence_band, component_bands) = covariance_external.as_ref().map_or_else(
        || (None, Vec::new()),
        |matrix| {
            confidence_bands(
                problem.model.as_ref(),
                evaluation_x,
                &external,
                &layout,
                matrix,
                &statistics,
                options.objective,
                options.confidence_sigma,
                &best_fit,
                &components,
                SolverProfile::Lmfit134,
            )
            .map_or_else(|_| (None, Vec::new()), |(total, all)| (Some(total), all))
        },
    );

    Ok(FitResult {
        termination: termination_status(report.as_ref()),
        parameters,
        statistics,
        residuals,
        raw_residuals,
        observation_x: problem.x.clone(),
        diagnostics: FitDiagnostics::default(),
        evaluation_x: evaluation_x.to_vec(),
        best_fit,
        components,
        covariance,
        confidence_band,
        component_bands,
    })
}

pub(crate) fn fit_with_starts(
    problem: &FitProblem,
    options: &FitOptions,
    starts: &[ParameterValues],
) -> Result<FitResult, FitError> {
    if options.profile == SolverProfile::Robust {
        robust::fit(problem, options, starts)
    } else {
        fit(problem, options)
    }
}

fn validate_problem(problem: &FitProblem, options: &FitOptions) -> Result<(), FitError> {
    if problem.x.len() != problem.y.len() {
        return Err(FitError::LengthMismatch {
            x: problem.x.len(),
            y: problem.y.len(),
        });
    }
    if problem.x.is_empty() {
        return Err(FitError::EmptyData);
    }
    if let Some(weights) = &problem.weights
        && weights.len() != problem.x.len()
    {
        return Err(FitError::LengthMismatch {
            x: problem.x.len(),
            y: weights.len(),
        });
    }
    if problem
        .x
        .iter()
        .chain(&problem.y)
        .any(|value| !value.is_finite())
        || problem
            .weights
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
    {
        return Err(FitError::NonFinite {
            context: "fit inputs".to_owned(),
        });
    }
    if options.objective == ObjectiveKind::PoissonDeviance
        && problem.y.iter().any(|value| *value < 0.0)
    {
        return Err(FitError::Domain {
            model: "poisson objective".to_owned(),
            message: "observations must be non-negative".to_owned(),
        });
    }
    if options.evaluation_patience == 0
        || !options.ftol.is_finite()
        || options.ftol < 0.0
        || !options.xtol.is_finite()
        || options.xtol < 0.0
        || !options.gtol.is_finite()
        || options.gtol < 0.0
        || !options.step_bound.is_finite()
        || options.step_bound <= 0.0
        || !options.epsfcn.is_finite()
        || options.epsfcn <= 0.0
        || !options.confidence_sigma.is_finite()
        || options.confidence_sigma <= 0.0
    {
        return Err(FitError::Solver {
            message: "invalid solver options".to_owned(),
        });
    }
    if let Some(evaluation_x) = &options.evaluation_x
        && evaluation_x.iter().any(|value| !value.is_finite())
    {
        return Err(FitError::NonFinite {
            context: "evaluation grid".to_owned(),
        });
    }
    check_allocation(
        problem.x.len(),
        problem.model.parameter_definitions().len().max(1),
    )
}

#[derive(Debug, Clone)]
struct LayoutEntry {
    definition: ParameterDefinition,
    free_index: Option<usize>,
}

struct ParameterLayout {
    entries: Vec<LayoutEntry>,
    free: Vec<usize>,
    by_name: BTreeMap<String, usize>,
}

impl ParameterLayout {
    fn new(definitions: Vec<ParameterDefinition>) -> Result<Self, FitError> {
        let mut by_name = BTreeMap::new();
        for (index, definition) in definitions.iter().enumerate() {
            if definition.name.is_empty()
                || by_name.insert(definition.name.clone(), index).is_some()
            {
                return Err(FitError::InvalidParameter {
                    parameter: definition.name.clone(),
                });
            }
            definition
                .bounds
                .validate(definition.initial, &definition.name)?;
        }
        for definition in &definitions {
            if let Some(ParameterBinding::EqualTo(source)) = &definition.binding
                && (!by_name.contains_key(source) || source == &definition.name)
            {
                return Err(FitError::InvalidParameter {
                    parameter: definition.name.clone(),
                });
            }
        }
        let mut free = Vec::new();
        let entries = definitions
            .into_iter()
            .enumerate()
            .map(|(index, definition)| {
                let free_index = (definition.vary && definition.binding.is_none()).then(|| {
                    let position = free.len();
                    free.push(index);
                    position
                });
                LayoutEntry {
                    definition,
                    free_index,
                }
            })
            .collect();
        Ok(Self {
            entries,
            free,
            by_name,
        })
    }

    fn initial_internal(&self) -> Vec<f64> {
        self.free
            .iter()
            .map(|index| {
                let definition = &self.entries[*index].definition;
                to_internal(definition.initial, definition.bounds)
            })
            .collect()
    }

    fn external_values(&self, internal: &[f64]) -> Result<ParameterValues, FitError> {
        self.values_with_transform(internal, from_internal)
    }

    fn values_with_transform(
        &self,
        internal: &[f64],
        transform: fn(f64, Bounds) -> f64,
    ) -> Result<ParameterValues, FitError> {
        if internal.len() != self.free.len() {
            return Err(FitError::LengthMismatch {
                x: self.free.len(),
                y: internal.len(),
            });
        }
        let mut values = ParameterValues::new();
        for entry in &self.entries {
            if entry.definition.binding.is_none() {
                let value = entry.free_index.map_or(entry.definition.initial, |index| {
                    transform(internal[index], entry.definition.bounds)
                });
                values.insert(entry.definition.name.clone(), value);
            }
        }
        for _ in 0..self.entries.len() {
            let mut changed = false;
            for entry in &self.entries {
                if values.get(&entry.definition.name).is_some() {
                    continue;
                }
                if let Some(ParameterBinding::EqualTo(source)) = &entry.definition.binding
                    && let Some(value) = values.get(source)
                {
                    values.insert(entry.definition.name.clone(), value);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        if let Some(entry) = self
            .entries
            .iter()
            .find(|entry| values.get(&entry.definition.name).is_none())
        {
            return Err(FitError::InvalidParameter {
                parameter: entry.definition.name.clone(),
            });
        }
        Ok(values)
    }

    fn external_covariance(&self, internal: &[f64], covariance: &DMatrix<f64>) -> DMatrix<f64> {
        let mut result = covariance.clone();
        for row in 0..self.free.len() {
            let row_bounds = self.entries[self.free[row]].definition.bounds;
            let row_gradient = bound_gradient(internal[row], row_bounds);
            for column in 0..self.free.len() {
                let column_bounds = self.entries[self.free[column]].definition.bounds;
                let column_gradient = bound_gradient(internal[column], column_bounds);
                result[(row, column)] *= row_gradient * column_gradient;
            }
        }
        result
    }

    fn free_index_for_name(&self, name: &str) -> Option<usize> {
        let mut current = name;
        let mut visited = BTreeSet::new();
        loop {
            if !visited.insert(current.to_owned()) {
                return None;
            }
            let entry = self
                .by_name
                .get(current)
                .map(|index| &self.entries[*index])?;
            if let Some(index) = entry.free_index {
                return Some(index);
            }
            match &entry.definition.binding {
                Some(ParameterBinding::EqualTo(source)) => current = source,
                None => return None,
            }
        }
    }

    fn set_free_external(
        &self,
        baseline: &ParameterValues,
        free_index: usize,
        value: f64,
    ) -> ParameterValues {
        let mut values = baseline.clone();
        let source_name = &self.entries[self.free[free_index]].definition.name;
        values.insert(source_name.clone(), value);
        for entry in &self.entries {
            if let Some(ParameterBinding::EqualTo(source)) = &entry.definition.binding
                && self.free_index_for_name(source) == Some(free_index)
            {
                values.insert(entry.definition.name.clone(), value);
            }
        }
        values
    }
}

struct InternalProblem<'a> {
    parameters: DVector<f64>,
    model: &'a dyn Model,
    x: &'a [f64],
    y: &'a [f64],
    weights: Option<&'a [f64]>,
    objective: ObjectiveKind,
    layout: &'a ParameterLayout,
    epsfcn: f64,
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for InternalProblem<'_> {
    type ParameterStorage = Owned<f64, Dyn>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;

    fn set_params(&mut self, parameters: &DVector<f64>) {
        for column in 0..self.parameters.len() {
            let bounds = self.layout.entries[self.layout.free[column]]
                .definition
                .bounds;
            self.parameters[column] = clamp_internal(parameters[column], bounds);
        }
    }

    fn params(&self) -> DVector<f64> {
        self.parameters.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        let external = self
            .layout
            .external_values(self.parameters.as_slice())
            .ok()?;
        let mut predicted = vec![0.0; self.x.len()];
        self.model
            .evaluate(self.x, &external, &mut predicted)
            .ok()?;
        Some(DVector::from_vec(objective_residuals(
            self.y,
            &predicted,
            self.weights,
            self.objective,
        )))
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        let rows = self.x.len();
        let columns = self.parameters.len();
        let mut jacobian = DMatrix::zeros(rows, columns);
        let baseline_values = self
            .layout
            .external_values(self.parameters.as_slice())
            .ok()?;
        let mut baseline_curve = vec![0.0; rows];
        self.model
            .evaluate(self.x, &baseline_values, &mut baseline_curve)
            .ok()?;
        let parameter_names = self
            .layout
            .free
            .iter()
            .map(|entry| self.layout.entries[*entry].definition.name.clone())
            .collect::<Vec<_>>();
        let mut model_jacobian = vec![0.0; rows.saturating_mul(columns)];
        if self
            .model
            .analytic_jacobian(
                self.x,
                &baseline_values,
                &parameter_names,
                &mut model_jacobian,
            )
            .ok()?
        {
            for row in 0..rows {
                let weight = self.weights.map_or(1.0, |weights| weights[row]);
                let residual_gradient = match self.objective {
                    ObjectiveKind::LeastSquares => -weight,
                    ObjectiveKind::PoissonDeviance => {
                        poisson_deviance_residual_gradient(self.y[row], baseline_curve[row])
                            * weight
                    }
                };
                for column in 0..columns {
                    let bounds = self.layout.entries[self.layout.free[column]]
                        .definition
                        .bounds;
                    jacobian[(row, column)] = model_jacobian[row * columns + column]
                        * residual_gradient
                        * bound_gradient(self.parameters[column], bounds);
                }
            }
            return Some(jacobian);
        }
        let baseline_residuals =
            objective_residuals(self.y, &baseline_curve, self.weights, self.objective);
        for column in 0..columns {
            let mut step = finite_difference_step(self.parameters[column], self.epsfcn);
            let mut shifted = self.parameters.clone();
            shifted[column] += step;
            let mut shifted_values = self.layout.external_values(shifted.as_slice()).ok()?;
            let source_name = &self.layout.entries[self.layout.free[column]]
                .definition
                .name;
            if shifted_values.get(source_name) == baseline_values.get(source_name) {
                // A bound transform can round a tiny internal step to the same
                // external f64. MINPACK uses epsilon itself for a zero value;
                // apply the same fallback when the transformed value is unchanged.
                step = self.epsfcn.max(f64::EPSILON).sqrt();
                shifted[column] = self.parameters[column] + step;
                shifted_values = self.layout.external_values(shifted.as_slice()).ok()?;
            }
            let mut shifted_curve = vec![0.0; rows];
            self.model
                .evaluate(self.x, &shifted_values, &mut shifted_curve)
                .ok()?;
            let shifted_residuals =
                objective_residuals(self.y, &shifted_curve, self.weights, self.objective);
            for row in 0..rows {
                jacobian[(row, column)] = (shifted_residuals[row] - baseline_residuals[row]) / step;
            }
        }
        jacobian
            .iter()
            .all(|value| value.is_finite())
            .then_some(jacobian)
    }
}

const POISSON_FLOOR: f64 = 1.0e-12;

fn objective_residuals(
    observations: &[f64],
    predicted: &[f64],
    weights: Option<&[f64]>,
    objective: ObjectiveKind,
) -> Vec<f64> {
    observations
        .iter()
        .zip(predicted)
        .enumerate()
        .map(|(index, (observed, predicted))| {
            let weight = weights.map_or(1.0, |all| all[index]);
            let residual = match objective {
                ObjectiveKind::LeastSquares => observed - predicted,
                ObjectiveKind::PoissonDeviance => poisson_deviance_residual(*observed, *predicted),
            };
            residual * weight
        })
        .collect()
}

fn poisson_deviance_residual(observed: f64, predicted: f64) -> f64 {
    let mean = predicted.max(POISSON_FLOOR);
    let deviance = if observed == 0.0 {
        2.0 * mean
    } else {
        let relative = (mean - observed) / observed;
        2.0 * observed * stable_poisson_term(relative)
    }
    .max(0.0);
    let signed = if observed >= mean { 1.0 } else { -1.0 };
    let base = signed * deviance.sqrt();
    if predicted >= POISSON_FLOOR {
        base
    } else {
        let barrier_slope = (observed.max(1.0) / POISSON_FLOOR).sqrt().min(1.0e6);
        let direction = if base < 0.0 { -1.0 } else { 1.0 };
        base + direction * (POISSON_FLOOR - predicted) * barrier_slope
    }
}

fn stable_poisson_term(relative: f64) -> f64 {
    if relative.abs() < 1.0e-4 {
        let square = relative * relative;
        square
            * (0.5
                + relative * (-1.0 / 3.0 + relative * (0.25 + relative * (-0.2 + relative / 6.0))))
    } else {
        relative - relative.ln_1p()
    }
}

fn poisson_deviance_residual_gradient(observed: f64, predicted: f64) -> f64 {
    if predicted < POISSON_FLOOR {
        let base = poisson_deviance_residual(observed, POISSON_FLOOR);
        let direction = if base < 0.0 { -1.0 } else { 1.0 };
        let barrier_slope = (observed.max(1.0) / POISSON_FLOOR).sqrt().min(1.0e6);
        return -direction * barrier_slope;
    }
    let mean = predicted;
    if observed == 0.0 {
        return -1.0 / (2.0 * mean).sqrt();
    }
    let relative = (mean - observed) / observed;
    if relative.abs() <= f64::EPSILON.sqrt() {
        return -1.0 / observed.sqrt();
    }
    let deviance = 2.0 * observed * stable_poisson_term(relative);
    let signed = if observed >= mean { 1.0 } else { -1.0 };
    signed * (1.0 - observed / mean) / deviance.sqrt()
}

fn objective_value(residuals: &[f64]) -> f64 {
    residuals.iter().map(|value| value * value).sum()
}

fn to_internal(external: f64, bounds: Bounds) -> f64 {
    let lower = bounds.lower.lower_value();
    let upper = bounds.upper.upper_value();
    match (lower.is_finite(), upper.is_finite()) {
        (false, false) => external,
        (true, false) => ((external - lower + 1.0).powi(2) - 1.0).max(0.0).sqrt(),
        (false, true) => ((upper - external + 1.0).powi(2) - 1.0).max(0.0).sqrt(),
        (true, true) => (2.0 * (external - lower) / (upper - lower) - 1.0)
            .clamp(-1.0, 1.0)
            .asin(),
    }
}

fn from_internal(internal: f64, bounds: Bounds) -> f64 {
    let internal = clamp_internal(internal, bounds);
    let lower = bounds.lower.lower_value();
    let upper = bounds.upper.upper_value();
    match (lower.is_finite(), upper.is_finite()) {
        (false, false) => internal,
        (true, false) => lower - 1.0 + internal.hypot(1.0),
        (false, true) => upper + 1.0 - internal.hypot(1.0),
        (true, true) => lower + (internal.sin() + 1.0) * (upper - lower) / 2.0,
    }
}

fn bound_gradient(internal: f64, bounds: Bounds) -> f64 {
    let internal = clamp_internal(internal, bounds);
    let lower = bounds.lower.lower_value();
    let upper = bounds.upper.upper_value();
    match (lower.is_finite(), upper.is_finite()) {
        (false, false) => 1.0,
        (true, false) => internal / internal.hypot(1.0),
        (false, true) => -internal / internal.hypot(1.0),
        (true, true) => internal.cos() * (upper - lower) / 2.0,
    }
}

fn clamp_internal(internal: f64, bounds: Bounds) -> f64 {
    let lower = bounds.lower.lower_value();
    let upper = bounds.upper.upper_value();
    match (lower.is_finite(), upper.is_finite()) {
        (false, false) => internal,
        (true, false) | (false, true) => internal.max(0.0),
        (true, true) => internal.clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2),
    }
}

fn finite_difference_step(value: f64, epsfcn: f64) -> f64 {
    let epsilon = epsfcn.max(f64::EPSILON).sqrt();
    let step = epsilon * value.abs();
    if step == 0.0 { epsilon } else { step }
}

fn invert_normal_matrix(jacobian: &DMatrix<f64>, scale: f64) -> Option<DMatrix<f64>> {
    let singular_values = jacobian.clone().svd(false, false).singular_values;
    let largest = singular_values.iter().copied().fold(0.0_f64, f64::max);
    let smallest = singular_values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let threshold = f64::EPSILON * jacobian.nrows().max(jacobian.ncols()).max(1) as f64 * largest;
    if !smallest.is_finite() || smallest <= threshold {
        return None;
    }
    let columns = jacobian.ncols();
    let r = jacobian.clone().qr().r();
    let square_r = r.rows(0, columns).into_owned();
    square_r.try_inverse().map(|inverse_r| {
        let inverse = &inverse_r * inverse_r.transpose();
        inverse * scale
    })
}

fn covariance_public(layout: &ParameterLayout, matrix: &DMatrix<f64>) -> Covariance {
    let names = layout
        .free
        .iter()
        .map(|entry| layout.entries[*entry].definition.name.clone())
        .collect::<Vec<_>>();
    let values = (0..matrix.nrows())
        .map(|row| {
            (0..matrix.ncols())
                .map(|column| matrix[(row, column)])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let correlations = (0..matrix.nrows())
        .map(|row| {
            (0..matrix.ncols())
                .map(|column| {
                    let denominator = (matrix[(row, row)] * matrix[(column, column)]).sqrt();
                    if denominator > 0.0 {
                        matrix[(row, column)] / denominator
                    } else if row == column {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();
    Covariance {
        parameter_names: names,
        matrix: values,
        correlations,
    }
}

fn base_estimates(
    layout: &ParameterLayout,
    values: &ParameterValues,
    covariance: Option<&DMatrix<f64>>,
) -> Vec<ParameterEstimate> {
    layout
        .entries
        .iter()
        .map(|entry| {
            let value = values
                .get(&entry.definition.name)
                .unwrap_or(entry.definition.initial);
            let kind = if entry.definition.binding.is_some() {
                ParameterKind::Shared
            } else if entry.free_index.is_some() {
                ParameterKind::Free
            } else {
                ParameterKind::Fixed
            };
            let standard_error = if kind == ParameterKind::Fixed {
                covariance.map(|_| 0.0)
            } else {
                covariance.and_then(|matrix| {
                    layout
                        .free_index_for_name(&entry.definition.name)
                        .map(|index| matrix[(index, index)].max(0.0).sqrt())
                })
            };
            ParameterEstimate {
                name: entry.definition.name.clone(),
                value,
                standard_error,
                kind,
                bounds: entry.definition.bounds,
                active_bound: entry.definition.bounds.active(value),
            }
        })
        .collect()
}

fn derived_estimates(
    layout: &ParameterLayout,
    derived: &[DerivedParameter],
    covariance: Option<&DMatrix<f64>>,
) -> Vec<ParameterEstimate> {
    derived
        .iter()
        .map(|parameter| {
            let standard_error = covariance.and_then(|matrix| {
                let mut gradient = DVector::zeros(layout.free.len());
                for (name, derivative) in &parameter.gradient {
                    if let Some(index) = layout.free_index_for_name(name) {
                        gradient[index] += derivative;
                    }
                }
                let variance = (gradient.transpose() * matrix * gradient)[0];
                variance.is_finite().then(|| variance.max(0.0).sqrt())
            });
            ParameterEstimate {
                name: parameter.name.clone(),
                value: parameter.value,
                standard_error,
                kind: ParameterKind::Derived,
                bounds: Bounds::unbounded(),
                active_bound: false,
            }
        })
        .collect()
}

fn statistics(
    observations: &[f64],
    predicted: &[f64],
    residuals: &[f64],
    variables: usize,
    evaluations: usize,
    objective: ObjectiveKind,
    initial_objective: f64,
) -> FitStatistics {
    let count = observations.len();
    let degrees_of_freedom = count - variables;
    let chi_square = residuals.iter().map(|value| value * value).sum::<f64>();
    let reduced_chi_square = chi_square / degrees_of_freedom.max(1) as f64;
    let objective_improvement = initial_objective
        .is_finite()
        .then(|| (initial_objective - chi_square) / initial_objective.abs().max(f64::EPSILON));
    let (aic, bic, deviance, reduced_deviance) = match objective {
        ObjectiveKind::LeastSquares => {
            let log_term = (chi_square / count as f64).ln();
            (
                log_term
                    .is_finite()
                    .then_some(count as f64 * log_term + 2.0 * variables as f64),
                log_term
                    .is_finite()
                    .then(|| count as f64 * log_term + (count as f64).ln() * variables as f64),
                None,
                None,
            )
        }
        ObjectiveKind::PoissonDeviance => (
            Some(chi_square + 2.0 * variables as f64),
            Some(chi_square + (count as f64).ln() * variables as f64),
            Some(chi_square),
            (degrees_of_freedom > 0).then_some(reduced_chi_square),
        ),
    };
    let mean = observations.iter().sum::<f64>() / count as f64;
    let total_sum = observations
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    let unweighted_sum = observations
        .iter()
        .zip(predicted)
        .map(|(observed, model)| (observed - model).powi(2))
        .sum::<f64>();
    let rmse = (unweighted_sum / count as f64).sqrt();
    let r_squared = (total_sum > 0.0).then(|| 1.0 - unweighted_sum / total_sum);
    let pearson_chi_square = (objective == ObjectiveKind::PoissonDeviance).then(|| {
        observations
            .iter()
            .zip(predicted)
            .map(|(observed, model)| (observed - model).powi(2) / model.max(POISSON_FLOOR))
            .sum::<f64>()
    });
    let reduced_pearson_chi_square = pearson_chi_square
        .filter(|_| degrees_of_freedom > 0)
        .map(|value| value / degrees_of_freedom as f64);
    let goodness_of_fit_p_value = deviance.and_then(|value| {
        ChiSquared::new(degrees_of_freedom as f64)
            .ok()
            .map(|distribution| (1.0 - distribution.cdf(value)).clamp(0.0, 1.0))
    });
    FitStatistics {
        objective,
        observations: count,
        variables,
        degrees_of_freedom,
        chi_square,
        reduced_chi_square,
        initial_objective: Some(initial_objective),
        final_objective: chi_square,
        objective_improvement,
        rmse,
        deviance,
        reduced_deviance,
        pearson_chi_square,
        reduced_pearson_chi_square,
        goodness_of_fit_p_value,
        aic,
        bic,
        r_squared,
        evaluations,
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "band evaluation keeps each numerical input explicit"
)]
fn confidence_bands(
    model: &dyn Model,
    x: &[f64],
    baseline: &ParameterValues,
    layout: &ParameterLayout,
    covariance: &DMatrix<f64>,
    statistics: &FitStatistics,
    objective: ObjectiveKind,
    sigma: f64,
    best_fit: &[f64],
    components: &[ComponentCurve],
    profile: SolverProfile,
) -> Result<(ConfidenceBand, Vec<(String, ConfidenceBand)>), FitError> {
    let scalars_per_row = layout
        .free
        .len()
        .saturating_mul(components.len() + 1)
        .max(1);
    let chunk_length = (MAX_VALUES / scalars_per_row).max(1);
    let mut total = empty_band(x.len())?;
    let mut component_bands = components
        .iter()
        .map(|component| Ok((component.name.clone(), empty_band(x.len())?)))
        .collect::<Result<Vec<_>, FitError>>()?;
    let scale = confidence_scale(statistics, objective, sigma)?;

    for start in (0..x.len()).step_by(chunk_length) {
        let end = (start + chunk_length).min(x.len());
        let chunk_x = &x[start..end];
        let (total_jacobian, component_jacobians) = evaluation_jacobians(
            model,
            chunk_x,
            baseline,
            layout,
            covariance,
            components.len(),
            profile,
        )?;
        append_band_from_jacobian(
            &mut total,
            chunk_x,
            &best_fit[start..end],
            &total_jacobian,
            covariance,
            scale,
        );
        for ((_, destination), (component, jacobian)) in component_bands
            .iter_mut()
            .zip(components.iter().zip(component_jacobians))
        {
            append_band_from_jacobian(
                destination,
                chunk_x,
                &component.values[start..end],
                &jacobian,
                covariance,
                scale,
            );
        }
    }
    Ok((total, component_bands))
}

fn empty_band(capacity: usize) -> Result<ConfidenceBand, FitError> {
    let reserved = || {
        let mut values = checked_zeros(capacity)?;
        values.clear();
        Ok(values)
    };
    Ok(ConfidenceBand {
        x: reserved()?,
        best_fit: reserved()?,
        uncertainty: reserved()?,
        lower: reserved()?,
        upper: reserved()?,
    })
}

fn evaluation_jacobians(
    model: &dyn Model,
    x: &[f64],
    baseline: &ParameterValues,
    layout: &ParameterLayout,
    covariance: &DMatrix<f64>,
    component_count: usize,
    profile: SolverProfile,
) -> Result<(DMatrix<f64>, Vec<DMatrix<f64>>), FitError> {
    check_allocation(
        x.len(),
        layout.free.len().saturating_mul(component_count + 1),
    )?;
    let mut total_jacobian = DMatrix::zeros(x.len(), layout.free.len());
    let mut component_jacobians = (0..component_count)
        .map(|_| DMatrix::zeros(x.len(), layout.free.len()))
        .collect::<Vec<_>>();
    let parameter_names = layout
        .free
        .iter()
        .map(|entry| layout.entries[*entry].definition.name.clone())
        .collect::<Vec<_>>();
    let steps = (0..layout.free.len())
        .map(|column| {
            let stderr = covariance[(column, column)].max(0.0).sqrt();
            let base_name = &parameter_names[column];
            let base_value = baseline.require(base_name)?;
            if profile == SolverProfile::Robust {
                let bounds = layout.entries[layout.free[column]].definition.bounds;
                let span = bounds.upper.upper_value() - bounds.lower.lower_value();
                let scale = span.min(base_value.abs().max(1.0));
                Ok((f64::EPSILON.cbrt() * scale).max(8.0 * f64::EPSILON * base_value.abs()))
            } else {
                Ok((stderr * 0.01).max(f64::EPSILON.sqrt() * base_value.abs().max(1.0)))
            }
        })
        .collect::<Result<Vec<_>, FitError>>()?;
    let mut compatibility = (0..component_count)
        .map(|_| checked_zeros(x.len().saturating_mul(layout.free.len())))
        .collect::<Result<Vec<_>, FitError>>()?;
    let analytic = if profile == SolverProfile::Robust {
        model.robust_component_jacobians(x, baseline, &parameter_names, &mut compatibility)?
    } else {
        model.compatibility_component_jacobians(
            x,
            baseline,
            &parameter_names,
            &steps,
            &mut compatibility,
        )?
    };
    if analytic {
        for (destination, values) in component_jacobians.iter_mut().zip(compatibility) {
            *destination = DMatrix::from_row_slice(x.len(), layout.free.len(), &values);
            total_jacobian += &*destination;
        }
        return Ok((total_jacobian, component_jacobians));
    }
    for column in 0..layout.free.len() {
        let base_name = &layout.entries[layout.free[column]].definition.name;
        let base_value = baseline.require(base_name)?;
        let step = steps[column];
        let bounds = layout.entries[layout.free[column]].definition.bounds;
        let (plus_value, minus_value) = if profile == SolverProfile::Robust {
            (
                (base_value + step).min(bounds.upper.upper_value()),
                (base_value - step).max(bounds.lower.lower_value()),
            )
        } else {
            (base_value + step, base_value - step)
        };
        let denominator = if profile == SolverProfile::Robust {
            plus_value - minus_value
        } else {
            2.0 * step
        };
        let plus = layout.set_free_external(baseline, column, plus_value);
        let minus = layout.set_free_external(baseline, column, minus_value);
        let plus_components = model.components(x, &plus)?;
        let minus_components = model.components(x, &minus)?;
        if plus_components.len() != component_count || minus_components.len() != component_count {
            return Err(FitError::Solver {
                message: "model component count changed during uncertainty evaluation".to_owned(),
            });
        }
        for row in 0..x.len() {
            let mut plus_total = 0.0;
            let mut minus_total = 0.0;
            for component in 0..component_count {
                let plus_value = plus_components[component].values[row];
                let minus_value = minus_components[component].values[row];
                plus_total += plus_value;
                minus_total += minus_value;
                component_jacobians[component][(row, column)] =
                    (plus_value - minus_value) / denominator;
            }
            total_jacobian[(row, column)] = (plus_total - minus_total) / denominator;
        }
    }
    Ok((total_jacobian, component_jacobians))
}

fn append_band_from_jacobian(
    destination: &mut ConfidenceBand,
    x: &[f64],
    best_fit: &[f64],
    jacobian: &DMatrix<f64>,
    covariance: &DMatrix<f64>,
    scale: f64,
) {
    let active_columns = (0..jacobian.ncols())
        .filter(|column| (0..jacobian.nrows()).any(|row| jacobian[(row, *column)] != 0.0))
        .collect::<Vec<_>>();
    for row in 0..x.len() {
        let mut variance = 0.0;
        for (position, &column) in active_columns.iter().enumerate() {
            let left = jacobian[(row, column)];
            variance += left * covariance[(column, column)] * left;
            for &other in &active_columns[..position] {
                let right = jacobian[(row, other)];
                variance += left * covariance[(column, other)] * right;
                variance += right * covariance[(other, column)] * left;
            }
        }
        let uncertainty = scale * variance.max(0.0).sqrt();
        destination.x.push(x[row]);
        destination.best_fit.push(best_fit[row]);
        destination.uncertainty.push(uncertainty);
        destination.lower.push(best_fit[row] - uncertainty);
        destination.upper.push(best_fit[row] + uncertainty);
    }
}

fn confidence_scale(
    statistics: &FitStatistics,
    objective: ObjectiveKind,
    sigma: f64,
) -> Result<f64, FitError> {
    if objective == ObjectiveKind::PoissonDeviance {
        return Ok(sigma);
    }
    let probability = (erf(sigma / std::f64::consts::SQRT_2) + 1.0) / 2.0;
    let distribution =
        StudentsT::new(0.0, 1.0, statistics.degrees_of_freedom as f64).map_err(|error| {
            FitError::Solver {
                message: format!("invalid Student-t distribution: {error}"),
            }
        })?;
    Ok(distribution.inverse_cdf(probability))
}

fn termination_status(report: Option<&MinimizationReport>) -> TerminationStatus {
    let Some(report) = report else {
        return TerminationStatus {
            reason: "no_variables".to_owned(),
            message: "model contains no varying parameters".to_owned(),
            success: false,
        };
    };
    let (reason, message) = match &report.termination {
        TerminationReason::User(message) => ("evaluation_failed", *message),
        TerminationReason::Numerical(message) => ("numerical_failure", *message),
        TerminationReason::ResidualsZero => ("residuals_zero", "residuals are exactly zero"),
        TerminationReason::Orthogonal => ("orthogonal", "residuals and Jacobian are orthogonal"),
        TerminationReason::Converged { ftol, xtol } => match (*ftol, *xtol) {
            (true, true) => ("converged", "ftol and xtol convergence criteria reached"),
            (true, false) => ("ftol", "ftol convergence criterion reached"),
            (false, true) => ("xtol", "xtol convergence criterion reached"),
            (false, false) => ("converged", "convergence criterion reached"),
        },
        TerminationReason::NoImprovementPossible(message) => ("no_improvement", *message),
        TerminationReason::LostPatience => ("max_evaluations", "maximum evaluations reached"),
        TerminationReason::NoParameters => ("no_variables", "model contains no variables"),
        TerminationReason::NoResiduals => ("no_residuals", "model contains no residuals"),
        TerminationReason::WrongDimensions(message) => ("wrong_dimensions", *message),
    };
    TerminationStatus {
        reason: reason.to_owned(),
        message: message.to_owned(),
        success: report.termination.was_successful(),
    }
}

fn check_allocation(rows: usize, columns: usize) -> Result<(), FitError> {
    let requested = rows.checked_mul(columns).ok_or(FitError::AllocationLimit {
        requested: usize::MAX,
        limit: MAX_VALUES,
    })?;
    if requested > MAX_VALUES {
        return Err(FitError::AllocationLimit {
            requested,
            limit: MAX_VALUES,
        });
    }
    Ok(())
}

fn checked_zeros(length: usize) -> Result<Vec<f64>, FitError> {
    check_allocation(length, 1)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_allocation_error| FitError::AllocationLimit {
            requested: length,
            limit: MAX_VALUES,
        })?;
    values.resize(length, 0.0);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::{
        bound_gradient, from_internal, poisson_deviance_residual,
        poisson_deviance_residual_gradient, to_internal,
    };
    use crate::Bounds;

    #[test]
    fn lmfit_bound_transforms_round_trip() {
        for (bounds, values) in [
            (Bounds::unbounded(), vec![-3.0, 0.0, 8.0]),
            (Bounds::lower_bounded(2.0), vec![2.0, 3.0, 20.0]),
            (
                Bounds {
                    lower: crate::Bound::Unbounded,
                    upper: crate::Bound::Inclusive(4.0),
                },
                vec![-5.0, 3.0, 4.0],
            ),
            (Bounds::finite(-2.0, 7.0), vec![-2.0, 0.5, 7.0]),
        ] {
            for value in values {
                let internal = to_internal(value, bounds);
                let round_trip = from_internal(internal, bounds);
                assert!((round_trip - value).abs() <= 1.0e-12);
                assert!(bound_gradient(internal, bounds).is_finite());
            }
        }
    }

    #[test]
    fn bounded_transforms_do_not_wrap_outside_their_monotonic_branch() {
        let finite = Bounds::finite(-2.0, 7.0);
        assert_eq!(from_internal(-100.0, finite), -2.0);
        assert_eq!(from_internal(100.0, finite), 7.0);
        let values = (-40..=40)
            .map(|index| from_internal(index as f64 * 0.1, finite))
            .collect::<Vec<_>>();
        assert!(values.windows(2).all(|pair| pair[0] <= pair[1]));

        let lower = Bounds::lower_bounded(2.0);
        assert_eq!(from_internal(-10.0, lower), 2.0);
        assert!(from_internal(3.0, lower) > from_internal(2.0, lower));
    }

    #[test]
    fn poisson_deviance_gradient_matches_central_differences() {
        for (observed, predicted) in [
            (0.0_f64, 0.2_f64),
            (0.0, 15.0),
            (0.5, 0.2),
            (1.0, 0.8),
            (10.0, 9.999),
            (10.0, 10.001),
            (1000.0, 820.0),
            (1000.0, 1250.0),
        ] {
            let step = predicted.abs().max(1.0) * 1.0e-6;
            let numerical = (poisson_deviance_residual(observed, predicted + step)
                - poisson_deviance_residual(observed, predicted - step))
                / (2.0 * step);
            let analytic = poisson_deviance_residual_gradient(observed, predicted);
            let scale = numerical.abs().max(analytic.abs()).max(1.0);
            assert!(
                (analytic - numerical).abs() <= 2.0e-6 * scale,
                "y={observed}, mu={predicted}: analytic={analytic}, numerical={numerical}",
            );
        }
    }

    #[test]
    fn poisson_deviance_is_smooth_at_observed_equals_predicted() {
        let observed = 1.0e6;
        let below = poisson_deviance_residual(observed, observed * (1.0 - 1.0e-10));
        let above = poisson_deviance_residual(observed, observed * (1.0 + 1.0e-10));
        assert!(below.is_finite() && above.is_finite());
        assert!(below > 0.0 && above < 0.0);
        let gradient = poisson_deviance_residual_gradient(observed, observed);
        assert!((gradient + 1.0 / observed.sqrt()).abs() < 1.0e-12);
    }
}
