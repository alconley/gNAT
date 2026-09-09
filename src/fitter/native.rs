//! Adapters between the UI's legacy storage structs and `spectrix-fitting`.

use spectrix_fitting::{
    BackgroundKind, BackgroundSeed, Bound, Bounds, FitResult as NativeFitResult,
    ParameterDefinition,
};

use super::{
    common::{Data, Parameter},
    main_fitter::{BackgroundModel, BackgroundResult},
    models::{
        exponential::ExponentialFitter, linear::LinearFitter, powerlaw::PowerLawFitter,
        quadratic::QuadraticFitter,
    },
};

pub(crate) fn background_kind(model: &BackgroundModel) -> BackgroundKind {
    match model {
        BackgroundModel::Constant(_) => BackgroundKind::Constant,
        BackgroundModel::Linear(_) => BackgroundKind::Linear,
        BackgroundModel::Quadratic(_) => BackgroundKind::Quadratic,
        BackgroundModel::PowerLaw(_) => BackgroundKind::PowerLaw,
        BackgroundModel::Exponential(_) => BackgroundKind::Exponential,
        BackgroundModel::LegacyAuto | BackgroundModel::None => BackgroundKind::None,
    }
}

pub(crate) fn concrete_background_model(kind: BackgroundKind) -> BackgroundModel {
    match kind {
        BackgroundKind::None => BackgroundModel::None,
        BackgroundKind::Constant => BackgroundModel::Constant(Parameter {
            name: "Constant".to_owned(),
            min: 0.0,
            ..Parameter::default()
        }),
        BackgroundKind::Linear => BackgroundModel::Linear(Default::default()),
        BackgroundKind::Quadratic => BackgroundModel::Quadratic(Default::default()),
        BackgroundKind::Exponential => BackgroundModel::Exponential(Default::default()),
        BackgroundKind::PowerLaw => BackgroundModel::PowerLaw(Default::default()),
    }
}

pub(crate) fn background_seed(
    model: &BackgroundModel,
    previous: Option<&BackgroundResult>,
) -> BackgroundSeed {
    let parameters = match model {
        BackgroundModel::LegacyAuto | BackgroundModel::None => {
            vec![ParameterDefinition::fixed("bg_c", 0.0)]
        }
        BackgroundModel::Constant(parameter) => {
            vec![parameter_definition("bg_c", parameter, None)]
        }
        BackgroundModel::Linear(parameters) => vec![
            parameter_definition(
                "bg_slope",
                &parameters.slope,
                previous.and_then(|result| match result {
                    BackgroundResult::Linear(fit) => fit.paramaters.slope.value,
                    _ => None,
                }),
            ),
            parameter_definition(
                "bg_intercept",
                &parameters.intercept,
                previous.and_then(|result| match result {
                    BackgroundResult::Linear(fit) => fit.paramaters.intercept.value,
                    _ => None,
                }),
            ),
        ],
        BackgroundModel::Quadratic(parameters) => vec![
            parameter_definition(
                "bg_a",
                &parameters.a,
                previous.and_then(|result| match result {
                    BackgroundResult::Quadratic(fit) => fit.paramaters.a.value,
                    _ => None,
                }),
            ),
            parameter_definition(
                "bg_b",
                &parameters.b,
                previous.and_then(|result| match result {
                    BackgroundResult::Quadratic(fit) => fit.paramaters.b.value,
                    _ => None,
                }),
            ),
            parameter_definition(
                "bg_c",
                &parameters.c,
                previous.and_then(|result| match result {
                    BackgroundResult::Quadratic(fit) => fit.paramaters.c.value,
                    _ => None,
                }),
            ),
        ],
        BackgroundModel::Exponential(parameters) => vec![
            parameter_definition(
                "bg_amplitude",
                &parameters.amplitude,
                previous.and_then(|result| match result {
                    BackgroundResult::Exponential(fit) => fit.paramaters.amplitude.value,
                    _ => None,
                }),
            ),
            parameter_definition(
                "bg_decay",
                &parameters.decay,
                previous.and_then(|result| match result {
                    BackgroundResult::Exponential(fit) => fit.paramaters.decay.value,
                    _ => None,
                }),
            ),
        ],
        BackgroundModel::PowerLaw(parameters) => vec![
            parameter_definition(
                "bg_amplitude",
                &parameters.amplitude,
                previous.and_then(|result| match result {
                    BackgroundResult::PowerLaw(fit) => fit.paramaters.amplitude.value,
                    _ => None,
                }),
            ),
            parameter_definition(
                "bg_exponent",
                &parameters.exponent,
                previous.and_then(|result| match result {
                    BackgroundResult::PowerLaw(fit) => fit.paramaters.exponent.value,
                    _ => None,
                }),
            ),
        ],
    };
    BackgroundSeed { parameters }
}

pub(crate) fn background_result_from_native(
    model: &BackgroundModel,
    result: &NativeFitResult,
    data: Data,
) -> Option<BackgroundResult> {
    background_result_from_values(model, result, data, &result.best_fit)
}

pub(crate) fn background_result_from_composite_native(
    model: &BackgroundModel,
    result: &NativeFitResult,
    data: Data,
) -> Option<BackgroundResult> {
    let background_values = result
        .components
        .iter()
        .find(|component| component.name == "background")
        .map(|component| component.values.as_slice())?;
    background_result_from_values(model, result, data, background_values)
}

fn background_result_from_values(
    model: &BackgroundModel,
    result: &NativeFitResult,
    data: Data,
    background_values: &[f64],
) -> Option<BackgroundResult> {
    let fit_points = result
        .evaluation_x
        .iter()
        .copied()
        .zip(background_values.iter().copied())
        .map(Into::into)
        .collect::<Vec<_>>();
    let report = fit_report(result);
    match model {
        BackgroundModel::Constant(parameter) => {
            let mut intercept = parameter.clone();
            apply_estimate(&mut intercept, result, "bg_c");
            let slope = Parameter {
                name: "Slope".to_owned(),
                vary: false,
                value: Some(0.0),
                ..Parameter::default()
            };
            Some(BackgroundResult::Constant(LinearFitter {
                data,
                paramaters: super::models::linear::LinearParameters { slope, intercept },
                fit_points,
                fit_report: report,
            }))
        }
        BackgroundModel::Linear(parameters) => {
            let mut parameters = parameters.clone();
            apply_estimate(&mut parameters.slope, result, "bg_slope");
            apply_estimate(&mut parameters.intercept, result, "bg_intercept");
            Some(BackgroundResult::Linear(LinearFitter {
                data,
                paramaters: parameters,
                fit_points,
                fit_report: report,
            }))
        }
        BackgroundModel::Quadratic(parameters) => {
            let mut parameters = parameters.clone();
            apply_estimate(&mut parameters.a, result, "bg_a");
            apply_estimate(&mut parameters.b, result, "bg_b");
            apply_estimate(&mut parameters.c, result, "bg_c");
            Some(BackgroundResult::Quadratic(QuadraticFitter {
                data,
                paramaters: parameters,
                fit_points,
                fit_report: report,
                covar: covariance_3(result, ["bg_a", "bg_b", "bg_c"]),
            }))
        }
        BackgroundModel::Exponential(parameters) => {
            let mut parameters = parameters.clone();
            apply_estimate(&mut parameters.amplitude, result, "bg_amplitude");
            apply_estimate(&mut parameters.decay, result, "bg_decay");
            Some(BackgroundResult::Exponential(ExponentialFitter {
                data,
                paramaters: parameters,
                fit_points,
                fit_report: report,
            }))
        }
        BackgroundModel::PowerLaw(parameters) => {
            let mut parameters = parameters.clone();
            apply_estimate(&mut parameters.amplitude, result, "bg_amplitude");
            apply_estimate(&mut parameters.exponent, result, "bg_exponent");
            Some(BackgroundResult::PowerLaw(PowerLawFitter {
                data,
                paramaters: parameters,
                fit_points,
                fit_report: report,
            }))
        }
        BackgroundModel::LegacyAuto | BackgroundModel::None => None,
    }
}

pub(crate) fn fit_report(result: &NativeFitResult) -> String {
    let statistics = &result.statistics;
    let mut report = format!(
        "[[Native {:?} fit]]\n\
         success = {}\n\
         termination = {} ({})\n\
         function evaluations = {}\n\
         data points = {}\n\
         variables = {}\n\
         chi-square = {:.15e}\n\
         reduced chi-square = {}\n\
         Akaike information criterion = {}\n\
         Bayesian information criterion = {}\n\
         R-squared = {}\n\
         [[Variables]]\n",
        statistics.objective,
        result.termination.success,
        result.termination.reason,
        result.termination.message,
        statistics.evaluations,
        statistics.observations,
        statistics.variables,
        statistics.chi_square,
        optional_number(
            (statistics.degrees_of_freedom > 0).then_some(statistics.reduced_chi_square)
        ),
        optional_number(statistics.aic),
        optional_number(statistics.bic),
        optional_number(statistics.r_squared),
    );
    for parameter in &result.parameters {
        let error = parameter
            .standard_error
            .map_or_else(|| "unavailable".to_owned(), |value| format!("{value:.15e}"));
        report.push_str(&format!(
            "    {}: {:.15e} +/- {} ({:?})\n",
            parameter.name, parameter.value, error, parameter.kind
        ));
    }
    if result.diagnostics.attempts > 0 {
        report.push_str(&format!(
            "[[Numerical diagnostics]]\nattempts (including polish) = {}\nprojected gradient = {}\nobjective improvement = {}\ncovariance = {}\nshared evaluation budget exhausted = {}\n",
            result.diagnostics.attempts,
            optional_number(result.diagnostics.optimality),
            optional_number(result.statistics.objective_improvement),
            covariance_description(result),
            result.diagnostics.budget_exhausted,
        ));
    }
    if let Some(error) = &result.diagnostics.confidence_band_error {
        report.push_str(&format!("Confidence bands unavailable: {error}\n"));
    }
    report
}

pub(crate) fn covariance_description(result: &NativeFitResult) -> String {
    use spectrix_fitting::CovarianceStatus;
    let reason = match result.diagnostics.covariance_status {
        CovarianceStatus::Available => {
            "Available; the model is stationary and has full numerical rank."
        }
        CovarianceStatus::Disabled => {
            "Uncertainties are not calculated for this background starting estimate."
        }
        CovarianceStatus::NoFreeParameters => "No independently varying parameters.",
        CovarianceStatus::InsufficientInformation => {
            "Covariance unavailable: no residual degrees of freedom remain to estimate noise. Use wider background windows or more observations for uncertainty estimation."
        }
        CovarianceStatus::NotConverged => {
            "Covariance unavailable: the solution has not converged. Review the starting values or fit region."
        }
        CovarianceStatus::ActiveBounds => {
            "Covariance unavailable: parameters are limited by their bounds. Review those limits before interpreting symmetric errors."
        }
        CovarianceStatus::RankDeficient => {
            "Covariance unavailable: the data cannot distinguish all parameters. Review overlapping peaks, shared widths, or background complexity."
        }
        CovarianceStatus::NumericalFailure => {
            "Covariance unavailable: model derivatives or information could not be evaluated reliably."
        }
        CovarianceStatus::Unknown => "Parameter covariance and uncertainties are unavailable.",
    };
    if result.diagnostics.affected_parameters.is_empty() {
        reason.to_owned()
    } else {
        format!(
            "{reason} Parameters: {}.",
            result.diagnostics.affected_parameters.join(", ")
        )
    }
}

fn optional_number(value: Option<f64>) -> String {
    value.map_or_else(|| "unavailable".to_owned(), |value| format!("{value:.15e}"))
}

pub(crate) fn parameter_definition(
    name: &str,
    parameter: &Parameter,
    value: Option<f64>,
) -> ParameterDefinition {
    ParameterDefinition {
        name: name.to_owned(),
        initial: value.unwrap_or(parameter.initial_guess),
        bounds: bounds(parameter.min, parameter.max),
        vary: parameter.vary,
        binding: None,
    }
}

fn bounds(minimum: f64, maximum: f64) -> Bounds {
    Bounds {
        lower: if minimum.is_finite() {
            Bound::Inclusive(minimum)
        } else {
            Bound::Unbounded
        },
        upper: if maximum.is_finite() {
            Bound::Inclusive(maximum)
        } else {
            Bound::Unbounded
        },
    }
}

pub(crate) fn apply_estimate(parameter: &mut Parameter, result: &NativeFitResult, name: &str) {
    if let Some(estimate) = result
        .parameters
        .iter()
        .find(|parameter| parameter.name == name)
    {
        parameter.value = Some(estimate.value);
        parameter.uncertainty = estimate.standard_error;
    }
}

pub(crate) fn covariance_3(result: &NativeFitResult, names: [&str; 3]) -> Option<[[f64; 3]; 3]> {
    let covariance = result.covariance.as_ref()?;
    let indices = names.map(|name| {
        covariance
            .parameter_names
            .iter()
            .position(|candidate| candidate == name)
    });
    let [Some(first), Some(second), Some(third)] = indices else {
        return None;
    };
    let indices = [first, second, third];
    Some(std::array::from_fn(|row| {
        std::array::from_fn(|column| covariance.matrix[indices[row]][indices[column]])
    }))
}
