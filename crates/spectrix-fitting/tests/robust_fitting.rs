use spectrix_fitting::{
    BackgroundCoupling, BackgroundKind, BackgroundSeed, Bounds, CompositeModel, ConstantModel,
    CovarianceStatus, FitOptions, FitProblem, ManualPeakBounds, ManualPeakSeed, ModelComponent,
    ObjectiveKind, ParameterDefinition, PeakFitRequest, SpectrumFitResult, evaluate_manual_peak,
    fit, fit_peaks,
};

fn parameter(result: &SpectrumFitResult, name: &str) -> f64 {
    result
        .fit
        .parameters
        .iter()
        .find(|parameter| parameter.name == name)
        .expect("parameter")
        .value
}

fn request(background: BackgroundKind) -> PeakFitRequest {
    let truth = ManualPeakSeed {
        center: 5.0,
        sigma: 0.6,
        amplitude: 100.0,
    };
    let x = (0..161)
        .map(|index| 1.0 + index as f64 * 0.05)
        .collect::<Vec<_>>();
    let y = x
        .iter()
        .map(|value| {
            let baseline = match background {
                BackgroundKind::None => 0.0,
                BackgroundKind::Constant => 3.0,
                BackgroundKind::Linear => 2.0 + 0.3 * value,
                BackgroundKind::Quadratic => 2.0 + 0.3 * value + 0.02 * value * value,
                BackgroundKind::Exponential => 10.0 * (-value / 5.0).exp(),
                BackgroundKind::PowerLaw => 10.0 * value.powf(-0.7),
            };
            baseline + evaluate_manual_peak(truth, *value, 0.05).expect("truth")
        })
        .collect();
    PeakFitRequest {
        x,
        y,
        bin_width: 0.05,
        region: [1.0, 9.0],
        peak_seeds: vec![ManualPeakSeed {
            center: 5.0,
            sigma: 0.1,
            amplitude: 70.0,
        }],
        peak_bounds: Some(vec![ManualPeakBounds {
            center: [4.0, 6.0],
            sigma: [0.1, 2.0],
            net_height: [0.0, 1000.0],
        }]),
        background_markers: Vec::new(),
        background,
        background_seed: None,
        background_coupling: BackgroundCoupling::PrefitJoint,
        equal_sigma: false,
        free_centers: true,
        sigma_bounds: None,
    }
}

#[test]
fn isolated_peak_escapes_a_boundary_and_refitting_does_not_improve_it() {
    for objective in [ObjectiveKind::LeastSquares, ObjectiveKind::PoissonDeviance] {
        for (center, sigma, amplitude) in [(5.0, 0.1, 70.0), (5.9, 0.1, 10.0), (4.1, 1.5, 500.0)] {
            let mut input = request(BackgroundKind::None);
            input.peak_seeds[0] = ManualPeakSeed {
                center,
                sigma,
                amplitude,
            };
            let options = FitOptions {
                objective,
                ..FitOptions::robust()
            };
            let result = fit_peaks(&input, &options).expect("fit");
            assert!(
                result.fit.termination.success,
                "{:?}",
                result.fit.diagnostics
            );
            assert!(
                result.fit.covariance.is_some(),
                "{:?}",
                result.fit.diagnostics
            );
            assert!((parameter(&result, "g0_sigma") - 0.6).abs() < 1.0e-6);
            assert!(result.fit.diagnostics.attempts >= 2);
            input.peak_seeds[0] = ManualPeakSeed {
                center: parameter(&result, "g0_center"),
                sigma: parameter(&result, "g0_sigma"),
                amplitude: parameter(&result, "g0_amplitude"),
            };
            let repeated = fit_peaks(&input, &options).expect("refit");
            assert!(
                result.fit.statistics.final_objective - repeated.fit.statistics.final_objective
                    <= 1.0e-8 * result.fit.statistics.final_objective.max(1.0)
            );
            assert_eq!(result.peak_seeds[0].sigma, sigma);
        }
    }
}

#[test]
fn every_background_can_be_fitted_without_windows() {
    for background in [
        BackgroundKind::Constant,
        BackgroundKind::Linear,
        BackgroundKind::Quadratic,
        BackgroundKind::Exponential,
        BackgroundKind::PowerLaw,
    ] {
        for objective in [ObjectiveKind::LeastSquares, ObjectiveKind::PoissonDeviance] {
            let input = request(background);
            let options = FitOptions {
                objective,
                ..FitOptions::robust()
            };
            let result = fit_peaks(&input, &options).expect("automatic background fit");
            assert!(
                result.fit.termination.success,
                "{background:?}/{objective:?}: {:?}",
                result.fit.diagnostics
            );
            assert!(
                result.fit.covariance.is_some(),
                "{background:?}/{objective:?}: {:?}",
                result.fit.diagnostics
            );
            assert!(
                (parameter(&result, "g0_sigma") - 0.6).abs() < 1.0e-5,
                "{background:?}/{objective:?}"
            );
            assert!(
                result.fit.statistics.final_objective < 1.0e-7,
                "{background:?}/{objective:?}: {}",
                result.fit.statistics.final_objective
            );
            assert_eq!(
                result.background_prefit.diagnostics.covariance_status,
                CovarianceStatus::Disabled
            );
        }
    }
}

#[test]
fn external_windows_anchor_joint_fits_and_overlap_is_counted_once() {
    let mut input = request(BackgroundKind::Linear);
    input.region = [3.0, 7.0];
    input.background_markers = vec![(1.0, 1.5), (1.3, 1.7), (3.2, 4.2), (8.5, 9.0)];
    for (y, x) in input.y.iter_mut().zip(&input.x) {
        *y += 0.01 * (7.0 * x).sin();
    }
    let result = fit_peaks(&input, &FitOptions::robust()).expect("window fit");
    let expected_x = input
        .x
        .iter()
        .copied()
        .filter(|x| (1.0..=1.7).contains(x) || (3.0..=7.0).contains(x) || (8.5..=9.0).contains(x))
        .collect::<Vec<_>>();
    assert_eq!(result.fit.observation_x, expected_x);
    assert_eq!(result.fit.raw_residuals.len(), expected_x.len());
    let peak = ManualPeakSeed {
        center: parameter(&result, "g0_center"),
        sigma: parameter(&result, "g0_sigma"),
        amplitude: parameter(&result, "g0_amplitude"),
    };
    for (x, residual) in result
        .fit
        .observation_x
        .iter()
        .zip(&result.fit.raw_residuals)
    {
        let index = input
            .x
            .iter()
            .position(|value| value == x)
            .expect("original bin");
        let model = parameter(&result, "bg_slope") * x
            + parameter(&result, "bg_intercept")
            + evaluate_manual_peak(peak, *x, input.bin_width).expect("peak prediction");
        assert!((residual - (input.y[index] - model)).abs() < 1.0e-9);
    }
    assert!(
        result
            .fit
            .evaluation_x
            .iter()
            .all(|x| *x >= 3.0 && *x <= 7.0)
    );
    let covariance = result.fit.covariance.expect("joint covariance");
    assert!(
        covariance
            .parameter_names
            .iter()
            .any(|name| name.starts_with("bg_"))
    );
    let bg = covariance
        .parameter_names
        .iter()
        .position(|name| name.starts_with("bg_"))
        .expect("fixture value");
    let height = covariance
        .parameter_names
        .iter()
        .position(|name| name == "g0_height")
        .expect("fixture value");
    assert_ne!(covariance.matrix[bg][height], 0.0);
}

#[test]
fn explicit_frozen_background_preserves_coefficients_and_region_sampling() {
    let mut input = request(BackgroundKind::Constant);
    input.region = [3.0, 7.0];
    input.background_markers = vec![(1.0, 2.0), (8.0, 9.0)];
    input.background_seed = Some(BackgroundSeed {
        parameters: vec![ParameterDefinition::fixed("bg_c", 3.0)],
    });
    input.background_coupling = BackgroundCoupling::PrefitFrozen;
    let result = fit_peaks(&input, &FitOptions::robust()).expect("locked fit");
    assert_eq!(parameter(&result, "bg_c"), 3.0);
    assert!(
        result
            .fit
            .observation_x
            .iter()
            .all(|x| *x >= 3.0 && *x <= 7.0)
    );
    assert!(
        !result
            .fit
            .covariance
            .expect("fixture value")
            .parameter_names
            .iter()
            .any(|name| name.starts_with("bg_"))
    );
}

#[test]
fn poisson_covariance_matches_constant_mean_information() {
    let y = vec![2.0, 4.0, 3.0, 7.0, 4.0];
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    let expected_variance = mean / y.len() as f64;
    let result = fit(
        &FitProblem::new(
            Box::new(ConstantModel::new("", [1.0])),
            (0..y.len()).map(|i| i as f64).collect(),
            y,
        ),
        &FitOptions {
            objective: ObjectiveKind::PoissonDeviance,
            ..FitOptions::robust()
        },
    )
    .expect("Poisson fit");
    assert!(
        (result.covariance.expect("information covariance").matrix[0][0] - expected_variance).abs()
            < 1.0e-6
    );
}

#[test]
fn genuine_bound_and_rank_failures_have_explanations() {
    let bounded = ConstantModel::new("", [0.0])
        .with_parameters([
            ParameterDefinition::varying("c", 0.0).with_bounds(Bounds::lower_bounded(0.0))
        ]);
    let result = fit(
        &FitProblem::new(Box::new(bounded), vec![0.0, 1.0, 2.0], vec![-2.0; 3]),
        &FitOptions::robust(),
    )
    .expect("bounded fit");
    assert!(result.termination.success);
    assert_eq!(
        result.diagnostics.covariance_status,
        CovarianceStatus::ActiveBounds
    );
    assert_eq!(result.diagnostics.affected_parameters, vec!["c"]);
    assert!(result.covariance.is_none());
    let mut model = CompositeModel::default();
    model
        .push(ModelComponent::new(
            "a",
            Box::new(ConstantModel::new("a_", [1.0])),
        ))
        .expect("fixture value");
    model
        .push(ModelComponent::new(
            "b",
            Box::new(ConstantModel::new("b_", [2.0])),
        ))
        .expect("fixture value");
    let singular = fit(
        &FitProblem::new(Box::new(model), vec![0.0, 1.0, 2.0, 3.0], vec![4.0; 4]),
        &FitOptions::robust(),
    )
    .expect("singular fit");
    assert_eq!(
        singular.diagnostics.covariance_status,
        CovarianceStatus::RankDeficient
    );
    assert_eq!(singular.diagnostics.affected_parameters.len(), 2);
    assert!(singular.covariance.is_none());
}

#[test]
fn recovery_obeys_one_budget() {
    let options = FitOptions {
        evaluation_patience: 1,
        ..FitOptions::robust()
    };
    let result = fit_peaks(&request(BackgroundKind::None), &options).expect("budgeted result");
    assert!(result.fit.statistics.evaluations <= result.fit.statistics.variables + 1);
    assert!(
        result.fit.statistics.final_objective
            <= result
                .fit
                .statistics
                .initial_objective
                .expect("fixture value")
    );
    assert!(!result.fit.termination.success);
    assert!(result.fit.covariance.is_none());
}

#[cfg(feature = "serde")]
#[test]
fn old_results_load_without_losing_values() {
    let result = fit_peaks(&request(BackgroundKind::None), &FitOptions::robust()).expect("fit");
    let mut encoded = serde_json::to_value(&result.fit).expect("fixture value");
    let object = encoded.as_object_mut().expect("fixture value");
    object.remove("diagnostics");
    object.remove("observation_x");
    let decoded: spectrix_fitting::FitResult =
        serde_json::from_value(encoded).expect("fixture value");
    assert_eq!(decoded.parameters, result.fit.parameters);
    assert_eq!(decoded.best_fit, result.fit.best_fit);
    assert_eq!(
        decoded.diagnostics.covariance_status,
        CovarianceStatus::Unknown
    );
    assert!(decoded.observation_x.is_empty());
}

struct DifferentScales;

impl spectrix_fitting::Model for DifferentScales {
    fn name(&self) -> &'static str {
        "different parameter scales"
    }
    fn parameter_definitions(&self) -> Vec<ParameterDefinition> {
        vec![
            ParameterDefinition::varying("large", 5.0e5),
            ParameterDefinition::varying("small", 2.0e-7),
        ]
    }
    fn evaluate(
        &self,
        x: &[f64],
        parameters: &spectrix_fitting::ParameterValues,
        output: &mut [f64],
    ) -> Result<(), spectrix_fitting::FitError> {
        for (value, x) in output.iter_mut().zip(x) {
            *value =
                parameters.require("large")? * 1.0e-6 + parameters.require("small")? * 1.0e6 * x;
        }
        Ok(())
    }
}

#[test]
fn scaled_custom_derivatives_and_covariance_match_linear_algebra() {
    let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let noise = [0.1, -0.2, 0.2, -0.2, 0.1];
    let y = x
        .iter()
        .zip(noise)
        .map(|(x, noise)| 1.0 + x + noise)
        .collect();
    let result = fit(
        &FitProblem::new(Box::new(DifferentScales), x, y),
        &FitOptions::robust(),
    )
    .expect("scaled custom model");
    assert!(result.termination.success, "{:?}", result.diagnostics);
    let covariance = result.covariance.expect("physical covariance");
    let variance = noise.iter().map(|value| value * value).sum::<f64>() / 3.0;
    for (actual, expected) in [
        (covariance.matrix[0][0], variance / (5.0e-12)),
        (covariance.matrix[1][1], variance / (10.0e12)),
    ] {
        assert!(
            (actual / expected - 1.0).abs() < 1.0e-6,
            "{actual} != {expected}"
        );
    }
    assert!((result.parameters[0].value - 1.0e6).abs() < 0.1);
    assert!((result.parameters[1].value - 1.0e-6).abs() < 1.0e-12);
}

#[test]
fn overlapping_peaks_preserve_fixed_centers_and_shared_or_independent_widths() {
    for equal_sigma in [false, true] {
        for free_centers in [false, true] {
            for objective in [ObjectiveKind::LeastSquares, ObjectiveKind::PoissonDeviance] {
                let mut input = request(BackgroundKind::Linear);
                input.equal_sigma = equal_sigma;
                input.free_centers = free_centers;
                let second = ManualPeakSeed {
                    center: 5.9,
                    sigma: if equal_sigma { 0.6 } else { 0.8 },
                    amplitude: 60.0,
                };
                for (y, x) in input.y.iter_mut().zip(&input.x) {
                    *y +=
                        evaluate_manual_peak(second, *x, input.bin_width).expect("second Gaussian");
                }
                input.peak_seeds[0].sigma = 0.4;
                input.peak_seeds.push(ManualPeakSeed {
                    sigma: 0.5,
                    ..second
                });
                input
                    .peak_bounds
                    .as_mut()
                    .expect("bounds")
                    .push(ManualPeakBounds {
                        center: [5.5, 6.5],
                        sigma: [0.1, 2.0],
                        net_height: [0.0, 1000.0],
                    });
                let result = fit_peaks(
                    &input,
                    &FitOptions {
                        objective,
                        ..FitOptions::robust()
                    },
                )
                .expect("overlap fit");
                assert!(
                    result.fit.termination.success,
                    "{equal_sigma}/{free_centers}/{objective:?}: {:?}",
                    result.fit.diagnostics
                );
                assert!(
                    result.fit.covariance.is_some(),
                    "{equal_sigma}/{free_centers}/{objective:?}: {:?}",
                    result.fit.diagnostics
                );
                assert_eq!(result.peak_seeds.len(), 2);
                if equal_sigma {
                    assert_eq!(
                        parameter(&result, "g0_sigma"),
                        parameter(&result, "g1_sigma")
                    );
                }
                if !free_centers {
                    assert_eq!(parameter(&result, "g0_center"), 5.0);
                    assert_eq!(parameter(&result, "g1_center"), 5.9);
                }
                assert!((parameter(&result, "g1_sigma") - second.sigma).abs() < 1.0e-4);
            }
        }
    }
}

#[test]
fn low_counts_and_zero_bins_retain_poisson_covariance() {
    let mut input = request(BackgroundKind::None);
    for count in &mut input.y {
        *count = (*count * 0.05).round();
    }
    assert!(input.y.contains(&0.0));
    let result = fit_peaks(
        &input,
        &FitOptions {
            objective: ObjectiveKind::PoissonDeviance,
            ..FitOptions::robust()
        },
    )
    .expect("low count fit");
    assert!(
        result.fit.termination.success,
        "{:?}",
        result.fit.diagnostics
    );
    assert!(
        result.fit.covariance.is_some(),
        "{:?}",
        result.fit.diagnostics
    );
    assert!(result.fit.confidence_band.is_some());
}

#[test]
fn all_background_families_support_window_unions_and_locked_coefficients() {
    for (background, coefficients) in [
        (BackgroundKind::Constant, vec![("bg_c", 3.0)]),
        (
            BackgroundKind::Linear,
            vec![("bg_slope", 0.3), ("bg_intercept", 2.0)],
        ),
        (
            BackgroundKind::Quadratic,
            vec![("bg_a", 0.02), ("bg_b", 0.3), ("bg_c", 2.0)],
        ),
        (
            BackgroundKind::Exponential,
            vec![("bg_amplitude", 10.0), ("bg_decay", 5.0)],
        ),
        (
            BackgroundKind::PowerLaw,
            vec![("bg_amplitude", 10.0), ("bg_exponent", -0.7)],
        ),
    ] {
        for objective in [ObjectiveKind::LeastSquares, ObjectiveKind::PoissonDeviance] {
            for locked in [false, true] {
                let mut input = request(background);
                input.region = [3.0, 7.0];
                input.background_markers = vec![(1.0, 3.5), (2.0, 4.0), (6.5, 9.0)];
                if locked {
                    input.background_seed = Some(BackgroundSeed {
                        parameters: coefficients
                            .iter()
                            .map(|(name, value)| ParameterDefinition::fixed(*name, *value))
                            .collect(),
                    });
                    input.background_coupling = BackgroundCoupling::PrefitFrozen;
                }
                let result = fit_peaks(
                    &input,
                    &FitOptions {
                        objective,
                        ..FitOptions::robust()
                    },
                )
                .expect("window fit");
                assert!(
                    result.fit.termination.success,
                    "{background:?}/{objective:?}/{locked}: {:?}",
                    result.fit.diagnostics
                );
                assert!(
                    result.fit.covariance.is_some(),
                    "{background:?}/{objective:?}/{locked}: {:?}",
                    result.fit.diagnostics
                );
                assert!((parameter(&result, "g0_sigma") - 0.6).abs() < 1.0e-5);
                for (name, value) in &coefficients {
                    assert!((parameter(&result, name) - value).abs() < 1.0e-4);
                }
                let expected_x = input
                    .x
                    .iter()
                    .copied()
                    .filter(|x| !locked || (3.0..=7.0).contains(x))
                    .collect::<Vec<_>>();
                assert_eq!(result.fit.observation_x, expected_x);
            }
        }
    }
}

#[test]
fn insufficient_observations_and_numerical_failure_are_distinct() {
    let insufficient = fit(
        &FitProblem::new(
            Box::new(spectrix_fitting::LinearModel::new("", [1.0, 1.0])),
            vec![0.0],
            vec![1.0],
        ),
        &FitOptions::robust(),
    );
    assert!(matches!(
        insufficient,
        Err(spectrix_fitting::FitError::InsufficientDegreesOfFreedom {
            observations: 1,
            variables: 2
        })
    ));
    let invalid_mean =
        ConstantModel::new("", [-1.0]).with_parameters([ParameterDefinition::fixed("c", -1.0)]);
    let result = fit(
        &FitProblem::new(Box::new(invalid_mean), vec![0.0, 1.0], vec![1.0, 2.0]),
        &FitOptions {
            objective: ObjectiveKind::PoissonDeviance,
            ..FitOptions::robust()
        },
    )
    .expect("invalid Poisson model diagnostic");
    assert_eq!(
        result.diagnostics.covariance_status,
        CovarianceStatus::NumericalFailure
    );
    assert!(!result.termination.success);
    assert!(result.covariance.is_none());
}

#[test]
fn two_window_bins_determine_a_line_without_manufactured_least_squares_errors() {
    let result = spectrix_fitting::fit_background(
        &spectrix_fitting::BackgroundFitRequest {
            x: vec![0.5, 1.5, 2.5, 3.5],
            y: vec![5.0, 200.0, 300.0, 2.0],
            bin_width: 1.0,
            region: [0.0, 4.0],
            markers: vec![(0.0, 1.0), (3.0, 4.0)],
            kind: BackgroundKind::Linear,
            seed: None,
        },
        &FitOptions::robust(),
    )
    .expect("line through the two window bins");
    assert!(result.termination.success);
    assert_eq!(result.observation_x, [0.5, 3.5]);
    assert_eq!(result.statistics.degrees_of_freedom, 0);
    assert_eq!(
        result.diagnostics.covariance_status,
        CovarianceStatus::InsufficientInformation
    );
    assert!(result.covariance.is_none());
    assert!(
        result
            .parameters
            .iter()
            .all(|parameter| parameter.standard_error.is_none())
    );
    assert!(
        result
            .raw_residuals
            .iter()
            .all(|residual| residual.abs() < 1.0e-8)
    );
    #[cfg(feature = "serde")]
    {
        let encoded = serde_json::to_string(&result).expect("save exact window fit");
        let restored: spectrix_fitting::FitResult =
            serde_json::from_str(&encoded).expect("restore exact window fit");
        assert_eq!(restored.parameters, result.parameters);
        assert_eq!(restored.statistics, result.statistics);
        assert_eq!(restored.diagnostics, result.diagnostics);
        assert_eq!(restored.observation_x, result.observation_x);
        assert_eq!(restored.best_fit.len(), result.best_fit.len());
        assert!(
            restored
                .best_fit
                .iter()
                .zip(&result.best_fit)
                .all(|(restored, original)| (restored - original).abs() < 1.0e-12)
        );
    }
}
