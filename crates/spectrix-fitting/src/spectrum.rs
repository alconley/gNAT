use std::cmp::Ordering;

use crate::{
    Bound, Bounds, CompositeModel, ConstantModel, DerivedParameter, ExponentialModel, FitError,
    FitOptions, FitProblem, FitResult, LinearModel, Model, ModelComponent, ObjectiveKind,
    ParameterBinding, ParameterDefinition, ParameterKind, ParameterValues, PowerLawModel,
    QuadraticModel, SolverProfile, fit,
};

/// How background seed values participate in the composite peak fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum BackgroundCoupling {
    /// Keep supplied background seed values fixed during the peak fit.
    #[default]
    PrefitFrozen,
    /// Robustly initialize and vary enabled background parameters jointly with peaks.
    PrefitJoint,
}

/// Background equations available in version 0.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum BackgroundKind {
    /// A fixed zero background.
    #[default]
    None,
    /// A varying constant background.
    Constant,
    /// `slope * x + intercept`.
    Linear,
    /// `a * x^2 + b * x + c`.
    Quadratic,
    /// `amplitude * exp(-x / decay)`.
    Exponential,
    /// `amplitude * x^exponent`.
    PowerLaw,
}

/// Advisory quality classification for a completed fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum FitQualityStatus {
    /// The solver converged without an objective-specific warning.
    #[default]
    Good,
    /// The result is usable but deserves inspection.
    Review,
    /// The result is statistically poor or worsened its starting objective.
    Poor,
    /// The solver or returned model is numerically invalid.
    Failed,
}

/// A structured, advisory fit-quality diagnostic.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum FitQualityIssue {
    /// The nonlinear solver did not report successful convergence.
    FailedConvergence {
        /// Stable solver reason or error detail.
        reason: String,
    },
    /// Uncertainty estimates could not be calculated.
    MissingCovariance,
    /// At least one fitted value or reported curve is not finite.
    NonFiniteResult,
    /// One or more free parameters stopped on a bound.
    ActiveBounds {
        /// Names of the free parameters touching their bounds.
        parameters: Vec<String>,
    },
    /// The final Poisson model has an invalid expected count in at least one bin.
    NonpositivePrediction,
    /// The returned parameters increased the selected objective.
    ObjectiveWorsened {
        /// Objective at the exact initial parameters.
        initial: f64,
        /// Objective at the returned parameters.
        final_value: f64,
    },
    /// Poisson deviance is unlikely under the fitted model.
    PoorGoodnessOfFit {
        /// Chi-square survival probability for the fitted deviance.
        p_value: f64,
    },
    /// Significant positive peaks remain in the fit residuals.
    UnmodeledResidualPeaks {
        /// X positions of the significant residual maxima.
        positions: Vec<f64>,
    },
}

/// Optional initial background parameter definitions.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct BackgroundSeed {
    /// Definitions keyed with lmfit-style `bg_` names (for example `bg_slope`).
    pub parameters: Vec<ParameterDefinition>,
}

/// Per-peak or shared sigma constraints.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct SigmaBounds {
    /// Lower bounds; one entry for equal sigma or one per peak otherwise.
    pub minima: Vec<f64>,
    /// Upper bounds; one entry for equal sigma or one per peak otherwise.
    pub maxima: Vec<f64>,
}

/// An explicit, user-visible starting point for one Gaussian component.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ManualPeakSeed {
    /// Initial Gaussian center. The fitter never snaps or replaces this value.
    pub center: f64,
    /// Initial Gaussian standard deviation.
    pub sigma: f64,
    /// Initial unit-normalized Gaussian amplitude (integral in x/y units).
    pub amplitude: f64,
}

/// User-visible convergence limits for one manual Gaussian seed.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ManualPeakBounds {
    /// Inclusive center interval.
    pub center: [f64; 2],
    /// Inclusive Gaussian sigma interval.
    pub sigma: [f64; 2],
    /// Inclusive background-subtracted bin-height interval at the seed center.
    pub net_height: [f64; 2],
}

/// Diagnostics produced while estimating a manual peak seed from marker-driven data.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ManualPeakEstimate {
    /// Explicit seed to display and pass to [`fit_peaks`].
    pub seed: ManualPeakSeed,
    /// Recommended visible convergence limits derived from the ownership cell and signal.
    pub bounds: ManualPeakBounds,
    /// Background-subtracted data height at the exact marker position.
    pub net_height: f64,
    /// Whether both interpolated half-height crossings were found in the ownership cell.
    pub clean_width: bool,
    /// Whether the estimate contains a positive, finite width and amplitude.
    pub valid: bool,
}

/// Marker-driven request for estimating visible manual Gaussian starting values.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ManualSeedEstimateRequest {
    /// Bin centers.
    pub x: Vec<f64>,
    /// Bin counts.
    pub y: Vec<f64>,
    /// Positive bin width.
    pub bin_width: f64,
    /// Inclusive fit region.
    pub region: [f64; 2],
    /// Exact user-supplied peak marker positions.
    pub peak_markers: Vec<f64>,
    /// Inclusive user-supplied background windows.
    pub background_markers: Vec<(f64, f64)>,
    /// Explicitly selected background family.
    pub background: BackgroundKind,
    /// Optional background parameter definitions and constraints.
    pub background_seed: Option<BackgroundSeed>,
    /// Whether the returned peak estimates should share the strongest clean width.
    pub equal_sigma: bool,
}

/// Visible manual peak estimates plus the background fit used to obtain them.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ManualSeedEstimate {
    /// Sorted estimates corresponding one-for-one with the supplied markers.
    pub peaks: Vec<ManualPeakEstimate>,
    /// Background-only fit over the explicit background windows.
    pub background_prefit: FitResult,
}

/// Evaluates one explicit manual peak with the same bin-integrated Gaussian used by fitting.
pub fn evaluate_manual_peak(seed: ManualPeakSeed, x: f64, bin_width: f64) -> Result<f64, FitError> {
    if !x.is_finite()
        || !bin_width.is_finite()
        || bin_width <= 0.0
        || !seed.center.is_finite()
        || !seed.sigma.is_finite()
        || seed.sigma <= 0.0
        || !seed.amplitude.is_finite()
        || seed.amplitude < 0.0
    {
        return Err(FitError::InvalidParameter {
            parameter: "manual peak evaluation".to_owned(),
        });
    }
    Ok(seed.amplitude * integrated_gaussian(x, seed.center, seed.sigma, bin_width))
}

/// Request for fitting only a spectrum background.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct BackgroundFitRequest {
    /// Bin centers or independent values.
    pub x: Vec<f64>,
    /// Counts or dependent values.
    pub y: Vec<f64>,
    /// Positive bin width.
    pub bin_width: f64,
    /// Inclusive region; marker order is normalized.
    pub region: [f64; 2],
    /// Inclusive background windows. The robust profile estimates from the region when empty.
    pub markers: Vec<(f64, f64)>,
    /// Background equation.
    pub kind: BackgroundKind,
    /// Optional bounds, initial values, and fixed/varying settings.
    pub seed: Option<BackgroundSeed>,
}

/// Request for Gaussian peak fitting with an optional background.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct PeakFitRequest {
    /// Bin centers.
    pub x: Vec<f64>,
    /// Counts.
    pub y: Vec<f64>,
    /// Positive bin width. Spectrix area is amplitude divided by this value.
    pub bin_width: f64,
    /// Inclusive fit region; marker order is normalized.
    pub region: [f64; 2],
    /// Explicit, user-visible Gaussian starting values. Empty is an error.
    pub peak_seeds: Vec<ManualPeakSeed>,
    /// Optional one-for-one visible convergence limits for the supplied seeds.
    pub peak_bounds: Option<Vec<ManualPeakBounds>>,
    /// Inclusive background windows. The robust profile estimates from the region when empty.
    pub background_markers: Vec<(f64, f64)>,
    /// Explicit background equation.
    pub background: BackgroundKind,
    /// Optional background initial values and constraints.
    pub background_seed: Option<BackgroundSeed>,
    /// Background/peak covariance behavior.
    pub background_coupling: BackgroundCoupling,
    /// Whether every Gaussian shares the first seed's sigma.
    pub equal_sigma: bool,
    /// Whether Gaussian centers may vary.
    pub free_centers: bool,
    /// Optional sigma constraints.
    pub sigma_bounds: Option<SigmaBounds>,
}

/// Deterministic manual peak fit plus its explicit background prefit.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct SpectrumFitResult {
    /// Composite Gaussian/background fit.
    pub fit: FitResult,
    /// Initialization from window bins or a peak-resistant region estimate.
    pub background_prefit: FitResult,
    /// Sorted inclusive region used by the fit.
    pub region: [f64; 2],
    /// Sorted, in-region peak markers used for initialization.
    pub peak_markers: Vec<f64>,
    /// Background coupling used by the composite fit.
    pub background_coupling: BackgroundCoupling,
    /// Exact sorted manual seeds used to construct the fit.
    pub peak_seeds: Vec<ManualPeakSeed>,
    /// Initial parameter definitions before composite optimization and recovery.
    pub initial_parameters: Vec<ParameterDefinition>,
    /// Advisory diagnostics attached to the completed result.
    pub quality_issues: Vec<FitQualityIssue>,
    /// Objective-aware advisory classification.
    #[cfg_attr(feature = "serde", serde(default))]
    pub quality_status: FitQualityStatus,
}

impl SpectrumFitResult {
    /// Returns true because fit-quality findings are advisory and never block storage.
    #[deprecated(note = "fit quality is advisory; inspect quality_status instead")]
    #[must_use]
    pub fn is_storable(&self) -> bool {
        true
    }
}

/// Fits the selected background family.
///
/// The robust profile uses a peak-resistant region estimate when no windows are supplied.
/// [`BackgroundKind::None`] evaluates a fixed zero baseline and needs no windows.
pub fn fit_background(
    request: &BackgroundFitRequest,
    options: &FitOptions,
) -> Result<FitResult, FitError> {
    validate_data(&request.x, &request.y, request.bin_width)?;
    let region = sorted_region(request.region)?;
    if options.profile == SolverProfile::Lmfit134 {
        require_manual_background_markers(request.kind, &request.markers)?;
    }
    let automatic = request.kind != BackgroundKind::None && request.markers.is_empty();
    let (x, y) = if request.kind == BackgroundKind::None || automatic {
        region_data(&request.x, &request.y, region)?
    } else if options.profile == SolverProfile::Robust {
        unique_window_data(&request.x, &request.y, &request.markers)?
    } else {
        background_data(&request.x, &request.y, &request.markers)?
    };
    validate_power_law_domain(request.kind, &x)?;
    if automatic {
        return automatic_background(request, options, region, &x, &y);
    }
    let initial_values = robust_background_values(request.kind, &x, &y, region);
    let definitions = background_definitions(
        request.kind,
        request.seed.as_ref(),
        Some(&initial_values),
        true,
    );
    let model = background_model(request.kind, &definitions, region);
    let mut background_options = options.clone();
    if background_options.evaluation_x.is_none() {
        background_options.evaluation_x = Some(linspace(region[0], region[1], 256)?);
    }
    fit(&FitProblem::new(model, x, y), &background_options)
}

/// Estimates explicit Gaussian starting values from user-supplied markers.
///
/// The marker positions remain exact. Only widths and amplitudes are inferred from the
/// background-subtracted data, and no components are inserted or removed.
pub fn estimate_manual_peak_seeds(
    request: &ManualSeedEstimateRequest,
    options: &FitOptions,
) -> Result<ManualSeedEstimate, FitError> {
    validate_data(&request.x, &request.y, request.bin_width)?;
    let region = sorted_region(request.region)?;
    let markers = validated_manual_markers(&request.peak_markers, region, request.bin_width)?;
    if options.profile == SolverProfile::Lmfit134 {
        require_manual_background_markers(request.background, &request.background_markers)?;
    }
    let (region_x, _) = region_data(&request.x, &request.y, region)?;
    validate_power_law_domain(request.background, &region_x)?;

    let background_prefit = fit_background(
        &BackgroundFitRequest {
            x: request.x.clone(),
            y: request.y.clone(),
            bin_width: request.bin_width,
            region,
            markers: request.background_markers.clone(),
            kind: request.background,
            seed: request.background_seed.clone(),
        },
        options,
    )?;
    let (x, y) = region_data(&request.x, &request.y, region)?;
    let background = evaluate_background_fit(request.background, &background_prefit, &x)?;
    let signal = y
        .iter()
        .zip(&background)
        .map(|(observed, baseline)| observed - baseline)
        .collect::<Vec<_>>();
    let peaks = estimate_manual_components(
        &x,
        &signal,
        &markers,
        region,
        request.bin_width,
        request.equal_sigma,
    );
    Ok(ManualSeedEstimate {
        peaks,
        background_prefit,
    })
}

/// Fits exactly the supplied Gaussian components. Robust fits polish the solution
/// and recover deterministically when the first solution is unreliable.
#[expect(
    clippy::too_many_lines,
    reason = "manual model assembly is intentionally explicit"
)]
pub fn fit_peaks(
    request: &PeakFitRequest,
    options: &FitOptions,
) -> Result<SpectrumFitResult, FitError> {
    validate_data(&request.x, &request.y, request.bin_width)?;
    let region = sorted_region(request.region)?;
    let (region_x, region_y) = region_data(&request.x, &request.y, region)?;
    if region_x.len() < 2 {
        return Err(FitError::InvalidRegion);
    }
    if options.profile == SolverProfile::Lmfit134 {
        require_manual_background_markers(request.background, &request.background_markers)?;
    }
    validate_power_law_domain(request.background, &region_x)?;
    let peak_seeds = validated_manual_seeds(&request.peak_seeds, region, request.bin_width)?;
    let peak_bounds = validated_manual_peak_bounds(
        request.peak_bounds.as_deref(),
        &request.peak_seeds,
        &peak_seeds,
        request.bin_width,
        region,
    )?;

    let background_prefit = fit_background(
        &BackgroundFitRequest {
            x: request.x.clone(),
            y: request.y.clone(),
            bin_width: request.bin_width,
            region,
            markers: request.background_markers.clone(),
            kind: request.background,
            seed: request.background_seed.clone(),
        },
        options,
    )?;
    let fitted_background_values = background_prefit
        .parameters
        .iter()
        .filter(|parameter| parameter.name.starts_with("bg_"))
        .map(|parameter| (parameter.name.clone(), parameter.value))
        .collect::<Vec<_>>();
    let vary_background = request.background_coupling == BackgroundCoupling::PrefitJoint;
    let background_definitions = definitions_from_background_fit(
        request.background,
        request.background_seed.as_ref(),
        &fitted_background_values,
        &background_prefit,
        vary_background,
    );

    let sigma_limits = normalized_sigma_bounds(
        request.sigma_bounds.as_ref(),
        request.equal_sigma,
        peak_seeds.len(),
    )?;
    let shared_sigma = peak_seeds[0].sigma;
    let states = peak_seeds
        .iter()
        .enumerate()
        .map(|(index, seed)| {
            let constraint_index = if request.equal_sigma { 0 } else { index };
            let suggested_sigma = peak_bounds.as_ref().map(|bounds| {
                if request.equal_sigma {
                    (
                        bounds
                            .iter()
                            .map(|bound| bound.sigma[0])
                            .fold(f64::NEG_INFINITY, f64::max),
                        bounds
                            .iter()
                            .map(|bound| bound.sigma[1])
                            .fold(f64::INFINITY, f64::min),
                    )
                } else {
                    (bounds[index].sigma[0], bounds[index].sigma[1])
                }
            });
            let explicit_sigma = sigma_limits
                .as_ref()
                .map(|(minimum, maximum)| (minimum[constraint_index], maximum[constraint_index]));
            let sigma_bounds = match (suggested_sigma, explicit_sigma) {
                (Some(suggested), Some(explicit)) => {
                    Bounds::finite(suggested.0.max(explicit.0), suggested.1.min(explicit.1))
                }
                (Some((minimum, maximum)), None) | (None, Some((minimum, maximum))) => {
                    Bounds::finite(minimum, maximum)
                }
                (None, None) => Bounds::lower_bounded(f64::EPSILON),
            };
            let center_bounds = peak_bounds.as_ref().map_or_else(
                || manual_center_bounds(index, &peak_seeds, region),
                |bounds| Bounds::finite(bounds[index].center[0], bounds[index].center[1]),
            );
            let center_response =
                integrated_gaussian(seed.center, seed.center, seed.sigma, request.bin_width);
            PeakState {
                height: seed.amplitude * center_response,
                height_bounds: peak_bounds.as_ref().map_or_else(
                    || Bounds::lower_bounded(0.0),
                    |bounds| {
                        Bounds::finite(bounds[index].net_height[0], bounds[index].net_height[1])
                    },
                ),
                center: seed.center,
                sigma: if request.equal_sigma {
                    shared_sigma
                } else {
                    seed.sigma
                },
                center_bounds,
                sigma_bounds,
            }
        })
        .collect::<Vec<_>>();
    let model = build_peak_model(
        request.background,
        &background_definitions,
        region,
        request.bin_width,
        &states,
        request.equal_sigma,
        request.free_centers,
        true,
    )?;
    let initial_parameters = model.parameter_definitions();
    let mut fit_options = options.clone();
    if fit_options.evaluation_x.is_none() {
        let evaluation_count = region_x.len().saturating_mul(4).max(region_x.len());
        fit_options.evaluation_x = Some(linspace(
            region_x[0],
            region_x[region_x.len() - 1],
            evaluation_count,
        )?);
    }
    let (observation_x, observation_y) = composite_data(request, options, region)?;
    validate_power_law_domain(request.background, &observation_x)?;
    let starts = if options.profile == SolverProfile::Robust {
        recovery_starts(
            request,
            region,
            &states,
            &background_definitions,
            &region_x,
            &region_y,
        )?
    } else {
        Vec::new()
    };
    let fit = crate::solver::fit_with_starts(
        &FitProblem::new(model, observation_x.clone(), observation_y.clone()),
        &fit_options,
        &starts,
    )?;
    let (quality_status, quality_issues) = assess_quality(
        &fit,
        &observation_x,
        &observation_y,
        request.bin_width,
        peak_seeds.len(),
    );
    Ok(SpectrumFitResult {
        fit,
        background_prefit,
        region,
        peak_markers: peak_seeds.iter().map(|seed| seed.center).collect(),
        background_coupling: request.background_coupling,
        peak_seeds,
        initial_parameters,
        quality_issues,
        quality_status,
    })
}

fn composite_data(
    request: &PeakFitRequest,
    options: &FitOptions,
    region: [f64; 2],
) -> Result<(Vec<f64>, Vec<f64>), FitError> {
    if options.profile != SolverProfile::Robust
        || request.background == BackgroundKind::None
        || request.background_coupling == BackgroundCoupling::PrefitFrozen
        || request.background_markers.is_empty()
    {
        return region_data(&request.x, &request.y, region);
    }
    // Select each original bin once, even if several windows overlap the region.
    let mut windows = request.background_markers.clone();
    windows.push((region[0], region[1]));
    unique_window_data(&request.x, &request.y, &windows)
}

fn unique_window_data(
    x: &[f64],
    y: &[f64],
    windows: &[(f64, f64)],
) -> Result<(Vec<f64>, Vec<f64>), FitError> {
    if windows
        .iter()
        .any(|(start, end)| !start.is_finite() || !end.is_finite())
    {
        return Err(FitError::NonFinite {
            context: "background windows".to_owned(),
        });
    }
    let mut selected = x
        .iter()
        .copied()
        .zip(y.iter().copied())
        .filter(|(x, _)| {
            windows
                .iter()
                .any(|(start, end)| *x >= start.min(*end) && *x <= start.max(*end))
        })
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(FitError::InvalidRegion);
    }
    selected.sort_by(|left, right| left.0.total_cmp(&right.0));
    Ok(selected.into_iter().unzip())
}

fn automatic_background(
    request: &BackgroundFitRequest,
    options: &FitOptions,
    region: [f64; 2],
    x: &[f64],
    y: &[f64],
) -> Result<FitResult, FitError> {
    let initial = robust_background_values(request.kind, x, y, region);
    let mut definitions =
        background_definitions(request.kind, request.seed.as_ref(), Some(&initial), true);
    let mut weights = vec![1.0; x.len()];
    let mut estimate_options = options.clone();
    estimate_options.objective = ObjectiveKind::LeastSquares;
    // This is an initialization, not an uncertainty model for the final fit.
    estimate_options.calculate_covariance = false;
    estimate_options.evaluation_x = Some(x.to_vec());
    let mut result = None;
    for _ in 0..10 {
        let fitted = fit(
            &FitProblem::new(
                background_model(request.kind, &definitions, region),
                x.to_vec(),
                y.to_vec(),
            )
            .with_weights(weights.clone()),
            &estimate_options,
        )?;
        for (weight, residual) in weights.iter_mut().zip(&fitted.raw_residuals) {
            // FitProblem weights multiply residuals, so use the square root of
            // the asymmetric least-squares weight.
            *weight = if *residual > 0.0 {
                0.05_f64.sqrt()
            } else {
                0.95_f64.sqrt()
            };
        }
        for definition in &mut definitions {
            if let Some(parameter) = fitted
                .parameters
                .iter()
                .find(|parameter| parameter.name == definition.name)
            {
                definition.initial = clamp_to_bounds(parameter.value, definition.bounds);
            }
        }
        result = Some(fitted);
    }
    result.ok_or_else(|| FitError::Solver {
        message: "automatic background produced no estimate".to_owned(),
    })
}

fn recovery_starts(
    request: &PeakFitRequest,
    region: [f64; 2],
    states: &[PeakState],
    background: &[ParameterDefinition],
    x: &[f64],
    y: &[f64],
) -> Result<Vec<ParameterValues>, FitError> {
    let baseline = evaluate_background_definitions(request.background, background, x)?;
    let signal = y
        .iter()
        .zip(baseline)
        .map(|(count, level)| count - level)
        .collect::<Vec<_>>();
    let markers = states.iter().map(|state| state.center).collect::<Vec<_>>();
    let estimated = estimate_manual_components(
        x,
        &signal,
        &markers,
        region,
        request.bin_width,
        request.equal_sigma,
    );
    let mut starts = Vec::new();
    for factor in [None, Some(0.5), Some(2.0)] {
        let mut alternative = states.to_vec();
        for (index, state) in alternative.iter_mut().enumerate() {
            let proposed = factor.map_or_else(
                || {
                    estimated
                        .get(index)
                        .filter(|estimate| estimate.valid)
                        .map_or(state.sigma, |estimate| estimate.seed.sigma)
                },
                |factor| state.sigma * factor,
            );
            state.sigma = clamp_to_bounds(proposed, state.sigma_bounds);
        }
        let model = build_peak_model(
            request.background,
            background,
            region,
            request.bin_width,
            &alternative,
            request.equal_sigma,
            request.free_centers,
            true,
        )?;
        let mut values = ParameterValues::new();
        for definition in model.parameter_definitions() {
            values.insert(definition.name, definition.initial);
        }
        starts.push(values);
    }
    Ok(starts)
}

fn require_manual_background_markers(
    background: BackgroundKind,
    markers: &[(f64, f64)],
) -> Result<(), FitError> {
    if background != BackgroundKind::None && markers.is_empty() {
        return Err(FitError::InvalidParameter {
            parameter: "at least one explicit background marker window is required".to_owned(),
        });
    }
    Ok(())
}

fn validated_manual_markers(
    markers: &[f64],
    region: [f64; 2],
    bin_width: f64,
) -> Result<Vec<f64>, FitError> {
    if markers.is_empty() {
        return Err(FitError::InvalidParameter {
            parameter: "at least one in-region peak marker is required".to_owned(),
        });
    }
    if let Some((index, marker)) = markers
        .iter()
        .enumerate()
        .find(|(_, marker)| !marker.is_finite())
    {
        return Err(FitError::InvalidParameter {
            parameter: format!("peak marker {} ({marker}) is not finite", index + 1),
        });
    }
    if let Some((index, marker)) = markers
        .iter()
        .enumerate()
        .find(|(_, marker)| **marker < region[0] || **marker > region[1])
    {
        return Err(FitError::InvalidParameter {
            parameter: format!(
                "peak marker {} ({marker:.6}) is outside fit region {:.6}..{:.6}",
                index + 1,
                region[0],
                region[1]
            ),
        });
    }
    let mut sorted = markers.to_vec();
    sorted.sort_by(f64::total_cmp);
    if let Some([first, second]) = sorted
        .windows(2)
        .find(|pair| (pair[1] - pair[0]).abs() < bin_width)
    {
        return Err(FitError::InvalidParameter {
            parameter: format!(
                "peak markers {first:.6} and {second:.6} are duplicate within one bin"
            ),
        });
    }
    Ok(sorted)
}

fn validated_manual_seeds(
    seeds: &[ManualPeakSeed],
    region: [f64; 2],
    bin_width: f64,
) -> Result<Vec<ManualPeakSeed>, FitError> {
    if seeds.is_empty() {
        return Err(FitError::InvalidParameter {
            parameter: "at least one explicit manual peak seed is required".to_owned(),
        });
    }
    for (index, seed) in seeds.iter().enumerate() {
        if !seed.center.is_finite()
            || !seed.sigma.is_finite()
            || !seed.amplitude.is_finite()
            || seed.sigma <= 0.0
            || seed.amplitude <= 0.0
        {
            return Err(FitError::InvalidParameter {
                parameter: format!(
                    "manual peak seed {} must have a finite center and positive finite sigma/amplitude",
                    index + 1
                ),
            });
        }
    }
    let markers = seeds.iter().map(|seed| seed.center).collect::<Vec<_>>();
    let sorted_markers = validated_manual_markers(&markers, region, bin_width)?;
    Ok(sorted_markers
        .into_iter()
        .map(|marker| {
            seeds
                .iter()
                .find(|seed| seed.center == marker)
                .copied()
                .expect("validated marker originates from a seed")
        })
        .collect())
}

fn validated_manual_peak_bounds(
    bounds: Option<&[ManualPeakBounds]>,
    original_seeds: &[ManualPeakSeed],
    sorted_seeds: &[ManualPeakSeed],
    bin_width: f64,
    region: [f64; 2],
) -> Result<Option<Vec<ManualPeakBounds>>, FitError> {
    let Some(bounds) = bounds else {
        return Ok(None);
    };
    if bounds.len() != original_seeds.len() {
        return Err(FitError::InvalidParameter {
            parameter: "manual peak bounds must correspond one-for-one with peak seeds".to_owned(),
        });
    }
    let sorted_bounds = sorted_seeds
        .iter()
        .map(|seed| {
            let index = original_seeds
                .iter()
                .position(|original| original.center == seed.center)
                .expect("validated seed originates from the request");
            bounds[index]
        })
        .collect::<Vec<_>>();
    for (index, (bound, seed)) in sorted_bounds.iter().zip(sorted_seeds).enumerate() {
        let response = integrated_gaussian(seed.center, seed.center, seed.sigma, bin_width);
        let seed_height = seed.amplitude * response;
        for (label, interval, value) in [
            ("center", bound.center, seed.center),
            ("sigma", bound.sigma, seed.sigma),
            ("net height", bound.net_height, seed_height),
        ] {
            if interval.iter().any(|entry| !entry.is_finite())
                || interval[0] >= interval[1]
                || value < interval[0]
                || value > interval[1]
                || (label != "center" && interval[0] < 0.0)
            {
                return Err(FitError::InvalidBounds {
                    parameter: format!("peak {} {label}", index + 1),
                });
            }
        }
        if bound.center[0] < region[0] || bound.center[1] > region[1] {
            return Err(FitError::InvalidBounds {
                parameter: format!("peak {} center outside fit region", index + 1),
            });
        }
    }
    Ok(Some(sorted_bounds))
}

fn manual_center_bounds(index: usize, seeds: &[ManualPeakSeed], region: [f64; 2]) -> Bounds {
    let lower = if index == 0 {
        region[0]
    } else {
        0.5 * (seeds[index - 1].center + seeds[index].center)
    };
    let upper = if index + 1 == seeds.len() {
        region[1]
    } else {
        0.5 * (seeds[index].center + seeds[index + 1].center)
    };
    Bounds::finite(lower, upper)
}

#[derive(Debug, Clone, Copy)]
struct ManualPeakDraft {
    center: f64,
    sigma: f64,
    net_height: f64,
    clean_width: bool,
    cell: [f64; 2],
    sigma_limits: [f64; 2],
}

#[expect(
    clippy::too_many_lines,
    reason = "the ordered seed-estimation stages share intermediate peak geometry and are kept together for auditability"
)]
fn estimate_manual_components(
    x: &[f64],
    signal: &[f64],
    markers: &[f64],
    region: [f64; 2],
    bin_width: f64,
    equal_sigma: bool,
) -> Vec<ManualPeakEstimate> {
    let smoothed = smooth_one_bin(signal);
    let mut drafts = markers
        .iter()
        .copied()
        .enumerate()
        .map(|(index, center)| {
            let cell_min = if index == 0 {
                region[0]
            } else {
                0.5 * (markers[index - 1] + center)
            };
            let cell_max = if index + 1 == markers.len() {
                region[1]
            } else {
                0.5 * (center + markers[index + 1])
            };
            let cell_width = (cell_max - cell_min).max(bin_width);
            let peak_index = nearest_index(x, center);
            // Smoothing is exclusively a width-detection aid. The visible height seed must be
            // the background-subtracted data value at the exact marker position.
            let net_height = interpolated_value(x, signal, center).max(0.0);
            let width_height = interpolated_value(x, &smoothed, center).max(0.0);
            let half_height = 0.5 * width_height;
            let cell_start = x.partition_point(|value| *value < cell_min);
            let cell_end = x.partition_point(|value| *value <= cell_max);
            let left = (net_height > 0.0)
                .then(|| {
                    interpolated_left_crossing(x, &smoothed, cell_start, peak_index, half_height)
                })
                .flatten();
            let right = (net_height > 0.0)
                .then(|| {
                    interpolated_right_crossing(x, &smoothed, peak_index, cell_end, half_height)
                })
                .flatten();
            let clean_width = matches!((left, right), (Some(left), Some(right)) if right > left);
            let crossing_sigma = match (left, right) {
                (Some(left), Some(right)) if right > left => Some((right - left) / 2.354_82),
                (Some(left), None) if center > left => Some(2.0 * (center - left) / 2.354_82),
                (None, Some(right)) if right > center => Some(2.0 * (right - center) / 2.354_82),
                _ => None,
            };
            let curvature_sigma = if peak_index > 0 && peak_index + 1 < smoothed.len() {
                let spacing = 0.5 * (x[peak_index + 1] - x[peak_index - 1]);
                let curvature = (smoothed[peak_index - 1] - 2.0 * smoothed[peak_index]
                    + smoothed[peak_index + 1])
                    / spacing.powi(2);
                (curvature < 0.0 && width_height > 0.0).then(|| (-width_height / curvature).sqrt())
            } else {
                None
            };
            let moment_sigma = positive_second_moment(x, signal, center, cell_min, cell_max);
            let fallback_sigma = (0.15 * cell_width).max(bin_width);
            let minimum_sigma = (0.5 * bin_width).max(f64::EPSILON);
            let maximum_sigma = (0.45 * cell_width).max(minimum_sigma * 1.01);
            let sigma = crossing_sigma
                .filter(|value| value.is_finite() && *value > 0.0)
                .or_else(|| curvature_sigma.filter(|value| value.is_finite() && *value > 0.0))
                .or_else(|| moment_sigma.filter(|value| value.is_finite() && *value > 0.0))
                .unwrap_or(fallback_sigma)
                .clamp(minimum_sigma, maximum_sigma);
            ManualPeakDraft {
                center,
                sigma,
                net_height,
                clean_width,
                cell: [cell_min, cell_max],
                sigma_limits: [minimum_sigma, maximum_sigma],
            }
        })
        .collect::<Vec<_>>();

    if equal_sigma
        && let Some(shared_sigma) = drafts
            .iter()
            .filter(|draft| draft.clean_width && draft.net_height > 0.0)
            .max_by(|left, right| left.net_height.total_cmp(&right.net_height))
            .or_else(|| {
                drafts
                    .iter()
                    .filter(|draft| draft.net_height > 0.0)
                    .max_by(|left, right| left.net_height.total_cmp(&right.net_height))
            })
            .map(|draft| draft.sigma)
    {
        let common_minimum = drafts
            .iter()
            .map(|draft| draft.sigma_limits[0])
            .fold(f64::NEG_INFINITY, f64::max);
        let common_maximum = drafts
            .iter()
            .map(|draft| draft.sigma_limits[1])
            .fold(f64::INFINITY, f64::min);
        let shared_sigma = if common_minimum < common_maximum {
            shared_sigma.clamp(common_minimum, common_maximum)
        } else {
            shared_sigma
        };
        for draft in &mut drafts {
            draft.sigma = shared_sigma;
        }
    }

    drafts
        .into_iter()
        .map(|draft| {
            let lower = draft.cell[0].max(draft.center - 3.0 * draft.sigma);
            let upper = draft.cell[1].min(draft.center + 3.0 * draft.sigma);
            let positive_area = x
                .iter()
                .zip(signal)
                .filter(|(independent, _)| **independent >= lower && **independent <= upper)
                .map(|(_, value)| value.max(0.0) * bin_width)
                .sum::<f64>();
            let denominator = draft.sigma * std::f64::consts::SQRT_2;
            let captured_fraction = 0.5
                * (statrs::function::erf::erf((upper - draft.center) / denominator)
                    - statrs::function::erf::erf((lower - draft.center) / denominator));
            let area_amplitude = (positive_area > 0.0
                && captured_fraction.is_finite()
                && captured_fraction > 1.0e-6)
                .then_some(positive_area / captured_fraction);
            let center_response =
                integrated_gaussian(draft.center, draft.center, draft.sigma, bin_width);
            let height_amplitude = (draft.net_height > 0.0 && center_response > 0.0)
                .then_some(draft.net_height / center_response);
            let amplitude = area_amplitude.or(height_amplitude).unwrap_or(0.0);
            let preview_height = amplitude * center_response;
            // The area seed is deliberately robust to a noisy apex, but its implied height can
            // differ from the actually observed marker height. Keep both inside the initial
            // search interval so a good width estimate cannot accidentally lock the solver onto
            // a poor area-derived height.
            let height_reference_min = preview_height.min(draft.net_height);
            let height_reference_max = preview_height.max(draft.net_height);
            let height_uncertainty = height_reference_max
                .max(1.0)
                .sqrt()
                .max(0.20 * height_reference_max);
            let fwhm = 2.354_82 * draft.sigma;
            let center_half_range = (0.5 * fwhm).max(0.5 * bin_width);
            let center_bounds = [
                (draft.center - center_half_range).max(draft.cell[0]),
                (draft.center + center_half_range).min(draft.cell[1]),
            ];
            let sigma_bounds = [
                (0.5 * draft.sigma).max(draft.sigma_limits[0]),
                (2.0 * draft.sigma).min(draft.sigma_limits[1]),
            ];
            let height_bounds = [
                (height_reference_min - height_uncertainty).max(0.0),
                height_reference_max + height_uncertainty,
            ];
            let valid = draft.sigma.is_finite()
                && draft.sigma > 0.0
                && amplitude.is_finite()
                && amplitude > 0.0;
            ManualPeakEstimate {
                seed: ManualPeakSeed {
                    center: draft.center,
                    sigma: draft.sigma,
                    amplitude,
                },
                bounds: ManualPeakBounds {
                    center: center_bounds,
                    sigma: sigma_bounds,
                    net_height: height_bounds,
                },
                net_height: draft.net_height,
                clean_width: draft.clean_width,
                valid,
            }
        })
        .collect()
}

fn interpolated_value(x: &[f64], y: &[f64], target: f64) -> f64 {
    if x.is_empty() || x.len() != y.len() {
        return 0.0;
    }
    let upper = x.partition_point(|value| *value < target);
    match upper {
        0 => y[0],
        upper if upper >= x.len() => y[y.len() - 1],
        upper => {
            let x0 = x[upper - 1];
            let x1 = x[upper];
            if (x1 - x0).abs() <= f64::EPSILON {
                y[upper - 1]
            } else {
                y[upper - 1] + (target - x0) / (x1 - x0) * (y[upper] - y[upper - 1])
            }
        }
    }
}

fn positive_second_moment(
    x: &[f64],
    signal: &[f64],
    center: f64,
    lower: f64,
    upper: f64,
) -> Option<f64> {
    let (weight, moment) = x
        .iter()
        .zip(signal)
        .filter(|(independent, _)| **independent >= lower && **independent <= upper)
        .fold((0.0, 0.0), |(weight, moment), (independent, value)| {
            let positive = value.max(0.0);
            (
                weight + positive,
                moment + positive * (*independent - center).powi(2),
            )
        });
    (weight > 0.0).then(|| (moment / weight).sqrt())
}

fn smooth_one_bin(values: &[f64]) -> Vec<f64> {
    const WEIGHTS: [f64; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];
    values
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let mut weighted = 0.0;
            let mut weight = 0.0;
            for (offset, coefficient) in WEIGHTS.iter().enumerate() {
                let source = index as isize + offset as isize - 2;
                if let Ok(source) = usize::try_from(source)
                    && source < values.len()
                {
                    weighted += coefficient * values[source];
                    weight += coefficient;
                }
            }
            weighted / weight
        })
        .collect()
}

fn robust_scale(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    let mut deviations = values
        .iter()
        .map(|value| (value - median).abs())
        .collect::<Vec<_>>();
    deviations.sort_by(f64::total_cmp);
    1.482_602_218_505_602 * deviations[deviations.len() / 2]
}

#[derive(Debug, Clone, Copy)]
struct PeakState {
    height: f64,
    height_bounds: Bounds,
    center: f64,
    sigma: f64,
    center_bounds: Bounds,
    sigma_bounds: Bounds,
}

fn definitions_from_background_fit(
    kind: BackgroundKind,
    seed: Option<&BackgroundSeed>,
    initial_values: &[(String, f64)],
    fitted: &FitResult,
    allow_vary: bool,
) -> Vec<ParameterDefinition> {
    let mut definitions = background_definitions(kind, seed, Some(initial_values), allow_vary);
    for definition in &mut definitions {
        if let Some(estimate) = fitted
            .parameters
            .iter()
            .find(|estimate| estimate.name == definition.name)
            && estimate.value.is_finite()
        {
            definition.initial = clamp_to_bounds(estimate.value, definition.bounds);
        }
        definition.vary &= allow_vary;
    }
    definitions
}

fn evaluate_background_fit(
    kind: BackgroundKind,
    fitted: &FitResult,
    x: &[f64],
) -> Result<Vec<f64>, FitError> {
    let definitions = default_background_parameters(kind)
        .into_iter()
        .map(|mut definition| {
            if let Some(estimate) = fitted
                .parameters
                .iter()
                .find(|estimate| estimate.name == definition.name)
            {
                definition.initial = estimate.value;
            }
            definition.vary = false;
            definition
        })
        .collect::<Vec<_>>();
    evaluate_background_definitions(kind, &definitions, x)
}

fn integrated_gaussian(x: f64, center: f64, sigma: f64, bin_width: f64) -> f64 {
    integrated_gaussian_response(x, center, sigma, bin_width).0
}

fn integrated_gaussian_response(
    x: f64,
    center: f64,
    sigma: f64,
    bin_width: f64,
) -> (f64, f64, f64) {
    let denominator = sigma * std::f64::consts::SQRT_2;
    let lower = (x - 0.5 * bin_width - center) / denominator;
    let upper = (x + 0.5 * bin_width - center) / denominator;
    let lower_exponential = (-lower * lower).exp();
    let upper_exponential = (-upper * upper).exp();
    let value =
        0.5 * (statrs::function::erf::erf(upper) - statrs::function::erf::erf(lower)) / bin_width;
    let center_derivative = (lower_exponential - upper_exponential)
        / (2.0 * std::f64::consts::PI).sqrt()
        / sigma
        / bin_width;
    let sigma_derivative = (lower * lower_exponential - upper * upper_exponential)
        / std::f64::consts::PI.sqrt()
        / sigma
        / bin_width;
    (value, center_derivative, sigma_derivative)
}

#[derive(Debug, Clone)]
struct HeightGaussianModel {
    prefix: String,
    height: ParameterDefinition,
    center: ParameterDefinition,
    sigma: ParameterDefinition,
    bin_width: f64,
}

impl Model for HeightGaussianModel {
    fn name(&self) -> &str {
        &self.prefix
    }

    fn parameter_definitions(&self) -> Vec<ParameterDefinition> {
        vec![self.height.clone(), self.center.clone(), self.sigma.clone()]
    }

    fn evaluate(
        &self,
        x: &[f64],
        parameters: &ParameterValues,
        output: &mut [f64],
    ) -> Result<(), FitError> {
        if output.len() != x.len() {
            return Err(FitError::LengthMismatch {
                x: x.len(),
                y: output.len(),
            });
        }
        let height = parameters.require(&self.height.name)?;
        let center = parameters.require(&self.center.name)?;
        let sigma = parameters.require(&self.sigma.name)?;
        let center_response = integrated_gaussian(center, center, sigma, self.bin_width);
        if !center_response.is_finite() || center_response <= 0.0 {
            return Err(FitError::Domain {
                model: self.prefix.clone(),
                message: "height-normalized Gaussian response must be positive".to_owned(),
            });
        }
        let amplitude = height / center_response;
        for (value, independent) in output.iter_mut().zip(x) {
            *value = amplitude * integrated_gaussian(*independent, center, sigma, self.bin_width);
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(FitError::NonFinite {
                context: format!("{} Gaussian evaluation", self.prefix),
            });
        }
        Ok(())
    }

    fn analytic_jacobian(
        &self,
        x: &[f64],
        parameters: &ParameterValues,
        parameter_names: &[String],
        output: &mut [f64],
    ) -> Result<bool, FitError> {
        let columns = parameter_names.len();
        let expected = x.len().saturating_mul(columns);
        if output.len() != expected {
            return Err(FitError::LengthMismatch {
                x: expected,
                y: output.len(),
            });
        }
        let height = parameters.require(&self.height.name)?;
        let center = parameters.require(&self.center.name)?;
        let sigma = parameters.require(&self.sigma.name)?;
        if !sigma.is_finite() || sigma <= 0.0 {
            return Ok(false);
        }
        let column_for = |definition: &ParameterDefinition| -> Option<Option<usize>> {
            if let Some(column) = parameter_names
                .iter()
                .position(|name| name == &definition.name)
            {
                return Some(Some(column));
            }
            match &definition.binding {
                Some(ParameterBinding::EqualTo(source)) => parameter_names
                    .iter()
                    .position(|name| name == source)
                    .map(Some),
                None => Some(None),
            }
        };
        let Some(height_column) = column_for(&self.height) else {
            return Ok(false);
        };
        let Some(center_column) = column_for(&self.center) else {
            return Ok(false);
        };
        let Some(sigma_column) = column_for(&self.sigma) else {
            return Ok(false);
        };
        output.fill(0.0);
        let (center_response, _, center_sigma_derivative) =
            integrated_gaussian_response(center, center, sigma, self.bin_width);
        if !center_response.is_finite() || center_response <= 0.0 {
            return Ok(false);
        }
        for (row, independent) in x.iter().enumerate() {
            let (response, response_center_derivative, response_sigma_derivative) =
                integrated_gaussian_response(*independent, center, sigma, self.bin_width);
            let offset = row * columns;
            if let Some(column) = height_column {
                output[offset + column] += response / center_response;
            }
            if let Some(column) = center_column {
                output[offset + column] += height * response_center_derivative / center_response;
            }
            if let Some(column) = sigma_column {
                output[offset + column] += height
                    * (response_sigma_derivative * center_response
                        - response * center_sigma_derivative)
                    / center_response.powi(2);
            }
        }
        Ok(output.iter().all(|value| value.is_finite()))
    }

    fn derived_parameters(
        &self,
        parameters: &ParameterValues,
    ) -> Result<Vec<DerivedParameter>, FitError> {
        let height = parameters.require(&self.height.name)?;
        let center = parameters.require(&self.center.name)?;
        let sigma = parameters.require(&self.sigma.name)?;
        let (response, _, response_derivative) =
            integrated_gaussian_response(center, center, sigma, self.bin_width);
        let amplitude = height / response;
        let amplitude_gradient = vec![
            (self.height.name.clone(), 1.0 / response),
            (
                self.sigma.name.clone(),
                -height * response_derivative / response.powi(2),
            ),
        ];
        Ok(vec![
            DerivedParameter {
                name: format!("{}amplitude", self.prefix),
                value: amplitude,
                gradient: amplitude_gradient.clone(),
            },
            DerivedParameter {
                name: format!("{}fwhm", self.prefix),
                value: 2.354_82 * sigma,
                gradient: vec![(self.sigma.name.clone(), 2.354_82)],
            },
            DerivedParameter {
                name: format!("{}area", self.prefix),
                value: amplitude / self.bin_width,
                gradient: amplitude_gradient
                    .into_iter()
                    .map(|(name, value)| (name, value / self.bin_width))
                    .collect(),
            },
        ])
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "model construction mirrors the public request"
)]
fn build_peak_model(
    background: BackgroundKind,
    background_definitions: &[ParameterDefinition],
    region: [f64; 2],
    bin_width: f64,
    states: &[PeakState],
    equal_sigma: bool,
    free_centers: bool,
    vary_shapes: bool,
) -> Result<Box<dyn Model>, FitError> {
    let mut composite = CompositeModel::default();
    composite.push(ModelComponent::new(
        "background",
        background_model(background, background_definitions, region),
    ))?;
    for (index, state) in states.iter().enumerate() {
        let prefix = format!("g{index}_");
        let height = ParameterDefinition::varying(format!("{prefix}height"), state.height)
            .with_bounds(state.height_bounds);
        let mut center = ParameterDefinition::varying(format!("{prefix}center"), state.center)
            .with_bounds(state.center_bounds);
        center.vary = vary_shapes && free_centers;
        let sigma_name = format!("{prefix}sigma");
        let sigma = if equal_sigma && index > 0 {
            ParameterDefinition::fixed(sigma_name, state.sigma).equal_to("g0_sigma")
        } else {
            let mut definition = ParameterDefinition::varying(sigma_name, state.sigma)
                .with_bounds(state.sigma_bounds);
            definition.vary = vary_shapes;
            definition
        };
        let gaussian = HeightGaussianModel {
            prefix: prefix.clone(),
            height,
            center,
            sigma,
            bin_width,
        };
        composite.push(ModelComponent::new(prefix, Box::new(gaussian)))?;
    }
    Ok(Box::new(composite))
}

#[expect(
    clippy::too_many_lines,
    reason = "fit-quality checks deliberately run in their displayed priority order against one fitted result"
)]
fn assess_quality(
    fitted: &FitResult,
    x: &[f64],
    y: &[f64],
    bin_width: f64,
    peak_count: usize,
) -> (FitQualityStatus, Vec<FitQualityIssue>) {
    let mut issues = Vec::new();
    let mut status = FitQualityStatus::Good;
    if !fitted.termination.success {
        status = FitQualityStatus::Failed;
        issues.push(FitQualityIssue::FailedConvergence {
            reason: fitted.termination.reason.clone(),
        });
    }
    if fitted.statistics.variables > 0 && fitted.covariance.is_none() {
        status = status.max(FitQualityStatus::Review);
        issues.push(FitQualityIssue::MissingCovariance);
    }
    if fitted
        .parameters
        .iter()
        .any(|value| !value.value.is_finite())
        || fitted.best_fit.iter().any(|value| !value.is_finite())
    {
        status = FitQualityStatus::Failed;
        issues.push(FitQualityIssue::NonFiniteResult);
    }
    let active = fitted
        .parameters
        .iter()
        .filter(|value| value.kind == ParameterKind::Free && value.active_bound)
        .map(|value| value.name.clone())
        .collect::<Vec<_>>();
    if !active.is_empty() {
        status = status.max(FitQualityStatus::Review);
        issues.push(FitQualityIssue::ActiveBounds { parameters: active });
    }
    let prediction = y
        .iter()
        .zip(&fitted.raw_residuals)
        .map(|(observed, residual)| observed - residual)
        .collect::<Vec<_>>();
    if fitted.statistics.objective == ObjectiveKind::PoissonDeviance
        && (prediction.len() != y.len()
            || prediction
                .iter()
                .zip(y)
                .any(|(model, observed)| *model < 0.0 || (*model == 0.0 && *observed > 0.0)))
    {
        status = FitQualityStatus::Failed;
        issues.push(FitQualityIssue::NonpositivePrediction);
    }
    if let (Some(initial), final_value) = (
        fitted.statistics.initial_objective,
        fitted.statistics.final_objective,
    ) && final_value > initial + 1.0e-8 * initial.abs().max(1.0)
    {
        status = status.max(FitQualityStatus::Poor);
        issues.push(FitQualityIssue::ObjectiveWorsened {
            initial,
            final_value,
        });
    }
    if let Some(p_value) = fitted.statistics.goodness_of_fit_p_value
        && p_value < 0.05
    {
        status = status.max(if p_value < 0.001 {
            FitQualityStatus::Poor
        } else {
            FitQualityStatus::Review
        });
        issues.push(FitQualityIssue::PoorGoodnessOfFit { p_value });
    }
    let fitted_peaks = (0..peak_count)
        .filter_map(|index| {
            let center = fitted
                .parameters
                .iter()
                .find(|value| value.name == format!("g{index}_center"))
                .map(|value| value.value)?;
            let sigma = fitted
                .parameters
                .iter()
                .find(|value| value.name == format!("g{index}_sigma"))
                .map(|value| value.value.abs())?;
            Some((center, sigma))
        })
        .collect::<Vec<_>>();
    let raw = &fitted.raw_residuals;
    let noise = robust_scale(raw).max(1.0);
    let mut unmodeled = Vec::new();
    for index in 1..raw.len().saturating_sub(1) {
        let threshold = (6.0
            * prediction
                .get(index)
                .copied()
                .unwrap_or(1.0)
                .max(1.0)
                .sqrt())
        .max(6.0 * noise);
        if raw[index] > threshold
            && (x[index] - x[index - 1]).abs() <= 1.5 * bin_width
            && (x[index + 1] - x[index]).abs() <= 1.5 * bin_width
            && raw[index] >= raw[index - 1]
            && raw[index] > raw[index + 1]
            && fitted_peaks.iter().all(|(center, sigma)| {
                (x[index] - center).abs() > (2.5 * sigma).max(3.0 * bin_width)
            })
        {
            unmodeled.push(x[index]);
        }
    }
    if !unmodeled.is_empty() {
        status = status.max(FitQualityStatus::Review);
        issues.push(FitQualityIssue::UnmodeledResidualPeaks {
            positions: unmodeled,
        });
    }
    (status, issues)
}

fn background_model(
    kind: BackgroundKind,
    definitions: &[ParameterDefinition],
    region: [f64; 2],
) -> Box<dyn Model> {
    match kind {
        BackgroundKind::None | BackgroundKind::Constant => Box::new(
            ConstantModel::new("bg_", [definitions[0].initial])
                .with_parameters([definitions[0].clone()]),
        ),
        BackgroundKind::Linear if can_condition_polynomial(definitions) => {
            Box::new(ConditionedPolynomialModel::new_linear(definitions, region))
        }
        BackgroundKind::Linear => Box::new(
            LinearModel::new("bg_", [definitions[0].initial, definitions[1].initial])
                .with_parameters([definitions[0].clone(), definitions[1].clone()]),
        ),
        BackgroundKind::Quadratic if can_condition_polynomial(definitions) => Box::new(
            ConditionedPolynomialModel::new_quadratic(definitions, region),
        ),
        BackgroundKind::Quadratic => Box::new(
            QuadraticModel::new(
                "bg_",
                [
                    definitions[0].initial,
                    definitions[1].initial,
                    definitions[2].initial,
                ],
            )
            .with_parameters([
                definitions[0].clone(),
                definitions[1].clone(),
                definitions[2].clone(),
            ]),
        ),
        BackgroundKind::Exponential => Box::new(
            ExponentialModel::new("bg_", [definitions[0].initial, definitions[1].initial])
                .with_parameters([definitions[0].clone(), definitions[1].clone()]),
        ),
        BackgroundKind::PowerLaw => Box::new(
            PowerLawModel::new("bg_", [definitions[0].initial, definitions[1].initial])
                .with_parameters([definitions[0].clone(), definitions[1].clone()]),
        ),
    }
}

fn background_definitions(
    kind: BackgroundKind,
    seed: Option<&BackgroundSeed>,
    initial_values: Option<&[(String, f64)]>,
    allow_vary: bool,
) -> Vec<ParameterDefinition> {
    default_background_parameters(kind)
        .into_iter()
        .map(|default| {
            let mut definition = seed
                .and_then(|all| find_definition(&all.parameters, &default.name))
                .cloned()
                .unwrap_or(default);
            if definition.vary
                && let Some(value) = initial_values.and_then(|values| {
                    values
                        .iter()
                        .find(|(name, _)| name == &definition.name)
                        .map(|(_, value)| *value)
                })
                && value.is_finite()
            {
                definition.initial = clamp_to_bounds(value, definition.bounds);
            }
            definition.vary &= allow_vary;
            definition
        })
        .collect()
}

fn default_background_parameters(kind: BackgroundKind) -> Vec<ParameterDefinition> {
    match kind {
        BackgroundKind::None => vec![ParameterDefinition::fixed("bg_c", 0.0)],
        BackgroundKind::Constant => vec![ParameterDefinition::varying("bg_c", 0.0)],
        BackgroundKind::Linear => vec![
            ParameterDefinition::varying("bg_slope", 0.0),
            ParameterDefinition::varying("bg_intercept", 0.0),
        ],
        BackgroundKind::Quadratic => vec![
            ParameterDefinition::varying("bg_a", 0.0),
            ParameterDefinition::varying("bg_b", 0.0),
            ParameterDefinition::varying("bg_c", 0.0),
        ],
        BackgroundKind::Exponential => vec![
            ParameterDefinition::varying("bg_amplitude", 0.0),
            ParameterDefinition::varying("bg_decay", 500.0),
        ],
        BackgroundKind::PowerLaw => vec![
            ParameterDefinition::varying("bg_amplitude", 0.0),
            ParameterDefinition::varying("bg_exponent", -1.0),
        ],
    }
}

fn find_definition<'a>(
    definitions: &'a [ParameterDefinition],
    expected_name: &str,
) -> Option<&'a ParameterDefinition> {
    let suffix = expected_name.strip_prefix("bg_").unwrap_or(expected_name);
    definitions
        .iter()
        .find(|definition| definition.name == expected_name || definition.name == suffix)
}

fn evaluate_background_definitions(
    kind: BackgroundKind,
    parameters: &[ParameterDefinition],
    x: &[f64],
) -> Result<Vec<f64>, FitError> {
    let value = |name: &str| {
        find_definition(parameters, name)
            .map(|parameter| parameter.initial)
            .ok_or_else(|| FitError::InvalidParameter {
                parameter: name.to_owned(),
            })
    };
    x.iter()
        .map(|independent| {
            let result = match kind {
                BackgroundKind::None => 0.0,
                BackgroundKind::Constant => value("bg_c")?,
                BackgroundKind::Linear => value("bg_slope")? * independent + value("bg_intercept")?,
                BackgroundKind::Quadratic => {
                    value("bg_a")? * independent * independent
                        + value("bg_b")? * independent
                        + value("bg_c")?
                }
                BackgroundKind::Exponential => {
                    value("bg_amplitude")? * (-independent / value("bg_decay")?).exp()
                }
                BackgroundKind::PowerLaw => {
                    value("bg_amplitude")? * independent.powf(value("bg_exponent")?)
                }
            };
            if result.is_finite() {
                Ok(result)
            } else {
                Err(FitError::NonFinite {
                    context: "background peak estimate".to_owned(),
                })
            }
        })
        .collect()
}

fn robust_background_values(
    kind: BackgroundKind,
    x: &[f64],
    y: &[f64],
    region: [f64; 2],
) -> Vec<(String, f64)> {
    let center = 0.5 * (region[0] + region[1]);
    let scale = (0.5 * (region[1] - region[0])).abs().max(f64::EPSILON);
    let scaled_x = x
        .iter()
        .map(|value| (*value - center) / scale)
        .collect::<Vec<_>>();
    match kind {
        BackgroundKind::None => vec![("bg_c".to_owned(), 0.0)],
        BackgroundKind::Constant => {
            let mut values = y.to_vec();
            values.sort_by(f64::total_cmp);
            vec![(
                "bg_c".to_owned(),
                values.get(values.len() / 2).copied().unwrap_or(0.0),
            )]
        }
        BackgroundKind::Linear => {
            let [level, slope] = polynomial_least_squares::<2>(&scaled_x, y);
            let original_slope = slope / scale;
            vec![
                ("bg_slope".to_owned(), original_slope),
                ("bg_intercept".to_owned(), level - original_slope * center),
            ]
        }
        BackgroundKind::Quadratic => {
            let [level, slope, curvature] = polynomial_least_squares::<3>(&scaled_x, y);
            let a = curvature / scale.powi(2);
            let b = slope / scale - 2.0 * a * center;
            let c = level - slope * center / scale + a * center.powi(2);
            vec![
                ("bg_a".to_owned(), a),
                ("bg_b".to_owned(), b),
                ("bg_c".to_owned(), c),
            ]
        }
        BackgroundKind::Exponential => {
            let positive = x
                .iter()
                .copied()
                .zip(y.iter().copied())
                .filter(|(_, value)| *value > 0.0)
                .map(|(independent, value)| (independent, value.ln()))
                .collect::<Vec<_>>();
            let (slope, intercept) = linear_regression(&positive);
            let decay = if slope < -f64::EPSILON {
                -1.0 / slope
            } else {
                (region[1] - region[0]).abs().max(1.0)
            };
            vec![
                ("bg_amplitude".to_owned(), intercept.exp()),
                ("bg_decay".to_owned(), decay),
            ]
        }
        BackgroundKind::PowerLaw => {
            let logged = x
                .iter()
                .copied()
                .zip(y.iter().copied())
                .filter(|(independent, value)| *independent > 0.0 && *value > 0.0)
                .map(|(independent, value)| (independent.ln(), value.ln()))
                .collect::<Vec<_>>();
            let (exponent, log_amplitude) = linear_regression(&logged);
            vec![
                ("bg_amplitude".to_owned(), log_amplitude.exp()),
                ("bg_exponent".to_owned(), exponent),
            ]
        }
    }
}

fn linear_regression(points: &[(f64, f64)]) -> (f64, f64) {
    if points.is_empty() {
        return (0.0, 0.0);
    }
    let count = points.len() as f64;
    let mean_x = points.iter().map(|point| point.0).sum::<f64>() / count;
    let mean_y = points.iter().map(|point| point.1).sum::<f64>() / count;
    let denominator = points
        .iter()
        .map(|point| (point.0 - mean_x).powi(2))
        .sum::<f64>();
    let slope = if denominator > f64::EPSILON {
        points
            .iter()
            .map(|point| (point.0 - mean_x) * (point.1 - mean_y))
            .sum::<f64>()
            / denominator
    } else {
        0.0
    };
    (slope, mean_y - slope * mean_x)
}

#[expect(
    clippy::needless_range_loop,
    reason = "small fixed-size Gaussian elimination is clearer with matrix indices"
)]
fn polynomial_least_squares<const N: usize>(x: &[f64], y: &[f64]) -> [f64; N] {
    let mut augmented = [[0.0; 4]; 3];
    for row in 0..N {
        for column in 0..N {
            augmented[row][column] = x
                .iter()
                .map(|value| value.powi((row + column) as i32))
                .sum();
        }
        augmented[row][N] = x
            .iter()
            .zip(y)
            .map(|(independent, dependent)| dependent * independent.powi(row as i32))
            .sum();
    }
    for pivot in 0..N {
        let best = (pivot..N)
            .max_by(|left, right| {
                augmented[*left][pivot]
                    .abs()
                    .total_cmp(&augmented[*right][pivot].abs())
            })
            .unwrap_or(pivot);
        augmented.swap(pivot, best);
        let divisor = augmented[pivot][pivot];
        if divisor.abs() <= f64::EPSILON {
            return [0.0; N];
        }
        for column in pivot..=N {
            augmented[pivot][column] /= divisor;
        }
        for row in 0..N {
            if row == pivot {
                continue;
            }
            let factor = augmented[row][pivot];
            for column in pivot..=N {
                augmented[row][column] -= factor * augmented[pivot][column];
            }
        }
    }
    std::array::from_fn(|index| augmented[index][N])
}

fn can_condition_polynomial(definitions: &[ParameterDefinition]) -> bool {
    definitions
        .first()
        .is_some_and(|definition| definition.vary)
        && definitions
            .iter()
            .all(|definition| definition.bounds == Bounds::unbounded())
        && definitions
            .windows(2)
            .all(|pair| pair[0].vary == pair[1].vary)
}

#[derive(Debug, Clone)]
struct ConditionedPolynomialModel {
    degree: usize,
    center: f64,
    scale: f64,
    parameters: Vec<ParameterDefinition>,
}

impl ConditionedPolynomialModel {
    fn new_linear(definitions: &[ParameterDefinition], region: [f64; 2]) -> Self {
        let center = 0.5 * (region[0] + region[1]);
        let scale = (0.5 * (region[1] - region[0])).abs().max(f64::EPSILON);
        let slope = definitions[0].initial;
        let intercept = definitions[1].initial;
        let vary = definitions[0].vary;
        Self {
            degree: 1,
            center,
            scale,
            parameters: vec![
                parameter_with_vary("bg_scaled_level", slope.mul_add(center, intercept), vary),
                parameter_with_vary("bg_scaled_slope", slope * scale, vary),
            ],
        }
    }

    fn new_quadratic(definitions: &[ParameterDefinition], region: [f64; 2]) -> Self {
        let center = 0.5 * (region[0] + region[1]);
        let scale = (0.5 * (region[1] - region[0])).abs().max(f64::EPSILON);
        let a = definitions[0].initial;
        let b = definitions[1].initial;
        let c = definitions[2].initial;
        let vary = definitions[0].vary;
        Self {
            degree: 2,
            center,
            scale,
            parameters: vec![
                parameter_with_vary(
                    "bg_scaled_level",
                    a.mul_add(center * center, b.mul_add(center, c)),
                    vary,
                ),
                parameter_with_vary("bg_scaled_slope", scale * (2.0 * a * center + b), vary),
                parameter_with_vary("bg_scaled_curvature", a * scale.powi(2), vary),
            ],
        }
    }
}

fn parameter_with_vary(name: &str, initial: f64, vary: bool) -> ParameterDefinition {
    if vary {
        ParameterDefinition::varying(name, initial)
    } else {
        ParameterDefinition::fixed(name, initial)
    }
}

impl Model for ConditionedPolynomialModel {
    fn name(&self) -> &'static str {
        "conditioned polynomial background"
    }

    fn parameter_definitions(&self) -> Vec<ParameterDefinition> {
        self.parameters.clone()
    }

    fn evaluate(
        &self,
        x: &[f64],
        parameters: &ParameterValues,
        output: &mut [f64],
    ) -> Result<(), FitError> {
        if x.len() != output.len() {
            return Err(FitError::LengthMismatch {
                x: x.len(),
                y: output.len(),
            });
        }
        let level = parameters.require("bg_scaled_level")?;
        let slope = parameters.require("bg_scaled_slope")?;
        let curvature = if self.degree == 2 {
            parameters.require("bg_scaled_curvature")?
        } else {
            0.0
        };
        for (result, independent) in output.iter_mut().zip(x) {
            let scaled = (*independent - self.center) / self.scale;
            *result = curvature.mul_add(scaled * scaled, slope.mul_add(scaled, level));
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(FitError::NonFinite {
                context: self.name().to_owned(),
            });
        }
        Ok(())
    }

    fn analytic_jacobian(
        &self,
        x: &[f64],
        _parameters: &ParameterValues,
        parameter_names: &[String],
        output: &mut [f64],
    ) -> Result<bool, FitError> {
        let expected = x.len().saturating_mul(parameter_names.len());
        if output.len() != expected {
            return Err(FitError::LengthMismatch {
                x: expected,
                y: output.len(),
            });
        }
        output.fill(0.0);
        for (row, independent) in x.iter().enumerate() {
            let scaled = (*independent - self.center) / self.scale;
            for (column, name) in parameter_names.iter().enumerate() {
                output[row * parameter_names.len() + column] = match name.as_str() {
                    "bg_scaled_level" => 1.0,
                    "bg_scaled_slope" => scaled,
                    "bg_scaled_curvature" if self.degree == 2 => scaled * scaled,
                    _ => 0.0,
                };
            }
        }
        Ok(true)
    }

    fn derived_parameters(
        &self,
        parameters: &ParameterValues,
    ) -> Result<Vec<DerivedParameter>, FitError> {
        let level = parameters.require("bg_scaled_level")?;
        let slope = parameters.require("bg_scaled_slope")?;
        if self.degree == 1 {
            let original_slope = slope / self.scale;
            return Ok(vec![
                DerivedParameter {
                    name: "bg_slope".to_owned(),
                    value: original_slope,
                    gradient: vec![("bg_scaled_slope".to_owned(), 1.0 / self.scale)],
                },
                DerivedParameter {
                    name: "bg_intercept".to_owned(),
                    value: level - original_slope * self.center,
                    gradient: vec![
                        ("bg_scaled_level".to_owned(), 1.0),
                        ("bg_scaled_slope".to_owned(), -self.center / self.scale),
                    ],
                },
            ]);
        }
        let curvature = parameters.require("bg_scaled_curvature")?;
        let a = curvature / self.scale.powi(2);
        let b = slope / self.scale - 2.0 * a * self.center;
        let c = level - slope * self.center / self.scale + a * self.center.powi(2);
        Ok(vec![
            DerivedParameter {
                name: "bg_a".to_owned(),
                value: a,
                gradient: vec![("bg_scaled_curvature".to_owned(), 1.0 / self.scale.powi(2))],
            },
            DerivedParameter {
                name: "bg_b".to_owned(),
                value: b,
                gradient: vec![
                    ("bg_scaled_slope".to_owned(), 1.0 / self.scale),
                    (
                        "bg_scaled_curvature".to_owned(),
                        -2.0 * self.center / self.scale.powi(2),
                    ),
                ],
            },
            DerivedParameter {
                name: "bg_c".to_owned(),
                value: c,
                gradient: vec![
                    ("bg_scaled_level".to_owned(), 1.0),
                    ("bg_scaled_slope".to_owned(), -self.center / self.scale),
                    (
                        "bg_scaled_curvature".to_owned(),
                        self.center.powi(2) / self.scale.powi(2),
                    ),
                ],
            },
        ])
    }
}

fn validate_data(x: &[f64], y: &[f64], bin_width: f64) -> Result<(), FitError> {
    if x.len() != y.len() {
        return Err(FitError::LengthMismatch {
            x: x.len(),
            y: y.len(),
        });
    }
    if x.is_empty() {
        return Err(FitError::EmptyData);
    }
    if !bin_width.is_finite() || bin_width <= 0.0 {
        return Err(FitError::Domain {
            model: "spectrum".to_owned(),
            message: "bin width must be positive and finite".to_owned(),
        });
    }
    if x.iter().chain(y).any(|value| !value.is_finite()) {
        return Err(FitError::NonFinite {
            context: "spectrum data".to_owned(),
        });
    }
    Ok(())
}

fn sorted_region(mut region: [f64; 2]) -> Result<[f64; 2], FitError> {
    if region.iter().any(|marker| !marker.is_finite()) || region[0] == region[1] {
        return Err(FitError::InvalidRegion);
    }
    if region[0] > region[1] {
        region.swap(0, 1);
    }
    Ok(region)
}

fn region_data(x: &[f64], y: &[f64], region: [f64; 2]) -> Result<(Vec<f64>, Vec<f64>), FitError> {
    let selected = x
        .iter()
        .copied()
        .zip(y.iter().copied())
        .filter(|(independent, _)| *independent >= region[0] && *independent <= region[1])
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(FitError::InvalidRegion);
    }
    Ok(selected.into_iter().unzip())
}

fn background_data(
    x: &[f64],
    y: &[f64],
    markers: &[(f64, f64)],
) -> Result<(Vec<f64>, Vec<f64>), FitError> {
    let mut selected_x = Vec::new();
    let mut selected_y = Vec::new();
    for &(first, second) in markers {
        if !first.is_finite() || !second.is_finite() {
            return Err(FitError::NonFinite {
                context: "background markers".to_owned(),
            });
        }
        let (lower, upper) = if first <= second {
            (first, second)
        } else {
            (second, first)
        };
        for (independent, dependent) in x.iter().zip(y) {
            if *independent >= lower && *independent <= upper {
                selected_x.push(*independent);
                selected_y.push(*dependent);
            }
        }
    }
    if selected_x.is_empty() {
        return Err(FitError::InvalidRegion);
    }
    Ok((selected_x, selected_y))
}

fn interpolate_crossing(x0: f64, y0: f64, x1: f64, y1: f64, level: f64) -> f64 {
    if (y1 - y0).abs() <= f64::EPSILON {
        return 0.5 * (x0 + x1);
    }
    x0 + (level - y0) / (y1 - y0) * (x1 - x0)
}

fn interpolated_left_crossing(
    x: &[f64],
    values: &[f64],
    start: usize,
    peak: usize,
    level: f64,
) -> Option<f64> {
    (start..peak).rev().find_map(|index| {
        (values[index] <= level && values[index + 1] > level).then(|| {
            interpolate_crossing(
                x[index],
                values[index],
                x[index + 1],
                values[index + 1],
                level,
            )
        })
    })
}

fn interpolated_right_crossing(
    x: &[f64],
    values: &[f64],
    peak: usize,
    end: usize,
    level: f64,
) -> Option<f64> {
    (peak..end.saturating_sub(1)).find_map(|index| {
        (values[index] > level && values[index + 1] <= level).then(|| {
            interpolate_crossing(
                x[index],
                values[index],
                x[index + 1],
                values[index + 1],
                level,
            )
        })
    })
}

fn nearest_index(values: &[f64], target: f64) -> usize {
    values
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            (*left - target)
                .abs()
                .partial_cmp(&(*right - target).abs())
                .unwrap_or(Ordering::Equal)
        })
        .map_or(0, |(index, _)| index)
}

type SigmaLimits = (Vec<f64>, Vec<f64>);

fn normalized_sigma_bounds(
    bounds: Option<&SigmaBounds>,
    equal_sigma: bool,
    peaks: usize,
) -> Result<Option<SigmaLimits>, FitError> {
    let Some(bounds) = bounds else {
        return Ok(None);
    };
    let valid_length = if equal_sigma {
        bounds.minima.len() == 1 && bounds.maxima.len() == 1
    } else {
        (bounds.minima.len() == peaks && bounds.maxima.len() == peaks)
            || (bounds.minima.len() == 1 && bounds.maxima.len() == 1)
    };
    if !valid_length
        || bounds
            .minima
            .iter()
            .chain(&bounds.maxima)
            .any(|value| !value.is_finite())
        || bounds
            .minima
            .iter()
            .zip(&bounds.maxima)
            .any(|(minimum, maximum)| minimum < &0.0 || minimum >= maximum)
    {
        return Err(FitError::InvalidBounds {
            parameter: "sigma".to_owned(),
        });
    }
    if !equal_sigma && bounds.minima.len() == 1 {
        Ok(Some((
            vec![bounds.minima[0]; peaks],
            vec![bounds.maxima[0]; peaks],
        )))
    } else {
        Ok(Some((bounds.minima.clone(), bounds.maxima.clone())))
    }
}

fn clamp_to_bounds(value: f64, bounds: Bounds) -> f64 {
    value
        .max(match bounds.lower {
            Bound::Unbounded => f64::NEG_INFINITY,
            Bound::Inclusive(lower) => lower,
        })
        .min(match bounds.upper {
            Bound::Unbounded => f64::INFINITY,
            Bound::Inclusive(upper) => upper,
        })
}

fn validate_power_law_domain(kind: BackgroundKind, x: &[f64]) -> Result<(), FitError> {
    if kind == BackgroundKind::PowerLaw && x.iter().any(|value| *value <= 0.0) {
        return Err(FitError::Domain {
            model: "power-law".to_owned(),
            message: "x must be positive for a real-valued variable exponent".to_owned(),
        });
    }
    Ok(())
}

fn linspace(start: f64, end: f64, count: usize) -> Result<Vec<f64>, FitError> {
    if count > 16_777_216 {
        return Err(FitError::AllocationLimit {
            requested: count,
            limit: 16_777_216,
        });
    }
    if count <= 1 {
        return Ok(vec![start]);
    }
    let denominator = (count - 1) as f64;
    Ok((0..count)
        .map(|index| start + (end - start) * index as f64 / denominator)
        .collect())
}

#[cfg(test)]
mod manual_tests {
    use super::{
        BackgroundCoupling, BackgroundKind, BackgroundSeed, ManualPeakBounds, ManualPeakSeed,
        ManualSeedEstimateRequest, PeakFitRequest, estimate_manual_peak_seeds, fit_peaks,
    };
    use crate::{
        FitOptions, FitQualityStatus, Model, ObjectiveKind, ParameterDefinition, ParameterValues,
    };

    fn gaussian_data(center: f64, sigma: f64, amplitude: f64) -> (Vec<f64>, Vec<f64>) {
        let bin_width = 0.1;
        let x = (0..=100)
            .map(|index| index as f64 * bin_width)
            .collect::<Vec<_>>();
        let seed = ManualPeakSeed {
            center,
            sigma,
            amplitude,
        };
        let y = x
            .iter()
            .map(|&independent| {
                3.0 + super::evaluate_manual_peak(seed, independent, bin_width).unwrap()
            })
            .collect();
        (x, y)
    }

    #[test]
    fn height_integrated_gaussian_analytic_jacobian_matches_central_difference() {
        let model = super::HeightGaussianModel {
            prefix: "g0_".to_owned(),
            height: ParameterDefinition::varying("g0_height", 120.0),
            center: ParameterDefinition::varying("g0_center", 5.07),
            sigma: ParameterDefinition::varying("g0_sigma", 0.43),
            bin_width: 0.2,
        };
        let x = (0..50)
            .map(|index| 0.1 + index as f64 * 0.2)
            .collect::<Vec<_>>();
        let names = vec![
            "g0_height".to_owned(),
            "g0_center".to_owned(),
            "g0_sigma".to_owned(),
        ];
        let mut parameters = ParameterValues::new();
        parameters.insert("g0_height", 120.0);
        parameters.insert("g0_center", 5.07);
        parameters.insert("g0_sigma", 0.43);
        let mut analytic = vec![0.0; x.len() * names.len()];
        assert!(
            model
                .analytic_jacobian(&x, &parameters, &names, &mut analytic)
                .expect("analytic Jacobian")
        );
        for (column, name) in names.iter().enumerate() {
            let baseline = parameters.require(name).unwrap();
            let step = baseline.abs().max(1.0) * 1.0e-6;
            let mut plus = parameters.clone();
            let mut minus = parameters.clone();
            plus.insert(name.clone(), baseline + step);
            minus.insert(name.clone(), baseline - step);
            let mut plus_curve = vec![0.0; x.len()];
            let mut minus_curve = vec![0.0; x.len()];
            model.evaluate(&x, &plus, &mut plus_curve).unwrap();
            model.evaluate(&x, &minus, &mut minus_curve).unwrap();
            for row in 0..x.len() {
                let numerical = (plus_curve[row] - minus_curve[row]) / (2.0 * step);
                let actual = analytic[row * names.len() + column];
                let scale = numerical.abs().max(actual.abs()).max(1.0);
                assert!(
                    (actual - numerical).abs() < 2.0e-6 * scale,
                    "{name} row {row}: analytic={actual}, numerical={numerical}",
                );
            }
        }
    }

    #[test]
    fn estimator_keeps_exact_marker_and_recovers_two_sided_width_and_area() {
        let (x, y) = gaussian_data(5.0, 0.4, 100.0);
        let estimate = estimate_manual_peak_seeds(
            &ManualSeedEstimateRequest {
                x,
                y,
                bin_width: 0.1,
                region: [2.0, 8.0],
                peak_markers: vec![4.93],
                background_markers: vec![(2.0, 3.0), (7.0, 8.0)],
                background: BackgroundKind::Constant,
                background_seed: None,
                equal_sigma: false,
            },
            &FitOptions::default(),
        )
        .expect("manual estimate");
        let peak = estimate.peaks[0];
        assert_eq!(peak.seed.center, 4.93);
        assert!(peak.clean_width);
        assert!((peak.seed.sigma - 0.4).abs() < 0.08, "{:?}", peak.seed);
        assert!(
            (peak.seed.amplitude - 100.0).abs() < 10.0,
            "{:?}",
            peak.seed
        );
    }

    #[test]
    fn estimator_uses_unsmoothed_exact_marker_height() {
        let x = (0..=20).map(|index| index as f64 * 0.1).collect::<Vec<_>>();
        let mut signal = vec![0.0; x.len()];
        signal[10] = 100.0;
        let estimate =
            super::estimate_manual_components(&x, &signal, &[1.0], [0.0, 2.0], 0.1, false);

        assert_eq!(estimate[0].net_height, 100.0);
    }

    #[test]
    fn estimator_mirrors_a_single_half_height_crossing() {
        let bin_width = 0.05;
        let sigma = 0.3;
        let seed = ManualPeakSeed {
            center: 0.0,
            sigma,
            amplitude: 80.0,
        };
        let x = (0..=60)
            .map(|index| index as f64 * bin_width)
            .collect::<Vec<_>>();
        let signal = x
            .iter()
            .map(|&value| super::evaluate_manual_peak(seed, value, bin_width).unwrap())
            .collect::<Vec<_>>();
        let estimate =
            super::estimate_manual_components(&x, &signal, &[0.0], [0.0, 3.0], bin_width, false);

        assert!(!estimate[0].clean_width);
        assert!((estimate[0].seed.sigma - sigma).abs() < 0.06);
    }

    #[test]
    fn strongest_clean_peak_supplies_shared_width_and_zero_signal_is_invalid() {
        let bin_width = 0.1;
        let x = (0..=120)
            .map(|index| index as f64 * bin_width)
            .collect::<Vec<_>>();
        let strong = ManualPeakSeed {
            center: 4.0,
            sigma: 0.32,
            amplitude: 110.0,
        };
        let weak = ManualPeakSeed {
            center: 8.0,
            sigma: 0.75,
            amplitude: 25.0,
        };
        let y = x
            .iter()
            .map(|&value| {
                2.0 + super::evaluate_manual_peak(strong, value, bin_width).unwrap()
                    + super::evaluate_manual_peak(weak, value, bin_width).unwrap()
            })
            .collect::<Vec<_>>();
        let estimate = estimate_manual_peak_seeds(
            &ManualSeedEstimateRequest {
                x: x.clone(),
                y,
                bin_width,
                region: [1.0, 11.0],
                peak_markers: vec![4.0, 8.0],
                background_markers: vec![(1.0, 2.0), (10.0, 11.0)],
                background: BackgroundKind::Constant,
                background_seed: None,
                equal_sigma: true,
            },
            &FitOptions::default(),
        )
        .expect("shared estimate");
        assert!((estimate.peaks[0].seed.sigma - estimate.peaks[1].seed.sigma).abs() < 1.0e-12);
        assert!((estimate.peaks[0].seed.sigma - 0.32).abs() < 0.08);

        let invalid = estimate_manual_peak_seeds(
            &ManualSeedEstimateRequest {
                x,
                y: vec![2.0; 121],
                bin_width,
                region: [1.0, 11.0],
                peak_markers: vec![6.0],
                background_markers: vec![(1.0, 2.0), (10.0, 11.0)],
                background: BackgroundKind::Constant,
                background_seed: None,
                equal_sigma: false,
            },
            &FitOptions::default(),
        )
        .expect("zero-signal estimate");
        assert!(!invalid.peaks[0].valid);
        assert_eq!(invalid.peaks[0].seed.amplitude, 0.0);
    }

    #[test]
    fn displayed_manual_seed_is_used_by_one_composite_solve() {
        let (x, y) = gaussian_data(5.0, 0.4, 100.0);
        let seed = ManualPeakSeed {
            center: 4.8,
            sigma: 0.5,
            amplitude: 90.0,
        };
        let result = fit_peaks(
            &PeakFitRequest {
                x,
                y,
                bin_width: 0.1,
                region: [2.0, 8.0],
                peak_seeds: vec![seed],
                peak_bounds: None,
                background_markers: vec![(2.0, 3.0), (7.0, 8.0)],
                background: BackgroundKind::Constant,
                background_seed: None,
                background_coupling: BackgroundCoupling::PrefitJoint,
                equal_sigma: true,
                free_centers: true,
                sigma_bounds: None,
            },
            &FitOptions {
                objective: ObjectiveKind::PoissonDeviance,
                ..FitOptions::default()
            },
        )
        .expect("manual fit");
        assert_eq!(result.peak_seeds, vec![seed]);
        assert_eq!(
            result
                .initial_parameters
                .iter()
                .find(|p| p.name == "g0_center")
                .unwrap()
                .initial,
            4.8
        );
    }

    #[test]
    fn poisson_and_least_squares_recover_the_same_fractional_bin_shape() {
        let bin_width = 0.25;
        let truth = ManualPeakSeed {
            center: 5.13,
            sigma: 0.47,
            amplitude: 95.0,
        };
        let x = (0..48)
            .map(|index| 0.125 + index as f64 * bin_width)
            .collect::<Vec<_>>();
        let y = x
            .iter()
            .map(|&value| super::evaluate_manual_peak(truth, value, bin_width).unwrap())
            .collect::<Vec<_>>();
        let seed = ManualPeakSeed {
            center: 4.85,
            sigma: 0.7,
            amplitude: 70.0,
        };
        let height = super::evaluate_manual_peak(seed, seed.center, bin_width).unwrap();
        let request = PeakFitRequest {
            x: x.clone(),
            y: y.clone(),
            bin_width,
            region: [2.0, 8.0],
            peak_seeds: vec![seed],
            peak_bounds: Some(vec![ManualPeakBounds {
                center: [4.0, 6.0],
                sigma: [0.2, 1.2],
                net_height: [0.2 * height, 3.0 * height],
            }]),
            background_markers: Vec::new(),
            background: BackgroundKind::None,
            background_seed: None,
            background_coupling: BackgroundCoupling::PrefitFrozen,
            equal_sigma: true,
            free_centers: true,
            sigma_bounds: None,
        };
        let least_squares = fit_peaks(
            &request,
            &FitOptions {
                objective: ObjectiveKind::LeastSquares,
                ..FitOptions::default()
            },
        )
        .expect("least-squares fit");
        let poisson = fit_peaks(
            &request,
            &FitOptions {
                objective: ObjectiveKind::PoissonDeviance,
                ..FitOptions::default()
            },
        )
        .expect("Poisson fit");

        let parameter = |result: &super::SpectrumFitResult, name: &str| {
            result
                .fit
                .parameters
                .iter()
                .find(|parameter| parameter.name == name)
                .unwrap()
                .value
        };
        for result in [&least_squares, &poisson] {
            assert!((parameter(result, "g0_center") - truth.center).abs() < 2.0e-5);
            assert!((parameter(result, "g0_sigma") - truth.sigma).abs() < 2.0e-5);
            assert!(result.fit.statistics.objective_improvement.unwrap() > 0.0);
        }
        let prediction = |result: &super::SpectrumFitResult| {
            y.iter()
                .zip(&result.fit.raw_residuals)
                .map(|(observed, residual)| observed - residual)
                .collect::<Vec<_>>()
        };
        let poisson_deviance = |model: &[f64]| {
            y.iter()
                .zip(model)
                .map(|(observed, predicted)| {
                    if *observed == 0.0 {
                        2.0 * predicted.max(1.0e-12)
                    } else {
                        2.0 * (predicted.max(1.0e-12) - observed
                            + observed * (observed / predicted.max(1.0e-12)).ln())
                    }
                })
                .sum::<f64>()
        };
        let sum_of_squares = |model: &[f64]| {
            y.iter()
                .zip(model)
                .map(|(observed, predicted)| (observed - predicted).powi(2))
                .sum::<f64>()
        };
        let least_squares_prediction = prediction(&least_squares);
        let poisson_prediction = prediction(&poisson);
        assert!(
            poisson_deviance(&poisson_prediction)
                <= poisson_deviance(&least_squares_prediction) + 1.0e-8
        );
        assert!(
            sum_of_squares(&least_squares_prediction)
                <= sum_of_squares(&poisson_prediction) + 1.0e-8
        );
    }

    #[test]
    fn fit_quality_is_objective_aware() {
        let bin_width = 0.2;
        let truth = ManualPeakSeed {
            center: 5.0,
            sigma: 0.35,
            amplitude: 80.0,
        };
        let x = (0..51)
            .map(|index| index as f64 * bin_width)
            .collect::<Vec<_>>();
        let exact = x
            .iter()
            .map(|&value| super::evaluate_manual_peak(truth, value, bin_width).unwrap())
            .collect::<Vec<_>>();
        let exact_result = fit_peaks(
            &PeakFitRequest {
                x: x.clone(),
                y: exact,
                bin_width,
                region: [1.0, 9.0],
                peak_seeds: vec![ManualPeakSeed {
                    center: 4.9,
                    sigma: 0.4,
                    amplitude: 75.0,
                }],
                peak_bounds: None,
                background_markers: Vec::new(),
                background: BackgroundKind::None,
                background_seed: None,
                background_coupling: BackgroundCoupling::PrefitFrozen,
                equal_sigma: true,
                free_centers: true,
                sigma_bounds: None,
            },
            &FitOptions::default(),
        )
        .expect("exact least-squares fit");
        assert_eq!(exact_result.quality_status, FitQualityStatus::Good);

        let poor = x
            .iter()
            .enumerate()
            .map(|(index, value)| {
                super::evaluate_manual_peak(truth, *value, bin_width).unwrap()
                    + if index % 2 == 0 { 80.0 } else { 0.0 }
            })
            .collect::<Vec<_>>();
        let truth_height = super::evaluate_manual_peak(truth, truth.center, bin_width).unwrap();
        let poor_result = fit_peaks(
            &PeakFitRequest {
                x,
                y: poor,
                bin_width,
                region: [1.0, 9.0],
                peak_seeds: vec![truth],
                peak_bounds: Some(vec![ManualPeakBounds {
                    center: [4.9, 5.1],
                    sigma: [0.3, 0.4],
                    net_height: [0.9 * truth_height, 1.1 * truth_height],
                }]),
                background_markers: Vec::new(),
                background: BackgroundKind::None,
                background_seed: None,
                background_coupling: BackgroundCoupling::PrefitFrozen,
                equal_sigma: true,
                free_centers: true,
                sigma_bounds: None,
            },
            &FitOptions {
                objective: ObjectiveKind::PoissonDeviance,
                ..FitOptions::default()
            },
        )
        .expect("poor Poisson fit still returns diagnostics");
        assert_ne!(poor_result.quality_status, FitQualityStatus::Good);
    }

    #[test]
    fn displayed_position_width_and_height_ranges_are_solver_bounds() {
        let (x, y) = gaussian_data(5.0, 0.4, 100.0);
        let seed = ManualPeakSeed {
            center: 4.9,
            sigma: 0.5,
            amplitude: 90.0,
        };
        let height = super::evaluate_manual_peak(seed, seed.center, 0.1).unwrap();
        let bounds = ManualPeakBounds {
            center: [4.4, 5.4],
            sigma: [0.25, 0.75],
            net_height: [0.75 * height, 1.25 * height],
        };
        let result = fit_peaks(
            &PeakFitRequest {
                x,
                y,
                bin_width: 0.1,
                region: [2.0, 8.0],
                peak_seeds: vec![seed],
                peak_bounds: Some(vec![bounds]),
                background_markers: vec![(2.0, 3.0), (7.0, 8.0)],
                background: BackgroundKind::Constant,
                background_seed: None,
                background_coupling: BackgroundCoupling::PrefitJoint,
                equal_sigma: true,
                free_centers: true,
                sigma_bounds: None,
            },
            &FitOptions::default(),
        )
        .expect("bounded manual fit");
        for (name, expected) in [
            (
                "g0_height",
                crate::Bounds::finite(bounds.net_height[0], bounds.net_height[1]),
            ),
            (
                "g0_center",
                crate::Bounds::finite(bounds.center[0], bounds.center[1]),
            ),
            (
                "g0_sigma",
                crate::Bounds::finite(bounds.sigma[0], bounds.sigma[1]),
            ),
        ] {
            assert_eq!(
                result
                    .initial_parameters
                    .iter()
                    .find(|parameter| parameter.name == name)
                    .unwrap()
                    .bounds,
                expected,
            );
        }
    }

    #[test]
    fn manual_fit_rejects_missing_seeds_and_required_background_windows() {
        let (x, y) = gaussian_data(5.0, 0.4, 100.0);
        let mut request = PeakFitRequest {
            x,
            y,
            bin_width: 0.1,
            region: [2.0, 8.0],
            peak_seeds: Vec::new(),
            peak_bounds: None,
            background_markers: Vec::new(),
            background: BackgroundKind::None,
            background_seed: None,
            background_coupling: BackgroundCoupling::PrefitJoint,
            equal_sigma: true,
            free_centers: true,
            sigma_bounds: None,
        };
        let error = fit_peaks(&request, &FitOptions::default()).expect_err("missing seed");
        assert!(error.to_string().contains("manual peak seed"));

        request.peak_seeds.push(ManualPeakSeed {
            center: 5.0,
            sigma: 0.4,
            amplitude: 100.0,
        });
        request.background = BackgroundKind::Linear;
        let error = fit_peaks(&request, &FitOptions::default()).expect_err("missing background");
        assert!(error.to_string().contains("background marker window"));
    }

    #[test]
    fn all_explicit_backgrounds_objectives_and_width_couplings_fit_overlapping_peaks() {
        let bin_width = 0.05;
        let x = (0..200)
            .map(|index| 1.025 + index as f64 * bin_width)
            .collect::<Vec<_>>();
        for background in [
            BackgroundKind::None,
            BackgroundKind::Constant,
            BackgroundKind::Linear,
            BackgroundKind::Quadratic,
            BackgroundKind::Exponential,
            BackgroundKind::PowerLaw,
        ] {
            let background_parameters = match background {
                BackgroundKind::None => Vec::new(),
                BackgroundKind::Constant => {
                    vec![ParameterDefinition::fixed("bg_c", 2.0)]
                }
                BackgroundKind::Linear => vec![
                    ParameterDefinition::fixed("bg_slope", 0.12),
                    ParameterDefinition::fixed("bg_intercept", 1.5),
                ],
                BackgroundKind::Quadratic => vec![
                    ParameterDefinition::fixed("bg_a", 0.008),
                    ParameterDefinition::fixed("bg_b", 0.04),
                    ParameterDefinition::fixed("bg_c", 1.2),
                ],
                BackgroundKind::Exponential => vec![
                    ParameterDefinition::fixed("bg_amplitude", 3.0),
                    ParameterDefinition::fixed("bg_decay", 20.0),
                ],
                BackgroundKind::PowerLaw => vec![
                    ParameterDefinition::fixed("bg_amplitude", 2.5),
                    ParameterDefinition::fixed("bg_exponent", -0.15),
                ],
            };
            for equal_sigma in [true, false] {
                let truth = [
                    ManualPeakSeed {
                        center: 4.8,
                        sigma: 0.28,
                        amplitude: 70.0,
                    },
                    ManualPeakSeed {
                        center: 5.55,
                        sigma: if equal_sigma { 0.28 } else { 0.42 },
                        amplitude: 55.0,
                    },
                ];
                let y = x
                    .iter()
                    .map(|&value| {
                        let baseline = match background {
                            BackgroundKind::None => 0.0,
                            BackgroundKind::Constant => 2.0,
                            BackgroundKind::Linear => 0.12 * value + 1.5,
                            BackgroundKind::Quadratic => 0.008 * value * value + 0.04 * value + 1.2,
                            BackgroundKind::Exponential => 3.0 * (-value / 20.0).exp(),
                            BackgroundKind::PowerLaw => 2.5 * value.powf(-0.15),
                        };
                        baseline
                            + truth
                                .iter()
                                .map(|seed| {
                                    super::evaluate_manual_peak(*seed, value, bin_width).unwrap()
                                })
                                .sum::<f64>()
                    })
                    .collect::<Vec<_>>();
                for objective in [ObjectiveKind::LeastSquares, ObjectiveKind::PoissonDeviance] {
                    let result = fit_peaks(
                        &PeakFitRequest {
                            x: x.clone(),
                            y: y.clone(),
                            bin_width,
                            region: [2.0, 9.0],
                            peak_seeds: vec![
                                ManualPeakSeed {
                                    center: 4.75,
                                    sigma: 0.33,
                                    amplitude: 65.0,
                                },
                                ManualPeakSeed {
                                    center: 5.6,
                                    sigma: 0.33,
                                    amplitude: 50.0,
                                },
                            ],
                            peak_bounds: None,
                            background_markers: if background == BackgroundKind::None {
                                Vec::new()
                            } else {
                                vec![(1.0, 2.0), (9.0, 11.0)]
                            },
                            background,
                            background_seed: (!background_parameters.is_empty()).then(|| {
                                BackgroundSeed {
                                    parameters: background_parameters.clone(),
                                }
                            }),
                            background_coupling: BackgroundCoupling::PrefitFrozen,
                            equal_sigma,
                            free_centers: true,
                            sigma_bounds: None,
                        },
                        &FitOptions {
                            objective,
                            ..FitOptions::default()
                        },
                    )
                    .unwrap_or_else(|error| {
                        panic!("{background:?}/{objective:?}/{equal_sigma}: {error}")
                    });
                    assert!(
                        result.fit.termination.success,
                        "{background:?}/{objective:?}/{equal_sigma}: {:?}",
                        result.fit.termination
                    );
                    assert_eq!(result.peak_seeds.len(), 2);
                }
            }
        }
    }
}
