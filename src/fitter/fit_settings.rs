use crate::fitter::common::Parameter;
use crate::fitter::main_fitter::{BackgroundModel, BackgroundResult};
use crate::fitter::models::exponential::ExponentialParameters;
use crate::fitter::models::linear::LinearParameters;
use crate::fitter::models::powerlaw::PowerLawParameters;
use crate::fitter::models::quadratic::QuadraticParameters;
use spectrix_fitting::{BackgroundCoupling, ObjectiveKind};

/// The objective requested by the histogram UI. `Auto` selects least squares
/// for high-count bins, where its Gaussian approximation is more stable, and
/// Poisson deviance otherwise.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub enum HistogramObjective {
    #[default]
    Auto,
    PoissonDeviance,
    LeastSquares,
}

impl HistogramObjective {
    pub const HIGH_COUNT_THRESHOLD: f64 = 100.0;

    pub fn resolve(self, counts: impl IntoIterator<Item = f64>) -> ObjectiveKind {
        match self {
            Self::PoissonDeviance => ObjectiveKind::PoissonDeviance,
            Self::LeastSquares => ObjectiveKind::LeastSquares,
            Self::Auto => {
                let maximum = counts
                    .into_iter()
                    .filter(|count| count.is_finite())
                    .fold(0.0_f64, f64::max);
                if maximum >= Self::HIGH_COUNT_THRESHOLD {
                    ObjectiveKind::LeastSquares
                } else {
                    ObjectiveKind::PoissonDeviance
                }
            }
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::PoissonDeviance => "Poisson",
            Self::LeastSquares => "Least squares",
        }
    }
}

/// How the histogram's displayed calibration coefficients are maintained.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub enum CalibrationMode {
    #[default]
    None,
    Linear,
    Quadratic,
    Manual,
}

impl CalibrationMode {
    pub const fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Linear => "Linear",
            Self::Quadratic => "Quadratic",
            Self::Manual => "Manual",
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct FitSettings {
    pub show_decomposition: bool,
    pub show_composition: bool,
    pub show_background: bool,
    pub show_fit_lines_area: bool,
    /// Show the active fit separately from stored fits.
    pub show_temp_fit: bool,
    pub uuid_label_size: f32,
    pub uuid_label_lift: f32,
    pub uuid_label_guides: bool,
    pub show_fit_stats: bool,
    pub fit_panel_popout: bool,
    pub equal_stddev: bool,
    pub free_position: bool,
    /// Whether the editable initial-parameter columns are visible in the peak table.
    pub show_initial_parameters: bool,
    /// Re-estimate width, height, and bounds when a peak's center is dragged.
    pub auto_estimate_moved_peak: bool,
    pub background_model: BackgroundModel,
    pub lock_background: bool,
    /// Legacy persisted coupling mode; active fits derive this from `lock_background`.
    pub background_coupling: BackgroundCoupling,
    pub linear_params: LinearParameters,
    pub quadratic_params: QuadraticParameters,
    pub power_law_params: PowerLawParameters,
    pub exponential_params: ExponentialParameters,
    pub constant_param: Parameter,
    pub objective: HistogramObjective,
    pub calibrated: bool,
    /// The selected calibration workflow. Older calibrated workspaces infer `Manual` in the UI.
    #[serde(default)]
    pub calibration_mode: CalibrationMode,
    /// Legacy global sigma limits retained only so older saved workspaces load without loss.
    /// Per-peak initial FWHM bounds are the active constraint mechanism.
    pub constrain_sigma: bool,
    pub sigma_min: f64,
    pub sigma_max: f64,
}

impl Default for FitSettings {
    fn default() -> Self {
        Self {
            show_decomposition: true,
            show_composition: true,
            show_background: true,
            show_fit_lines_area: true,
            show_temp_fit: true,
            uuid_label_size: 14.0,
            uuid_label_lift: 1.6,
            uuid_label_guides: true,
            show_fit_stats: false,
            fit_panel_popout: false,
            equal_stddev: true,
            free_position: true,
            show_initial_parameters: true,
            auto_estimate_moved_peak: false,
            background_model: BackgroundModel::None,
            lock_background: false,
            background_coupling: BackgroundCoupling::PrefitFrozen,
            linear_params: LinearParameters::default(),
            quadratic_params: QuadraticParameters::default(),
            power_law_params: PowerLawParameters::default(),
            exponential_params: ExponentialParameters::default(),
            constant_param: Parameter {
                name: "Constant".to_owned(),
                min: 0.0,
                ..Parameter::default()
            },
            objective: HistogramObjective::Auto,
            calibrated: false,
            calibration_mode: CalibrationMode::None,
            constrain_sigma: false,
            sigma_min: 0.1,
            sigma_max: 10.0,
        }
    }
}

impl FitSettings {
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        background_parameters_are_current: bool,
        manual_background_available: bool,
        has_background_windows: bool,
    ) {
        ui.separator();

        let background_label = match &self.background_model {
            BackgroundModel::Constant(_) => "Constant",
            BackgroundModel::Linear(_) => "Linear",
            BackgroundModel::Quadratic(_) => "Quadratic",
            BackgroundModel::PowerLaw(_) => "Power Law",
            BackgroundModel::Exponential(_) => "Exponential",
            BackgroundModel::None | BackgroundModel::LegacyAuto => "None",
        };
        egui::ComboBox::from_label("Background model")
            .selected_text(background_label)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.background_model, BackgroundModel::None, "None");
                ui.selectable_value(
                    &mut self.background_model,
                    BackgroundModel::Constant(self.constant_param.clone()),
                    "Constant",
                );
                ui.selectable_value(
                    &mut self.background_model,
                    BackgroundModel::Linear(self.linear_params.clone()),
                    "Linear",
                );
                ui.selectable_value(
                    &mut self.background_model,
                    BackgroundModel::Quadratic(self.quadratic_params.clone()),
                    "Quadratic",
                );
                ui.selectable_value(
                    &mut self.background_model,
                    BackgroundModel::PowerLaw(self.power_law_params.clone()),
                    "Power Law",
                );
                ui.selectable_value(
                    &mut self.background_model,
                    BackgroundModel::Exponential(self.exponential_params.clone()),
                    "Exponential",
                );
            })
            .response
            .on_hover_text("Choose the background function. With windows, their fitted background stays fixed during peak fitting. Without windows, background and peaks are fitted jointly.");

        if let BackgroundModel::Constant(parameter) = &mut self.background_model {
            parameter.ui(ui);
            self.constant_param = parameter.clone();
        }

        if let BackgroundModel::Linear(params) = &mut self.background_model {
            params.ui(ui, background_parameters_are_current);
            self.linear_params = params.clone();
        }

        if let BackgroundModel::Quadratic(params) = &mut self.background_model {
            params.ui(ui, background_parameters_are_current);
            self.quadratic_params = params.clone();
        }

        if let BackgroundModel::PowerLaw(params) = &mut self.background_model {
            params.ui(ui, background_parameters_are_current);
            self.power_law_params = params.clone();
        }

        if let BackgroundModel::Exponential(params) = &mut self.background_model {
            params.ui(ui, background_parameters_are_current);
            self.exponential_params = params.clone();
        }

        if !matches!(
            self.background_model,
            BackgroundModel::None | BackgroundModel::LegacyAuto
        ) {
            if has_background_windows {
                ui.label("Window background stays fixed during peak fitting.")
                    .on_hover_text("Move or resize the windows to change the background. Its line spans the region and all background windows. Peak uncertainties are conditional on that fitted baseline.");
            }
            ui.add_enabled(
                manual_background_available,
                egui::Checkbox::new(&mut self.lock_background, "Lock manual background"),
            )
            .on_hover_text(if manual_background_available {
                "Keep an explicit background fit fixed, including without windows. Window-based backgrounds already stay fixed. Peak uncertainties exclude uncertainty in a fixed background."
            } else {
                "Run an explicit background fit first to make a lockable background available."
            });
            if !manual_background_available {
                self.lock_background = false;
            }
        }

        ui.separator();

        ui.collapsing("Fit Behavior", |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.toggle_value(&mut self.equal_stddev, "Shared FWHM")
                    .on_hover_text(
                        "When on, all peaks share one FWHM. Turn off for independent FWHM values.",
                    );
                ui.checkbox(&mut self.free_position, "Free Position")
                    .on_hover_text("Allow the position of the Gaussian to be free");
                ui.checkbox(&mut self.show_initial_parameters, "Show initial parameters")
                    .on_hover_text(
                        "Show the editable min ≤ value ≤ max columns in the peak table. Hide them to focus on fitted results; this does not change any initial values.",
                    );
                ui.checkbox(
                    &mut self.auto_estimate_moved_peak,
                    "Auto-estimate moved peak",
                )
                .on_hover_text(
                    "When a peak center is dragged, re-estimate only that peak's width, height, and initial bounds from the histogram. This leaves the dragged center where you placed it.",
                );
                egui::ComboBox::from_label("Fit objective")
                    .selected_text(self.objective.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.objective, HistogramObjective::Auto, "Auto")
                            .on_hover_text("Uses least squares when any selected bin has 100 or more counts; otherwise uses Poisson deviance.");
                        ui.selectable_value(
                            &mut self.objective,
                            HistogramObjective::PoissonDeviance,
                            "Poisson",
                        );
                        ui.selectable_value(
                            &mut self.objective,
                            HistogramObjective::LeastSquares,
                            "Least squares",
                        );
                    })
                    .response
                    .on_hover_text("Choose the objective used by background and Gaussian fits.");
            });
        });

        ui.separator();
    }

    pub fn apply_background_fit(&mut self, result: &BackgroundResult) {
        fn update(target: &mut Parameter, fitted: &Parameter) {
            if let Some(value) = fitted.value.filter(|value| value.is_finite()) {
                target.initial_guess = value;
            }
        }

        match (&mut self.background_model, result) {
            (BackgroundModel::Constant(parameter), BackgroundResult::Constant(fit)) => {
                update(parameter, &fit.paramaters.intercept);
                self.constant_param = parameter.clone();
            }
            (BackgroundModel::Linear(parameters), BackgroundResult::Linear(fit)) => {
                update(&mut parameters.slope, &fit.paramaters.slope);
                update(&mut parameters.intercept, &fit.paramaters.intercept);
                self.linear_params = parameters.clone();
            }
            (BackgroundModel::Quadratic(parameters), BackgroundResult::Quadratic(fit)) => {
                update(&mut parameters.a, &fit.paramaters.a);
                update(&mut parameters.b, &fit.paramaters.b);
                update(&mut parameters.c, &fit.paramaters.c);
                self.quadratic_params = parameters.clone();
            }
            (BackgroundModel::PowerLaw(parameters), BackgroundResult::PowerLaw(fit)) => {
                update(&mut parameters.amplitude, &fit.paramaters.amplitude);
                update(&mut parameters.exponent, &fit.paramaters.exponent);
                self.power_law_params = parameters.clone();
            }
            (BackgroundModel::Exponential(parameters), BackgroundResult::Exponential(fit)) => {
                update(&mut parameters.amplitude, &fit.paramaters.amplitude);
                update(&mut parameters.decay, &fit.paramaters.decay);
                self.exponential_params = parameters.clone();
            }
            _ => {}
        }
    }

    pub fn background_parameters_match(&self, result: &BackgroundResult) -> bool {
        let matches = |initial_guess: f64, fitted: &Parameter| {
            fitted
                .value
                .is_some_and(|value| value.is_finite() && initial_guess == value)
        };
        match (&self.background_model, result) {
            (BackgroundModel::Constant(parameter), BackgroundResult::Constant(fit)) => {
                matches(parameter.initial_guess, &fit.paramaters.intercept)
            }
            (BackgroundModel::Linear(parameters), BackgroundResult::Linear(fit)) => {
                matches(parameters.slope.initial_guess, &fit.paramaters.slope)
                    && matches(
                        parameters.intercept.initial_guess,
                        &fit.paramaters.intercept,
                    )
            }
            (BackgroundModel::Quadratic(parameters), BackgroundResult::Quadratic(fit)) => {
                matches(parameters.a.initial_guess, &fit.paramaters.a)
                    && matches(parameters.b.initial_guess, &fit.paramaters.b)
                    && matches(parameters.c.initial_guess, &fit.paramaters.c)
            }
            (BackgroundModel::PowerLaw(parameters), BackgroundResult::PowerLaw(fit)) => {
                matches(
                    parameters.amplitude.initial_guess,
                    &fit.paramaters.amplitude,
                ) && matches(parameters.exponent.initial_guess, &fit.paramaters.exponent)
            }
            (BackgroundModel::Exponential(parameters), BackgroundResult::Exponential(fit)) => {
                matches(
                    parameters.amplitude.initial_guess,
                    &fit.paramaters.amplitude,
                ) && matches(parameters.decay.initial_guess, &fit.paramaters.decay)
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CalibrationMode, FitSettings, HistogramObjective};
    use crate::fitter::{
        main_fitter::{BackgroundModel, BackgroundResult},
        models::quadratic::QuadraticFitter,
    };
    #[test]
    fn legacy_equal_stddev_remains_the_width_coupling() {
        let shared: FitSettings =
            serde_json::from_str(r#"{"equal_stddev":true}"#).expect("legacy shared settings");
        let independent: FitSettings =
            serde_json::from_str(r#"{"equal_stddev":false}"#).expect("legacy independent settings");
        assert!(shared.equal_stddev);
        assert!(!independent.equal_stddev);
        assert!(!shared.auto_estimate_moved_peak);
        assert!(!independent.auto_estimate_moved_peak);
        assert!(shared.show_initial_parameters);
        assert!(independent.show_initial_parameters);
        assert!(shared.show_temp_fit);
        assert_eq!(shared.calibration_mode, CalibrationMode::None);
    }

    #[test]
    fn auto_estimate_moved_peak_round_trips() {
        let settings = FitSettings {
            auto_estimate_moved_peak: true,
            ..FitSettings::default()
        };
        let encoded = serde_json::to_string(&settings).expect("serialize fit settings");
        let decoded: FitSettings =
            serde_json::from_str(&encoded).expect("deserialize fit settings");
        assert!(decoded.auto_estimate_moved_peak);
        assert!(decoded.show_initial_parameters);
        assert!(decoded.show_temp_fit);
        assert_eq!(decoded.calibration_mode, CalibrationMode::None);
    }

    #[test]
    fn calibration_mode_round_trips_without_changing_legacy_calibrated_flag() {
        let settings = FitSettings {
            calibrated: true,
            calibration_mode: CalibrationMode::Quadratic,
            ..FitSettings::default()
        };
        let encoded = serde_json::to_string(&settings).expect("serialize fit settings");
        let decoded: FitSettings =
            serde_json::from_str(&encoded).expect("deserialize fit settings");
        assert!(decoded.calibrated);
        assert_eq!(decoded.calibration_mode, CalibrationMode::Quadratic);
    }

    #[test]
    fn fitted_background_values_become_visible_next_guesses() {
        let mut settings = FitSettings {
            background_model: BackgroundModel::Quadratic(Default::default()),
            ..FitSettings::default()
        };
        let fitted = BackgroundResult::Quadratic(QuadraticFitter::new_from_parameters(
            (0.125, 0.01),
            (-2.5, 0.02),
            (40.0, 0.5),
            0.0,
            10.0,
        ));

        settings.apply_background_fit(&fitted);

        let BackgroundModel::Quadratic(parameters) = &settings.background_model else {
            panic!("expected quadratic settings");
        };
        assert_eq!(parameters.a.initial_guess, 0.125);
        assert_eq!(parameters.b.initial_guess, -2.5);
        assert_eq!(parameters.c.initial_guess, 40.0);
        assert!(settings.background_parameters_match(&fitted));
        let BackgroundModel::Quadratic(parameters) = &mut settings.background_model else {
            panic!("expected quadratic settings");
        };
        parameters.a.initial_guess = 0.2;
        assert!(!settings.background_parameters_match(&fitted));
    }

    #[test]
    fn auto_objective_uses_least_squares_for_high_count_bins() {
        assert_eq!(
            HistogramObjective::Auto.resolve([0.0, 99.0]),
            spectrix_fitting::ObjectiveKind::PoissonDeviance
        );
        assert_eq!(
            HistogramObjective::Auto.resolve([0.0, 100.0]),
            spectrix_fitting::ObjectiveKind::LeastSquares
        );
    }
}
