use crate::fitter::common::{Data, Parameter};

#[derive(PartialEq, Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ExponentialParameters {
    pub amplitude: Parameter,
    pub decay: Parameter,
}

impl Default for ExponentialParameters {
    fn default() -> Self {
        Self {
            amplitude: named_parameter("amplitude", 0.0),
            decay: named_parameter("decay", 500.0),
        }
    }
}

impl ExponentialParameters {
    pub fn ui(&mut self, ui: &mut egui::Ui, parameters_are_current: bool) {
        ui.horizontal(|ui| {
            ui.label(if parameters_are_current {
                "Current Background Fit Parameters"
            } else {
                "Background Starting Parameters"
            });
            if ui.small_button("Reset").clicked() {
                *self = Self::default();
            }
        });
        egui::Grid::new("exponential_params_grid")
            .striped(true)
            .num_columns(5)
            .show(ui, |ui| {
                ui.label("Parameter");
                ui.label("Current / Next Guess");
                ui.label("Min");
                ui.label("Max");
                ui.label("Vary");
                ui.end_row();
                self.amplitude.ui(ui);
                ui.end_row();
                self.decay.ui(ui);
            });
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ExponentialFitter {
    pub data: Data,
    pub paramaters: ExponentialParameters,
    pub fit_points: Vec<[f64; 2]>,
    pub fit_report: String,
}

impl ExponentialFitter {
    pub fn new(data: Data) -> Self {
        Self {
            data,
            paramaters: ExponentialParameters::default(),
            fit_points: Vec::new(),
            fit_report: String::new(),
        }
    }

    pub fn new_from_parameters(
        amplitude: (f64, f64),
        decay: (f64, f64),
        min_x: f64,
        max_x: f64,
    ) -> Self {
        let mut fitter = Self {
            data: Data::default(),
            paramaters: ExponentialParameters {
                amplitude: fitted_parameter("amplitude", amplitude),
                decay: fitted_parameter("decay", decay),
            },
            fit_points: Vec::new(),
            fit_report: "Fitted with another native model".to_owned(),
        };
        fitter.fit_points = sampled_points(min_x, max_x, |x| fitter.evaluate(x));
        fitter
    }

    pub fn fit(&mut self) -> Result<(), spectrix_fitting::FitError> {
        let model = spectrix_fitting::ExponentialModel::new(
            "",
            [
                self.paramaters.amplitude.initial_guess,
                self.paramaters.decay.initial_guess,
            ],
        )
        .with_parameters([
            crate::fitter::native::parameter_definition(
                "amplitude",
                &self.paramaters.amplitude,
                None,
            ),
            crate::fitter::native::parameter_definition("decay", &self.paramaters.decay, None),
        ]);
        let result = spectrix_fitting::fit(
            &spectrix_fitting::FitProblem::new(
                Box::new(model),
                self.data.x.clone(),
                self.data.y.clone(),
            ),
            &spectrix_fitting::FitOptions::robust(),
        )?;
        crate::fitter::native::apply_estimate(&mut self.paramaters.amplitude, &result, "amplitude");
        crate::fitter::native::apply_estimate(&mut self.paramaters.decay, &result, "decay");
        self.fit_points = result
            .evaluation_x
            .iter()
            .copied()
            .zip(result.best_fit.iter().copied())
            .map(Into::into)
            .collect();
        self.fit_report = crate::fitter::native::fit_report(&result);
        Ok(())
    }

    pub fn evaluate(&self, x: f64) -> f64 {
        self.paramaters.amplitude.value.unwrap_or(1.0)
            * (-x / self.paramaters.decay.value.unwrap_or(1.0)).exp()
    }

    pub fn ui(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            parameter_label(ui, "Amplitude", &self.paramaters.amplitude);
            ui.separator();
            parameter_label(ui, "Decay", &self.paramaters.decay);
            ui.separator();
            ui.menu_button("Fit Report", |ui| {
                ui.horizontal_wrapped(|ui| ui.label(&self.fit_report));
            });
        });
    }
}

fn sampled_points(min_x: f64, max_x: f64, evaluate: impl Fn(f64) -> f64) -> Vec<[f64; 2]> {
    (0..=100)
        .map(|index| {
            let x = (max_x - min_x).mul_add(index as f64 / 100.0, min_x);
            [x, evaluate(x)]
        })
        .collect()
}

fn named_parameter(name: &str, initial_guess: f64) -> Parameter {
    Parameter {
        name: name.to_owned(),
        initial_guess,
        ..Default::default()
    }
}

fn fitted_parameter(name: &str, estimate: (f64, f64)) -> Parameter {
    Parameter {
        name: name.to_owned(),
        min: f64::NEG_INFINITY,
        max: f64::INFINITY,
        initial_guess: estimate.0,
        vary: true,
        value: Some(estimate.0),
        uncertainty: Some(estimate.1),
        calibrated_value: None,
        calibrated_uncertainty: None,
    }
}

fn parameter_label(ui: &mut egui::Ui, name: &str, parameter: &Parameter) {
    if let Some(value) = parameter.value {
        ui.label(format!(
            "{name}: {value:.3} ± {:.3}",
            parameter.uncertainty.unwrap_or(0.0)
        ));
    }
}
