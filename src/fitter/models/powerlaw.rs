use crate::fitter::common::{Data, Parameter};

#[derive(PartialEq, Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct PowerLawParameters {
    pub amplitude: Parameter,
    pub exponent: Parameter,
}

impl Default for PowerLawParameters {
    fn default() -> Self {
        Self {
            amplitude: named_parameter("amplitude", 0.0),
            exponent: named_parameter("exponent", -1.0),
        }
    }
}

impl PowerLawParameters {
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
        egui::Grid::new("power_law_params_grid")
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
                self.exponent.ui(ui);
            });
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct PowerLawFitter {
    pub data: Data,
    pub paramaters: PowerLawParameters,
    pub fit_points: Vec<[f64; 2]>,
    pub fit_report: String,
}

impl PowerLawFitter {
    pub fn new(data: Data) -> Self {
        Self {
            data,
            paramaters: PowerLawParameters::default(),
            fit_points: Vec::new(),
            fit_report: String::new(),
        }
    }

    pub fn new_from_parameters(
        amplitude: (f64, f64),
        exponent: (f64, f64),
        min_x: f64,
        max_x: f64,
    ) -> Self {
        let mut fitter = Self {
            data: Data::default(),
            paramaters: PowerLawParameters {
                amplitude: fitted_parameter("amplitude", amplitude),
                exponent: fitted_parameter("exponent", exponent),
            },
            fit_points: Vec::new(),
            fit_report: "Fitted with another native model".to_owned(),
        };
        fitter.fit_points = sampled_points(min_x, max_x, |x| fitter.evaluate(x));
        fitter
    }

    pub fn fit(&mut self) -> Result<(), spectrix_fitting::FitError> {
        let model = spectrix_fitting::PowerLawModel::new(
            "",
            [
                self.paramaters.amplitude.initial_guess,
                self.paramaters.exponent.initial_guess,
            ],
        )
        .with_parameters([
            crate::fitter::native::parameter_definition(
                "amplitude",
                &self.paramaters.amplitude,
                None,
            ),
            crate::fitter::native::parameter_definition(
                "exponent",
                &self.paramaters.exponent,
                None,
            ),
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
        crate::fitter::native::apply_estimate(&mut self.paramaters.exponent, &result, "exponent");
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
            * x.powf(self.paramaters.exponent.value.unwrap_or(-1.0))
    }

    pub fn ui(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            parameter_label(ui, "Amplitude", &self.paramaters.amplitude);
            ui.separator();
            parameter_label(ui, "Exponent", &self.paramaters.exponent);
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
