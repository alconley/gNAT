use crate::fitter::common::{Data, Parameter};

#[derive(PartialEq, Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct LinearParameters {
    pub slope: Parameter,
    pub intercept: Parameter,
}

impl Default for LinearParameters {
    fn default() -> Self {
        Self {
            slope: Parameter {
                name: "slope".to_owned(),
                ..Default::default()
            },
            intercept: Parameter {
                name: "intercept".to_owned(),
                ..Default::default()
            },
        }
    }
}

impl LinearParameters {
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
        egui::Grid::new("linear_params_grid")
            .striped(true)
            .num_columns(5)
            .show(ui, |ui| {
                ui.label("Parameter");
                ui.label("Current / Next Guess");
                ui.label("Min");
                ui.label("Max");
                ui.label("Vary");
                ui.end_row();
                self.slope.ui(ui);
                ui.end_row();
                self.intercept.ui(ui);
            });
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct LinearFitter {
    pub data: Data,
    pub paramaters: LinearParameters,
    pub fit_points: Vec<[f64; 2]>,
    pub fit_report: String,
}

impl LinearFitter {
    pub fn new(data: Data) -> Self {
        Self {
            data,
            paramaters: LinearParameters::default(),
            fit_points: Vec::new(),
            fit_report: String::new(),
        }
    }

    pub fn new_from_parameters(
        slope: (f64, f64),
        intercept: (f64, f64),
        min_x: f64,
        max_x: f64,
    ) -> Self {
        let fit_points = vec![
            [min_x, slope.0.mul_add(min_x, intercept.0)],
            [max_x, slope.0.mul_add(max_x, intercept.0)],
        ];
        let paramaters = LinearParameters {
            slope: fitted_parameter("slope", slope),
            intercept: fitted_parameter("intercept", intercept),
        };
        Self {
            data: Data::default(),
            paramaters,
            fit_points,
            fit_report: "Fitted with another native model".to_owned(),
        }
    }

    pub fn fit(&mut self) -> Result<(), spectrix_fitting::FitError> {
        let model = spectrix_fitting::LinearModel::new(
            "",
            [
                self.paramaters.slope.initial_guess,
                self.paramaters.intercept.initial_guess,
            ],
        )
        .with_parameters([
            crate::fitter::native::parameter_definition("slope", &self.paramaters.slope, None),
            crate::fitter::native::parameter_definition(
                "intercept",
                &self.paramaters.intercept,
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
        crate::fitter::native::apply_estimate(&mut self.paramaters.slope, &result, "slope");
        crate::fitter::native::apply_estimate(&mut self.paramaters.intercept, &result, "intercept");
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
        self.paramaters
            .slope
            .value
            .unwrap_or(0.0)
            .mul_add(x, self.paramaters.intercept.value.unwrap_or(0.0))
    }

    pub fn ui(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if let Some(slope) = self.paramaters.slope.value {
                ui.label(format!(
                    "Slope: {slope:.3} ± {:.3}",
                    self.paramaters.slope.uncertainty.unwrap_or(0.0)
                ));
            }
            ui.separator();
            if let Some(intercept) = self.paramaters.intercept.value {
                ui.label(format!(
                    "Intercept: {intercept:.3} ± {:.3}",
                    self.paramaters.intercept.uncertainty.unwrap_or(0.0)
                ));
            }
            ui.separator();
            ui.menu_button("Fit Report", |ui| {
                ui.horizontal_wrapped(|ui| ui.label(&self.fit_report));
            });
        });
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
