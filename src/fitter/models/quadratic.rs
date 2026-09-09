use crate::fitter::common::{Data, Parameter};

#[derive(PartialEq, Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct QuadraticParameters {
    pub a: Parameter,
    pub b: Parameter,
    pub c: Parameter,
}

impl Default for QuadraticParameters {
    fn default() -> Self {
        Self {
            a: named_parameter("a"),
            b: named_parameter("b"),
            c: named_parameter("c"),
        }
    }
}

impl QuadraticParameters {
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
        egui::Grid::new("quadratic_params_grid")
            .striped(true)
            .num_columns(5)
            .show(ui, |ui| {
                ui.label("Parameter");
                ui.label("Current / Next Guess");
                ui.label("Min");
                ui.label("Max");
                ui.label("Vary");
                ui.end_row();
                self.a.ui(ui);
                ui.end_row();
                self.b.ui(ui);
                ui.end_row();
                self.c.ui(ui);
            });
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct QuadraticFitter {
    pub data: Data,
    pub paramaters: QuadraticParameters,
    pub fit_points: Vec<[f64; 2]>,
    pub fit_report: String,
    pub covar: Option<[[f64; 3]; 3]>,
}

impl QuadraticFitter {
    pub fn new(data: Data) -> Self {
        Self {
            data,
            paramaters: QuadraticParameters::default(),
            fit_points: Vec::new(),
            fit_report: String::new(),
            covar: None,
        }
    }

    pub fn new_from_parameters(
        a: (f64, f64),
        b: (f64, f64),
        c: (f64, f64),
        min_x: f64,
        max_x: f64,
    ) -> Self {
        let mut fitter = Self {
            data: Data::default(),
            paramaters: QuadraticParameters {
                a: fitted_parameter("a", a),
                b: fitted_parameter("b", b),
                c: fitted_parameter("c", c),
            },
            fit_points: Vec::new(),
            fit_report: "Fitted with another native model".to_owned(),
            covar: None,
        };
        fitter.fit_points = sampled_points(min_x, max_x, |x| fitter.evaluate(x));
        fitter
    }

    pub fn fit(&mut self) -> Result<(), spectrix_fitting::FitError> {
        let model = spectrix_fitting::QuadraticModel::new(
            "",
            [
                self.paramaters.a.initial_guess,
                self.paramaters.b.initial_guess,
                self.paramaters.c.initial_guess,
            ],
        )
        .with_parameters([
            crate::fitter::native::parameter_definition("a", &self.paramaters.a, None),
            crate::fitter::native::parameter_definition("b", &self.paramaters.b, None),
            crate::fitter::native::parameter_definition("c", &self.paramaters.c, None),
        ]);
        let result = spectrix_fitting::fit(
            &spectrix_fitting::FitProblem::new(
                Box::new(model),
                self.data.x.clone(),
                self.data.y.clone(),
            ),
            &spectrix_fitting::FitOptions::robust(),
        )?;
        crate::fitter::native::apply_estimate(&mut self.paramaters.a, &result, "a");
        crate::fitter::native::apply_estimate(&mut self.paramaters.b, &result, "b");
        crate::fitter::native::apply_estimate(&mut self.paramaters.c, &result, "c");
        self.covar = crate::fitter::native::covariance_3(&result, ["a", "b", "c"]);
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
        self.paramaters.a.value.unwrap_or(0.0).mul_add(
            x * x,
            self.paramaters
                .b
                .value
                .unwrap_or(0.0)
                .mul_add(x, self.paramaters.c.value.unwrap_or(0.0)),
        )
    }

    pub fn ui(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            parameter_label(ui, "a", &self.paramaters.a);
            ui.separator();
            parameter_label(ui, "b", &self.paramaters.b);
            ui.separator();
            parameter_label(ui, "c", &self.paramaters.c);
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

fn named_parameter(name: &str) -> Parameter {
    Parameter {
        name: name.to_owned(),
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
