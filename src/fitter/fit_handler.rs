use rfd::FileDialog;

use std::fs::File;
use std::io::{Read, Write};

use super::egui_line::EguiLine;
use super::gaussian::GaussianFitter;
use super::linear::LinearFitter;

use crate::fitter::background_fitter::BackgroundFitter;

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub enum FitModel {
    Gaussian(Vec<f64>), // put the initial peak locations in here
    Linear,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub enum FitResult {
    Gaussian(GaussianFitter),
    Linear(LinearFitter),
}
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Fitter {
    pub x_data: Vec<f64>,
    pub y_data: Vec<f64>,
    pub y_err: Option<Vec<f64>>,
    pub background: Option<BackgroundFitter>,
    pub model: FitModel,
    pub result: Option<FitResult>,
    pub deconvoluted_lines: Vec<EguiLine>,
    pub convoluted_line: EguiLine,
}

impl Fitter {
    // Constructor to create a new Fitter with empty data and specified model
    pub fn new(model: FitModel, background: Option<BackgroundFitter>) -> Self {
        Fitter {
            x_data: Vec::new(),
            y_data: Vec::new(),
            y_err: None,
            background,
            model,
            result: None,
            deconvoluted_lines: Vec::new(),
            convoluted_line: EguiLine::new("Convoluted".to_string(), egui::Color32::BLUE),
        }
    }

    fn subtract_background(&self) -> Vec<f64> {
        if let Some(bg_fitter) = &self.background {
            if let Some(bg_result) = bg_fitter.get_background(&self.x_data) {
                self.y_data
                    .iter()
                    .zip(bg_result.iter())
                    .map(|(y, bg)| y - bg)
                    .collect()
            } else {
                self.y_data.clone()
            }
        } else {
            self.y_data.clone()
        }
    }

    pub fn get_peak_markers(&self) -> Vec<f64> {
        if let Some(FitResult::Gaussian(fit)) = &self.result {
            fit.peak_markers.clone()
        } else if let FitModel::Gaussian(peak_markers) = &self.model {
            peak_markers.clone()
        } else {
            Vec::new()
        }
    }

    pub fn fit(&mut self) {
        // Fit the background if it's defined and there is no background result
        if let Some(bg_fitter) = &mut self.background {
            if bg_fitter.result.is_none() {
                bg_fitter.fit();
            }
        }

        // Perform the background subtraction if necessary
        let y_data_corrected = self.subtract_background();

        // Perform the fit based on the model
        match &self.model {
            FitModel::Gaussian(peak_markers) => {
                // Perform Gaussian fit
                let mut fit = GaussianFitter::new(
                    self.x_data.clone(),
                    y_data_corrected,
                    peak_markers.clone(),
                );

                fit.multi_gauss_fit();

                // get the fit_lines and store them in the deconvoluted_lines
                let deconvoluted_default_color = egui::Color32::from_rgb(255, 0, 255);
                if let Some(fit_lines) = &fit.fit_lines {
                    for (i, line) in fit_lines.iter().enumerate() {
                        let mut fit_line = EguiLine::new(format!("Peak {}", i), deconvoluted_default_color);
                        fit_line.points = line.clone();
                        self.deconvoluted_lines.push(fit_line);
                    }
                }

                self.result = Some(FitResult::Gaussian(fit));
            }

            FitModel::Linear => {
                // Perform Linear fit
                let mut fit = LinearFitter::new(self.x_data.clone(), y_data_corrected);

                fit.perform_linear_fit();

                self.result = Some(FitResult::Linear(fit));
            }
        }
    }

    pub fn fitter_stats(&self, ui: &mut egui::Ui) {
        if let Some(fit) = &self.result {
            match fit {
                FitResult::Gaussian(fit) => fit.fit_params_ui(ui),
                FitResult::Linear(fit) => fit.fit_params_ui(ui),
            }
        }
    }

    pub fn draw(&self, plot_ui: &mut egui_plot::PlotUi, log_y_scale: bool) {
        // Draw the fit lines
        if let Some(fit) = &self.result {
            match fit {
                FitResult::Gaussian(fit) => {
                    // Draw the deconvoluted lines
                    for line in &self.deconvoluted_lines {
                        line.draw(plot_ui);
                    }

                    if let Some(background) = &self.background {
                        // Draw the background fit
                        background.draw(plot_ui);

                        // Draw the convoluted line if background fit is available
                        // if self.convoluted_line.draw {
                        //     if let Some((slope, intercept)) = background.get_slope_intercept() {
                        //         let convoluted_points = fit
                        //             .calculate_convoluted_fit_points_with_linear_background(
                        //                 slope,
                        //                 intercept,
                        //                 log_y_scale,
                        //             );
                        //         let line = Line::new(egui::PlotPoints::Owned(convoluted_points))
                        //             .color(self.convoluted_line.color)
                        //             .stroke(Stroke::new(1.0, self.convoluted_line.color));
                        //         plot_ui.line(line);
                        //     }
                        // }
                    }
                }

                FitResult::Linear(fit) => {
                    log::info!("Drawing linear fit");
                }
            }
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Fits {
    pub temp_fit: Option<Fitter>,
    pub temp_background_fit: Option<BackgroundFitter>,
    pub stored_fits: Vec<Fitter>,
}

impl Default for Fits {
    fn default() -> Self {
        Self::new()
    }
}

impl Fits {
    pub fn new() -> Self {
        Fits {
            temp_fit: None,
            temp_background_fit: None,
            stored_fits: Vec::new(),
        }
    }

    fn save_to_file(&self) {
        if let Some(path) = FileDialog::new().add_filter("JSON", &["json"]).save_file() {
            let file = File::create(path);
            match file {
                Ok(mut file) => {
                    let json = serde_json::to_string(self).expect("Failed to serialize fits");
                    file.write_all(json.as_bytes())
                        .expect("Failed to write file");
                }
                Err(e) => {
                    log::error!("Error creating file: {:?}", e);
                }
            }
        }
    }

    fn load_from_file(&mut self) {
        if let Some(path) = FileDialog::new().add_filter("JSON", &["json"]).pick_file() {
            let file = File::open(path);
            match file {
                Ok(mut file) => {
                    let mut contents = String::new();
                    file.read_to_string(&mut contents)
                        .expect("Failed to read file");
                    let loaded_fits: Fits =
                        serde_json::from_str(&contents).expect("Failed to deserialize fits");
                    self.stored_fits.extend(loaded_fits.stored_fits); // Append loaded fits to current stored fits
                    self.temp_fit = loaded_fits.temp_fit; // override temp_fit
                    self.temp_background_fit = loaded_fits.temp_background_fit; // override temp_background_fit
                }
                Err(e) => {
                    log::error!("Error opening file: {:?}", e);
                }
            }
        }
    }

    pub fn save_and_load_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("Save Fits").clicked() {
                self.save_to_file();
            }

            ui.separator();

            if ui.button("Load Fits").clicked() {
                self.load_from_file();
            }
        });
    }

    pub fn remove_temp_fits(&mut self) {
        self.temp_fit = None;
        self.temp_background_fit = None;
    }

    pub fn draw(&self, plot_ui: &mut egui_plot::PlotUi, log_y_scale: bool) {
        if let Some(temp_fit) = &self.temp_fit {
            temp_fit.draw(plot_ui, log_y_scale);
        }

        if let Some(temp_background_fit) = &self.temp_background_fit {
            temp_background_fit.draw(plot_ui);
        }

        for fit in self.stored_fits.iter() {
            fit.draw(plot_ui, log_y_scale);
        }
    }

    pub fn fit_stats_grid_ui(&mut self, ui: &mut egui::Ui) {
        // only show the grid if there is something to show
        if self.temp_fit.is_none() && self.stored_fits.is_empty() {
            return;
        }

        let mut to_remove = None;

        egui::Grid::new("fit_params_grid")
            .striped(true)
            .show(ui, |ui| {
                ui.label("Fit");
                ui.label("Peak");
                ui.label("Mean");
                ui.label("FWHM");
                ui.label("Area");
                ui.end_row();

                if self.temp_fit.is_some() {
                    ui.label("Current");

                    if let Some(temp_fit) = &self.temp_fit {
                        temp_fit.fitter_stats(ui);
                    }
                }

                if !self.stored_fits.is_empty() {
                    for (i, fit) in self.stored_fits.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(format!("{}", i));

                            ui.separator();

                            if ui.button("X").clicked() {
                                to_remove = Some(i);
                            }

                            ui.separator();
                        });
                        fit.fitter_stats(ui);
                    }
                }
            });

        if let Some(index) = to_remove {
            self.stored_fits.remove(index);
        }
    }

    pub fn fit_context_menu_ui(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Fits", |ui| {
            self.save_and_load_ui(ui);

            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                self.fit_stats_grid_ui(ui);
            });
        });
    }
}
