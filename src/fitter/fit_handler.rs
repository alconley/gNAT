use rfd::FileDialog;

use std::fs::File;
use std::io::{Read, Write};

use super::background_fitter::BackgroundFitter;
use super::fit_settings::FitSettings;
use super::main_fitter::Fitter;

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Fits {
    pub temp_fit: Option<Fitter>,
    pub temp_background_fit: Option<BackgroundFitter>,
    pub stored_fits: Vec<Fitter>,
    pub settings: FitSettings,
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
            settings: FitSettings::default(),
        }
    }

    pub fn store_temp_fit(&mut self) {
        if let Some(temp_fit) = &mut self.temp_fit.take() {
            temp_fit.set_background_color(egui::Color32::DARK_GREEN);
            temp_fit.set_composition_color(egui::Color32::DARK_BLUE);
            temp_fit.set_decomposition_color(egui::Color32::from_rgb(150, 0, 255));

            temp_fit.set_name(format!("Fit {}", self.stored_fits.len()));

            self.stored_fits.push(temp_fit.clone());
        }

        self.temp_background_fit = None;
    }

    pub fn set_log(&mut self, log_y: bool, log_x: bool) {
        if let Some(temp_fit) = &mut self.temp_fit {
            temp_fit.set_log(log_y, log_x);
        }

        if let Some(temp_background_fit) = &mut self.temp_background_fit {
            temp_background_fit.fit_line.log_y = log_y;
            temp_background_fit.fit_line.log_x = log_x;
        }

        for fit in &mut self.stored_fits {
            fit.set_log(log_y, log_x);
        }
    }

    pub fn set_stored_fits_background_color(&mut self, color: egui::Color32) {
        for fit in &mut self.stored_fits {
            if let Some(background) = &mut fit.background {
                background.fit_line.color = color;
            }
        }
    }

    pub fn set_stored_fits_composition_color(&mut self, color: egui::Color32) {
        for fit in &mut self.stored_fits {
            fit.composition_line.color = color;
        }
    }

    pub fn set_stored_fits_decomposition_color(&mut self, color: egui::Color32) {
        for fit in &mut self.stored_fits {
            for line in &mut fit.decomposition_lines {
                line.color = color;
            }
        }
    }

    pub fn update_visibility(&mut self) {
        if let Some(temp_fit) = &mut self.temp_fit {
            temp_fit.show_decomposition(self.settings.show_decomposition);
            temp_fit.show_composition(self.settings.show_composition);
            temp_fit.show_background(self.settings.show_background);
        }

        for fit in &mut self.stored_fits {
            fit.show_decomposition(self.settings.show_decomposition);
            fit.show_composition(self.settings.show_composition);
            fit.show_background(self.settings.show_background);
        }
    }

    pub fn apply_visibility_settings(&mut self) {
        self.update_visibility();
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

    pub fn draw(&mut self, plot_ui: &mut egui_plot::PlotUi) {
        self.apply_visibility_settings();

        if let Some(temp_fit) = &self.temp_fit {
            temp_fit.draw(plot_ui);
        }

        if let Some(temp_background_fit) = &self.temp_background_fit {
            temp_background_fit.draw(plot_ui);
        }

        for fit in &mut self.stored_fits.iter() {
            fit.draw(plot_ui);
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

    pub fn fit_stats_ui(&mut self, ui: &mut egui::Ui) {
        if self.settings.show_fit_stats {
            ui.separator();

            egui::ScrollArea::vertical()
                .max_height(self.settings.fit_stats_height)
                .show(ui, |ui| {
                    self.fit_stats_grid_ui(ui);
                });
        }
    }

    pub fn fit_lines_ui(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.vertical(|ui| {
                ui.label("Fit Lines");

                if let Some(temp_fit) = &mut self.temp_fit {
                    temp_fit.lines_ui(ui);
                }

                for fit in &mut self.stored_fits {
                    fit.lines_ui(ui);
                }
            });
        });
    }

    pub fn fit_context_menu_ui(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Fits", |ui| {
            self.save_and_load_ui(ui);

            ui.separator();

            self.settings.menu_ui(ui);

            egui::ScrollArea::vertical()
                .max_height(300.0)
                .id_salt("Context menu fit stats grid")
                .show(ui, |ui| {
                    self.fit_stats_grid_ui(ui);
                });

            ui.separator();

            egui::ScrollArea::vertical()
                .max_height(300.0)
                .show(ui, |ui| {
                    self.fit_lines_ui(ui);
                });
        });
    }
}
