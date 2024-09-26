use super::configure_lazyframes::LazyFrameInfo;

#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub enum HistoConfig {
    AddHisto1d(AddHisto1d),
    AddHisto2d(AddHisto2d),
    FillHisto1d(FillHisto1d),
    FillHisto2d(FillHisto2d),
}

impl HistoConfig {
    pub fn add_ui(&mut self, ui: &mut egui::Ui, grids: Vec<String>) {
        match self {
            HistoConfig::AddHisto1d(config) => {
                config.ui(ui, grids);
            }
            HistoConfig::AddHisto2d(config) => {
                config.ui(ui, grids);
            }
            _ => {}
        }
    }

    pub fn fill_ui(
        &mut self,
        ui: &mut egui::Ui,
        lazyframe_info: LazyFrameInfo,
        histogram_names: Vec<String>,
    ) {
        match self {
            HistoConfig::FillHisto1d(config) => {
                config.ui(ui, lazyframe_info, histogram_names);
            }
            HistoConfig::FillHisto2d(config) => {
                config.ui(ui, lazyframe_info, histogram_names);
            }
            _ => {}
        }
    }

    pub fn set_name(&mut self, new_name: &str) {
        match self {
            HistoConfig::AddHisto1d(config) => config.name = new_name.to_string(),
            HistoConfig::AddHisto2d(config) => config.name = new_name.to_string(),
            HistoConfig::FillHisto1d(config) => config.name = new_name.to_string(),
            HistoConfig::FillHisto2d(config) => config.name = new_name.to_string(),
        }
    }

    pub fn name(&self) -> String {
        match self {
            HistoConfig::AddHisto1d(config) => config.name.clone(),
            HistoConfig::AddHisto2d(config) => config.name.clone(),
            HistoConfig::FillHisto1d(config) => config.name.clone(),
            HistoConfig::FillHisto2d(config) => config.name.clone(),
        }
    }
}

#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub struct AddHisto1d {
    pub name: String,
    pub bins: usize,
    pub range: (f64, f64),
    pub grid: Option<String>,
    pub id: usize,
}

impl Default for AddHisto1d {
    fn default() -> Self {
        Self {
            name: "1D Histogram".to_string(),
            bins: 512,
            range: (0.0, 4096.0),
            grid: None,
            id: 0,
        }
    }
}

impl AddHisto1d {
    pub fn new(id: usize) -> Self {
        Self {
            name: format!("Histogram {}", id),
            bins: 512,
            range: (0.0, 4096.0),
            grid: None,
            id,
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, grids: Vec<String>) {
        ui.text_edit_singleline(&mut self.name);

        ui.add(
            egui::DragValue::new(&mut self.bins)
                .speed(1.0)
                .range(1..=usize::MAX),
        );

        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut self.range.0)
                    .speed(1.0)
                    .prefix("(")
                    .suffix(","),
            );
            ui.add(
                egui::DragValue::new(&mut self.range.1)
                    .speed(1.0)
                    .suffix(")"),
            );
        });

        let mut grid_options = vec!["None".to_string()];
        grid_options.extend(grids.clone());

        if let Some(selected_grid) = &self.grid {
            if !grids.contains(selected_grid) {
                self.grid = None;
            }
        }

        egui::ComboBox::from_id_salt(format!("Add grid selector {}", self.id))
            .selected_text(self.grid.clone().unwrap_or_else(|| "None".to_string()))
            .show_ui(ui, |ui| {
                for grid in grid_options {
                    ui.selectable_value(
                        &mut self.grid,
                        if grid == "None" {
                            None
                        } else {
                            Some(grid.clone())
                        },
                        grid,
                    );
                }
            });
    }
}

#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub struct AddHisto2d {
    pub name: String,
    pub bins: (usize, usize),
    pub range: ((f64, f64), (f64, f64)),
    pub grid: Option<String>,
    pub id: usize,
}

impl Default for AddHisto2d {
    fn default() -> Self {
        Self {
            name: "2D Histogram".to_string(),
            bins: (512, 512),
            range: ((0.0, 4096.0), (0.0, 4096.0)),
            grid: None,
            id: 0,
        }
    }
}

impl AddHisto2d {
    pub fn new(id: usize) -> Self {
        Self {
            name: format!("Histogram {}", id),
            bins: (512, 512),
            range: ((0.0, 4096.0), (0.0, 4096.0)),
            grid: None,
            id,
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, grids: Vec<String>) {
        ui.text_edit_singleline(&mut self.name);

        ui.vertical(|ui| {
            ui.add(
                egui::DragValue::new(&mut self.bins.0)
                    .speed(1.0)
                    .range(1..=usize::MAX),
            );

            ui.add(
                egui::DragValue::new(&mut self.bins.1)
                    .speed(1.0)
                    .range(1..=usize::MAX),
            );
        });

        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut self.range.0 .0)
                        .speed(1.0)
                        .prefix("(")
                        .suffix(","),
                );
                ui.add(
                    egui::DragValue::new(&mut self.range.0 .1)
                        .speed(1.0)
                        .suffix(")"),
                );
            });

            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut self.range.1 .0)
                        .speed(1.0)
                        .prefix("(")
                        .suffix(","),
                );
                ui.add(
                    egui::DragValue::new(&mut self.range.1 .1)
                        .speed(1.0)
                        .suffix(")"),
                );
            });
        });

        let mut grid_options = vec!["None".to_string()];
        grid_options.extend(grids.clone());

        if let Some(selected_grid) = &self.grid {
            if !grids.contains(selected_grid) {
                self.grid = None;
            }
        }

        egui::ComboBox::from_id_salt(format!("Add Grid selector {}", self.id))
            .selected_text(self.grid.clone().unwrap_or_else(|| "None".to_string()))
            .show_ui(ui, |ui| {
                for grid in grid_options {
                    ui.selectable_value(
                        &mut self.grid,
                        if grid == "None" {
                            None
                        } else {
                            Some(grid.clone())
                        },
                        grid,
                    );
                }
            });
    }
}

#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub struct FillHisto1d {
    pub name: String,
    pub lazyframe: String,
    pub column: String,
    pub calculate: bool,
    pub id: usize,
}

impl Default for FillHisto1d {
    fn default() -> Self {
        Self {
            name: "Xavg".to_string(),
            lazyframe: "Raw".to_string(),
            column: "Xavg".to_string(),
            calculate: true,
            id: 0,
        }
    }
}

impl FillHisto1d {
    pub fn new(id: usize) -> Self {
        Self {
            name: format!("Histogram {}", id),
            lazyframe: "Raw".to_string(),
            column: "".to_string(),
            calculate: true,
            id,
        }
    }
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        lazyframe_info: LazyFrameInfo,
        histogram_names: Vec<String>,
    ) {
        //combobox for histogram name
        egui::ComboBox::from_id_salt(format!("Fill name selector {}", self.id))
            .selected_text(&self.name)
            .show_ui(ui, |ui| {
                for name in &histogram_names {
                    ui.selectable_value(&mut self.name, name.clone(), name.clone());
                }
            });

        egui::ComboBox::from_id_salt(format!("Fill lazyframe selector {}", self.id))
            .selected_text(&self.lazyframe)
            .show_ui(ui, |ui| {
                for lf in &lazyframe_info.lfs {
                    ui.selectable_value(&mut self.lazyframe, lf.clone(), lf.clone());
                }
            });

        ui.horizontal(|ui| {
            ui.label("X:");
            egui::ComboBox::from_id_salt(format!("Fill Column selector {}", self.id))
                .selected_text(&self.column)
                .show_ui(ui, |ui| {
                    for column in &lazyframe_info.columns {
                        ui.selectable_value(&mut self.column, column.clone(), column.clone());
                    }
                });
        });

        ui.checkbox(&mut self.calculate, "");
    }
}

#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub struct FillHisto2d {
    pub name: String,
    pub lazyframe: String,
    pub x_column: String,
    pub y_column: String,
    pub calculate: bool,
    pub id: usize,
}

impl Default for FillHisto2d {
    fn default() -> Self {
        Self {
            name: "PID".to_string(),
            lazyframe: "Raw".to_string(),
            x_column: "ScintLeftEnergy".to_string(),
            y_column: "AnodeBackEnergy".to_string(),
            calculate: true,
            id: 0,
        }
    }
}

impl FillHisto2d {
    pub fn new(id: usize) -> Self {
        Self {
            name: format!("Histogram {}", id),
            lazyframe: "Raw".to_string(),
            x_column: "".to_string(),
            y_column: "".to_string(),
            calculate: true,
            id,
        }
    }

    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        lazyframe_info: LazyFrameInfo,
        histogram_names: Vec<String>,
    ) {
        egui::ComboBox::from_id_salt(format!("Fill name selector {}", self.id))
            .selected_text(&self.name)
            .show_ui(ui, |ui| {
                for name in &histogram_names {
                    ui.selectable_value(&mut self.name, name.clone(), name.clone());
                }
            });

        egui::ComboBox::from_id_salt(format!("Fill lazyframe selector {}", self.id))
            .selected_text(&self.lazyframe)
            .show_ui(ui, |ui| {
                for lf in &lazyframe_info.lfs {
                    ui.selectable_value(&mut self.lazyframe, lf.clone(), lf.clone());
                }
            });

        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.label("X:");
                egui::ComboBox::from_id_salt(format!("Fill {}-X Column ComboBox", self.id))
                    .selected_text(&self.x_column)
                    .show_ui(ui, |ui| {
                        for column in &lazyframe_info.columns {
                            ui.selectable_value(&mut self.x_column, column.clone(), column.clone());
                        }
                    });
            });

            ui.horizontal(|ui| {
                ui.label("Y:");
                egui::ComboBox::from_id_salt(format!("Fill {}-Y Column ComboBox", self.id))
                    .selected_text(&self.y_column)
                    .show_ui(ui, |ui| {
                        for column in &lazyframe_info.columns {
                            ui.selectable_value(&mut self.y_column, column.clone(), column.clone());
                        }
                    });
            });
        });

        ui.checkbox(&mut self.calculate, "");
    }
}
