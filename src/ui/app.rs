// use super::pane::Pane;
// use super::tree::TreeBehavior;
use crate::util::processer::Processer;

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)] // if we add new fields, give them default values when deserializing old state
pub struct GNATApp {
    processer: Processer,
    left_side_panel_open: bool,
    right_side_panel_open: bool,
}

impl Default for GNATApp {
    fn default() -> Self {
        Self {
            processer: Processer::new(),
            left_side_panel_open: true,
            right_side_panel_open: true,
        }
    }
}

impl GNATApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        if let Some(storage) = cc.storage {
            return eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
        }

        Default::default()
    }
}

impl eframe::App for GNATApp {
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("nat_top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.label("Show: ");
                ui.checkbox(&mut self.left_side_panel_open, "Info Panel");
                ui.checkbox(&mut self.right_side_panel_open, "Histogram Script");
            });
        });

        egui::SidePanel::left("nat_left_panel").show_animated(
            ctx,
            self.left_side_panel_open,
            |ui| {
                ui.horizontal(|ui| {
                    ui.heading("gNAT");

                    egui::global_theme_preference_switch(ui);

                    if ui.button("Reset").clicked() {
                        self.processer.reset();
                    }
                });

                egui::ScrollArea::vertical()
                    .id_salt("LeftPanel")
                    .show(ui, |ui| {
                        ui.separator();

                        self.processer.ui(ui);
                    });
            },
        );

        egui::SidePanel::right("nat_right_panel").show_animated(
            ctx,
            self.right_side_panel_open,
            |ui| {
                self.processer.histogram_script_ui(ui);
            },
        );

        egui::CentralPanel::default().show(ctx, |ui| {
            self.processer.histogrammer.ui(ui);
        });
    }
}
