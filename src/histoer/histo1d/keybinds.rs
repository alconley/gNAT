use super::histogram1d::Histogram;

impl Histogram {
    // Handles the interactive elements of the histogram
    pub fn keybinds(&mut self, ui: &mut egui::Ui) {
        self.plot_settings.markers.cursor_position = self.plot_settings.cursor_position;

        if let Some(cursor_position) = self.plot_settings.cursor_position {
            if ui.input(|i| i.key_pressed(egui::Key::P)) {
                self.plot_settings
                    .markers
                    .add_peak_marker(cursor_position.x);
            }

            if ui.input(|i| i.key_pressed(egui::Key::B)) {
                self.plot_settings
                    .markers
                    .add_background_marker(cursor_position.x);
            }

            if ui.input(|i| i.key_pressed(egui::Key::R)) {
                if self.plot_settings.markers.region_markers.len() >= 2 {
                    self.plot_settings.markers.clear_region_markers();
                }
                self.plot_settings
                    .markers
                    .add_region_marker(cursor_position.x);
            }

            if ui.input(|i| i.key_pressed(egui::Key::Minus)) {
                self.plot_settings.markers.delete_closest_marker();
            }

            if ui.input(|i| i.key_pressed(egui::Key::Delete)) {
                self.plot_settings.markers.clear_background_markers();
                self.plot_settings.markers.clear_peak_markers();
                self.plot_settings.markers.clear_region_markers();
            }

            if ui.input(|i| i.key_pressed(egui::Key::Minus) || i.key_pressed(egui::Key::Delete)) {
                self.fits.remove_temp_fits();
            }

            if ui.input(|i| i.key_pressed(egui::Key::G)) {
                self.fit_background();
            }

            if ui.input(|i| i.key_pressed(egui::Key::F)) {
                self.fit_gaussians();
            }

            if ui.input(|i| i.key_pressed(egui::Key::S)) {
                self.fits.store_temp_fit();
            }

            if ui.input(|i| i.key_pressed(egui::Key::I)) {
                self.plot_settings.stats_info = !self.plot_settings.stats_info;
            }

            if ui.input(|i| i.key_pressed(egui::Key::L)) {
                self.plot_settings.egui_settings.log_y = !self.plot_settings.egui_settings.log_y;
            }

            if ui.input(|i| i.key_pressed(egui::Key::O)) {
                self.find_peaks();
            }
        }
    }

    // create a ui function to show the keybinds in the context menu
    pub fn keybinds_ui(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Keybind Help", |ui| {
            egui::ScrollArea::vertical()
            .id_salt("keybinds_scroll")
            .max_height(300.0)
            .show(ui, |ui| {
                ui.heading("Keybinds");
                ui.separator();
                ui.label("Markers");
                ui.label("P: Add Marker");
                ui.label("B: Add Background Marker");
                ui.label("R: Add Region Marker");
                ui.label("-: Remove Marker Closest to Cursor");
                ui.label("Delete: Remove All Markers & Temp Fits");
                ui.label("Left click/Drag to Move Marker").on_hover_text("Markers can be dragged to new positions with the left clicking and dragingong when hovered over center point");
                ui.separator();
                ui.label("Fitting");
                ui.label("G: Fit Background").on_hover_text("Fit a linear background using the background markers");
                ui.label("F: Fit Gaussians").on_hover_text("Fit gaussians at the peak markers give some region with a linear background");
                ui.label("S: Store Fit").on_hover_text("Store the current fit as a permanent fit which can be saved and loaded later");
                ui.separator();
                ui.label("Plot");
                ui.label("I: Toggle Stats");
                ui.label("L: Toggle Log Y");
                ui.separator();
                ui.label("Peak Finder");
                ui.label("O: Detect Peaks").on_hover_text("Detect peaks in the spectrum using the peak finding parameters");

            });
        });
    }
}
