use super::histogram1d::Histogram;

fn consume_plot_key(ui: &egui::Ui, key: egui::Key) -> bool {
    ui.input_mut(|input| input.consume_key(egui::Modifiers::NONE, key))
}

impl Histogram {
    // Handles the interactive elements of the histogram
    pub fn keybinds(&mut self, ui: &mut egui::Ui) {
        self.plot_settings.markers.cursor_position = self.plot_settings.cursor_position;

        if let Some(cursor_position) = self.plot_settings.cursor_position {
            let cursor_x_raw = self.display_x_to_raw_x(cursor_position.x);
            let add_peak = consume_plot_key(ui, egui::Key::P);
            let add_background = consume_plot_key(ui, egui::Key::B);
            let add_cut = consume_plot_key(ui, egui::Key::C);
            let add_region = consume_plot_key(ui, egui::Key::R);
            let remove_closest = consume_plot_key(ui, egui::Key::Minus);
            let remove_all = consume_plot_key(ui, egui::Key::Delete);
            let fit_background = consume_plot_key(ui, egui::Key::G);
            let fit_gaussians = consume_plot_key(ui, egui::Key::F);
            let store_fit = consume_plot_key(ui, egui::Key::S);
            let toggle_stats = consume_plot_key(ui, egui::Key::I);
            let toggle_log_y = consume_plot_key(ui, egui::Key::L);
            let toggle_auto_y = consume_plot_key(ui, egui::Key::Y);
            let detect_peaks = consume_plot_key(ui, egui::Key::O);

            if add_peak {
                self.plot_settings.markers.add_peak_marker(cursor_x_raw);
            }

            if add_background {
                self.add_background_marker_at(cursor_x_raw);
            }

            if add_cut {
                self.new_cut();
            }

            if add_region {
                if self.plot_settings.markers.region_markers.len() >= 2 {
                    self.plot_settings.markers.clear_region_markers();
                }
                self.plot_settings.markers.add_region_marker(cursor_x_raw);
            }

            if remove_closest {
                let regions_before = self.plot_settings.markers.get_region_marker_positions();
                let peaks_before = self.plot_settings.markers.get_peak_marker_positions();
                self.plot_settings
                    .markers
                    .delete_closest_marker(cursor_x_raw);
                if regions_before != self.plot_settings.markers.get_region_marker_positions()
                    || peaks_before != self.plot_settings.markers.get_peak_marker_positions()
                {
                    self.fits.remove_temp_fits();
                    self.refresh_manual_peak_guesses();
                }
            }

            if remove_all {
                self.plot_settings.markers.clear_background_markers();
                self.plot_settings.markers.clear_peak_markers();
                self.plot_settings.markers.clear_region_markers();
                self.fits.remove_temp_fits();
            }

            if fit_background {
                self.request_fit(super::fit_worker::FitTask::Background);
            }

            if fit_gaussians {
                self.request_fit(super::fit_worker::FitTask::Peaks);
            }

            if store_fit && self.fits.temp_fit_is_storable() {
                self.fits.store_temp_fit();
                self.plot_settings.markers.clear_peak_markers();
                self.plot_settings.markers.preview_background.clear();
                self.plot_settings.markers.estimate_error = None;
            }

            if toggle_stats {
                self.plot_settings.stats_info = !self.plot_settings.stats_info;
            }

            if toggle_log_y {
                self.plot_settings.egui_settings.log_y = !self.plot_settings.egui_settings.log_y;
            }

            if toggle_auto_y {
                self.plot_settings.auto_fit_y_to_visible_range =
                    !self.plot_settings.auto_fit_y_to_visible_range;
            }

            if detect_peaks {
                self.find_peaks();
            }
        }
    }

    // create a ui function to show the keybinds in the context menu
    pub fn keybinds_ui(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
        .id_salt("keybinds_scroll")
        .max_height(300.0)
        .show(ui, |ui| {
            ui.heading("Keybinds");
            ui.separator();
            ui.label("Markers");
            ui.label("P: Add Marker");
            ui.label("B: Add Background Marker").on_hover_text("Add a one-bin window snapped to the hovered bin's edges; fitting uses its center. Dragged background edges snap on release: cross a bin center to include or exclude that bin. Valid changes update the background automatically without G.");
            ui.label("R: Add Region Marker");
            ui.label("-: Remove Marker Closest to Cursor");
            ui.label("Delete: Remove All Markers & Temp Fits");
            ui.label("Left click/Drag to Move Marker").on_hover_text("Markers can be dragged to new positions with the left clicking and dragingong when hovered over center point");
            ui.separator();
            ui.label("Cuts");
            ui.label("C: Add 1D Cut");
            ui.label("Left click/Drag cut lines or the region between them")
                .on_hover_text("The cut span can be moved by dragging between the vertical lines while keeping the width fixed.");
            ui.separator();
            ui.label("Fitting");
            ui.label("G: Fit Background").on_hover_text("Estimate the selected background in a worker and make it available for locking. Uses your background windows when present; otherwise estimates from the region.");
            ui.label("F: Fit Gaussians").on_hover_text("Fit the marked peaks and selected background in a worker, with automatic polishing and recovery. Background windows are optional. With windows, their background stays fixed; without windows, peaks and background are fitted jointly.");
            ui.label("S: Store Fit").on_hover_text("Store the current fit as a permanent fit which can be saved and loaded later");
            ui.separator();
            ui.label("Peak Finder");
            ui.label("O: Detect Peaks").on_hover_text("Populate editable peak markers with the original peak finder. This never starts a fit.");
            ui.separator();
            ui.label("Plot");
            ui.label("I: Toggle Stats");
            ui.label("L: Toggle Log Y");
            ui.label("Y: Toggle Auto Fit Y");

        });
    }
}

#[cfg(test)]
mod tests {
    use super::Histogram;

    fn press_b(histogram: &mut Histogram, display_x: f64) {
        histogram.plot_settings.cursor_position = Some(egui_plot::PlotPoint::new(display_x, 0.0));
        let input = egui::RawInput {
            events: vec![egui::Event::Key {
                key: egui::Key::B,
                physical_key: Some(egui::Key::B),
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        };
        let mut output = egui::Context::default().run_ui(input, |ui| histogram.keybinds(ui));
        output.textures_delta.clear();
    }

    #[test]
    fn b_snaps_negative_position_and_samples_and_highlights_only_its_bin() {
        let mut histogram = Histogram::new("negative bins", 4, (-103.0, -99.0));
        histogram.bins = vec![10, 20, 30, 900];
        press_b(&mut histogram, -100.75);

        assert_eq!(
            histogram
                .plot_settings
                .markers
                .get_background_marker_positions(),
            vec![(-101.0, -100.0)]
        );
        assert_eq!(
            histogram.plot_settings.markers.background_markers[0].average_x(),
            -100.5
        );
        let input = histogram.background_fit_input().expect("one bin selected");
        assert_eq!(input.data.x, vec![-100.5]);
        assert_eq!(input.data.y, vec![30.0]);

        histogram.update_background_pair_lines();
        assert_eq!(
            histogram.plot_settings.markers.background_markers[0]
                .histogram_line
                .points,
            vec![[-101.0, 30.0], [-100.0, 30.0]]
        );
    }

    #[test]
    fn b_uses_offset_grid_and_current_rebinned_width_without_moving_existing_windows() {
        let mut histogram = Histogram::new("offset bins", 8, (0.25, 4.25));
        press_b(&mut histogram, 1.12);
        assert_eq!(
            histogram
                .plot_settings
                .markers
                .get_background_marker_positions(),
            vec![(0.75, 1.25)]
        );
        assert_eq!(
            histogram.background_fit_input().expect("sample").data.x,
            vec![1.0]
        );

        histogram.plot_settings.rebin_factor = 2;
        histogram.rebin();
        press_b(&mut histogram, 1.12);
        assert_eq!(
            histogram
                .plot_settings
                .markers
                .get_background_marker_positions(),
            vec![(0.75, 1.25), (0.25, 1.25)]
        );
        assert_eq!(
            histogram.background_fit_input().expect("sample").data.x,
            vec![0.75]
        );
    }

    #[test]
    fn b_handles_bin_edges_and_does_not_add_windows_outside_the_histogram() {
        let mut histogram = Histogram::new("bin edges", 4, (0.0, 4.0));
        for x in [0.0, 1.0, 4.0, -1.0, 5.0] {
            press_b(&mut histogram, x);
        }
        assert_eq!(
            histogram
                .plot_settings
                .markers
                .get_background_marker_positions(),
            vec![(0.0, 1.0), (1.0, 2.0), (3.0, 4.0)]
        );
        assert_eq!(
            histogram.background_fit_input().expect("samples").data.x,
            vec![0.5, 1.5, 3.5]
        );
    }

    #[test]
    fn b_places_fractional_internal_edges_in_the_bin_to_the_right() {
        let mut histogram = Histogram::new("fractional edges", 10, (0.25, 1.25));
        let edges = histogram.get_bin_edges();
        press_b(&mut histogram, edges[1]);
        assert_eq!(
            histogram
                .plot_settings
                .markers
                .get_background_marker_positions(),
            vec![(edges[1], edges[2])]
        );
        assert_eq!(
            histogram.background_fit_input().expect("sample").data.x,
            vec![histogram.get_bin_centers()[1]]
        );
    }

    #[test]
    fn b_snaps_in_raw_bin_coordinates_on_calibrated_and_logarithmic_axes() {
        for log_x in [false, true] {
            let mut histogram = Histogram::new("calibrated bins", 8, (-104.0, -96.0));
            histogram.fits.calibration.b.value = 2.0;
            histogram.fits.calibration.c.value = 300.0;
            histogram.fits.settings.calibrated = true;
            histogram.plot_settings.egui_settings.log_x = log_x;
            let display_x = histogram.fits.calibration.calibrate(-100.75);
            press_b(
                &mut histogram,
                if log_x { display_x.log10() } else { display_x },
            );
            assert_eq!(
                histogram
                    .plot_settings
                    .markers
                    .get_background_marker_positions(),
                vec![(-101.0, -100.0)]
            );
            assert_eq!(
                histogram.background_fit_input().expect("sample").data.x,
                vec![-100.5]
            );
        }
    }
}
