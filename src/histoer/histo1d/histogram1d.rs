use super::live_background::LiveBackgroundState;
use super::plot_settings::PlotSettings;
use crate::defaults::{Histogram1DDefaults, apply_plot_defaults};
use crate::egui_plot_stuff::egui_line::EguiLine;
use crate::fitter::common::Calibration;
use crate::fitter::fit_handler::Fits;

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Histogram {
    pub name: String,
    pub bins: Vec<u64>,
    pub range: (f64, f64),
    pub overflow: u64,
    pub underflow: u64,
    pub bin_width: f64,
    pub line: EguiLine,
    pub plot_settings: PlotSettings,
    pub fits: Fits,
    pub original_bins: Vec<u64>,
    #[serde(default)]
    pub generation_defaults: Histogram1DDefaults,
    #[serde(default = "default_true")]
    pub follow_theme_colors: bool,
    #[serde(skip)]
    pub(crate) live_background: LiveBackgroundState,
    #[serde(skip)]
    pub(crate) fit_worker: super::fit_worker::FitWorkState,
}

const fn default_true() -> bool {
    true
}

impl Histogram {
    // Create a new Histogram with specified min, max, and number of bins
    pub fn new(name: &str, number_of_bins: usize, range: (f64, f64)) -> Self {
        Self::new_with_defaults(name, number_of_bins, range, &Histogram1DDefaults::default())
    }

    pub fn new_with_defaults(
        name: &str,
        number_of_bins: usize,
        range: (f64, f64),
        defaults: &Histogram1DDefaults,
    ) -> Self {
        let mut histogram = Self {
            name: name.to_owned(),
            bins: vec![0; number_of_bins],
            range,
            overflow: 0,
            underflow: 0,
            bin_width: (range.1 - range.0) / number_of_bins as f64,
            line: EguiLine {
                name: name.to_owned(),
                ..Default::default()
            },
            plot_settings: PlotSettings::default(),
            fits: Fits::new(),
            original_bins: vec![0; number_of_bins],
            generation_defaults: defaults.clone(),
            follow_theme_colors: true,
            live_background: LiveBackgroundState::default(),
            fit_worker: super::fit_worker::FitWorkState::default(),
        };
        histogram.apply_generation_defaults(defaults);
        histogram
    }

    pub fn apply_generation_defaults(&mut self, defaults: &Histogram1DDefaults) {
        self.generation_defaults = defaults.clone();
        defaults.line.apply_to(&mut self.line);
        self.follow_theme_colors = true;
        apply_plot_defaults(&defaults.plot, &mut self.plot_settings.egui_settings);
        self.plot_settings.stats_info = defaults.show_statistics;
        self.plot_settings.auto_fit_y_to_visible_range = defaults.auto_fit_y_to_visible_range;
        self.plot_settings.auto_fit_y_max_multiplier_linear =
            defaults.auto_fit_y_multiplier_linear.max(1.0);
        self.plot_settings.auto_fit_y_max_multiplier_log =
            defaults.auto_fit_y_multiplier_log.max(1.0);
        self.plot_settings.markers.apply_defaults(&defaults.markers);
        for (index, cut) in self.plot_settings.cuts.iter_mut().enumerate() {
            let color = defaults.cuts.palette[index % defaults.cuts.palette.len()];
            cut.apply_defaults(&defaults.cuts, color);
        }
        self.fits
            .apply_generation_defaults(&defaults.fit, &defaults.fit_palette);
    }

    pub fn reset(&mut self) {
        self.bins = vec![0; self.original_bins.len()];
        self.original_bins = vec![0; self.original_bins.len()];
        self.plot_settings.rebin_factor = 1;
        self.bin_width = (self.range.1 - self.range.0) / self.bins.len() as f64;
        self.overflow = 0;
        self.underflow = 0;
    }

    pub fn update_line_points(&mut self) {
        self.line.set_points(
            self.bins
                .iter()
                .enumerate()
                .flat_map(|(index, &count)| {
                    let start = self.range.0 + index as f64 * self.bin_width;
                    let end = start + self.bin_width;
                    let y_value = count as f64;
                    [[start, y_value], [end, y_value]]
                })
                .collect(),
        );
    }

    pub(crate) fn display_calibration(&self) -> Option<&Calibration> {
        self.fits
            .settings
            .calibrated
            .then_some(&self.fits.calibration)
            .filter(|calibration| calibration.is_display_safe_on(self.range))
    }

    fn calibration_warning_message(&self) -> Option<&'static str> {
        if !self.fits.settings.calibrated {
            None
        } else if !self.fits.calibration.coefficients_are_finite() {
            Some("Calibration contains invalid values; using the raw X axis for this histogram.")
        } else if !self.fits.calibration.is_display_safe_on(self.range) {
            Some("Calibration is not safely invertible over this histogram; using the raw X axis.")
        } else {
            None
        }
    }

    fn display_x_to_raw_x_with_fallback(
        &self,
        display_x: f64,
        hint_raw: f64,
        fallback_raw: f64,
    ) -> f64 {
        let linear_display_x = if self.plot_settings.egui_settings.log_x {
            10_f64.powf(display_x)
        } else {
            display_x
        };

        if !linear_display_x.is_finite() {
            return fallback_raw;
        }

        if let Some(calibration) = self.display_calibration() {
            calibration
                .invert_in_range_with_hint(linear_display_x, self.range, Some(hint_raw))
                .unwrap_or(fallback_raw)
        } else {
            linear_display_x
        }
    }

    fn current_raw_center_hint(&self) -> f64 {
        self.plot_settings
            .current_plot_bounds
            .map(|(raw_min, raw_max)| (raw_min + raw_max) * 0.5)
            .unwrap_or((self.range.0 + self.range.1) * 0.5)
            .clamp(self.range.0, self.range.1)
    }

    pub(crate) fn display_x_to_raw_x(&self, display_x: f64) -> f64 {
        let hint_raw = self.current_raw_center_hint();
        self.display_x_to_raw_x_with_fallback(display_x, hint_raw, hint_raw)
    }

    pub(crate) fn display_x_bounds_to_raw_bounds(&self, x_min: f64, x_max: f64) -> (f64, f64) {
        let (hint_min, hint_max) = self.plot_settings.current_plot_bounds.unwrap_or(self.range);
        let raw_x_min = self.display_x_to_raw_x_with_fallback(
            x_min,
            hint_min.clamp(self.range.0, self.range.1),
            self.range.0,
        );
        let raw_x_max = self.display_x_to_raw_x_with_fallback(
            x_max,
            hint_max.clamp(self.range.0, self.range.1),
            self.range.1,
        );

        if raw_x_min <= raw_x_max {
            (raw_x_min, raw_x_max)
        } else {
            (raw_x_max, raw_x_min)
        }
    }

    fn current_raw_x_bounds(&self, plot_ui: &egui_plot::PlotUi<'_>) -> (f64, f64) {
        let plot_bounds = plot_ui.plot_bounds();
        self.display_x_bounds_to_raw_bounds(plot_bounds.min()[0], plot_bounds.max()[0])
    }

    pub fn draw(&mut self, plot_ui: &mut egui_plot::PlotUi<'_>) {
        // update the histogram and fit lines with the log setting and draw
        let log_y = self.plot_settings.egui_settings.log_y;
        let log_x = self.plot_settings.egui_settings.log_x;

        self.line.log_y = log_y;
        self.line.log_x = log_x;
        let calibration = self.display_calibration().cloned();
        let calibration_ref = calibration.as_ref();

        self.line.draw(plot_ui, calibration_ref);
        if self.fits.temp_fit.is_none()
            && self.fits.settings.show_background
            && let Some(preview) = &mut self.live_background.preview
        {
            self.fits.style_temporary_fit(preview);
            preview.background_line.name = "Live Background Estimate".to_owned();
            preview.set_log(log_y, log_x);
            preview.background_line.draw(plot_ui, calibration_ref);
        }
        self.plot_settings.markers.draw_all_markers(
            plot_ui,
            calibration_ref,
            log_x,
            log_y,
            self.bin_width,
        );

        self.fits.set_log(log_y, log_x);
        self.fits.draw(
            plot_ui,
            &self.bins,
            self.range,
            self.bin_width,
            calibration_ref,
        );
        self.show_stats(plot_ui);

        self.update_background_pair_lines();
        for bg_pair in &mut self.plot_settings.markers.background_markers {
            bg_pair.histogram_line.log_x = log_x;
            bg_pair.histogram_line.log_y = log_y;
        }

        if plot_ui.response().hovered() {
            self.plot_settings.cursor_position = plot_ui.pointer_coordinate();
        } else {
            self.plot_settings.cursor_position = None;
        }

        self.plot_settings.draw(plot_ui, calibration_ref);

        self.custom_plot_manipulation_update(plot_ui);
        self.plot_settings.current_plot_bounds = Some(self.current_raw_x_bounds(plot_ui));

        // self.plot_settings.egui_settings.y_label = format!("Counts/{:.}", self.bin_width);
    }

    pub fn draw_other_histograms(
        &mut self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        histograms: &[Self],
    ) {
        for histogram in histograms {
            let mut hist = histogram.clone();
            hist.draw(plot_ui);
        }
    }

    pub fn render(&mut self, ui: &mut egui::Ui) {
        // Establish a loaded fit's baseline before the first keyboard or marker edit.
        if self.live_background.last_attempt.is_none() && self.fits.temp_fit.is_some() {
            self.live_background.last_attempt = Some(self.live_background_signature());
        }
        // Process plot shortcuts before any fit-panel DragValue/TextEdit can see the same key
        // event. On Windows, letting a still-focused numeric editor reject a letter first can
        // play the system notification sound even if the plot consumes that key later.
        if self.plot_settings.cursor_position.is_some() {
            self.keybinds(ui);
        }

        if self.follow_theme_colors {
            let color = if ui.visuals().dark_mode {
                self.generation_defaults.dark_theme_color
            } else {
                self.generation_defaults.light_theme_color
            };
            self.line.set_color(color);
        }

        self.apply_fit_worker();
        self.apply_live_background();
        self.refresh_live_background(ui.ctx().clone());
        self.update_line_points();
        self.refresh_manual_peak_guesses();

        if self.fits.ui(
            ui,
            &self.name,
            self.range,
            &mut self.plot_settings.markers,
            self.bin_width,
        ) {
            self.invalidate_manual_gaussian_preview();
            self.refresh_manual_peak_guesses();
        }
        if let Some(status) = &self.live_background.status {
            ui.small(status).on_hover_text(
                "The selected background is estimated from the region when there are no windows. Supplied windows determine a background that stays fixed during peak fitting. Temporary fits update in the worker; stored fits change only when explicitly refitted.",
            );
        }
        self.apply_refit_all_request();
        self.apply_modify_fit_request();
        self.refresh_fit_worker(ui.ctx().clone());
        if let Some(status) = &self.fit_worker.status {
            ui.small(status).on_hover_text("Fits run in a worker and keep the lowest objective found within one shared evaluation budget. Input changes discard an outdated result; see the fit report for convergence and uncertainty details.");
        }

        if let Some(message) = self.calibration_warning_message() {
            ui.colored_label(egui::Color32::from_rgb(200, 120, 0), message);
        }

        let width = ui.available_width();
        let plot_id = egui::Id::new(("histogram1d-plot", std::ptr::from_ref(self) as usize));
        let mut plot = egui_plot::Plot::new(self.name.clone())
            .id(plot_id)
            .width(width);
        let mut effective_plot_settings = self.plot_settings.egui_settings.clone();
        effective_plot_settings.allow_drag &= !self.plot_settings.interactions_dragging;
        effective_plot_settings.allow_double_click_reset &= !self.plot_settings.cuts_clicking;
        plot = effective_plot_settings.apply_to_plot(plot);

        let (scroll, _pointer_down, _modifiers) = ui.input(|i| {
            let scroll = i.events.iter().find_map(|e| match e {
                egui::Event::MouseWheel { delta, .. } => Some(*delta),
                _ => None,
            });
            (scroll, i.pointer.primary_down(), i.modifiers)
        });

        let plot_response = plot.show(ui, |plot_ui| {
            self.draw(plot_ui);

            if self.plot_settings.progress.is_some()
                && !self.plot_settings.auto_fit_y_to_visible_range
            {
                let y_max = self.bins.iter().max().copied().unwrap_or(0) as f64;
                let mut plot_bounds = plot_ui.plot_bounds();
                plot_bounds.extend_with_y(y_max * 1.1);
                plot_ui.set_plot_bounds(plot_bounds);
            }

            if self.plot_settings.egui_settings.reset_axis {
                plot_ui.auto_bounds();
                self.plot_settings.egui_settings.reset_axis = false;
            }

            if self.plot_settings.cursor_position.is_some()
                && let Some(delta_pos) = scroll
            {
                let zoom_factor = if delta_pos.y > 0.0 || delta_pos.x > 0.0 {
                    1.1
                } else {
                    0.9
                };
                plot_ui.zoom_bounds_around_hovered(egui::Vec2::new(zoom_factor, 1.0));
            }
        });

        plot_response.response.context_menu(|ui| {
            self.context_menu(ui);
        });

        let calibration = self.display_calibration().cloned();
        let calibration_ref = calibration.as_ref();
        let background_markers_before =
            self.plot_settings.markers.get_background_marker_positions();
        let regions_before = self.plot_settings.markers.get_region_marker_positions();
        let peak_seeds_before = self.plot_settings.markers.get_peak_seeds();
        let peak_bounds_before = self.plot_settings.markers.get_peak_bounds();

        let markers_changed = self.plot_settings.interactive_response(
            &plot_response,
            calibration_ref,
            self.range,
            self.plot_settings.egui_settings.log_x,
            self.plot_settings.egui_settings.log_y,
            self.bin_width,
            self.fits.settings.equal_stddev,
            self.fits.settings.auto_estimate_moved_peak,
        );
        if markers_changed {
            let background_only_change = background_markers_before
                != self.plot_settings.markers.get_background_marker_positions()
                && regions_before == self.plot_settings.markers.get_region_marker_positions()
                && peak_seeds_before == self.plot_settings.markers.get_peak_seeds()
                && peak_bounds_before == self.plot_settings.markers.get_peak_bounds();
            if !background_only_change {
                self.invalidate_manual_gaussian_preview();
                self.refresh_manual_peak_guesses();
            }
            // Region and peak edits immediately invalidate a temporary peak fit. Background-only
            // edits retain it until the worker has produced a fresh background and can rebuild
            // the composite result from the current marker inputs.
        }
        // This is a no-op until the marker/model signature changes. During a drag the request is
        // deferred; the first frame after release captures the completed marker positions.
        self.refresh_live_background(ui.ctx().clone());

        if plot_response.response.hovered() {
            // Keep keyboard focus on the plot for the next frame's early shortcut pass.
            plot_response.response.request_focus();
        }
    }
}
