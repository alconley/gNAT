//! Background marker updates: snapshot on the UI thread, solve on a worker, install atomically.

use super::histogram1d::Histogram;
use super::utility::bin_center_range;
use crate::fitter::common::Data;
use crate::fitter::main_fitter::{BackgroundModel, FitResult, Fitter};
use std::hash::{Hash as _, Hasher as _};
use std::sync::{Mutex, mpsc};

#[derive(Debug, Default)]
pub(crate) struct LiveBackgroundState {
    pub(super) last_attempt: Option<u64>,
    pub(super) preview: Option<Fitter>,
    pub(super) status: Option<String>,
    pending: Option<PendingBackground>,
}

// Histogram clones must never share a receiver or consume each other's results.
impl Clone for LiveBackgroundState {
    fn clone(&self) -> Self {
        Self {
            last_attempt: if self.pending.is_some() {
                None
            } else {
                self.last_attempt
            },
            preview: self.preview.clone(),
            status: self.status.clone(),
            pending: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Target {
    Preview,
    Background(u64),
    Gaussian(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RequestKey {
    background: u64,
    target: Target,
    peaks: u64,
}

#[derive(Debug)]
struct PendingBackground {
    key: RequestKey,
    receiver: Mutex<mpsc::Receiver<Result<BackgroundUpdate, String>>>,
}

#[derive(Debug)]
struct BackgroundUpdate {
    background: Fitter,
    gaussian: Option<Fitter>,
}

pub(super) fn calculate_background(mut input: Fitter, range: (f64, f64)) -> Result<Fitter, String> {
    input.fit_background();
    if let Some(error) = input.last_fit_error.take() {
        return Err(error);
    }
    if input.background_result.is_none() {
        return Err(
            "Select a background model and a valid region or background windows.".to_owned(),
        );
    }
    input.set_background_display_range(range);
    Ok(input)
}

fn calculate_update(
    input: Fitter,
    mut gaussian: Option<Fitter>,
    range: (f64, f64),
) -> Result<BackgroundUpdate, String> {
    let background = calculate_background(input, range)?;
    if let Some(fit) = &mut gaussian {
        // Even a locked background must use the NEW marker fit, not the previous coefficients.
        fit.background_result
            .clone_from(&background.background_result);
        fit.native_background_result
            .clone_from(&background.native_background_result);
        fit.fit();
        if let Some(error) = fit.last_fit_error.take() {
            return Err(error);
        }
        if fit.fit_result.is_none() {
            return Err(
                "Temporary peak fit could not be rebuilt from the current markers.".to_owned(),
            );
        }
    }
    Ok(BackgroundUpdate {
        background,
        gaussian,
    })
}

impl Histogram {
    pub(super) fn background_display_range(&self) -> (f64, f64) {
        crate::fitter::main_fitter::background_display_range(
            &self.plot_settings.markers.get_region_marker_positions(),
            &self.plot_settings.markers.get_background_marker_positions(),
        )
        .unwrap_or(self.range)
    }

    pub(super) fn background_update_pending(&self) -> bool {
        self.live_background.pending.is_some()
            || (self
                .live_background
                .last_attempt
                .is_some_and(|signature| signature != self.live_background_signature())
                && !matches!(
                    self.fits.settings.background_model,
                    BackgroundModel::None | BackgroundModel::LegacyAuto
                ))
    }

    pub(super) fn live_background_signature(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.bin_width.to_bits().hash(&mut hasher);
        self.range.0.to_bits().hash(&mut hasher);
        self.range.1.to_bits().hash(&mut hasher);
        self.bins.hash(&mut hasher);
        // The region also determines Auto's objective, even with background windows.
        for value in self.plot_settings.markers.get_region_marker_positions() {
            value.to_bits().hash(&mut hasher);
        }
        format!("{:?}", self.fits.settings.background_model).hash(&mut hasher);
        format!("{:?}", self.fits.settings.objective).hash(&mut hasher);
        for (start, end) in self.plot_settings.markers.get_background_marker_positions() {
            start.to_bits().hash(&mut hasher);
            end.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    fn live_request_key(&self) -> RequestKey {
        let target = match &self.fits.temp_fit {
            None => Target::Preview,
            Some(fit) if matches!(fit.fit_result, Some(FitResult::Gaussian(_))) => {
                Target::Gaussian(self.fits.temp_fit_revision)
            }
            Some(_) => Target::Background(self.fits.temp_fit_revision),
        };
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        if matches!(target, Target::Gaussian(_)) {
            format!(
                "{:?}",
                self.plot_settings.markers.get_region_marker_positions()
            )
            .hash(&mut hasher);
            format!("{:?}", self.plot_settings.markers.get_peak_seeds()).hash(&mut hasher);
            format!("{:?}", self.plot_settings.markers.get_peak_bounds()).hash(&mut hasher);
            self.fits.settings.equal_stddev.hash(&mut hasher);
            self.fits.settings.free_position.hash(&mut hasher);
            self.fits.settings.lock_background.hash(&mut hasher);
            format!("{:?}", self.fits.calibration).hash(&mut hasher);
        }
        RequestKey {
            background: self.live_background_signature(),
            target,
            peaks: hasher.finish(),
        }
    }

    /// Both G and automatic fitting use the same region or unique window bins.
    pub(super) fn background_fit_input(&self) -> Result<Fitter, String> {
        if self.bins.is_empty() || !self.bin_width.is_finite() || self.bin_width <= 0.0 {
            return Err("The histogram has no valid bins for a background fit.".to_owned());
        }
        let windows = self.plot_settings.markers.get_background_marker_positions();
        let automatic = windows.is_empty();
        let windows = if automatic {
            let region = self.plot_settings.markers.get_region_marker_positions();
            if region.len() != 2 {
                return Err("Place two region markers to estimate the selected background, or add background windows.".to_owned());
            }
            vec![(region[0], region[1])]
        } else {
            windows
        };
        let centers = self.get_bin_centers();
        let mut selected = vec![false; self.bins.len()];
        for (start, end) in windows {
            if !start.is_finite() || !end.is_finite() {
                return Err("Background marker positions must be finite.".to_owned());
            }
            selected[bin_center_range(&centers, start, end)].fill(true);
        }
        let mut data = Data::default();
        for (index, selected) in selected.into_iter().enumerate() {
            if selected {
                data.x.push(centers[index]);
                data.y.push(self.bins[index] as f64);
            }
        }
        if data.x.is_empty() {
            return Err("Background windows contain no histogram bins.".to_owned());
        }
        let region = self.plot_settings.markers.get_region_marker_positions();
        let objective = if region.len() == 2 {
            self.resolved_objective(Some((region[0], region[1])))
        } else {
            self.fits.settings.objective.resolve(data.y.iter().copied())
        };
        let mut input = Fitter::new(data);
        input.background_model = self.fits.settings.background_model.clone();
        input.objective = objective;
        input.automatic_background = automatic;
        Ok(input)
    }

    pub fn refresh_live_background(&mut self, repaint_context: egui::Context) {
        if self.fit_worker.is_pending() {
            return;
        }
        let key = self.live_request_key();
        if matches!(
            self.fits.settings.background_model,
            BackgroundModel::None | BackgroundModel::LegacyAuto
        ) {
            self.live_background.preview = None;
            self.live_background.last_attempt = Some(key.background);
            self.live_background.status = None;
            return;
        }
        // Opening an existing saved temporary fit is not a marker edit.
        if self.live_background.last_attempt.is_none() && self.fits.temp_fit.is_some() {
            self.live_background.last_attempt = Some(key.background);
            return;
        }
        if self.plot_settings.markers.is_dragging()
            || self.live_background.pending.is_some()
            || self.live_background.last_attempt == Some(key.background)
        {
            return;
        }
        self.live_background.last_attempt = Some(key.background);
        let input = self.background_fit_input().and_then(|background| {
            let gaussian = if matches!(key.target, Target::Gaussian(_)) {
                Some(self.prepare_gaussian_fitter()?)
            } else {
                None
            };
            Ok((background, gaussian))
        });
        let (background, gaussian) = match input {
            Ok(input) => input,
            Err(error) => {
                self.live_background.status = Some(error);
                return;
            }
        };
        let range = self.background_display_range();
        let (sender, receiver) = mpsc::channel();
        self.live_background.pending = Some(PendingBackground {
            key,
            receiver: Mutex::new(receiver),
        });
        self.live_background.status = Some("Updating background…".to_owned());
        std::thread::spawn(move || {
            let result = calculate_update(background, gaussian, range);
            if sender.send(result).is_ok() {
                repaint_context.request_repaint();
            }
        });
    }

    pub fn apply_live_background(&mut self) {
        if self.plot_settings.markers.is_dragging() {
            return;
        }
        let Some(pending) = &self.live_background.pending else {
            return;
        };
        let received = pending.receiver.lock().map_or_else(
            |_| Err(mpsc::TryRecvError::Disconnected),
            |receiver| receiver.try_recv(),
        );
        let result = match received {
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => {
                Err("Background worker stopped before returning a fit.".to_owned())
            }
            Ok(result) => result,
        };
        let key = pending.key;
        self.live_background.pending = None;
        if key != self.live_request_key() {
            self.live_background.status =
                Some("Background inputs changed; discarded the earlier result.".to_owned());
            return;
        }
        match result {
            Ok(update) => self.install_live_background(update),
            Err(error) => self.live_background.status = Some(error),
        }
    }

    fn install_live_background(&mut self, update: BackgroundUpdate) {
        let mut background = update.background;
        let fitted_peaks = update.gaussian.is_some();
        if let Some(gaussian) = update.gaussian {
            let show_stats = self.fits.settings.show_fit_stats;
            self.install_gaussian_fit(gaussian);
            self.fits.settings.show_fit_stats = show_stats;
        } else if let Some(previous) = &self.fits.temp_fit {
            let points = background.background_line.points.clone();
            background.background_line = previous.background_line.clone();
            background.background_line.points = points;
            background.set_name(previous.name.clone());
            background.calibration = self.fits.calibration.clone();
            background.background_was_fit_manually = previous.background_was_fit_manually;
            background.background_coupling = previous.background_coupling;
            self.fits.replace_temp_fit(Some(background.clone()));
        }
        if let Some(fit) = &self.fits.temp_fit
            && let Some(result) = &fit.background_result
        {
            self.fits.settings.apply_background_fit(result);
        }
        // Preview-only estimates do not make a background manually fitted or change settings.
        background.background_was_fit_manually = false;
        self.live_background.preview = Some(background);
        self.plot_settings.markers.estimate_signature = if fitted_peaks {
            self.manual_estimate_signature()
        } else {
            0
        };
        self.live_background.last_attempt = Some(self.live_background_signature());
        self.live_background.status = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitter::fit_settings::HistogramObjective;
    use crate::histoer::histo1d::markers::GuessSource;
    use spectrix_fitting::{
        BackgroundCoupling, ManualPeakBounds, ManualPeakSeed, ParameterKind, evaluate_manual_peak,
    };
    use std::time::{Duration, Instant};

    fn histogram() -> Histogram {
        let mut histogram = Histogram::new("live-background-test", 100, (0.0, 100.0));
        histogram.bins = (0..100)
            .map(|index| {
                if index < 50 {
                    100 + index
                } else {
                    200 + 2 * index
                }
            })
            .collect();
        histogram.fits.settings.background_model = BackgroundModel::Linear(Default::default());
        histogram.fits.settings.objective = HistogramObjective::LeastSquares;
        histogram
            .plot_settings
            .markers
            .set_background_marker_positions(&[(2.0, 12.0)]);
        histogram
    }

    fn finish_worker(histogram: &mut Histogram) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while histogram.live_background.pending.is_some() {
            histogram.apply_live_background();
            assert!(Instant::now() < deadline, "worker did not finish");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn run_update(histogram: &mut Histogram) {
        histogram.refresh_live_background(egui::Context::default());
        assert!(
            histogram.live_background.pending.is_some(),
            "{:?}",
            histogram.live_background.status
        );
        finish_worker(histogram);
        assert!(histogram.live_background.status.is_none());
    }

    fn background_at(histogram: &Histogram, x: f64) -> f64 {
        histogram
            .fits
            .temp_fit
            .as_ref()
            .or(histogram.live_background.preview.as_ref())
            .and_then(|fit| fit.background_result.as_ref())
            .expect("background result")
            .evaluate(x)
    }

    fn render_frame(histogram: &mut Histogram, context: &egui::Context) {
        let mut output = context.run_ui(egui::RawInput::default(), |ui| histogram.render(ui));
        assert!(!output.shapes.is_empty());
        output.textures_delta.clear();
    }

    #[test]
    fn single_bin_windows_fit_at_centers_without_neighboring_counts() {
        let mut histogram = histogram();
        histogram.plot_settings.markers.clear_background_markers();
        // Isolated samples on y = 2*x + 10; high adjacent bins must not enter the fit.
        histogram.bins.fill(10_000);
        for (index, cursor) in [(2, 2.8), (10, 10.1), (20, 20.9)] {
            histogram.bins[index] = 2 * index as u64 + 11;
            histogram.add_background_marker_at(cursor);
        }
        let input = histogram.background_fit_input().expect("three samples");
        assert_eq!(input.data.x, vec![2.5, 10.5, 20.5]);
        assert_eq!(input.data.y, vec![15.0, 31.0, 51.0]);
        run_update(&mut histogram);
        assert!((background_at(&histogram, 50.0) - 110.0).abs() < 1e-6);
        assert!(histogram.fits.temp_fit.is_none());
        let live_result = background_at(&histogram, 50.0);
        histogram.fit_background();
        assert!((background_at(&histogram, 50.0) - live_result).abs() < 1e-9);
    }

    #[test]
    fn background_sampling_uses_centers_for_reversed_and_overlapping_windows() {
        let mut histogram = histogram();
        let windows = [(3.0, 1.0), (2.0, 4.0), (7.4, 7.6)];
        histogram
            .plot_settings
            .markers
            .set_background_marker_positions(&windows);
        let input = histogram.background_fit_input().expect("samples");
        assert_eq!(input.data.x, vec![1.5, 2.5, 3.5, 7.5]);
        assert_eq!(input.data.y, vec![101.0, 102.0, 103.0, 107.0]);
        histogram.update_background_pair_lines();
        assert_eq!(
            histogram.plot_settings.markers.background_markers[0]
                .histogram_line
                .points,
            vec![[1.0, 101.0], [2.0, 101.0], [2.0, 102.0], [3.0, 102.0]]
        );
        assert_eq!(
            histogram
                .plot_settings
                .markers
                .get_background_marker_positions(),
            windows
        );

        // A sub-bin window that contains no center does not sample either neighbor.
        histogram
            .plot_settings
            .markers
            .set_background_marker_positions(&[(7.8, 8.2)]);
        assert!(histogram.background_fit_input().is_err());
        histogram.update_background_pair_lines();
        assert!(
            histogram.plot_settings.markers.background_markers[0]
                .histogram_line
                .points
                .is_empty()
        );
    }

    #[test]
    fn preview_survives_real_render_frames_without_peak_or_region_markers() {
        let mut histogram = histogram();
        let settings = serde_json::to_value(&histogram.fits.settings).expect("settings");
        run_update(&mut histogram);
        let expected = background_at(&histogram, 50.0);
        let context = egui::Context::default();
        for _ in 0..3 {
            render_frame(&mut histogram, &context);
        }
        assert_eq!(background_at(&histogram, 50.0), expected);
        assert!(histogram.fits.temp_fit.is_none());
        assert_eq!(
            serde_json::to_value(&histogram.fits.settings).expect("settings"),
            settings
        );
        assert!(
            histogram.live_background.pending.is_none(),
            "no repeated fits for unchanged inputs"
        );
    }

    #[test]
    fn background_only_fit_updates_after_release_and_after_adding_a_window_like_g() {
        let mut histogram = histogram();
        histogram.fit_background();
        let before = background_at(&histogram, 50.0);
        histogram.plot_settings.markers.background_markers[0]
            .start
            .is_dragging = true;
        histogram.plot_settings.markers.background_markers[0]
            .start
            .x_value = 60.0;
        histogram.plot_settings.markers.background_markers[0]
            .end
            .x_value = 70.0;
        histogram.refresh_live_background(egui::Context::default());
        assert!(histogram.live_background.pending.is_none());
        histogram.plot_settings.markers.background_markers[0]
            .start
            .is_dragging = false;
        run_update(&mut histogram);
        assert!((background_at(&histogram, 50.0) - before).abs() > 50.0);
        assert!(
            histogram
                .fits
                .temp_fit
                .as_ref()
                .expect("background-only temp")
                .fit_result
                .is_none()
        );
        histogram
            .plot_settings
            .markers
            .add_background_pair(10.0, 10.0 + histogram.bin_width);
        let mut explicit = histogram.clone();
        explicit.fit_background();
        run_update(&mut histogram);
        for x in [0.0, 50.0, 100.0] {
            assert!((background_at(&histogram, x) - background_at(&explicit, x)).abs() < 1e-9);
        }
        let result = histogram
            .fits
            .temp_fit
            .as_ref()
            .expect("temporary background");
        assert!(result.native_background_result.is_some());
        assert!(
            histogram.fits.settings.background_parameters_match(
                result.background_result.as_ref().expect("coefficients")
            )
        );
        histogram.refresh_live_background(egui::Context::default());
        assert!(
            histogram.live_background.pending.is_none(),
            "applying coefficients must not restart fitting"
        );
    }

    #[test]
    fn latest_edit_wins_and_failed_estimates_keep_the_last_good_fit() {
        let mut histogram = histogram();
        run_update(&mut histogram);
        let original = background_at(&histogram, 50.0);
        histogram
            .plot_settings
            .markers
            .set_background_marker_positions(&[(60.0, 70.0)]);
        histogram.refresh_live_background(egui::Context::default());
        histogram
            .plot_settings
            .markers
            .set_background_marker_positions(&[(20.0, 30.0)]);
        finish_worker(&mut histogram);
        assert_eq!(background_at(&histogram, 50.0), original);
        run_update(&mut histogram);
        histogram
            .plot_settings
            .markers
            .set_background_marker_positions(&[(0.0, 1.0)]);
        histogram.refresh_live_background(egui::Context::default());
        finish_worker(&mut histogram);
        assert!(
            histogram
                .live_background
                .status
                .as_deref()
                .expect("error")
                .contains("degrees of freedom")
        );
        assert!((background_at(&histogram, 50.0) - original).abs() < 1e-8);
        histogram.refresh_live_background(egui::Context::default());
        assert!(
            histogram.live_background.pending.is_none(),
            "failed requests are not retried every frame"
        );
    }

    fn gaussian_histogram(locked: bool) -> Histogram {
        let mut histogram = histogram();
        let seed = ManualPeakSeed {
            center: 50.0,
            sigma: 4.0,
            amplitude: 2000.0,
        };
        histogram.bins = histogram
            .get_bin_centers()
            .into_iter()
            .map(|x| {
                (100.0
                    + x
                    + if x < 15.0 { 20.0 } else { 0.0 }
                    + evaluate_manual_peak(seed, x, 1.0).expect("Gaussian"))
                .round() as u64
            })
            .collect();
        histogram
            .plot_settings
            .markers
            .set_background_marker_positions(&[(2.0, 10.0), (90.0, 98.0)]);
        histogram.fit_background();
        histogram.fits.settings.lock_background = locked;
        histogram.plot_settings.markers.add_region_marker(0.0);
        histogram.plot_settings.markers.add_region_marker(100.0);
        histogram.plot_settings.markers.add_peak_seed(
            seed,
            GuessSource::Manual,
            1.0,
            Some(ManualPeakBounds {
                center: [45.0, 55.0],
                sigma: [1.0, 8.0],
                net_height: [1.0, 1000.0],
            }),
        );
        histogram.fit_gaussians();
        let Some(FitResult::Gaussian(gaussian)) = histogram
            .fits
            .temp_fit
            .as_mut()
            .and_then(|fit| fit.fit_result.as_mut())
        else {
            panic!("initial Gaussian fit")
        };
        gaussian.update_uuid_for_peak(0, 42).expect("UUID");
        gaussian
            .update_energy_for_peak(0, 1332.5, 0.1)
            .expect("energy");
        histogram
    }

    #[test]
    fn gaussian_refresh_uses_new_background_and_preserves_manual_bounds_and_assignments() {
        for locked in [false, true] {
            let mut histogram = gaussian_histogram(locked);
            let bounds = histogram.plot_settings.markers.get_peak_bounds();
            let old_background = background_at(&histogram, 50.0);
            histogram
                .plot_settings
                .markers
                .set_background_marker_positions(&[(20.0, 28.0), (72.0, 80.0)]);
            let expected = calculate_background(
                histogram.background_fit_input().expect("input"),
                histogram.range,
            )
            .expect("background");
            run_update(&mut histogram);
            assert_eq!(histogram.plot_settings.markers.get_peak_bounds(), bounds);
            assert_eq!(histogram.fits.settings.lock_background, locked);
            let fitted = histogram.fits.temp_fit.as_ref().expect("updated temp");
            let Some(FitResult::Gaussian(gaussian)) = &fitted.fit_result else {
                panic!("Gaussian result")
            };
            assert_eq!(gaussian.fit_result[0].uuid, 42);
            assert_eq!(gaussian.fit_result[0].energy.value, Some(1332.5));
            assert!((background_at(&histogram, 50.0) - old_background).abs() > 1.0);
            assert!(
                (background_at(&histogram, 50.0)
                    - expected
                        .background_result
                        .as_ref()
                        .expect("new coefficients")
                        .evaluate(50.0))
                .abs()
                    < 1e-9
            );
            assert_eq!(fitted.background_coupling, BackgroundCoupling::PrefitFrozen);
            let native = gaussian.native_result.as_ref().expect("native composite");
            assert!(
                native
                    .fit
                    .parameters
                    .iter()
                    .filter(|parameter| parameter.name.starts_with("bg_"))
                    .all(|parameter| parameter.kind == ParameterKind::Fixed)
            );
        }
    }

    #[test]
    fn storing_removing_or_replacing_a_temp_fit_rejects_the_pending_update() {
        let mut histogram = gaussian_histogram(false);
        histogram
            .plot_settings
            .markers
            .add_background_pair(30.0, 31.0);
        histogram.refresh_live_background(egui::Context::default());
        histogram.fits.store_temp_fit();
        let saved = serde_json::to_value(&histogram.fits.stored_fits).expect("saved fit");
        finish_worker(&mut histogram);
        assert!(histogram.fits.temp_fit.is_none());
        assert_eq!(
            serde_json::to_value(&histogram.fits.stored_fits).expect("saved fit"),
            saved
        );
        histogram.fit_background();
        histogram
            .plot_settings
            .markers
            .add_background_pair(40.0, 41.0);
        histogram.refresh_live_background(egui::Context::default());
        histogram.fit_background();
        let fitted = serde_json::to_value(&histogram.fits.temp_fit).expect("manual result");
        finish_worker(&mut histogram);
        assert_eq!(
            serde_json::to_value(&histogram.fits.temp_fit).expect("manual result"),
            fitted
        );
        histogram
            .plot_settings
            .markers
            .add_background_pair(42.0, 43.0);
        histogram.refresh_live_background(egui::Context::default());
        histogram.fits.remove_temp_fits();
        finish_worker(&mut histogram);
        assert!(histogram.fits.temp_fit.is_none());
    }

    #[test]
    fn clearing_windows_cancels_the_result_and_outside_windows_do_not_select_all_bins() {
        let mut histogram = histogram();
        histogram.refresh_live_background(egui::Context::default());
        histogram.plot_settings.markers.clear_background_markers();
        histogram.refresh_live_background(egui::Context::default());
        finish_worker(&mut histogram);
        assert!(histogram.live_background.preview.is_none());
        histogram
            .plot_settings
            .markers
            .set_background_marker_positions(&[(200.0, 210.0)]);
        histogram.refresh_live_background(egui::Context::default());
        assert!(
            histogram
                .live_background
                .status
                .as_deref()
                .expect("error")
                .contains("no histogram bins")
        );
        assert!(histogram.live_background.pending.is_none());
    }

    #[test]
    fn worker_wakes_the_ui_and_histogram_clones_do_not_share_pending_results() {
        let mut histogram = histogram();
        let context = egui::Context::default();
        let (sender, repaint) = mpsc::channel();
        context.set_request_repaint_callback(move |_| {
            if sender.send(()).is_err() {
                // The test has already observed the completion repaint.
            }
        });
        histogram.refresh_live_background(context);
        let cloned = histogram.clone();
        assert!(cloned.live_background.pending.is_none());
        repaint
            .recv_timeout(Duration::from_secs(5))
            .expect("worker completion must wake the UI");
        finish_worker(&mut histogram);
    }
}
