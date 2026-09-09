//! Explicit fit requests run off the UI thread and install only against their input snapshot.

use std::hash::{Hash as _, Hasher as _};
use std::sync::{Mutex, mpsc};

use super::histogram1d::Histogram;
use crate::fitter::main_fitter::Fitter;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FitTask {
    Peaks,
    Background,
    RefitAll,
}

#[derive(Debug, Default)]
pub(crate) struct FitWorkState {
    request: Option<FitTask>,
    pending: Option<PendingFit>,
    pub(super) status: Option<String>,
}

impl Clone for FitWorkState {
    fn clone(&self) -> Self {
        // A cloned histogram owns neither the original receiver nor its queued work.
        Self::default()
    }
}

impl FitWorkState {
    pub(super) fn is_pending(&self) -> bool {
        self.pending.is_some()
    }
}

#[derive(Debug)]
struct PendingFit {
    task: FitTask,
    signature: u64,
    receiver: Mutex<mpsc::Receiver<Result<FitUpdate, String>>>,
}

enum FitInput {
    Peaks(Box<Fitter>),
    Background(Box<Fitter>, (f64, f64)),
    RefitAll(Box<Histogram>),
}

#[derive(Debug)]
enum FitUpdate {
    Peaks(Box<Fitter>),
    Background(Box<Fitter>),
    RefitAll(Vec<Fitter>),
}

impl FitInput {
    fn calculate(self) -> Result<FitUpdate, String> {
        match self {
            Self::Peaks(mut fit) => {
                fit.fit();
                if let Some(error) = fit.last_fit_error.take() {
                    return Err(error);
                }
                if fit.fit_result.is_none() {
                    return Err("The peak fit did not produce a result.".to_owned());
                }
                Ok(FitUpdate::Peaks(fit))
            }
            Self::Background(fit, range) => {
                super::live_background::calculate_background(*fit, range)
                    .map(|fit| FitUpdate::Background(Box::new(fit)))
            }
            Self::RefitAll(mut histogram) => {
                let count = histogram.fits.stored_fits.len();
                for _ in 0..count {
                    if !matches!(
                        histogram.fits.stored_fits[0].fit_result,
                        Some(crate::fitter::main_fitter::FitResult::Gaussian(_))
                    ) {
                        // Background-only records have no peak-region metadata.
                        // Preserve them instead of reusing an unrelated peak snapshot.
                        histogram.fits.stored_fits.rotate_left(1);
                        continue;
                    }
                    histogram.fits.pending_modify_fit = Some(0);
                    histogram.apply_modify_fit_request();
                    histogram.refresh_manual_peak_guesses();
                    let mut fit = histogram.prepare_gaussian_fitter()?;
                    fit.fit();
                    if let Some(error) = fit.last_fit_error.take() {
                        return Err(error);
                    }
                    if fit.fit_result.is_none() {
                        return Err("A stored fit did not produce a result.".to_owned());
                    }
                    histogram.install_gaussian_fit(fit);
                    histogram.fits.store_temp_fit();
                }
                Ok(FitUpdate::RefitAll(histogram.fits.stored_fits))
            }
        }
    }
}

impl Histogram {
    pub(super) fn request_fit(&mut self, task: FitTask) {
        self.fit_worker.request = Some(task);
        self.fit_worker.status = Some("Fit queued…".to_owned());
    }

    fn fit_work_signature(&self, task: FitTask) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.live_background_signature().hash(&mut hasher);
        self.fits.temp_fit_revision.hash(&mut hasher);
        format!(
            "{:?}",
            self.plot_settings.markers.get_region_marker_positions()
        )
        .hash(&mut hasher);
        format!("{:?}", self.plot_settings.markers.get_peak_seeds()).hash(&mut hasher);
        format!("{:?}", self.plot_settings.markers.get_peak_bounds()).hash(&mut hasher);
        format!("{:?}", self.fits.calibration).hash(&mut hasher);
        self.fits.settings.equal_stddev.hash(&mut hasher);
        self.fits.settings.free_position.hash(&mut hasher);
        self.fits.settings.lock_background.hash(&mut hasher);
        self.fits.settings.calibrated.hash(&mut hasher);
        if task == FitTask::RefitAll {
            // Only evaluated at dispatch/completion; it protects stored user edits
            // without serializing the collection on every in-flight frame.
            format!("{:?}", self.fits.stored_fits).hash(&mut hasher);
        }
        hasher.finish()
    }

    pub(super) fn refresh_fit_worker(&mut self, context: egui::Context) {
        if self.fit_worker.is_pending() {
            context.request_repaint();
            return;
        }
        let Some(task) = self.fit_worker.request else {
            return;
        };
        if self.background_update_pending() || self.plot_settings.markers.is_dragging() {
            self.fit_worker.status = Some("Waiting for the current marker update…".to_owned());
            return;
        }
        self.refresh_manual_peak_guesses();
        self.fit_worker.request = None;
        let input = match task {
            FitTask::Peaks => self
                .prepare_gaussian_fitter()
                .map(|fit| FitInput::Peaks(Box::new(fit))),
            FitTask::Background => self
                .background_fit_input()
                .map(|fit| FitInput::Background(Box::new(fit), self.background_display_range())),
            FitTask::RefitAll => Ok(FitInput::RefitAll(Box::new(self.clone()))),
        };
        let input = match input {
            Ok(input) => input,
            Err(error) => {
                self.fit_worker.status = Some(error);
                return;
            }
        };
        let (sender, receiver) = mpsc::channel();
        self.fit_worker.pending = Some(PendingFit {
            task,
            signature: self.fit_work_signature(task),
            receiver: Mutex::new(receiver),
        });
        self.fit_worker.status = Some(match task {
            FitTask::Peaks => "Fitting peaks, polishing, and checking covariance…".to_owned(),
            FitTask::Background => "Estimating background…".to_owned(),
            FitTask::RefitAll => format!("Refitting {} stored fits…", self.fits.stored_fits.len()),
        });
        std::thread::spawn(move || {
            let result = input.calculate();
            if sender.send(result).is_ok() {
                context.request_repaint();
            }
        });
    }

    pub(super) fn apply_fit_worker(&mut self) {
        if self.plot_settings.markers.is_dragging() {
            return;
        }
        let Some(pending) = &self.fit_worker.pending else {
            return;
        };
        let received = pending.receiver.lock().map_or_else(
            |_| Err(mpsc::TryRecvError::Disconnected),
            |receiver| receiver.try_recv(),
        );
        let result = match received {
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => {
                Err("Fit worker stopped before returning a result.".to_owned())
            }
            Ok(result) => result,
        };
        let task = pending.task;
        let signature = pending.signature;
        self.fit_worker.pending = None;
        if signature != self.fit_work_signature(task) {
            self.fit_worker.status = Some("Fit inputs changed; discarded the earlier result. Press F to fit the current inputs.".to_owned());
            return;
        }
        match result {
            Ok(FitUpdate::Peaks(fit)) => {
                let status = match &fit.fit_result {
                    Some(crate::fitter::main_fitter::FitResult::Gaussian(gaussian))
                        if gaussian.fit_warning.is_some() =>
                    {
                        "Fit returned with warnings. Review fit quality and parameter limits in the fit report."
                    }
                    _ => "Fit complete. Convergence and covariance details are in the fit report.",
                }.to_owned();
                self.install_gaussian_fit(*fit);
                self.live_background.last_attempt = Some(self.live_background_signature());
                self.fit_worker.status = Some(status);
            }
            Ok(FitUpdate::Background(fit)) => {
                self.install_background_fit(*fit);
                self.fit_worker.status = None;
            }
            Ok(FitUpdate::RefitAll(fits)) => {
                self.fits.stored_fits = fits;
                self.fit_worker.status = Some("Stored fits updated.".to_owned());
            }
            Err(error) => self.fit_worker.status = Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FitTask, Histogram};
    use crate::fitter::fit_settings::HistogramObjective;
    use crate::fitter::main_fitter::{BackgroundModel, FitResult};
    use crate::histoer::histo1d::markers::GuessSource;
    use spectrix_fitting::{ManualPeakBounds, ManualPeakSeed, evaluate_manual_peak};
    use std::time::{Duration, Instant};

    fn histogram(background: bool) -> Histogram {
        let mut histogram = Histogram::new("worker regression", 160, (1.0, 9.0));
        let seed = ManualPeakSeed {
            center: 5.0,
            sigma: 0.6,
            amplitude: 100.0,
        };
        histogram.bins = histogram
            .get_bin_centers()
            .iter()
            .map(|x| {
                let level = if background { 2.0 + 0.3 * x } else { 0.0 };
                (level + evaluate_manual_peak(seed, *x, 0.05).expect("Gaussian")).round() as u64
            })
            .collect();
        histogram.fits.settings.objective = HistogramObjective::LeastSquares;
        histogram.fits.settings.background_model = if background {
            BackgroundModel::Linear(Default::default())
        } else {
            BackgroundModel::None
        };
        histogram.plot_settings.markers.add_region_marker(2.0);
        histogram.plot_settings.markers.add_region_marker(8.0);
        histogram.plot_settings.markers.add_peak_seed(
            ManualPeakSeed { sigma: 0.4, ..seed },
            GuessSource::Manual,
            0.05,
            Some(ManualPeakBounds {
                center: [4.0, 6.0],
                sigma: [0.1, 2.0],
                net_height: [1.0, 500.0],
            }),
        );
        histogram
    }

    fn finish(histogram: &mut Histogram) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            histogram.apply_fit_worker();
            histogram.apply_live_background();
            histogram.refresh_live_background(egui::Context::default());
            histogram.refresh_fit_worker(egui::Context::default());
            if histogram.fit_worker.pending.is_none() && histogram.fit_worker.request.is_none() {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "{:?}",
                histogram.fit_worker.status
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn explicit_fit_installs_atomically_and_preserves_submitted_guesses() {
        let mut histogram = histogram(false);
        let original = histogram.plot_settings.markers.get_peak_seeds();
        let bounds = histogram.plot_settings.markers.get_peak_bounds();
        histogram.request_fit(FitTask::Peaks);
        histogram.refresh_fit_worker(egui::Context::default());
        assert!(histogram.fit_worker.is_pending());
        assert!(histogram.fits.temp_fit.is_none());
        let clone = histogram.clone();
        assert!(!clone.fit_worker.is_pending());
        finish(&mut histogram);
        let Some(FitResult::Gaussian(fit)) = &histogram
            .fits
            .temp_fit
            .as_ref()
            .expect("installed fit")
            .fit_result
        else {
            panic!("Gaussian result");
        };
        assert_eq!(
            fit.fit_metadata
                .as_ref()
                .expect("metadata")
                .submitted_peak_seeds,
            original
        );
        assert_eq!(
            fit.fit_metadata
                .as_ref()
                .expect("metadata")
                .submitted_peak_bounds,
            bounds
        );
        assert_eq!(histogram.plot_settings.markers.get_peak_bounds(), bounds);
        assert!(
            fit.native_result
                .as_ref()
                .expect("native result")
                .fit
                .covariance
                .is_some()
        );
    }

    #[test]
    fn changed_inputs_discard_in_flight_results() {
        let mut histogram = histogram(false);
        histogram.request_fit(FitTask::Peaks);
        histogram.refresh_fit_worker(egui::Context::default());
        histogram.bins[60] += 10;
        finish(&mut histogram);
        assert!(histogram.fits.temp_fit.is_none());
        assert!(
            histogram
                .fit_worker
                .status
                .as_deref()
                .is_some_and(|status| status.contains("discarded"))
        );
        histogram.request_fit(FitTask::Peaks);
        finish(&mut histogram);
        assert!(histogram.fits.temp_fit.is_some());
    }

    #[test]
    fn fitted_handles_do_not_reestimate_submitted_bounds_on_the_next_frame() {
        let mut histogram = histogram(false);
        histogram.plot_settings.markers.peak_markers[0].bounds_source = GuessSource::Estimated;
        histogram.plot_settings.markers.peak_markers[0].width_source = GuessSource::Estimated;
        histogram.plot_settings.markers.peak_markers[0].amplitude_source = GuessSource::Estimated;
        histogram.refresh_manual_peak_guesses();
        let bounds = histogram.plot_settings.markers.get_peak_bounds();
        histogram.request_fit(FitTask::Peaks);
        finish(&mut histogram);
        assert!(
            histogram.fits.temp_fit.is_some(),
            "{:?}",
            histogram.fit_worker.status
        );
        histogram.refresh_manual_peak_guesses();
        assert_eq!(histogram.plot_settings.markers.get_peak_bounds(), bounds);
    }

    #[test]
    fn automatic_background_and_queued_peak_fit_share_the_windowless_workflow() {
        let mut histogram = histogram(true);
        histogram.refresh_live_background(egui::Context::default());
        histogram.request_fit(FitTask::Peaks);
        finish(&mut histogram);
        let Some(FitResult::Gaussian(fit)) = &histogram
            .fits
            .temp_fit
            .as_ref()
            .expect("windowless fit")
            .fit_result
        else {
            panic!("Gaussian result");
        };
        assert!(fit.background_markers.is_empty());
        assert!(
            fit.native_result
                .as_ref()
                .expect("native result")
                .fit
                .covariance
                .is_some()
        );
    }

    #[test]
    fn stored_refits_preserve_assignments_and_current_markers() {
        let mut histogram = histogram(true);
        if let BackgroundModel::Linear(parameters) = &mut histogram.fits.settings.background_model {
            parameters.slope.min = 0.1;
            parameters.slope.max = 0.6;
            parameters.intercept.initial_guess = 2.0;
            parameters.intercept.vary = false;
        }
        histogram.fits.settings.equal_stddev = true;
        histogram.fits.settings.free_position = false;
        histogram.request_fit(FitTask::Peaks);
        finish(&mut histogram);
        histogram.fits.store_temp_fit();
        let markers = histogram.plot_settings.markers.get_peak_seeds();
        let before = match &histogram.fits.stored_fits[0].fit_result {
            Some(FitResult::Gaussian(fit)) => fit.fit_result[0].uuid,
            None => panic!("stored Gaussian"),
        };
        histogram.fits.settings.equal_stddev = false;
        histogram.fits.settings.free_position = true;
        let background_record = crate::fitter::main_fitter::Fitter::default();
        let retained_record = ron::to_string(&background_record).expect("background record");
        histogram.fits.stored_fits.push(background_record);
        histogram.request_fit(FitTask::RefitAll);
        finish(&mut histogram);
        assert_eq!(histogram.fits.stored_fits.len(), 2);
        assert_eq!(
            ron::to_string(&histogram.fits.stored_fits[1]).expect("retained record"),
            retained_record
        );
        assert_eq!(histogram.plot_settings.markers.get_peak_seeds(), markers);
        let Some(FitResult::Gaussian(fit)) = &histogram.fits.stored_fits[0].fit_result else {
            panic!("refitted Gaussian");
        };
        assert_eq!(fit.fit_result[0].uuid, before);
        assert!(fit.fit_settings.equal_stdev);
        assert!(!fit.fit_settings.free_position);
        assert!(!histogram.fits.settings.equal_stddev);
        assert!(histogram.fits.settings.free_position);
        let BackgroundModel::Linear(parameters) = &fit.background_model else {
            panic!("stored linear background");
        };
        assert_eq!(parameters.slope.min, 0.1);
        assert_eq!(parameters.slope.max, 0.6);
        assert!(!parameters.intercept.vary);
        assert_eq!(parameters.intercept.initial_guess, 2.0);
        assert!(
            histogram
                .fit_worker
                .status
                .as_deref()
                .is_some_and(|status| status == "Stored fits updated.")
        );
    }

    #[test]
    fn constrained_fit_has_no_display_uncertainty_band() {
        let mut histogram = histogram(true);
        if let BackgroundModel::Linear(parameters) = &mut histogram.fits.settings.background_model {
            parameters.slope.min = 0.0;
            parameters.slope.max = 0.1;
        }
        histogram.request_fit(FitTask::Peaks);
        finish(&mut histogram);
        let Some(FitResult::Gaussian(fit)) =
            &histogram.fits.temp_fit.as_ref().expect("fit").fit_result
        else {
            panic!("Gaussian fit");
        };
        assert!(
            fit.native_result
                .as_ref()
                .expect("native")
                .fit
                .covariance
                .is_none()
        );
        assert!(fit.uncertainty_band.xs.is_empty());
        assert!(fit.fit_report.contains("bg_slope"));
        let metadata = fit.fit_metadata.as_ref().expect("metadata");
        let seed = metadata
            .submitted_background_seed
            .as_ref()
            .expect("submitted background");
        assert_eq!(
            seed.parameters[0].bounds,
            spectrix_fitting::Bounds::finite(0.0, 0.1)
        );
        let mut legacy = serde_json::to_value(metadata).expect("metadata serialization");
        for name in [
            "submitted_peak_seeds",
            "submitted_peak_bounds",
            "submitted_background_seed",
        ] {
            legacy
                .as_object_mut()
                .expect("metadata object")
                .remove(name);
        }
        let restored: crate::fitter::models::gaussian::GaussianFitMetadata =
            serde_json::from_value(legacy).expect("older metadata");
        assert_eq!(restored.peak_seeds, metadata.peak_seeds);
        assert_eq!(restored.peak_bounds, metadata.peak_bounds);
        assert_eq!(restored.background_model, metadata.background_model);
        assert!(restored.submitted_background_seed.is_none());
    }

    #[test]
    fn failed_fit_leaves_the_previous_result_and_persisted_state_intact() {
        let mut histogram = histogram(false);
        histogram.request_fit(FitTask::Peaks);
        finish(&mut histogram);
        let original = ron::to_string(&histogram.fits.temp_fit).expect("serialize previous fit");
        histogram.plot_settings.markers.clear_peak_markers();
        histogram.request_fit(FitTask::Peaks);
        finish(&mut histogram);
        assert_eq!(
            ron::to_string(&histogram.fits.temp_fit).expect("serialize retained fit"),
            original
        );
        let encoded = ron::to_string(&histogram).expect("serialize histogram");
        let mut loaded: Histogram = ron::from_str(&encoded).expect("restore histogram");
        loaded.refresh_live_background(egui::Context::default());
        assert!(!loaded.fit_worker.is_pending());
        assert_eq!(
            ron::to_string(&loaded.fits.temp_fit).expect("serialize loaded fit"),
            original
        );
    }
}
