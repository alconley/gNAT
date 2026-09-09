use super::histogram1d::Histogram;

use crate::fitter::common::Data;
use crate::fitter::main_fitter::{BackgroundModel, FitModel, FitResult, Fitter};
use spectrix_fitting::{
    BackgroundCoupling, FitOptions as NativeFitOptions, ManualSeedEstimateRequest,
    estimate_manual_peak_seeds,
};
use std::hash::{Hash as _, Hasher as _};

fn background_coupling_for_fit(
    background_model: &BackgroundModel,
    has_background_windows: bool,
    lock_manual_background: bool,
) -> BackgroundCoupling {
    if (has_background_windows || lock_manual_background)
        && !matches!(
            background_model,
            BackgroundModel::None | BackgroundModel::LegacyAuto
        )
    {
        BackgroundCoupling::PrefitFrozen
    } else {
        BackgroundCoupling::PrefitJoint
    }
}

fn reusable_manual_background(temp_fit: Option<&Fitter>) -> Option<&Fitter> {
    temp_fit.filter(|fit| fit.background_was_fit_manually && fit.background_result.is_some())
}

impl Histogram {
    pub(super) fn resolved_objective(
        &self,
        range: Option<(f64, f64)>,
    ) -> spectrix_fitting::ObjectiveKind {
        let counts = range.map_or_else(
            || self.bins.iter().map(|count| *count as f64).collect(),
            |(start, end)| self.get_bin_counts_between(start, end),
        );
        self.fits.settings.objective.resolve(counts)
    }

    pub(super) fn manual_estimate_signature(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.bin_width.to_bits().hash(&mut hasher);
        self.bins.hash(&mut hasher);
        format!("{:?}", self.fits.settings.background_model).hash(&mut hasher);
        format!("{:?}", self.fits.settings.objective).hash(&mut hasher);
        self.fits.settings.equal_stddev.hash(&mut hasher);
        for value in self
            .plot_settings
            .markers
            .get_region_marker_positions()
            .into_iter()
            .chain(self.plot_settings.markers.get_peak_marker_positions())
        {
            value.to_bits().hash(&mut hasher);
        }
        for (start, end) in self.plot_settings.markers.get_background_marker_positions() {
            start.to_bits().hash(&mut hasher);
            end.to_bits().hash(&mut hasher);
        }
        hasher.finish().max(1)
    }

    pub fn refresh_manual_peak_guesses(&mut self) {
        // Background marker edits are solved by the worker. Do not run a second background
        // optimization here, or modify peak seeds while the composite snapshot is in flight.
        if self.background_update_pending() || self.fit_worker.is_pending() {
            return;
        }
        if matches!(
            self.fits.settings.background_model,
            BackgroundModel::LegacyAuto
        ) {
            self.fits.settings.background_model = BackgroundModel::None;
        }
        self.plot_settings.markers.ensure_peak_ids();
        if self.plot_settings.markers.peak_markers.is_empty() {
            self.plot_settings.markers.preview_background.clear();
            self.plot_settings.markers.estimate_error = None;
            self.plot_settings.markers.invalid_estimate_signature = 0;
            return;
        }
        let signature = self.manual_estimate_signature();
        let invalid_peak_ids = self
            .plot_settings
            .markers
            .peak_markers
            .iter()
            .filter(|guess| !guess.valid)
            .map(|guess| guess.id)
            .collect::<Vec<_>>();
        let retry_invalid_estimates = !invalid_peak_ids.is_empty()
            && self.plot_settings.markers.invalid_estimate_signature != signature;
        if self.plot_settings.markers.estimate_signature == signature && !retry_invalid_estimates {
            return;
        }

        let region_markers = self.plot_settings.markers.get_region_marker_positions();
        if region_markers.len() != 2 {
            self.plot_settings.markers.estimate_error =
                Some("Place exactly two region markers to estimate peak guesses.".to_owned());
            self.plot_settings.markers.preview_background.clear();
            self.plot_settings.markers.estimate_signature = signature;
            self.plot_settings.markers.invalid_estimate_signature = signature;
            return;
        }
        let background_markers = self.plot_settings.markers.get_background_marker_positions();

        if retry_invalid_estimates {
            for guess in &mut self.plot_settings.markers.peak_markers {
                if invalid_peak_ids.contains(&guess.id) {
                    guess.reset_estimates();
                }
            }
        }

        let equal_sigma = self.fits.settings.equal_stddev;
        let displayed_widths_disagree =
            self.plot_settings
                .markers
                .peak_markers
                .windows(2)
                .any(|pair| {
                    let scale = pair[0].fwhm.abs().max(pair[1].fwhm.abs()).max(1.0);
                    (pair[0].fwhm - pair[1].fwhm).abs() > f64::EPSILON * scale * 16.0
                });
        let force_shared_width = equal_sigma
            && (self.plot_settings.markers.last_equal_sigma == Some(false)
                || displayed_widths_disagree);
        let mut background_seed = crate::fitter::native::background_seed(
            &self.fits.settings.background_model,
            self.live_background
                .preview
                .as_ref()
                .and_then(|fit| fit.background_result.as_ref()),
        );
        if self.live_background.preview.is_some() {
            for parameter in &mut background_seed.parameters {
                parameter.vary = false;
            }
        }
        let request = ManualSeedEstimateRequest {
            x: self.get_bin_centers(),
            y: self.bins.iter().map(|count| *count as f64).collect(),
            bin_width: self.bin_width,
            region: [region_markers[0], region_markers[1]],
            peak_markers: self.plot_settings.markers.get_peak_marker_positions(),
            background_markers,
            background: crate::fitter::native::background_kind(
                &self.fits.settings.background_model,
            ),
            background_seed: Some(background_seed),
            equal_sigma,
        };
        let options = NativeFitOptions {
            objective: self.resolved_objective(Some((region_markers[0], region_markers[1]))),
            ..NativeFitOptions::robust()
        };
        match estimate_manual_peak_seeds(&request, &options) {
            Ok(estimate) => {
                self.plot_settings.markers.preview_background = estimate
                    .background_prefit
                    .evaluation_x
                    .iter()
                    .copied()
                    .zip(estimate.background_prefit.best_fit.iter().copied())
                    .map(Into::into)
                    .collect();
                for (guess, estimated) in self
                    .plot_settings
                    .markers
                    .peak_markers
                    .iter_mut()
                    .zip(estimate.peaks)
                {
                    if force_shared_width
                        || guess.width_source
                            == crate::histoer::histo1d::markers::GuessSource::Estimated
                    {
                        let fwhm = 2.354_82 * estimated.seed.sigma;
                        if guess.amplitude_source
                            != crate::histoer::histo1d::markers::GuessSource::Estimated
                        {
                            guess.set_fwhm_preserving_height(fwhm, self.bin_width);
                        } else {
                            guess.fwhm = fwhm;
                        }
                        guess.width_source =
                            crate::histoer::histo1d::markers::GuessSource::Estimated;
                    }
                    if guess.amplitude_source
                        == crate::histoer::histo1d::markers::GuessSource::Estimated
                    {
                        guess.amplitude = estimated.seed.amplitude;
                    }
                    if guess.bounds_source
                        == crate::histoer::histo1d::markers::GuessSource::Estimated
                    {
                        guess.center_min = estimated.bounds.center[0];
                        guess.center_max = estimated.bounds.center[1];
                        guess.fwhm_min = 2.354_82 * estimated.bounds.sigma[0];
                        guess.fwhm_max = 2.354_82 * estimated.bounds.sigma[1];
                        guess.net_height_min = estimated.bounds.net_height[0];
                        guess.net_height_max = estimated.bounds.net_height[1];
                    }
                    guess.net_height = guess.model_height(self.bin_width);
                    guess.clean_width = estimated.clean_width;
                    guess.valid = guess.fwhm.is_finite()
                        && guess.fwhm > 0.0
                        && guess.amplitude.is_finite()
                        && guess.amplitude > 0.0
                        && guess.bounds_valid();
                }
                self.plot_settings.markers.estimate_error = None;
                self.plot_settings.markers.invalid_estimate_signature = if self
                    .plot_settings
                    .markers
                    .peak_markers
                    .iter()
                    .any(|guess| !guess.valid)
                {
                    signature
                } else {
                    0
                };
            }
            Err(error) => {
                self.plot_settings.markers.preview_background.clear();
                self.plot_settings.markers.estimate_error = Some(error.to_string());
                for guess in &mut self.plot_settings.markers.peak_markers {
                    if guess.width_source
                        == crate::histoer::histo1d::markers::GuessSource::Estimated
                        || guess.amplitude_source
                            == crate::histoer::histo1d::markers::GuessSource::Estimated
                    {
                        guess.valid = false;
                    }
                }
                self.plot_settings.markers.invalid_estimate_signature = signature;
            }
        }
        self.plot_settings.markers.last_equal_sigma = Some(equal_sigma);
        self.plot_settings.markers.estimate_signature = signature;
    }

    pub fn invalidate_manual_gaussian_preview(&mut self) {
        self.plot_settings.markers.estimate_signature = 0;
        if self
            .fits
            .temp_fit
            .as_ref()
            .is_some_and(|fit| matches!(&fit.fit_result, Some(FitResult::Gaussian(_))))
        {
            self.fits.remove_temp_fits();
        }
    }

    fn manual_fit_readiness(&self) -> Result<(), String> {
        let markers = &self.plot_settings.markers;
        if markers.region_markers.len() != 2 {
            return Err("Place exactly two region markers before fitting.".to_owned());
        }
        if markers.peak_markers.is_empty() {
            return Err("Place at least one peak marker before fitting.".to_owned());
        }
        if let Some(error) = &markers.estimate_error {
            return Err(error.clone());
        }
        if markers.peak_markers.iter().any(|guess| !guess.valid) {
            return Err(
                "Every peak needs a positive width and amplitude; move it or re-estimate it."
                    .to_owned(),
            );
        }
        Ok(())
    }

    pub fn apply_refit_all_request(&mut self) {
        if self.fits.take_pending_refit_all() {
            self.request_fit(super::fit_worker::FitTask::RefitAll);
        }
    }

    pub fn apply_modify_fit_request(&mut self) {
        let Some(fit_idx) = self.fits.take_pending_modify_fit() else {
            return;
        };

        let Some((
            metadata,
            metadata_found,
            fallback_background_model,
            background_coupling,
            mut moved_fit,
        )) = self.fits.stored_fits.get(fit_idx).and_then(|stored_fit| {
            if let Some(FitResult::Gaussian(gaussian)) = &stored_fit.fit_result {
                let (metadata, metadata_found) = gaussian.fit_metadata_with_fallback();
                Some((
                    metadata,
                    metadata_found,
                    stored_fit.background_model.clone(),
                    gaussian.background_coupling,
                    stored_fit.clone(),
                ))
            } else {
                None
            }
        })
        else {
            log::warn!("Modify fit requested for non-Gaussian fit.");
            return;
        };

        if !metadata_found {
            log::warn!(
                "Fit metadata was not found; using fallback marker data derived from Gaussian parameters."
            );
        }

        self.plot_settings.markers.clear_background_markers();
        self.plot_settings.markers.clear_peak_markers();
        self.plot_settings.markers.clear_region_markers();

        for marker in metadata.region_markers {
            self.plot_settings.markers.add_region_marker(marker);
        }
        if metadata.peak_seeds.is_empty() {
            for marker in metadata.peak_markers {
                self.plot_settings.markers.add_peak_marker(marker);
            }
        } else {
            for (index, seed) in metadata.peak_seeds.iter().copied().enumerate() {
                self.plot_settings.markers.add_peak_seed(
                    seed,
                    crate::histoer::histo1d::markers::GuessSource::Fitted,
                    self.bin_width,
                    metadata.peak_bounds.get(index).copied(),
                );
            }
        }

        self.fits.settings.background_model =
            if fallback_background_model.type_name() == metadata.background_model {
                fallback_background_model
            } else {
                match metadata.background_model.as_str() {
                    "auto" | "Auto" | "None" => BackgroundModel::None,
                    "constant" => crate::fitter::native::concrete_background_model(
                        spectrix_fitting::BackgroundKind::Constant,
                    ),
                    "linear" => BackgroundModel::Linear(Default::default()),
                    "quadratic" => BackgroundModel::Quadratic(Default::default()),
                    "exponential" => BackgroundModel::Exponential(Default::default()),
                    "powerlaw" => BackgroundModel::PowerLaw(Default::default()),
                    _ => fallback_background_model,
                }
            };
        if let Some(background) = &moved_fit.background_result {
            self.fits.settings.apply_background_fit(background);
        }
        if let Some(FitResult::Gaussian(gaussian)) = &moved_fit.fit_result {
            self.fits.settings.equal_stddev = gaussian.fit_settings.equal_stdev;
            self.fits.settings.free_position = gaussian.fit_settings.free_position;
            self.fits.settings.objective = match gaussian.fit_settings.objective {
                spectrix_fitting::ObjectiveKind::LeastSquares => {
                    crate::fitter::fit_settings::HistogramObjective::LeastSquares
                }
                spectrix_fitting::ObjectiveKind::PoissonDeviance => {
                    crate::fitter::fit_settings::HistogramObjective::PoissonDeviance
                }
            };
        }
        self.fits.settings.background_coupling = background_coupling;
        self.fits.settings.lock_background = background_coupling
            == BackgroundCoupling::PrefitFrozen
            && moved_fit.background_was_fit_manually;

        if matches!(self.fits.settings.background_model, BackgroundModel::None) {
            self.plot_settings.markers.clear_background_markers();
        } else {
            self.plot_settings
                .markers
                .set_background_marker_positions(&metadata.background_markers);
            self.update_background_pair_lines();
        }

        if fit_idx < self.fits.stored_fits.len() {
            self.fits.stored_fits.remove(fit_idx);
        }

        moved_fit.name = format!("{} (Temp)", moved_fit.name);
        self.fits.replace_temp_fit(Some(moved_fit));
    }

    pub fn fit_background(&mut self) {
        let result = self.background_fit_input().and_then(|input| {
            super::live_background::calculate_background(input, self.background_display_range())
        });
        match result {
            Ok(fitter) => self.install_background_fit(fitter),
            Err(error) => {
                log::warn!("Background fit failed: {error}");
                self.live_background.status = Some(error);
            }
        }
    }

    pub(super) fn install_background_fit(&mut self, mut fitter: Fitter) {
        if let Some(background) = &fitter.background_result {
            self.fits.settings.apply_background_fit(background);
        }
        fitter.set_name(self.name.clone());
        self.fits.style_temporary_fit(&mut fitter);
        self.live_background.preview = Some(fitter.clone());
        self.fits.replace_temp_fit(Some(fitter));
        self.live_background.last_attempt = Some(self.live_background_signature());
        self.live_background.status = None;
    }

    pub fn fit_gaussians(&mut self) {
        self.refresh_manual_peak_guesses();
        let mut fitter = match self.prepare_gaussian_fitter() {
            Ok(fitter) => fitter,
            Err(error) => {
                log::error!("Fit aborted: {error}");
                return;
            }
        };
        fitter.fit();
        if let Some(error) = &fitter.last_fit_error {
            log::error!("{error}");
            return;
        }
        self.install_gaussian_fit(fitter);
        self.live_background.last_attempt = Some(self.live_background_signature());
    }

    pub(super) fn prepare_gaussian_fitter(&self) -> Result<Fitter, String> {
        self.manual_fit_readiness()?;
        let region_markers = self.plot_settings.markers.get_region_marker_positions();
        let peak_seeds = self.plot_settings.markers.get_peak_seeds();
        let peak_bounds = self.plot_settings.markers.get_peak_bounds();
        let background_markers = self.plot_settings.markers.get_background_marker_positions();

        let centers = self.get_bin_centers();
        let counts = self.bins.clone();

        let data = Data {
            x: centers,
            y: counts.iter().map(|&c| c as f64).collect(),
        };

        let mut fitter = Fitter::new(data);

        let background_model = self.fits.settings.background_model.clone();

        let manual_background = reusable_manual_background(self.fits.temp_fit.as_ref());
        let lock_manual_background =
            self.fits.settings.lock_background && manual_background.is_some();
        let background_result = if let Some(temp_fit) = manual_background
            && (background_markers.is_empty() || lock_manual_background)
        {
            fitter.background_line = temp_fit.background_line.clone();
            temp_fit.background_result.clone()
        } else {
            None
        };

        let equal_stdev = self.fits.settings.equal_stddev;
        let free_position = self.fits.settings.free_position;

        fitter.calibration = self.fits.calibration.clone();

        fitter.background_model = background_model;
        fitter.background_result = background_result;
        fitter.background_was_fit_manually = manual_background.is_some();
        // Windows determine the baseline. Only windowless fits jointly refine it,
        // unless an explicit manual background has been locked.
        fitter.background_coupling = background_coupling_for_fit(
            &fitter.background_model,
            !background_markers.is_empty(),
            lock_manual_background,
        );
        fitter.objective = self.resolved_objective(Some((region_markers[0], region_markers[1])));
        fitter.manual_peak_bounds = Some(peak_bounds);

        fitter.fit_model = FitModel::Gaussian(
            region_markers.clone(),
            peak_seeds,
            background_markers.clone(),
            equal_stdev,
            free_position,
            None,  // Per-peak initial FWHM bounds are the only active width constraints.
            false, // No global calibrated sigma limits are applied.
        );

        Ok(fitter)
    }

    pub(super) fn install_gaussian_fit(&mut self, mut fitter: Fitter) {
        let previous_peak_assignments = self
            .fits
            .temp_fit
            .as_ref()
            .and_then(|temp_fit| match &temp_fit.fit_result {
                Some(FitResult::Gaussian(g)) => Some(
                    g.fit_result
                        .iter()
                        .filter_map(|p| {
                            p.mean.value.map(|m| {
                                (
                                    m,
                                    p.uuid,
                                    p.energy.value.unwrap_or(-1.0),
                                    p.energy.uncertainty.unwrap_or(0.0),
                                )
                            })
                        })
                        .collect::<Vec<_>>(),
                ),
                _ => None,
            })
            .unwrap_or_default();

        if fitter.background_was_fit_manually
            && let Some(background) = &fitter.background_result
        {
            self.fits.settings.apply_background_fit(background);
        }

        if let Some(FitResult::Gaussian(gaussian)) = &fitter.fit_result {
            self.plot_settings.markers.preview_background = gaussian
                .background_result
                .as_ref()
                .map_or_else(Vec::new, |background| background.get_fit_points());
            for (guess, fitted) in self
                .plot_settings
                .markers
                .peak_markers
                .iter_mut()
                .zip(&gaussian.fit_result)
            {
                if let (Some(center), Some(fwhm), Some(amplitude)) =
                    (fitted.mean.value, fitted.fwhm.value, fitted.amplitude.value)
                {
                    guess.center.x_value = center;
                    guess.fwhm = fwhm;
                    guess.amplitude = amplitude;
                    guess.net_height = guess.model_height(self.bin_width);
                    guess.width_source = crate::histoer::histo1d::markers::GuessSource::Fitted;
                    guess.amplitude_source = crate::histoer::histo1d::markers::GuessSource::Fitted;
                    guess.valid = center.is_finite()
                        && fwhm.is_finite()
                        && fwhm > 0.0
                        && amplitude.is_finite()
                        && amplitude > 0.0;
                }
            }
            self.plot_settings
                .markers
                .peak_markers
                .sort_by(|left, right| left.center.x_value.total_cmp(&right.center.x_value));
        }

        fitter.set_name(self.name.clone());
        self.fits.style_temporary_fit(&mut fitter);
        self.fits.replace_temp_fit(Some(fitter));
        self.fits.settings.show_fit_stats = true;

        // Preserve UUID and energy assignments across modify -> refit workflows.
        if !previous_peak_assignments.is_empty()
            && let Some(temp_fit) = &mut self.fits.temp_fit
            && let Some(FitResult::Gaussian(g)) = &mut temp_fit.fit_result
        {
            let mut prev_sorted = previous_peak_assignments.clone();
            prev_sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let mut new_sorted: Vec<(usize, f64)> = g
                .fit_result
                .iter()
                .enumerate()
                .filter_map(|(idx, p)| p.mean.value.map(|m| (idx, m)))
                .collect();
            new_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            for ((new_idx, _), (_, uuid, energy, energy_unc)) in
                new_sorted.into_iter().zip(prev_sorted)
            {
                if let Err(e) = g.update_uuid_for_peak(new_idx, uuid) {
                    log::warn!("Failed to preserve UUID for peak {new_idx}: {e}");
                }
                if let Err(e) = g.update_energy_for_peak(new_idx, energy, energy_unc) {
                    log::warn!("Failed to preserve energy for peak {new_idx}: {e}");
                }
            }
        }

        // calibrate temp fit if calibration is enabled
        if self.fits.settings.calibrated
            && let Some(temp_fit) = &mut self.fits.temp_fit
        {
            temp_fit.calibrate(&self.fits.calibration);
        }
        // Fitted handles are display updates, not a request to re-estimate constraints.
        // A subsequent user edit will still invalidate this signature normally.
        self.plot_settings.markers.estimate_signature = self.manual_estimate_signature();
    }
}

#[cfg(test)]
mod tests {
    use super::{background_coupling_for_fit, reusable_manual_background};
    use crate::fitter::{
        main_fitter::{BackgroundModel, BackgroundResult, Fitter},
        models::linear::LinearFitter,
    };
    use spectrix_fitting::BackgroundCoupling;

    #[test]
    fn windows_fix_the_background_and_windowless_fits_refine_it() {
        for model in [
            BackgroundModel::Linear(Default::default()),
            BackgroundModel::Quadratic(Default::default()),
            BackgroundModel::PowerLaw(Default::default()),
            BackgroundModel::Exponential(Default::default()),
        ] {
            assert_eq!(
                background_coupling_for_fit(&model, false, false),
                BackgroundCoupling::PrefitJoint
            );
            assert_eq!(
                background_coupling_for_fit(&model, true, false),
                BackgroundCoupling::PrefitFrozen
            );
        }
        assert_eq!(
            background_coupling_for_fit(
                &BackgroundModel::Quadratic(Default::default()),
                false,
                true
            ),
            BackgroundCoupling::PrefitFrozen
        );
    }

    #[test]
    fn only_explicit_background_fits_are_reused() {
        let mut fitter =
            Fitter {
                background_result: Some(BackgroundResult::Linear(
                    LinearFitter::new_from_parameters((0.2, 0.01), (3.0, 0.1), 0.0, 10.0),
                )),
                ..Fitter::default()
            };

        assert!(reusable_manual_background(Some(&fitter)).is_none());
        fitter.background_was_fit_manually = true;
        assert!(reusable_manual_background(Some(&fitter)).is_some());
    }
}
