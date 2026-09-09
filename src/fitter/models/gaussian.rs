use crate::egui_plot_stuff::egui_filled_area::EguiFilledArea;
use crate::fitter::common::Calibration;
use crate::fitter::common::{Data, Parameter, fit_measurement_label};
use crate::fitter::main_fitter::{BackgroundModel, BackgroundResult};
use crate::fitter::native;

use spectrix_fitting::{
    BackgroundCoupling, FitError as NativeFitError, FitOptions as NativeFitOptions,
    ManualPeakBounds, ManualPeakSeed, ObjectiveKind, PeakFitRequest, SigmaBounds,
    SpectrumFitResult as NativeSpectrumFitResult, fit_peaks as fit_native_peaks,
};

fn auto_fmt(value: Option<f64>, unc: Option<f64>, units: Option<&str>) -> String {
    match value {
        Some(val) => {
            let unc = unc.unwrap_or(0.0);

            if unc > 0.0 && unc.is_finite() {
                // Get order of magnitude of the uncertainty
                let digits = if unc == 0.0 {
                    2
                } else {
                    let exp = unc.abs().log10().floor() as i32;
                    // show 2 significant figures in uncertainty
                    (-(exp) + 1).max(0) as usize
                };

                if let Some(units) = units {
                    format!("{val:.digits$} ± {unc:.digits$} {units}")
                } else {
                    format!("{val:.digits$} ± {unc:.digits$}")
                }
            } else if let Some(units) = units {
                format!("{val:.3} {units}")
            } else {
                format!("{val:.3}")
            }
        }
        None => "—".to_owned(),
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UuidDrawOptions {
    pub calibrate: bool,
    pub log_x: bool,
    pub log_y: bool,
    pub label_size: f32,
    pub label_lift: f32,
    pub draw_label_guide: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct HistogramDrawContext<'a> {
    pub bins: &'a [u64],
    pub range: (f64, f64),
    pub bin_width: f64,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct GaussianParameters {
    pub amplitude: Parameter,
    pub height: Parameter,
    pub mean: Parameter,
    pub sigma: Parameter,
    pub fwhm: Parameter,
    pub area: Parameter,
    pub uuid: usize,
    pub energy: Parameter,
    pub fit_points: Vec<[f64; 2]>, // Vector of (x, y) points representing the Gaussian curve
}

impl Default for GaussianParameters {
    fn default() -> Self {
        Self {
            amplitude: Parameter {
                name: "amplitude".to_owned(),
                ..Default::default()
            },
            height: Parameter {
                name: "height".to_owned(),
                ..Default::default()
            },
            mean: Parameter {
                name: "mean".to_owned(),
                ..Default::default()
            },
            sigma: Parameter {
                name: "sigma".to_owned(),
                ..Default::default()
            },
            fwhm: Parameter {
                name: "fwhm".to_owned(),
                ..Default::default()
            },
            area: Parameter {
                name: "area".to_owned(),
                ..Default::default()
            },
            uuid: 0,
            energy: Parameter {
                name: "energy".to_owned(),
                vary: false,
                ..Default::default()
            },
            fit_points: Vec::new(),
        }
    }
}

impl GaussianParameters {
    pub fn new(
        amp: (f64, f64),
        mean: (f64, f64),
        sigma: (f64, f64),
        fwhm: (f64, f64),
        area: (f64, f64),
    ) -> Self {
        Self {
            amplitude: Parameter {
                name: "amplitude".to_owned(),
                value: Some(amp.0),
                uncertainty: Some(amp.1),
                ..Default::default()
            },
            height: Parameter {
                name: "height".to_owned(),
                ..Default::default()
            },
            mean: Parameter {
                name: "mean".to_owned(),
                value: Some(mean.0),
                uncertainty: Some(mean.1),
                ..Default::default()
            },
            sigma: Parameter {
                name: "sigma".to_owned(),
                value: Some(sigma.0),
                uncertainty: Some(sigma.1),
                ..Default::default()
            },
            fwhm: Parameter {
                name: "fwhm".to_owned(),
                value: Some(fwhm.0),
                uncertainty: Some(fwhm.1),
                ..Default::default()
            },
            area: Parameter {
                name: "area".to_owned(),
                value: Some(area.0),
                uncertainty: Some(area.1),
                ..Default::default()
            },
            uuid: 0,
            energy: Parameter {
                name: "energy".to_owned(),
                value: None,
                uncertainty: None,
                ..Default::default()
            },
            fit_points: Vec::new(),
        }
    }

    /// Function to generate fit points 5 sigma out from the mean.
    /// Fit points are generated in the range [mean - 5 * sigma, mean + 5 * sigma].
    pub fn generate_fit_points(&mut self, num_points: usize) {
        let Some(xs) = self.sample_curve_xs(num_points) else {
            self.fit_points.clear();
            return;
        };

        self.fit_points = xs
            .iter()
            .filter_map(|&x| self.evaluate(x).map(|y| [x, y]))
            .collect();
    }

    fn sample_curve_xs(&self, num_points: usize) -> Option<Vec<f64>> {
        let mean = self.mean.value?;
        let sigma = self.sigma.value?;

        if !mean.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
            return None;
        }

        let range_min = mean - 5.0 * sigma;
        let range_max = mean + 5.0 * sigma;
        let step_size = (range_max - range_min) / (num_points as f64);

        Some(
            (0..=num_points)
                .map(|i| range_min + i as f64 * step_size)
                .collect(),
        )
    }

    fn evaluate(&self, x: f64) -> Option<f64> {
        let amplitude = self.amplitude.value?;
        let mean = self.mean.value?;
        let sigma = self.sigma.value?;

        if !amplitude.is_finite() || !mean.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
            return None;
        }

        let gaussian_norm = (1.0 / (2.0 * std::f64::consts::PI).sqrt()) / sigma;
        let exponent = -((x - mean).powi(2)) / (2.0 * sigma.powi(2));

        Some(amplitude * gaussian_norm * exponent.exp())
    }

    pub fn params_ui(&mut self, ui: &mut egui::Ui, calibrate: bool) {
        if calibrate {
            fit_measurement_label(
                ui,
                self.mean.calibrated_value,
                self.mean.calibrated_uncertainty,
            );
            fit_measurement_label(
                ui,
                self.fwhm.calibrated_value,
                self.fwhm.calibrated_uncertainty,
            );
            fit_measurement_label(
                ui,
                self.area.calibrated_value,
                self.area.calibrated_uncertainty,
            );
            fit_measurement_label(
                ui,
                self.amplitude.calibrated_value,
                self.amplitude.calibrated_uncertainty,
            );
            fit_measurement_label(
                ui,
                self.sigma.calibrated_value,
                self.sigma.calibrated_uncertainty,
            );
        } else {
            fit_measurement_label(ui, self.mean.value, self.mean.uncertainty);
            fit_measurement_label(ui, self.fwhm.value, self.fwhm.uncertainty);
            fit_measurement_label(ui, self.area.value, self.area.uncertainty);
            fit_measurement_label(ui, self.amplitude.value, self.amplitude.uncertainty);
            fit_measurement_label(ui, self.sigma.value, self.sigma.uncertainty);
        }
    }

    pub fn summary_string(
        &self,
        calibrated_units: Option<&str>,
        uncalibrated_units: Option<&str>,
    ) -> String {
        let mut out = String::new();

        out.push_str("\nGaussian Fit Parameters:\n");
        out.push_str(&format!(
            "Mean: {}\n",
            auto_fmt(self.mean.value, self.mean.uncertainty, uncalibrated_units)
        ));
        out.push_str(&format!(
            "Amplitude: {}\n",
            auto_fmt(self.amplitude.value, self.amplitude.uncertainty, None)
        ));
        out.push_str(&format!(
            "Sigma: {}\n",
            auto_fmt(self.sigma.value, self.sigma.uncertainty, uncalibrated_units)
        ));
        out.push_str(&format!(
            "FWHM: {}\n",
            auto_fmt(self.fwhm.value, self.fwhm.uncertainty, uncalibrated_units)
        ));
        out.push_str(&format!(
            "Area: {}\n",
            auto_fmt(self.area.value, self.area.uncertainty, None)
        ));

        out.push_str("\nCalibrated Parameters:\n");
        if let Some(e) = self.energy.value
            && e != -1.0
        {
            out.push_str(&format!(
                "Assigned Energy: {}\n",
                auto_fmt(Some(e), self.energy.uncertainty, calibrated_units)
            ));
        }

        if let Some(cal_mean) = self.mean.calibrated_value {
            out.push_str(&format!(
                "Calibrated Mean: {}\n",
                auto_fmt(
                    Some(cal_mean),
                    self.mean.calibrated_uncertainty,
                    calibrated_units
                )
            ));
        }

        if let Some(cal_fwhm) = self.fwhm.calibrated_value {
            out.push_str(&format!(
                "Calibrated FWHM: {}\n",
                auto_fmt(
                    Some(cal_fwhm),
                    self.fwhm.calibrated_uncertainty,
                    calibrated_units
                )
            ));
        }

        out.trim_end().to_owned()
    }
}

#[derive(Default, Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct GaussianFitSettings {
    pub equal_stdev: bool,
    pub free_position: bool,
    pub objective: ObjectiveKind,
    pub sigma_bounds: Option<(f64, f64)>,
    pub per_peak_sigma_bounds: Option<(Vec<f64>, Vec<f64>)>,
}

#[derive(Default, Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct GaussianFitMetadata {
    pub region_markers: Vec<f64>,
    pub peak_markers: Vec<f64>,
    pub peak_seeds: Vec<ManualPeakSeed>,
    pub peak_bounds: Vec<ManualPeakBounds>,
    pub background_markers: Vec<(f64, f64)>,
    pub background_model: String,
    /// Original inputs retained independently of the fitted marker values.
    pub submitted_peak_seeds: Vec<ManualPeakSeed>,
    pub submitted_peak_bounds: Vec<ManualPeakBounds>,
    pub submitted_background_seed: Option<spectrix_fitting::BackgroundSeed>,
}

#[derive(Default, Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct GaussianFitter {
    pub data: Data,
    pub region_markers: Vec<f64>,
    pub peak_markers: Vec<f64>,
    pub manual_peak_seeds: Vec<ManualPeakSeed>,
    pub manual_peak_bounds: Option<Vec<ManualPeakBounds>>,
    pub background_markers: Vec<(f64, f64)>,
    pub fit_settings: GaussianFitSettings,
    pub background_model: BackgroundModel,
    pub background_result: Option<BackgroundResult>,
    pub fit_result: Vec<GaussianParameters>,
    pub fit_points: Vec<[f64; 2]>,
    pub uncertainty_band: EguiFilledArea,
    pub fit_report: String,
    pub fit_warning: Option<String>,
    #[serde(default, skip_serializing)]
    pub lmfit_result: Option<String>,
    pub native_result: Option<NativeSpectrumFitResult>,
    pub background_coupling: BackgroundCoupling,
    pub sigma_bounds: Option<(Vec<f64>, Vec<f64>)>, // (mins_x, maxs_x)
    pub fit_metadata: Option<GaussianFitMetadata>,
}

impl GaussianFitter {
    // Plotting more than roughly one point per horizontal screen pixel adds no
    // visible detail, but it makes every stored fit expensive to tessellate.
    const MAX_COMPOSITION_DISPLAY_POINTS: usize = 2048;
    const MAX_COMPONENT_DISPLAY_POINTS: usize = 512;
    const MAX_UNCERTAINTY_DISPLAY_POINTS: usize = 512;

    #[expect(clippy::too_many_arguments)]
    pub fn new(
        data: Data,
        region_markers: Vec<f64>,
        manual_peak_seeds: Vec<ManualPeakSeed>,
        background_markers: Vec<(f64, f64)>,
        background_model: BackgroundModel,
        background_result: Option<BackgroundResult>,
        equal_stdev: bool,
        free_position: bool,
    ) -> Self {
        Self {
            data,
            region_markers,
            peak_markers: manual_peak_seeds.iter().map(|seed| seed.center).collect(),
            manual_peak_seeds,
            manual_peak_bounds: None,
            background_markers,
            background_model,
            background_result,
            fit_settings: GaussianFitSettings {
                equal_stdev,
                free_position,
                ..Default::default()
            },
            fit_result: Vec::new(),
            fit_points: Vec::new(),
            uncertainty_band: EguiFilledArea::default(),
            fit_report: String::new(),
            fit_warning: None,
            lmfit_result: None,
            native_result: None,
            background_coupling: BackgroundCoupling::PrefitFrozen,
            sigma_bounds: None,
            fit_metadata: None,
        }
    }

    pub fn fit_metadata_with_fallback(&self) -> (GaussianFitMetadata, bool) {
        if let Some(metadata) = &self.fit_metadata {
            return (metadata.clone(), true);
        }

        let region_markers = if self.region_markers.len() >= 2 {
            self.region_markers.clone()
        } else if let (Some(min), Some(max)) = (
            self.data.x.iter().copied().reduce(f64::max),
            self.data.x.iter().copied().reduce(f64::min),
        ) {
            vec![min, max]
        } else {
            Vec::new()
        };

        let peak_markers = if !self.peak_markers.is_empty() {
            self.peak_markers.clone()
        } else {
            self.fit_result
                .iter()
                .filter_map(|p| p.mean.value)
                .collect::<Vec<_>>()
        };

        (
            GaussianFitMetadata {
                region_markers,
                peak_markers,
                peak_seeds: self.manual_peak_seeds_with_fallback(),
                peak_bounds: self.manual_peak_bounds.clone().unwrap_or_default(),
                background_markers: self.background_markers.clone(),
                background_model: self.background_model.type_name(),
                submitted_peak_seeds: Vec::new(),
                submitted_peak_bounds: self.manual_peak_bounds.clone().unwrap_or_default(),
                submitted_background_seed: None,
            },
            false,
        )
    }

    fn manual_peak_seeds_with_fallback(&self) -> Vec<ManualPeakSeed> {
        if !self.manual_peak_seeds.is_empty() {
            return self.manual_peak_seeds.clone();
        }
        self.peak_markers
            .iter()
            .enumerate()
            .filter_map(|(index, &center)| {
                let fitted = self.fit_result.get(index);
                let sigma = fitted
                    .and_then(|peak| peak.sigma.value)
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .unwrap_or(1.0);
                let amplitude = fitted
                    .and_then(|peak| peak.amplitude.value)
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .unwrap_or(1.0);
                center.is_finite().then_some(ManualPeakSeed {
                    center,
                    sigma,
                    amplitude,
                })
            })
            .collect()
    }

    fn build_uncertainty_band(
        xs: &[f64],
        ys: &[f64],
        uncertainties: &[f64],
    ) -> Option<EguiFilledArea> {
        if xs.len() != ys.len() || ys.len() != uncertainties.len() {
            return None;
        }

        let sanitized_uncertainties = uncertainties
            .iter()
            .map(|&uncertainty| {
                if uncertainty.is_finite() && uncertainty > 0.0 {
                    uncertainty
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let upper_signal = ys
            .iter()
            .zip(&sanitized_uncertainties)
            .map(|(&y, &uncertainty)| y + uncertainty)
            .collect::<Vec<_>>();
        let indices =
            Self::display_sample_indices(xs, &upper_signal, Self::MAX_UNCERTAINTY_DISPLAY_POINTS);
        let band_xs = indices.iter().map(|&index| xs[index]).collect();
        let lower = indices
            .iter()
            .map(|&index| (ys[index] - sanitized_uncertainties[index]).max(0.0))
            .collect();
        let upper = indices
            .iter()
            .map(|&index| ys[index] + sanitized_uncertainties[index])
            .collect();

        Some(EguiFilledArea::new(band_xs, lower, upper))
    }

    fn display_sample_indices(xs: &[f64], ys: &[f64], limit: usize) -> Vec<usize> {
        let point_count = xs.len();
        if point_count <= limit || point_count != ys.len() || limit < 3 {
            return (0..point_count).collect();
        }

        // Largest-Triangle-Three-Buckets retains sharp Gaussian maxima and
        // shoulders much better than a fixed stride while bounding plot cost.
        let bucket_width = (point_count - 2) as f64 / (limit - 2) as f64;
        let mut sampled = Vec::with_capacity(limit);
        sampled.push(0);
        let mut anchor = 0;

        for bucket in 0..(limit - 2) {
            let average_start =
                (((bucket + 1) as f64 * bucket_width).floor() as usize + 1).min(point_count - 1);
            let average_end =
                (((bucket + 2) as f64 * bucket_width).floor() as usize + 1).min(point_count);
            let (average_x, average_y) = if average_start < average_end {
                let count = (average_end - average_start) as f64;
                (
                    xs[average_start..average_end].iter().sum::<f64>() / count,
                    ys[average_start..average_end].iter().sum::<f64>() / count,
                )
            } else {
                (xs[point_count - 1], ys[point_count - 1])
            };

            let range_start =
                ((bucket as f64 * bucket_width).floor() as usize + 1).min(point_count - 1);
            let range_end = (((bucket + 1) as f64 * bucket_width).floor() as usize + 1)
                .min(point_count - 1)
                .max(range_start + 1);
            let anchor_x = xs[anchor];
            let anchor_y = ys[anchor];
            let mut selected = range_start;
            let mut largest_area = f64::NEG_INFINITY;
            for index in range_start..range_end {
                let area = ((anchor_x - average_x) * (ys[index] - anchor_y)
                    - (anchor_x - xs[index]) * (average_y - anchor_y))
                    .abs();
                if area > largest_area {
                    largest_area = area;
                    selected = index;
                }
            }
            sampled.push(selected);
            anchor = selected;
        }

        sampled.push(point_count - 1);
        sampled
    }

    fn decimate_curve(xs: &[f64], ys: &[f64]) -> Vec<[f64; 2]> {
        Self::decimate_curve_to_limit(xs, ys, Self::MAX_COMPONENT_DISPLAY_POINTS)
    }

    fn decimate_curve_to_limit(xs: &[f64], ys: &[f64], limit: usize) -> Vec<[f64; 2]> {
        if xs.len() != ys.len() {
            return Vec::new();
        }
        Self::display_sample_indices(xs, ys, limit)
            .into_iter()
            .map(|index| [xs[index], ys[index]])
            .collect()
    }

    fn decimate_composition_arrays(
        xs: Vec<f64>,
        ys: Vec<f64>,
        uncertainties: Vec<f64>,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        if xs.len() != ys.len() || xs.len() != uncertainties.len() {
            return (xs, ys, uncertainties);
        }
        let indices = Self::display_sample_indices(&xs, &ys, Self::MAX_COMPOSITION_DISPLAY_POINTS);
        (
            indices.iter().map(|&index| xs[index]).collect(),
            indices.iter().map(|&index| ys[index]).collect(),
            indices.iter().map(|&index| uncertainties[index]).collect(),
        )
    }

    fn select_display_values(values: &mut Vec<f64>, indices: &[usize], expected_length: usize) {
        if values.len() == expected_length {
            *values = indices.iter().map(|&index| values[index]).collect();
        }
    }

    fn compact_native_evaluation(result: &mut NativeSpectrumFitResult) {
        let expected_length = result.fit.evaluation_x.len();
        if expected_length <= Self::MAX_COMPOSITION_DISPLAY_POINTS
            || result.fit.best_fit.len() != expected_length
        {
            return;
        }
        let indices = Self::display_sample_indices(
            &result.fit.evaluation_x,
            &result.fit.best_fit,
            Self::MAX_COMPOSITION_DISPLAY_POINTS,
        );

        Self::select_display_values(&mut result.fit.evaluation_x, &indices, expected_length);
        Self::select_display_values(&mut result.fit.best_fit, &indices, expected_length);
        for component in &mut result.fit.components {
            Self::select_display_values(&mut component.values, &indices, expected_length);
        }
        if let Some(band) = &mut result.fit.confidence_band {
            Self::select_display_values(&mut band.x, &indices, expected_length);
            Self::select_display_values(&mut band.best_fit, &indices, expected_length);
            Self::select_display_values(&mut band.uncertainty, &indices, expected_length);
            Self::select_display_values(&mut band.lower, &indices, expected_length);
            Self::select_display_values(&mut band.upper, &indices, expected_length);
        }
        for (_, band) in &mut result.fit.component_bands {
            Self::select_display_values(&mut band.x, &indices, expected_length);
            Self::select_display_values(&mut band.best_fit, &indices, expected_length);
            Self::select_display_values(&mut band.uncertainty, &indices, expected_length);
            Self::select_display_values(&mut band.lower, &indices, expected_length);
            Self::select_display_values(&mut band.upper, &indices, expected_length);
        }
    }

    pub(crate) fn compact_display_data(&mut self) {
        self.extend_background_display();
        if self.fit_points.len() > Self::MAX_COMPOSITION_DISPLAY_POINTS {
            let (xs, ys): (Vec<_>, Vec<_>) = self
                .fit_points
                .iter()
                .map(|point| (point[0], point[1]))
                .unzip();
            self.fit_points =
                Self::decimate_curve_to_limit(&xs, &ys, Self::MAX_COMPOSITION_DISPLAY_POINTS);
        }
        for peak in &mut self.fit_result {
            if peak.fit_points.len() > Self::MAX_COMPONENT_DISPLAY_POINTS {
                let (xs, ys): (Vec<_>, Vec<_>) = peak
                    .fit_points
                    .iter()
                    .map(|point| (point[0], point[1]))
                    .unzip();
                peak.fit_points = Self::decimate_curve(&xs, &ys);
            }
        }
        let band = &mut self.uncertainty_band;
        if band.xs.len() > Self::MAX_UNCERTAINTY_DISPLAY_POINTS
            && band.xs.len() == band.lower.len()
            && band.xs.len() == band.upper.len()
        {
            let indices = Self::display_sample_indices(
                &band.xs,
                &band.upper,
                Self::MAX_UNCERTAINTY_DISPLAY_POINTS,
            );
            band.xs = indices.iter().map(|&index| band.xs[index]).collect();
            band.lower = indices.iter().map(|&index| band.lower[index]).collect();
            band.upper = indices.iter().map(|&index| band.upper[index]).collect();
        }
        if let Some(native_result) = &mut self.native_result {
            Self::compact_native_evaluation(native_result);
        }
    }

    pub fn get_calibration_data(&self) -> Vec<(f64, f64, f64, f64)> {
        let mut calibration_data = Vec::new();

        for params in &self.fit_result {
            if let (Some(energy), Some(energy_unc), Some(mean), Some(mean_unc)) = (
                params.energy.value,
                params.energy.uncertainty,
                params.mean.value,
                params.mean.uncertainty,
            ) && energy != -1.0
                && mean.is_finite()
                && mean_unc.is_finite()
                && energy.is_finite()
                && energy_unc.is_finite()
            {
                calibration_data.push((mean, mean_unc.max(0.0), energy, energy_unc.max(0.0)));
            }
        }

        calibration_data
    }

    pub fn calibrate_parameters(&mut self, calibration: &Calibration) {
        for param in &mut self.fit_result {
            // param.amplitude.calibrate(calibration);
            param.mean.calibrate_energy(calibration);
            param
                .sigma
                .calibrate_sigma(calibration, param.mean.value.unwrap_or(0.0));
            param
                .fwhm
                .calibrate_fwhm(calibration, param.mean.value.unwrap_or(0.0));

            param.energy.calibrated_value = param.energy.value;
            param.energy.calibrated_uncertainty = param.energy.uncertainty;

            param.amplitude.calibrated_value = param.amplitude.value;
            param.amplitude.calibrated_uncertainty = param.amplitude.uncertainty;

            param.height.calibrated_value = param.height.value;
            param.height.calibrated_uncertainty = param.height.uncertainty;

            param.area.calibrated_value = param.area.value;
            param.area.calibrated_uncertainty = param.area.uncertainty;
        }
    }

    pub fn calibrate(&mut self, calibration: &Calibration) {
        log::info!("Calibrating");
        self.calibrate_parameters(calibration);
    }

    fn extend_background_display(&mut self) {
        let Some(range) = crate::fitter::main_fitter::background_display_range(
            &self.region_markers,
            &self.background_markers,
        ) else {
            return;
        };
        if let Some(background) = &mut self.background_result {
            let points = background.get_fit_points();
            if points.first().map(|point| point[0]) != Some(range.0)
                || points.last().map(|point| point[0]) != Some(range.1)
            {
                background.set_display_range(range);
            }
        }
    }

    /// Fits the configured Gaussian/background model using the native Rust backend.
    pub fn fit_native(&mut self) -> Result<(), NativeFitError> {
        if self.region_markers.len() != 2 {
            return Err(NativeFitError::InvalidRegion);
        }
        let sigma_bounds = self
            .sigma_bounds
            .as_ref()
            .map(|(minima, maxima)| SigmaBounds {
                minima: minima.clone(),
                maxima: maxima.clone(),
            });
        let mut background_seed =
            native::background_seed(&self.background_model, self.background_result.as_ref());
        // An explicitly fitted and locked background is the only fixed-background path.
        if self.background_result.is_some()
            && self.background_coupling == BackgroundCoupling::PrefitFrozen
        {
            for parameter in &mut background_seed.parameters {
                parameter.vary = false;
            }
        }
        let request = PeakFitRequest {
            x: self.data.x.clone(),
            y: self.data.y.clone(),
            bin_width: self
                .data
                .x
                .windows(2)
                .next()
                .map_or(1.0, |pair| (pair[1] - pair[0]).abs()),
            region: [self.region_markers[0], self.region_markers[1]],
            peak_seeds: self.manual_peak_seeds_with_fallback(),
            peak_bounds: self.manual_peak_bounds.clone(),
            background_markers: self.background_markers.clone(),
            background: native::background_kind(&self.background_model),
            background_seed: Some(background_seed),
            background_coupling: self.background_coupling,
            equal_sigma: self.fit_settings.equal_stdev,
            free_centers: self.fit_settings.free_position,
            sigma_bounds,
        };
        let options = NativeFitOptions {
            objective: self.fit_settings.objective,
            ..NativeFitOptions::robust()
        };
        let mut native_result = fit_native_peaks(&request, &options)?;
        if matches!(self.background_model, BackgroundModel::LegacyAuto) {
            self.background_model = BackgroundModel::None;
        }

        let estimate = |name: &str| {
            native_result
                .fit
                .parameters
                .iter()
                .find(|parameter| parameter.name == name)
        };
        self.fit_result.clear();
        self.peak_markers.clear();
        for index in 0..native_result.peak_markers.len() {
            let prefix = format!("g{index}_");
            let amplitude = estimate(&format!("{prefix}amplitude")).ok_or_else(|| {
                NativeFitError::InvalidParameter {
                    parameter: format!("{prefix}amplitude"),
                }
            })?;
            let height = estimate(&format!("{prefix}height")).ok_or_else(|| {
                NativeFitError::InvalidParameter {
                    parameter: format!("{prefix}height"),
                }
            })?;
            let center = estimate(&format!("{prefix}center")).ok_or_else(|| {
                NativeFitError::InvalidParameter {
                    parameter: format!("{prefix}center"),
                }
            })?;
            let sigma = estimate(&format!("{prefix}sigma")).ok_or_else(|| {
                NativeFitError::InvalidParameter {
                    parameter: format!("{prefix}sigma"),
                }
            })?;
            let fwhm = estimate(&format!("{prefix}fwhm")).ok_or_else(|| {
                NativeFitError::InvalidParameter {
                    parameter: format!("{prefix}fwhm"),
                }
            })?;
            let area = estimate(&format!("{prefix}area")).ok_or_else(|| {
                NativeFitError::InvalidParameter {
                    parameter: format!("{prefix}area"),
                }
            })?;

            let mut parameters = GaussianParameters::default();
            parameters.amplitude.value = Some(amplitude.value);
            parameters.amplitude.uncertainty = amplitude.standard_error;
            parameters.height.value = Some(height.value);
            parameters.height.uncertainty = height.standard_error;
            parameters.mean.value = Some(center.value);
            parameters.mean.uncertainty = center.standard_error;
            parameters.sigma.value = Some(sigma.value);
            parameters.sigma.uncertainty = sigma.standard_error;
            parameters.fwhm.value = Some(fwhm.value);
            parameters.fwhm.uncertainty = fwhm.standard_error;
            parameters.area.value = Some(area.value);
            parameters.area.uncertainty = area.standard_error;
            parameters.fit_points = native_result
                .fit
                .components
                .iter()
                .find(|component| component.name == prefix)
                .map(|component| {
                    Self::decimate_curve(&native_result.fit.evaluation_x, &component.values)
                })
                .unwrap_or_default();
            self.peak_markers.push(center.value);
            self.fit_result.push(parameters);
        }
        self.manual_peak_seeds = self
            .fit_result
            .iter()
            .filter_map(|peak| {
                Some(ManualPeakSeed {
                    center: peak.mean.value?,
                    sigma: peak.sigma.value?,
                    amplitude: peak.amplitude.value?,
                })
            })
            .collect();

        let uncertainties = native_result.fit.confidence_band.as_ref().map_or_else(
            || vec![0.0; native_result.fit.best_fit.len()],
            |band| band.uncertainty.clone(),
        );
        let (composition_x, composition_y, composition_uncertainty) =
            Self::decimate_composition_arrays(
                native_result.fit.evaluation_x.clone(),
                native_result.fit.best_fit.clone(),
                uncertainties,
            );
        self.fit_points = composition_x
            .iter()
            .copied()
            .zip(composition_y.iter().copied())
            .map(Into::into)
            .collect();
        self.uncertainty_band = native_result
            .fit
            .confidence_band
            .as_ref()
            .and_then(|_| {
                Self::build_uncertainty_band(
                    &composition_x,
                    &composition_y,
                    &composition_uncertainty,
                )
            })
            .unwrap_or_default();
        self.fit_report = native::fit_report(&native_result.fit);
        if self.background_coupling == BackgroundCoupling::PrefitFrozen
            && !matches!(self.background_model, BackgroundModel::None)
        {
            if !self.background_markers.is_empty() {
                self.fit_report.push_str("\nBackground windows determine the fixed baseline; peak-region bins do not change it.\n");
            }
            self.fit_report.push_str("\nPeak uncertainties are conditional on the fixed background; background uncertainty is excluded.\n");
        }
        self.fit_warning = (!native_result.quality_issues.is_empty()).then(|| {
            native_result
                .quality_issues
                .iter()
                .map(|issue| match issue {
                    spectrix_fitting::FitQualityIssue::FailedConvergence { reason } => {
                        format!("The optimizer did not converge: {reason}")
                    }
                    spectrix_fitting::FitQualityIssue::MissingCovariance => {
                        native::covariance_description(&native_result.fit)
                    }
                    spectrix_fitting::FitQualityIssue::NonFiniteResult => {
                        "The fit contains a non-finite parameter or curve value.".to_owned()
                    }
                    spectrix_fitting::FitQualityIssue::ActiveBounds { parameters } => {
                        format!("Parameters stopped on active bounds: {}", parameters.join(", "))
                    }
                    spectrix_fitting::FitQualityIssue::NonpositivePrediction => {
                        "The Poisson model has an invalid expected count in at least one bin."
                            .to_owned()
                    }
                    spectrix_fitting::FitQualityIssue::ObjectiveWorsened {
                        initial,
                        final_value,
                    } => {
                        format!("The objective worsened from {initial:.6} to {final_value:.6}.")
                    }
                    spectrix_fitting::FitQualityIssue::PoorGoodnessOfFit { p_value } => {
                        format!("The Poisson goodness-of-fit p-value is {p_value:.4e}.")
                    }
                    spectrix_fitting::FitQualityIssue::UnmodeledResidualPeaks { positions } => {
                        format!("Possible unmarked peak remains near {positions:?}; add a peak marker and refit.")
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        });
        if let Some(warning) = &self.fit_warning {
            log::warn!("{warning}");
            self.fit_report.push_str("\n[[Fit warning]]\n");
            self.fit_report.push_str(warning);
            self.fit_report.push('\n');
        }
        // The display background for a joint fit comes from the composite component, so compact
        // the dense evaluation grid before copying that curve into the UI-facing result.
        Self::compact_native_evaluation(&mut native_result);
        self.background_result = native::background_result_from_composite_native(
            &self.background_model,
            &native_result.fit,
            self.data.clone(),
        );
        self.extend_background_display();
        self.fit_metadata = Some(GaussianFitMetadata {
            region_markers: native_result.region.to_vec(),
            peak_markers: native_result.peak_markers.clone(),
            peak_seeds: self.manual_peak_seeds.clone(),
            peak_bounds: self.manual_peak_bounds.clone().unwrap_or_default(),
            background_markers: if matches!(self.background_model, BackgroundModel::None) {
                Vec::new()
            } else {
                self.background_markers.clone()
            },
            background_model: self.background_model.type_name(),
            submitted_peak_seeds: native_result.peak_seeds.clone(),
            submitted_peak_bounds: self.manual_peak_bounds.clone().unwrap_or_default(),
            submitted_background_seed: request.background_seed.clone(),
        });
        self.lmfit_result = None;
        // The full parameter covariance, residuals, and statistics are retained.
        // Dense evaluation/component grids are display payload, so keeping hundreds
        // of thousands of redundant samples per stored fit only wastes memory.
        self.native_result = Some(native_result);
        Ok(())
    }

    #[expect(
        dead_code,
        reason = "structured native quality issues supersede the legacy warning"
    )]
    fn unmodeled_peak_warning(&self, result: &NativeSpectrumFitResult) -> Option<String> {
        let region_min = self.region_markers[0].min(self.region_markers[1]);
        let region_max = self.region_markers[0].max(self.region_markers[1]);
        let observations = self
            .data
            .x
            .iter()
            .copied()
            .zip(self.data.y.iter().copied())
            .filter(|(x, _)| *x >= region_min && *x <= region_max)
            .collect::<Vec<_>>();
        if observations.len() != result.fit.residuals.len() || observations.len() < 3 {
            return None;
        }
        let mut absolute_residuals = result
            .fit
            .residuals
            .iter()
            .map(|residual| residual.abs())
            .collect::<Vec<_>>();
        absolute_residuals.sort_by(f64::total_cmp);
        let robust_noise = absolute_residuals[absolute_residuals.len() / 2].max(1.0);
        let fitted_peaks = self
            .fit_result
            .iter()
            .filter_map(|peak| Some((peak.mean.value?, peak.sigma.value?.abs())))
            .collect::<Vec<_>>();
        let candidates = (1..observations.len() - 1)
            .filter(|index| {
                let residual = result.fit.residuals[*index];
                let model_count = (observations[*index].1 - residual).max(1.0);
                let threshold = (5.0 * model_count.sqrt()).max(6.0 * robust_noise);
                residual > threshold
                    && residual >= result.fit.residuals[*index - 1]
                    && residual >= result.fit.residuals[*index + 1]
                    && fitted_peaks.iter().all(|(center, sigma)| {
                        (observations[*index].0 - center).abs() > (2.5 * sigma).max(1.0e-12)
                    })
            })
            .count();
        (candidates > 0).then(|| {
            format!(
                "Possible unmarked peak{} remain{} in the fit residuals ({candidates}). Add peak marker{} and refit.",
                if candidates == 1 { "" } else { "s" },
                if candidates == 1 { "s" } else { "" },
                if candidates == 1 { "" } else { "s" },
            )
        })
    }

    pub fn update_uuid_for_peak(
        &mut self,
        peak_index: usize,
        new_uuid: usize,
    ) -> Result<(), String> {
        let peak = self
            .fit_result
            .get_mut(peak_index)
            .ok_or_else(|| format!("peak index {peak_index} is out of range"))?;
        peak.uuid = new_uuid;
        Ok(())
    }

    pub fn update_energy_for_peak(
        &mut self,
        peak_index: usize,
        new_energy: f64,
        new_uncertainty: f64,
    ) -> Result<(), String> {
        if !new_energy.is_finite() || !new_uncertainty.is_finite() || new_uncertainty < 0.0 {
            return Err(
                "energy must be finite and uncertainty must be finite and non-negative".to_owned(),
            );
        }
        let peak = self
            .fit_result
            .get_mut(peak_index)
            .ok_or_else(|| format!("peak index {peak_index} is out of range"))?;
        peak.energy.value = Some(new_energy);
        peak.energy.uncertainty = Some(new_uncertainty);
        peak.energy.calibrated_value = Some(new_energy);
        peak.energy.calibrated_uncertainty = Some(new_uncertainty);
        Ok(())
    }

    fn composition_height_at_raw_x(&self, raw_x: f64) -> Option<f64> {
        self.fit_points
            .iter()
            .min_by(|a, b| {
                let da = (a[0] - raw_x).abs();
                let db = (b[0] - raw_x).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|point| point[1])
    }

    fn nearby_histogram_height_at_raw_x(
        raw_x: f64,
        histogram_bins: &[u64],
        histogram_range: (f64, f64),
        histogram_bin_width: f64,
    ) -> Option<f64> {
        if histogram_bins.is_empty() || histogram_bin_width <= 0.0 {
            return None;
        }

        if raw_x < histogram_range.0 || raw_x > histogram_range.1 {
            return None;
        }

        let mut bin_index = ((raw_x - histogram_range.0) / histogram_bin_width).floor() as usize;
        let last_bin = histogram_bins.len().saturating_sub(1);
        bin_index = bin_index.min(last_bin);

        let start = bin_index.saturating_sub(1);
        let end = (bin_index + 1).min(last_bin);

        histogram_bins[start..=end]
            .iter()
            .copied()
            .max()
            .map(|value| value as f64)
    }

    pub fn draw_uuid(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        options: UuidDrawOptions,
        histogram: HistogramDrawContext<'_>,
    ) {
        use egui::Align2;
        use egui::Color32;
        use egui::Id;
        use egui::RichText;
        use egui_plot::{Line, LineStyle, PlotPoints, Text};

        let plot_bounds = plot_ui.plot_bounds();
        let y_span = (plot_bounds.max()[1] - plot_bounds.min()[1]).abs();
        let label_size = f64::from(options.label_size.clamp(8.0, 32.0));
        let label_lift = f64::from(options.label_lift.clamp(0.0, 3.0));
        let linear_label_offset =
            ((y_span * (0.02 + label_size * 0.001)).max(label_size * 0.25)) * label_lift;
        let log_label_multiplier = 1.0 + label_size * 0.025 * label_lift;
        let label_color = if plot_ui.ctx().theme() == egui::Theme::Dark {
            Color32::WHITE
        } else {
            Color32::BLACK
        };

        for params in &self.fit_result {
            if params.uuid == 0 {
                continue; // Skip if UUID is not set
            }

            let Some(raw_mean) = params.mean.value else {
                continue;
            };

            let mut x_position = if options.calibrate {
                params.mean.calibrated_value.unwrap_or(raw_mean)
            } else {
                raw_mean
            };

            let composition_height = self.composition_height_at_raw_x(raw_mean);
            let histogram_height = Self::nearby_histogram_height_at_raw_x(
                raw_mean,
                histogram.bins,
                histogram.range,
                histogram.bin_width,
            );
            let Some(reference_height) = composition_height
                .into_iter()
                .chain(histogram_height)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            else {
                continue;
            };

            let guide_y = composition_height.and_then(|height| {
                if options.log_y {
                    if height <= 0.0 {
                        None
                    } else {
                        Some(height.log10().max(0.0001))
                    }
                } else {
                    Some(height)
                }
            });

            let y_position = if options.log_y {
                if reference_height <= 0.0 {
                    continue;
                }
                let lifted_y = reference_height * log_label_multiplier.max(1.0);
                if lifted_y <= 0.0 {
                    continue;
                }
                lifted_y.log10().max(0.0001)
            } else {
                reference_height + linear_label_offset
            };

            if options.log_x {
                if x_position <= 0.0 {
                    continue;
                }
                x_position = x_position.log10().max(0.0001);
            }

            let label = Text::new(
                "",
                [x_position, y_position].into(),
                RichText::new(params.uuid.to_string()).size(label_size as f32),
            )
            .anchor(Align2::CENTER_BOTTOM)
            .color(label_color)
            .allow_hover(false);

            if let Some(guide_y) = guide_y
                && options.draw_label_guide
                && (y_position - guide_y).abs() > f64::EPSILON
            {
                plot_ui.line(
                    Line::new(
                        "",
                        PlotPoints::Owned(vec![
                            [x_position, guide_y].into(),
                            [x_position, y_position].into(),
                        ]),
                    )
                    .allow_hover(false)
                    .color(label_color)
                    .width(1.0)
                    .style(LineStyle::Dashed {
                        length: (label_size as f32 * 0.45).max(4.0),
                    })
                    .id(Id::new((
                        "uuid_label_guide",
                        params.uuid,
                        x_position.to_bits(),
                        y_position.to_bits(),
                    ))),
                );
            }

            plot_ui.text(label);
        }
    }

    pub fn fit_params_ui(&mut self, ui: &mut egui::Ui, _skip_one: bool, calibrate: bool) {
        let mut uuid_updates = Vec::new();
        let mut energy_updates = Vec::new();

        for (i, params) in self.fit_result.iter_mut().enumerate() {
            // if skip_one && i != 0 {
            //     ui.label("");
            // }
            ui.label(format!("{i}"));
            params.params_ui(ui, calibrate);

            let mut uuid = params.uuid;
            if ui.add(egui::DragValue::new(&mut uuid).speed(1)).changed() {
                uuid_updates.push((i, uuid)); // defer the update
            }

            let mut energy = params.energy.value.unwrap_or(-1.0);
            let mut uncertainty = params.energy.uncertainty.unwrap_or(0.0);

            ui.horizontal(|ui| {
                let mut changed = false;
                changed |= ui
                    .add(egui::DragValue::new(&mut energy).speed(0.1))
                    .changed();
                ui.label("±");
                changed |= ui
                    .add(egui::DragValue::new(&mut uncertainty).speed(0.1))
                    .changed();

                if changed {
                    energy_updates.push((i, energy, uncertainty));
                }
            });

            if i == 0 {
                ui.vertical(|ui| {
                    if let Some(warning) = &self.fit_warning {
                        ui.colored_label(egui::Color32::YELLOW, warning);
                    }
                    ui.menu_button("Fit Report", |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                ui.label(self.fit_report.clone());
                            });
                        });
                    });
                });
            }

            ui.end_row();
        }

        for (index, new_uuid) in uuid_updates {
            println!("Updating UUID for peak {index}: {new_uuid}");
            if let Err(e) = self.update_uuid_for_peak(index, new_uuid) {
                eprintln!("UUID update failed: {e}");
            }
        }

        for (index, new_energy, new_uncertainty) in energy_updates {
            println!(
                "Updating energy for peak {index}: {new_energy}, uncertainty: {new_uncertainty}"
            );
            if let Err(e) = self.update_energy_for_peak(index, new_energy, new_uncertainty) {
                eprintln!("Energy update failed: {e}");
            }
        }
    }

    pub fn get_fit_report(&self) -> String {
        self.fit_report.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::{GaussianFitter, GaussianParameters};
    use crate::fitter::{
        common::{Calibration, Data},
        main_fitter::{BackgroundModel, BackgroundResult},
        models::linear::LinearFitter,
    };
    use spectrix_fitting::{BackgroundCoupling, ManualPeakSeed, ParameterKind};

    #[test]
    fn decimate_composition_arrays_limits_density_and_keeps_endpoints() {
        let point_count = GaussianFitter::MAX_COMPOSITION_DISPLAY_POINTS + 5000;
        let xs = (0..point_count).map(|i| i as f64).collect::<Vec<_>>();
        let ys = xs.iter().map(|x| x * 2.0).collect::<Vec<_>>();
        let uncertainties = xs.iter().map(|x| x * 0.1).collect::<Vec<_>>();

        let (xs, ys, uncertainties) =
            GaussianFitter::decimate_composition_arrays(xs, ys, uncertainties);

        assert!(xs.len() <= GaussianFitter::MAX_COMPOSITION_DISPLAY_POINTS);
        assert_eq!(xs.len(), ys.len());
        assert_eq!(xs.len(), uncertainties.len());
        assert_eq!(xs.first().copied(), Some(0.0));
        assert_eq!(ys.first().copied(), Some(0.0));
        assert_eq!(uncertainties.first().copied(), Some(0.0));
        assert_eq!(xs.last().copied(), Some((point_count - 1) as f64));
        assert_eq!(ys.last().copied(), Some(((point_count - 1) as f64) * 2.0));
        assert_eq!(
            uncertainties.last().copied(),
            Some(((point_count - 1) as f64) * 0.1)
        );
    }

    #[test]
    fn display_decimation_preserves_narrow_peak_and_limits_component_and_band_density() {
        let point_count = 20_000;
        let peak_index = 12_345;
        let xs = (0..point_count)
            .map(|index| index as f64)
            .collect::<Vec<_>>();
        let mut ys = vec![0.0; point_count];
        ys[peak_index] = 100.0;

        let curve = GaussianFitter::decimate_curve(&xs, &ys);
        assert!(curve.len() <= GaussianFitter::MAX_COMPONENT_DISPLAY_POINTS);
        assert_eq!(curve.first().copied(), Some([0.0, 0.0]));
        assert_eq!(curve.last().copied(), Some([(point_count - 1) as f64, 0.0]));
        assert!(curve.iter().any(|point| point[1] == 100.0));

        let uncertainties = vec![1.0; point_count];
        let band = GaussianFitter::build_uncertainty_band(&xs, &ys, &uncertainties)
            .expect("valid uncertainty band");
        assert!(band.xs.len() <= GaussianFitter::MAX_UNCERTAINTY_DISPLAY_POINTS);
        assert!(
            band.xs
                .iter()
                .zip(&band.upper)
                .any(|(&x, &upper)| x == peak_index as f64 && upper == 101.0)
        );
    }

    #[test]
    fn legacy_lmfit_payload_loads_but_is_omitted_from_new_json() {
        let legacy = serde_json::json!({
            "fit_result": [GaussianParameters::default()],
            "fit_points": [[1.0, 2.0], [2.0, 3.0]],
            "lmfit_result": "legacy model result"
        });
        let decoded: GaussianFitter =
            serde_json::from_value(legacy).expect("legacy Gaussian fit should load");
        assert_eq!(decoded.lmfit_result.as_deref(), Some("legacy model result"));
        assert_eq!(decoded.fit_points, vec![[1.0, 2.0], [2.0, 3.0]]);

        let encoded = serde_json::to_value(&decoded).expect("native Gaussian fit should save");
        assert!(encoded.get("lmfit_result").is_none());
        assert!(encoded.get("fit_points").is_some());
    }

    #[test]
    fn native_fit_preserves_workflow_metadata_uncertainty_and_assignments() {
        let x = (0..=100)
            .map(|index| index as f64 * 0.1)
            .collect::<Vec<_>>();
        let y = x
            .iter()
            .enumerate()
            .map(|(index, independent)| {
                1.5 + 0.25 * independent
                    + 85.0 / ((2.0 * std::f64::consts::PI).sqrt() * 0.32)
                        * (-0.5 * ((*independent - 5.0) / 0.32).powi(2)).exp()
                    + 0.01 * (index as f64).sin()
            })
            .collect();
        let mut fitter = GaussianFitter::new(
            Data { x, y },
            vec![2.0, 8.0],
            vec![ManualPeakSeed {
                center: 5.0,
                sigma: 0.4,
                amplitude: 85.0,
            }],
            vec![(2.0, 3.0), (7.0, 8.0)],
            BackgroundModel::Linear(Default::default()),
            None,
            true,
            true,
        );
        fitter.background_coupling = BackgroundCoupling::PrefitJoint;
        fitter.fit_native().expect("native Gaussian fit");

        let native = fitter.native_result.as_ref().expect("native payload");
        assert!(native.fit.termination.success);
        assert!(native.fit.covariance.is_some());
        assert!(native.fit.confidence_band.is_some());
        assert!(native.fit.evaluation_x.len() <= GaussianFitter::MAX_COMPOSITION_DISPLAY_POINTS);
        assert_eq!(native.fit.best_fit.len(), native.fit.evaluation_x.len());
        assert!(
            native
                .fit
                .components
                .iter()
                .all(|component| component.values.len() == native.fit.evaluation_x.len())
        );
        assert!(
            native
                .fit
                .component_bands
                .iter()
                .all(|(_, band)| band.x.len() == native.fit.evaluation_x.len())
        );
        assert!(fitter.uncertainty_band.xs.len() <= GaussianFitter::MAX_UNCERTAINTY_DISPLAY_POINTS);
        assert!(
            fitter
                .fit_result
                .iter()
                .all(|peak| peak.fit_points.len() <= GaussianFitter::MAX_COMPONENT_DISPLAY_POINTS)
        );
        assert!(
            native
                .fit
                .component_bands
                .iter()
                .any(|(name, _)| name == "g0_")
        );
        let native_background = native
            .fit
            .components
            .iter()
            .find(|component| component.name == "background")
            .expect("joint background component");
        let BackgroundResult::Linear(background) = fitter
            .background_result
            .as_ref()
            .expect("displayed joint background")
        else {
            panic!("expected linear background");
        };
        assert_eq!(
            background.fit_points,
            native
                .fit
                .evaluation_x
                .iter()
                .copied()
                .zip(native_background.values.iter().copied())
                .map(Into::into)
                .collect::<Vec<[f64; 2]>>()
        );
        assert_eq!(
            fitter.fit_metadata.as_ref().expect("metadata").peak_markers,
            vec![5.0]
        );

        fitter.update_uuid_for_peak(0, 42).expect("UUID update");
        fitter
            .update_energy_for_peak(0, 1332.5, 0.2)
            .expect("energy update");
        let mut calibration = Calibration::default();
        calibration.b.value = 2.0;
        calibration.c.value = 1.0;
        fitter.calibrate(&calibration);
        let peak = &fitter.fit_result[0];
        assert_eq!(peak.uuid, 42);
        assert_eq!(peak.energy.value, Some(1332.5));
        let expected_calibrated_mean = 2.0 * peak.mean.value.expect("mean") + 1.0;
        assert!(
            (peak.mean.calibrated_value.expect("calibrated mean") - expected_calibrated_mean).abs()
                < 1.0e-10
        );

        let encoded = serde_json::to_string(&fitter).expect("serialize native fit");
        let decoded: GaussianFitter =
            serde_json::from_str(&encoded).expect("deserialize native fit");
        assert_eq!(decoded.background_coupling, BackgroundCoupling::PrefitJoint);
        assert_eq!(decoded.fit_result[0].uuid, 42);
        assert_eq!(decoded.fit_result[0].energy.value, Some(1332.5));
        assert!(decoded.native_result.is_some());
    }

    #[test]
    fn frozen_manual_background_uses_explicit_background_markers() {
        let x = (0..=100)
            .map(|index| index as f64 * 0.1 + 0.05)
            .collect::<Vec<_>>();
        let y = x
            .iter()
            .map(|independent| {
                1.5 + 0.25 * independent
                    + 85.0 / ((2.0 * std::f64::consts::PI).sqrt() * 0.32)
                        * (-0.5 * ((*independent - 5.0) / 0.32).powi(2)).exp()
            })
            .collect();
        let background = BackgroundResult::Linear(LinearFitter::new_from_parameters(
            (0.25, 0.01),
            (1.5, 0.02),
            0.05,
            10.05,
        ));
        let mut fitter = GaussianFitter::new(
            Data { x, y },
            vec![2.03, 7.97],
            vec![ManualPeakSeed {
                center: 5.0,
                sigma: 0.4,
                amplitude: 85.0,
            }],
            vec![(2.03, 3.0), (7.0, 7.97)],
            BackgroundModel::Linear(Default::default()),
            Some(background),
            true,
            true,
        );
        fitter
            .fit_native()
            .expect("fit using frozen manual background");

        let native = fitter.native_result.as_ref().expect("native payload");
        for (name, expected) in [("bg_slope", 0.25), ("bg_intercept", 1.5)] {
            let parameter = native
                .fit
                .parameters
                .iter()
                .find(|parameter| parameter.name == name)
                .unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(parameter.kind, ParameterKind::Fixed);
            assert!((parameter.value - expected).abs() < 1.0e-12);
        }
        assert!(native.fit.termination.success);
        assert_eq!(native.peak_markers.len(), 1);
    }

    #[test]
    fn unmarked_residual_peak_is_reported_without_adding_a_gaussian() {
        let x = (0..=200)
            .map(|index| index as f64 * 0.05)
            .collect::<Vec<_>>();
        let y = x
            .iter()
            .map(|value| {
                3.0 + 80.0 / (2.506_628_274_631_000_2 * 0.2)
                    * (-0.5 * ((*value - 3.0) / 0.2).powi(2)).exp()
                    + 60.0 / (2.506_628_274_631_000_2 * 0.25)
                        * (-0.5 * ((*value - 8.0) / 0.25).powi(2)).exp()
            })
            .collect::<Vec<_>>();
        let mut fitter = GaussianFitter::new(
            Data { x, y },
            vec![0.0, 10.0],
            vec![ManualPeakSeed {
                center: 3.0,
                sigma: 0.25,
                amplitude: 80.0,
            }],
            Vec::new(),
            BackgroundModel::None,
            None,
            true,
            true,
        );
        fitter.background_coupling = BackgroundCoupling::PrefitJoint;

        fitter.fit_native().expect("single marked peak fit");

        assert_eq!(fitter.fit_result.len(), 1);
        assert!(
            fitter
                .fit_warning
                .as_deref()
                .is_some_and(|warning| warning.contains("unmarked peak"))
        );
    }
}
