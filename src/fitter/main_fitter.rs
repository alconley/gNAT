use super::common::{Data, Parameter};
use super::models::exponential::{ExponentialFitter, ExponentialParameters};
use super::models::gaussian::GaussianFitter;
use super::models::linear::{LinearFitter, LinearParameters};
use super::models::powerlaw::{PowerLawFitter, PowerLawParameters};
use super::models::quadratic::{QuadraticFitter, QuadraticParameters};
use crate::defaults::LineDefaults;
use crate::egui_plot_stuff::{egui_filled_area::EguiFilledArea, egui_line::EguiLine};
use crate::fitter::common::Calibration;
use crate::fitter::native;
use spectrix_fitting::{
    BackgroundCoupling, BackgroundFitRequest, FitOptions as NativeFitOptions,
    FitResult as NativeFitResult, ManualPeakBounds, ManualPeakSeed, ObjectiveKind,
    fit_background as fit_native_background,
};

fn fit_display_line(color: egui::Color32) -> EguiLine {
    let mut line = EguiLine::new(color);
    line.allow_hover = false;
    line
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, PartialEq)]
pub enum FitModel {
    Gaussian(
        Vec<f64>,
        Vec<ManualPeakSeed>,
        Vec<(f64, f64)>,
        bool,
        bool,
        Option<(f64, f64)>,
        bool,
    ),
    None,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub enum FitResult {
    Gaussian(GaussianFitter),
}

impl FitResult {
    pub fn get_calibration_data(&self) -> Vec<(f64, f64, f64, f64)> {
        match self {
            Self::Gaussian(fit) => fit.get_calibration_data(),
        }
    }
}

#[derive(Default, PartialEq, Debug, Clone, serde::Deserialize, serde::Serialize)]
pub enum BackgroundModel {
    /// Migration-only representation of the removed automatic background choice.
    /// All fitting paths treat it as `None`.
    #[serde(rename = "Auto")]
    LegacyAuto,
    Constant(Parameter),
    Linear(LinearParameters),
    Quadratic(QuadraticParameters),
    PowerLaw(PowerLawParameters),
    Exponential(ExponentialParameters),
    #[default]
    None,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub enum BackgroundResult {
    Constant(LinearFitter),
    Linear(LinearFitter),
    Quadratic(QuadraticFitter),
    PowerLaw(PowerLawFitter),
    Exponential(ExponentialFitter),
}

impl BackgroundResult {
    pub fn get_fit_points(&self) -> Vec<[f64; 2]> {
        match self {
            Self::Constant(fit) | Self::Linear(fit) => fit.fit_points.clone(),
            Self::Quadratic(fit) => fit.fit_points.clone(),
            Self::PowerLaw(fit) => fit.fit_points.clone(),
            Self::Exponential(fit) => fit.fit_points.clone(),
        }
    }

    pub fn evaluate(&self, x: f64) -> f64 {
        match self {
            Self::Constant(fit) | Self::Linear(fit) => fit.evaluate(x),
            Self::Quadratic(fit) => fit.evaluate(x),
            Self::PowerLaw(fit) => fit.evaluate(x),
            Self::Exponential(fit) => fit.evaluate(x),
        }
    }

    pub(crate) fn set_display_range(&mut self, range: (f64, f64)) {
        let (minimum, maximum) = if range.0 <= range.1 {
            range
        } else {
            (range.1, range.0)
        };
        if !minimum.is_finite() || !maximum.is_finite() || minimum == maximum {
            return;
        }

        const DISPLAY_POINTS: usize = 256;
        let step = (maximum - minimum) / (DISPLAY_POINTS - 1) as f64;
        let fit_points = (0..DISPLAY_POINTS)
            .map(|index| {
                let x = if index + 1 == DISPLAY_POINTS {
                    maximum
                } else {
                    minimum + index as f64 * step
                };
                [x, self.evaluate(x)]
            })
            .collect();

        match self {
            Self::Constant(fit) | Self::Linear(fit) => fit.fit_points = fit_points,
            Self::Quadratic(fit) => fit.fit_points = fit_points,
            Self::PowerLaw(fit) => fit.fit_points = fit_points,
            Self::Exponential(fit) => fit.fit_points = fit_points,
        }
    }
}

pub(crate) fn background_display_range(
    region: &[f64],
    windows: &[(f64, f64)],
) -> Option<(f64, f64)> {
    let mut positions = region
        .iter()
        .copied()
        .chain(windows.iter().flat_map(|(start, end)| [*start, *end]))
        .filter(|position| position.is_finite());
    let first = positions.next()?;
    let range = positions.fold((first, first), |(minimum, maximum), position| {
        (minimum.min(position), maximum.max(position))
    });
    (range.0 < range.1).then_some(range)
}

impl BackgroundModel {
    pub fn type_name(&self) -> String {
        match self {
            Self::LegacyAuto | Self::None => "None",
            Self::Constant(_) => "constant",
            Self::Linear(_) => "linear",
            Self::Quadratic(_) => "quadratic",
            Self::PowerLaw(_) => "powerlaw",
            Self::Exponential(_) => "exponential",
        }
        .to_owned()
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct Fitter {
    pub name: String,

    pub data: Data,

    pub background_model: BackgroundModel,
    pub background_result: Option<BackgroundResult>,
    pub native_background_result: Option<NativeFitResult>,
    /// True only when the user explicitly ran a background-only fit.
    pub background_was_fit_manually: bool,
    /// Transient input mode for a region estimate without manual windows.
    #[serde(skip)]
    pub automatic_background: bool,
    pub background_coupling: BackgroundCoupling,
    pub manual_peak_bounds: Option<Vec<ManualPeakBounds>>,
    pub objective: ObjectiveKind,
    #[serde(skip)]
    pub last_fit_error: Option<String>,

    pub fit_model: FitModel,
    pub fit_result: Option<FitResult>,

    pub background_line: EguiLine,
    pub composition_line: EguiLine,
    pub decomposition_lines: Vec<EguiLine>,

    pub calibration: Calibration,
}

impl Default for Fitter {
    fn default() -> Self {
        Self {
            name: "Fit".to_owned(),

            data: Data::default(),

            background_model: BackgroundModel::None,
            background_result: None,
            native_background_result: None,
            background_was_fit_manually: false,
            automatic_background: false,
            background_coupling: BackgroundCoupling::PrefitFrozen,
            manual_peak_bounds: None,
            objective: ObjectiveKind::PoissonDeviance,
            last_fit_error: None,

            fit_model: FitModel::None,
            fit_result: None,

            background_line: fit_display_line(egui::Color32::GREEN),
            composition_line: fit_display_line(egui::Color32::BLUE),
            decomposition_lines: Vec::new(),

            calibration: Calibration::default(),
        }
    }
}

impl Fitter {
    pub fn apply_line_defaults(
        &mut self,
        background: &LineDefaults,
        composition: &LineDefaults,
        decomposition: &LineDefaults,
    ) {
        background.apply_to(&mut self.background_line);
        composition.apply_to(&mut self.composition_line);
        for line in &mut self.decomposition_lines {
            decomposition.apply_to(line);
        }
    }

    // Constructor to create a new Fitter with empty data and specified model
    pub fn new(data: Data) -> Self {
        Self {
            name: "Fit".to_owned(),

            data,

            background_model: BackgroundModel::None,
            background_result: None,
            native_background_result: None,
            background_was_fit_manually: false,
            automatic_background: false,
            background_coupling: BackgroundCoupling::PrefitFrozen,
            manual_peak_bounds: None,
            objective: ObjectiveKind::PoissonDeviance,
            last_fit_error: None,

            fit_model: FitModel::None,
            fit_result: None,

            background_line: fit_display_line(egui::Color32::GREEN),
            composition_line: fit_display_line(egui::Color32::BLUE),
            decomposition_lines: Vec::new(),

            calibration: Calibration::default(),
        }
    }

    pub fn fit(&mut self) {
        self.last_fit_error = None;
        match &self.fit_model {
            FitModel::Gaussian(
                region_markers,
                peak_seeds,
                background_markrs,
                equal_stdev,
                free_position,
                sigma_bounds_ui,
                bounds_are_calibrated,
            ) => {
                if region_markers.len() != 2 || peak_seeds.is_empty() {
                    log::error!(
                        "Fit aborted: manual Gaussian fitting requires exactly two region markers and at least one valid peak seed."
                    );
                    return;
                }

                let xs = peak_seeds
                    .iter()
                    .map(|seed| seed.center)
                    .collect::<Vec<_>>();

                // build x-space bounds (or None)
                let sigma_bounds_x: Option<(Vec<f64>, Vec<f64>)> = if let Some((min_e, max_e)) =
                    sigma_bounds_ui
                {
                    if *bounds_are_calibrated {
                        // convert energy bounds to x using dE/dx = 2ax + b
                        let a = self.calibration.a.value;
                        let b = self.calibration.b.value;

                        // avoid zero derivative; clamp tiny to epsilon
                        let eps = 1e-12;
                        let deds: Vec<f64> = xs
                            .iter()
                            .map(|&x| (2.0 * a * x + b).abs().max(eps))
                            .collect();

                        if *equal_stdev {
                            let min_x = deds
                                .iter()
                                .map(|d| min_e / *d)
                                .fold(f64::NEG_INFINITY, f64::max);
                            let max_x = deds
                                .iter()
                                .map(|d| max_e / *d)
                                .fold(f64::INFINITY, f64::min);
                            if min_x.is_finite() && max_x.is_finite() && min_x <= max_x {
                                Some((vec![min_x], vec![max_x]))
                            } else {
                                self.last_fit_error = Some(
                                    "The calibrated width limits have no common range for a shared width. Adjust the limits or use independent widths.".to_owned(),
                                );
                                return;
                            }
                        } else {
                            let mins_x: Vec<f64> = deds.iter().map(|d| min_e / *d).collect();
                            let maxs_x: Vec<f64> = deds.iter().map(|d| max_e / *d).collect();
                            Some((mins_x, maxs_x))
                        }
                    } else {
                        // bounds entered in x already; broadcast as needed
                        if *equal_stdev {
                            Some((vec![*min_e], vec![*max_e]))
                        } else {
                            let n = xs.len().max(1);
                            Some((vec![*min_e; n], vec![*max_e; n]))
                        }
                    }
                } else {
                    None
                };

                let mut fit = GaussianFitter::new(
                    self.data.clone(),
                    region_markers.clone(),
                    peak_seeds.clone(),
                    background_markrs.clone(),
                    self.background_model.clone(),
                    self.background_result.clone(),
                    *equal_stdev,
                    *free_position,
                );

                fit.sigma_bounds = sigma_bounds_x;
                fit.manual_peak_bounds = self.manual_peak_bounds.clone();
                fit.background_coupling = self.background_coupling;
                fit.fit_settings.objective = self.objective;

                match fit.fit_native() {
                    Ok(_) => {
                        self.apply_gaussian_fit_visuals(&fit);

                        if let Some(background_result) = &fit.background_result {
                            self.background_line
                                .set_points(background_result.get_fit_points());
                            self.background_result = Some(background_result.clone());
                        }

                        self.fit_result = Some(FitResult::Gaussian(fit));
                    }
                    Err(e) => {
                        let message = format!("Fit failed: {e}");
                        log::error!("{message}");
                        self.last_fit_error = Some(message);
                    }
                }
            }
            FitModel::None => {
                log::info!("No fitting required for 'None'");
            }
        }
    }

    pub fn calibrate(&mut self, calibration: &Calibration) {
        log::info!("Calibrating");
        // Calibration logic goes here
        self.calibration = calibration.clone();

        // update gaussian fit parameters
        if let Some(fit_result) = &mut self.fit_result {
            match fit_result {
                FitResult::Gaussian(fit) => {
                    fit.calibrate(calibration);
                }
            }
        }
    }

    pub fn sync_calibration_values(&mut self, calibration: &Calibration) {
        self.calibration = calibration.clone();

        if let Some(FitResult::Gaussian(fit)) = &mut self.fit_result {
            fit.calibrate_parameters(calibration);
        }
    }

    pub fn fit_background(&mut self) {
        log::info!("Fitting background");
        if matches!(
            self.background_model,
            BackgroundModel::None | BackgroundModel::LegacyAuto
        ) {
            self.background_model = BackgroundModel::None;
            self.background_result = None;
            self.native_background_result = None;
            self.background_was_fit_manually = false;
            log::info!("No background fitting required for 'None'");
            return;
        }

        let Some(minimum) = self.data.x.iter().copied().reduce(f64::min) else {
            log::error!("Cannot fit an empty background");
            return;
        };
        let Some(maximum) = self.data.x.iter().copied().reduce(f64::max) else {
            log::error!("Cannot fit an empty background");
            return;
        };
        let bin_width = self
            .data
            .x
            .windows(2)
            .filter_map(|pair| {
                let difference = (pair[1] - pair[0]).abs();
                (difference > 0.0 && difference.is_finite()).then_some(difference)
            })
            .reduce(f64::min)
            .unwrap_or(1.0);
        let model = self.background_model.clone();
        let options = NativeFitOptions {
            objective: self.objective,
            ..NativeFitOptions::robust()
        };
        let request = BackgroundFitRequest {
            x: self.data.x.clone(),
            y: self.data.y.clone(),
            bin_width,
            region: if minimum == maximum {
                [minimum - 0.5 * bin_width, maximum + 0.5 * bin_width]
            } else {
                [minimum, maximum]
            },
            markers: if self.automatic_background {
                Vec::new()
            } else {
                vec![(minimum, maximum)]
            },
            kind: native::background_kind(&model),
            seed: Some(native::background_seed(
                &model,
                self.background_result.as_ref(),
            )),
        };
        match fit_native_background(&request, &options) {
            Ok(result) => {
                self.background_model = model;
                self.background_result = native::background_result_from_native(
                    &self.background_model,
                    &result,
                    self.data.clone(),
                );
                if let Some(background) = &self.background_result {
                    self.background_line.set_points(background.get_fit_points());
                    self.background_was_fit_manually = true;
                }
                self.native_background_result = Some(result);
            }
            Err(error) => {
                log::warn!("Background fit failed: {error}");
                self.last_fit_error = Some(error.to_string());
            }
        }
        log::info!("Finished fitting background");
    }

    pub fn set_background_display_range(&mut self, range: (f64, f64)) {
        if let Some(background) = &mut self.background_result {
            background.set_display_range(range);
            self.background_line.set_points(background.get_fit_points());
        }
    }

    pub fn get_peak_markers(&self) -> Vec<f64> {
        if self.fit_result.is_none() {
            match &self.fit_model {
                FitModel::Gaussian(_, peak_seeds, _, _, _, _, _) => {
                    peak_seeds.iter().map(|seed| seed.center).collect()
                }
                FitModel::None => Vec::new(),
            }
        } else {
            match &self.fit_result {
                Some(FitResult::Gaussian(fit)) => fit.peak_markers.clone(),
                None => Vec::new(),
            }
        }
    }

    pub fn set_background_color(&mut self, color: egui::Color32) {
        self.background_line.color = color;
    }

    pub fn set_composition_color(&mut self, color: egui::Color32) {
        self.composition_line.color = color;
    }

    pub fn set_decomposition_color(&mut self, color: egui::Color32) {
        for line in &mut self.decomposition_lines {
            line.color = color;
        }
    }

    pub fn show_decomposition(&mut self, show: bool) {
        for line in &mut self.decomposition_lines {
            line.draw = show;
        }
    }

    pub fn show_composition(&mut self, show: bool) {
        self.composition_line.draw = show;
    }

    pub fn show_background(&mut self, show: bool) {
        self.background_line.draw = show;
    }

    pub fn set_name(&mut self, name: String) {
        self.composition_line.name = format!("{name}-Composition");

        for (i, line) in self.decomposition_lines.iter_mut().enumerate() {
            line.name = format!("{name}-Peak {i}");
        }

        self.background_line.name = format!("{name}-Background");
        self.name = name;
    }

    pub fn apply_gaussian_fit_visuals(&mut self, fit: &GaussianFitter) {
        self.composition_line.set_points(fit.fit_points.clone());
        self.decomposition_lines.clear();

        for gaussian in &fit.fit_result {
            let mut line = fit_display_line(egui::Color32::from_rgb(150, 0, 255));
            line.set_points(gaussian.fit_points.clone());
            self.decomposition_lines.push(line);
        }

        if let Some(background_result) = &fit.background_result {
            self.background_line
                .set_points(background_result.get_fit_points());
        }

        self.set_name(self.name.clone());
    }

    pub fn compact_display_data(&mut self) {
        let Some(FitResult::Gaussian(fit)) = &mut self.fit_result else {
            return;
        };
        fit.compact_display_data();
        self.composition_line.allow_hover = false;
        self.background_line.allow_hover = false;
        self.composition_line.set_points(fit.fit_points.clone());
        self.decomposition_lines.truncate(fit.fit_result.len());
        while self.decomposition_lines.len() < fit.fit_result.len() {
            self.decomposition_lines
                .push(fit_display_line(egui::Color32::from_rgb(150, 0, 255)));
        }
        for (line, peak) in self.decomposition_lines.iter_mut().zip(&fit.fit_result) {
            line.allow_hover = false;
            line.set_points(peak.fit_points.clone());
        }
        if let Some(background_result) = &fit.background_result {
            self.background_line
                .set_points(background_result.get_fit_points());
        }
        self.set_name(self.name.clone());
    }

    pub fn fit_result_ui(&mut self, ui: &mut egui::Ui, calibrate: bool) {
        ui.collapsing(self.name.clone(), |ui| {
            egui::ScrollArea::vertical()
                .min_scrolled_height(300.0)
                .show(ui, |ui| {
                    ui.separator();

                    if let Some(background_result) = &self.background_result {
                        ui.label("Background");
                        match background_result {
                            BackgroundResult::Constant(fit) | BackgroundResult::Linear(fit) => {
                                fit.ui(ui);
                            }
                            BackgroundResult::Quadratic(fit) => {
                                fit.ui(ui);
                            }
                            BackgroundResult::PowerLaw(fit) => {
                                fit.ui(ui);
                            }
                            BackgroundResult::Exponential(fit) => {
                                fit.ui(ui);
                            }
                        }
                        ui.horizontal(|ui| {
                            ui.label("Line");
                            self.background_line.menu_button(ui);
                        });
                    }

                    if self.fit_result.is_some() {
                        egui::Grid::new("fit_params_grid")
                            .striped(true)
                            .show(ui, |ui| {
                                ui.label("Peak");
                                ui.label("Mean");
                                ui.label("FWHM");
                                ui.label("Area");
                                ui.label("Amplitude");
                                ui.label("Sigma");
                                ui.label("Energy");

                                ui.end_row();

                                self.fitter_stats(ui, false, calibrate);
                            });

                        // for line in &mut self.decomposition_lines {
                        //     line.menu_button(ui);
                        // }

                        // self.composition_line.menu_button(ui);
                    }
                });
        });
    }

    pub fn fitter_stats(&mut self, ui: &mut egui::Ui, skip_one: bool, calibrate: bool) {
        if let Some(fit_result) = &mut self.fit_result {
            match fit_result {
                FitResult::Gaussian(fit) => {
                    fit.fit_params_ui(ui, skip_one, calibrate);
                }
            }
        }
    }

    fn draw_uncertainty_band(
        plot_ui: &mut egui_plot::PlotUi<'_>,
        band: &EguiFilledArea,
        calibration: Option<&Calibration>,
        name: &str,
        line: &EguiLine,
        fill_alpha: f32,
    ) {
        band.draw(plot_ui, calibration, name, line, fill_alpha);
    }

    pub fn overlaps_visible_x(
        &self,
        plot_ui: &egui_plot::PlotUi<'_>,
        calibration: Option<&Calibration>,
    ) -> bool {
        let primary = if self.composition_line.points.len() >= 2 {
            &self.composition_line
        } else {
            &self.background_line
        };
        primary.overlaps_visible_x(plot_ui, calibration)
    }

    pub fn draw(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        calibration: Option<&Calibration>,
        show_fit_lines_area: bool,
    ) {
        if show_fit_lines_area
            && let Some(FitResult::Gaussian(fit)) = &self.fit_result
            && self.composition_line.draw
            && self
                .composition_line
                .overlaps_visible_x(plot_ui, calibration)
        {
            Self::draw_uncertainty_band(
                plot_ui,
                &fit.uncertainty_band,
                calibration,
                &format!("{}-Composition-Area", self.name),
                &self.composition_line,
                0.22,
            );
        }

        for line in &self.decomposition_lines {
            line.draw(plot_ui, calibration);
        }

        self.composition_line.draw(plot_ui, calibration);

        self.background_line.draw(plot_ui, calibration);
    }

    pub fn set_log(&mut self, log_y: bool, log_x: bool) {
        for line in &mut self.decomposition_lines {
            line.log_y = log_y;
            line.log_x = log_x;
        }

        self.composition_line.log_y = log_y;
        self.composition_line.log_x = log_x;

        self.background_line.log_y = log_y;
        self.background_line.log_x = log_x;
    }

    pub fn get_fit_report(&self) -> String {
        if let Some(fit_result) = &self.fit_result {
            match fit_result {
                FitResult::Gaussian(fit) => fit.get_fit_report(),
            }
        } else {
            "No fit result available.".to_owned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BackgroundResult, Fitter};
    use crate::fitter::models::linear::LinearFitter;

    #[test]
    fn background_display_range_extends_curve_without_refitting() {
        let mut fitter = Fitter::default();
        fitter.background_result = Some(BackgroundResult::Linear(
            LinearFitter::new_from_parameters((2.0, 0.0), (1.0, 0.0), 4.0, 5.0),
        ));

        fitter.set_background_display_range((0.0, 10.0));

        assert_eq!(fitter.background_line.points.len(), 256);
        assert_eq!(fitter.background_line.points.first(), Some(&[0.0, 1.0]));
        assert_eq!(fitter.background_line.points.last(), Some(&[10.0, 21.0]));
    }

    #[test]
    fn loaded_peak_fit_displays_background_through_all_windows_without_refitting() {
        let background = BackgroundResult::Linear(LinearFitter::new_from_parameters(
            (0.2, 0.01),
            (20.0, 0.1),
            -15.0,
            0.0,
        ));
        let composition = vec![[-15.0, 25.0], [0.0, 30.0]];
        let gaussian = crate::fitter::models::gaussian::GaussianFitter {
            region_markers: vec![-15.0, 0.0],
            background_markers: vec![(2.0, 1.0), (-54.0, -55.0)],
            background_result: Some(background.clone()),
            fit_points: composition.clone(),
            ..Default::default()
        };
        let fitter = Fitter {
            background_result: Some(background),
            fit_result: Some(super::FitResult::Gaussian(gaussian)),
            ..Fitter::default()
        };
        let encoded = ron::to_string(&fitter).expect("saved fit");
        let mut loaded: Fitter = ron::from_str(&encoded).expect("load fit");
        loaded.compact_display_data();
        assert_eq!(
            loaded.background_line.points.first().map(|point| point[0]),
            Some(-55.0)
        );
        assert_eq!(
            loaded.background_line.points.last().map(|point| point[0]),
            Some(2.0)
        );
        assert_eq!(loaded.composition_line.points, composition);
        let Some(super::FitResult::Gaussian(gaussian)) = &loaded.fit_result else {
            panic!("retained Gaussian fit");
        };
        assert!(
            gaussian.native_result.is_none(),
            "display changes must not refit"
        );
        assert_eq!(gaussian.region_markers, [-15.0, 0.0]);
        assert_eq!(
            gaussian
                .background_result
                .as_ref()
                .expect("background")
                .evaluate(0.0),
            20.0
        );
    }

    #[test]
    fn incompatible_calibrated_shared_width_limits_are_not_discarded() {
        let mut fitter = Fitter::default();
        fitter.calibration.a.value = 0.5;
        fitter.calibration.b.value = 0.0;
        fitter.fit_model = super::FitModel::Gaussian(
            vec![0.0, 3.0],
            [1.0, 2.0]
                .into_iter()
                .map(|center| spectrix_fitting::ManualPeakSeed {
                    center,
                    sigma: 1.0,
                    amplitude: 100.0,
                })
                .collect(),
            Vec::new(),
            true,
            true,
            Some((1.0, 1.1)),
            true,
        );
        fitter.fit();
        assert!(fitter.fit_result.is_none());
        assert!(
            fitter
                .last_fit_error
                .as_deref()
                .is_some_and(|message| message.contains("no common range"))
        );
    }
}
