use rfd::FileDialog;

use egui::{Align2, Color32};
use egui_extras::{Column, TableBuilder};
use egui_plot::{FilledArea, Line, MarkerShape, Plot, PlotUi, Points, Text};

use std::fs::File;
use std::io::Write as _;
use std::path::PathBuf;

use super::fit_settings::{CalibrationMode, FitSettings};
use super::main_fitter::{FitResult, Fitter};

use super::models::gaussian::{HistogramDrawContext, UuidDrawOptions};

use super::common::{Calibration, fit_measurement_label};

use crate::custom_analysis::se_sps_analysis::uuid_map::FitUUID;
use crate::defaults::FitPaletteDefaults;
use crate::fitter::common::Data;
use crate::fitter::models::linear::LinearFitter;
use crate::fitter::models::quadratic;
use crate::histoer::histo1d::markers::FitMarkers;
use spectrix_fitting::{
    Bound, FitQualityIssue, FitQualityStatus, ManualPeakBounds, ManualPeakSeed, ObjectiveKind,
};

use std::collections::HashMap;

#[derive(Debug, Clone)]
struct CalibrationPlotPoint {
    mean: f64,
    mean_uncertainty: f64,
    energy: f64,
    energy_uncertainty: f64,
    uuid: usize,
    fit_name: String,
    peak_index: usize,
}

fn padded_range(mut min: f64, mut max: f64) -> (f64, f64) {
    if !min.is_finite() || !max.is_finite() {
        return (0.0, 1.0);
    }

    if min > max {
        std::mem::swap(&mut min, &mut max);
    }

    let span = max - min;
    if span.abs() < f64::EPSILON {
        let pad = min.abs().max(1.0) * 0.1;
        (min - pad, max + pad)
    } else {
        let pad = span * 0.05;
        (min - pad, max + pad)
    }
}

fn unique_value_count(values: &[f64]) -> usize {
    let mut unique = values.to_vec();
    unique.sort_by(|left, right| left.total_cmp(right));
    unique.dedup_by(|left, right| {
        let scale = left.abs().max(right.abs()).max(1.0);
        (*left - *right).abs() <= scale * 1e-9
    });
    unique.len()
}

fn exact_linear_calibration(mean: &[f64], energy: &[f64]) -> Option<Calibration> {
    if mean.len() != 2 || energy.len() != 2 {
        return None;
    }
    let denominator = mean[1] - mean[0];
    let scale = mean[0].abs().max(mean[1].abs()).max(1.0);
    if !denominator.is_finite() || denominator.abs() <= scale * 1e-9 {
        return None;
    }
    let slope = (energy[1] - energy[0]) / denominator;
    let intercept = energy[0] - slope * mean[0];
    let candidate = Calibration {
        a: crate::fitter::common::Value {
            value: 0.0,
            uncertainty: 0.0,
        },
        b: crate::fitter::common::Value {
            value: slope,
            uncertainty: 0.0,
        },
        c: crate::fitter::common::Value {
            value: intercept,
            uncertainty: 0.0,
        },
        cov: None,
        uncertainty_available: false,
    };
    candidate.coefficients_are_finite().then_some(candidate)
}

fn draw_plot_segment(
    plot_ui: &mut PlotUi<'_>,
    id_source: impl std::hash::Hash + std::fmt::Debug,
    points: Vec<[f64; 2]>,
    color: Color32,
    width: f32,
) {
    plot_ui.line(
        Line::new("", points)
            .allow_hover(false)
            .color(color)
            .width(width)
            .id(egui::Id::new(id_source)),
    );
}

#[derive(Debug, Clone, Copy)]
struct CrossErrorBars<'a> {
    id_prefix: &'a str,
    index: usize,
    x: f64,
    y: f64,
    x_uncertainty: f64,
    y_uncertainty: f64,
    x_cap_half_width: f64,
    y_cap_half_height: f64,
    color: Color32,
}

fn draw_cross_error_bars(plot_ui: &mut PlotUi<'_>, error_bar: CrossErrorBars<'_>) {
    let x_uncertainty = error_bar.x_uncertainty.max(0.0);
    let y_uncertainty = error_bar.y_uncertainty.max(0.0);

    if x_uncertainty.is_finite() && x_uncertainty > 0.0 {
        draw_plot_segment(
            plot_ui,
            (error_bar.id_prefix, error_bar.index, "x_bar"),
            vec![
                [error_bar.x - x_uncertainty, error_bar.y],
                [error_bar.x + x_uncertainty, error_bar.y],
            ],
            error_bar.color,
            1.0,
        );

        if error_bar.y_cap_half_height > 0.0 {
            draw_plot_segment(
                plot_ui,
                (error_bar.id_prefix, error_bar.index, "x_cap_left"),
                vec![
                    [
                        error_bar.x - x_uncertainty,
                        error_bar.y - error_bar.y_cap_half_height,
                    ],
                    [
                        error_bar.x - x_uncertainty,
                        error_bar.y + error_bar.y_cap_half_height,
                    ],
                ],
                error_bar.color,
                1.0,
            );
            draw_plot_segment(
                plot_ui,
                (error_bar.id_prefix, error_bar.index, "x_cap_right"),
                vec![
                    [
                        error_bar.x + x_uncertainty,
                        error_bar.y - error_bar.y_cap_half_height,
                    ],
                    [
                        error_bar.x + x_uncertainty,
                        error_bar.y + error_bar.y_cap_half_height,
                    ],
                ],
                error_bar.color,
                1.0,
            );
        }
    }

    if y_uncertainty.is_finite() && y_uncertainty > 0.0 {
        draw_plot_segment(
            plot_ui,
            (error_bar.id_prefix, error_bar.index, "y_bar"),
            vec![
                [error_bar.x, error_bar.y - y_uncertainty],
                [error_bar.x, error_bar.y + y_uncertainty],
            ],
            error_bar.color,
            1.0,
        );

        if error_bar.x_cap_half_width > 0.0 {
            draw_plot_segment(
                plot_ui,
                (error_bar.id_prefix, error_bar.index, "y_cap_bottom"),
                vec![
                    [
                        error_bar.x - error_bar.x_cap_half_width,
                        error_bar.y - y_uncertainty,
                    ],
                    [
                        error_bar.x + error_bar.x_cap_half_width,
                        error_bar.y - y_uncertainty,
                    ],
                ],
                error_bar.color,
                1.0,
            );
            draw_plot_segment(
                plot_ui,
                (error_bar.id_prefix, error_bar.index, "y_cap_top"),
                vec![
                    [
                        error_bar.x - error_bar.x_cap_half_width,
                        error_bar.y + y_uncertainty,
                    ],
                    [
                        error_bar.x + error_bar.x_cap_half_width,
                        error_bar.y + y_uncertainty,
                    ],
                ],
                error_bar.color,
                1.0,
            );
        }
    }
}

fn fmt_value_uncertainty(value: f64, uncertainty: f64, precision: usize) -> String {
    if uncertainty.is_finite() && uncertainty > 0.0 {
        format!("{value:.precision$} ± {uncertainty:.precision$}")
    } else {
        format!("{value:.precision$}")
    }
}

const MANUAL_FWHM_FACTOR: f64 = 2.354_82;

#[derive(Clone)]
struct ManualInitialTableRow {
    id: u64,
    seed: ManualPeakSeed,
    bounds: ManualPeakBounds,
    net_height: f64,
    source: String,
    valid: bool,
}

#[derive(Clone)]
struct ManualTempTableRow {
    center: (f64, Option<f64>),
    fwhm: (f64, Option<f64>),
    height: (f64, Option<f64>),
    area: (f64, Option<f64>),
    amplitude: (f64, Option<f64>),
    sigma: (f64, Option<f64>),
}

fn bound_value(bound: Bound, fallback: f64) -> f64 {
    match bound {
        Bound::Inclusive(value) => value,
        Bound::Unbounded => fallback,
    }
}

fn decimal_places_for_resolution(resolution: f64) -> usize {
    let resolution = resolution.abs();
    if !resolution.is_finite() || resolution >= 1.0 {
        return 2;
    }
    let places = (0..=6)
        .find(|places| {
            let scaled = resolution * 10_f64.powi(*places as i32);
            (scaled - scaled.round()).abs() <= 1.0e-8
        })
        .unwrap_or(6);
    // Keep two digits beyond the bin resolution: e.g. 1.0-wide bins use
    // two decimals, while 0.2-wide bins use three.
    (places + 2).min(8)
}

fn range_label(ui: &mut egui::Ui, minimum: f64, guess: f64, maximum: f64, decimals: usize) {
    ui.label(format!(
        "{minimum:.decimals$} ≤ {guess:.decimals$} ≤ {maximum:.decimals$}"
    ))
    .on_hover_text("Initial lower bound ≤ initial solver value ≤ initial upper bound");
}

fn calibrated_center_range(
    calibration: &Calibration,
    minimum: f64,
    guess: f64,
    maximum: f64,
) -> Option<(f64, f64, f64)> {
    let minimum = calibration.calibrate_checked(minimum)?;
    let guess = calibration.calibrate_checked(guess)?;
    let maximum = calibration.calibrate_checked(maximum)?;
    Some((minimum.min(maximum), guess, minimum.max(maximum)))
}

fn calibrated_fwhm_range(
    calibration: &Calibration,
    center: f64,
    minimum: f64,
    guess: f64,
    maximum: f64,
) -> Option<(f64, f64, f64)> {
    let scale = calibration.derivative_checked(center)?.abs();
    (scale.is_finite() && scale > f64::EPSILON).then_some((
        minimum * scale,
        guess * scale,
        maximum * scale,
    ))
}

fn calibrated_center_measurement(
    calibration: &Calibration,
    value: f64,
    uncertainty: Option<f64>,
) -> (f64, Option<f64>) {
    let Some(calibrated_value) = calibration.calibrate_checked(value) else {
        return (value, uncertainty);
    };
    let calibrated_uncertainty = calibration
        .derivative_checked(value)
        .and_then(|derivative| {
            let fit_uncertainty = uncertainty.unwrap_or(0.0);
            let calibration_uncertainty = calibration.curve_uncertainty_checked(value)?;
            (derivative * fit_uncertainty)
                .hypot(calibration_uncertainty)
                .is_finite()
                .then_some((derivative * fit_uncertainty).hypot(calibration_uncertainty))
        });
    (calibrated_value, calibrated_uncertainty)
}

fn calibrated_width_measurement(
    calibration: &Calibration,
    center: f64,
    value: f64,
    uncertainty: Option<f64>,
) -> (f64, Option<f64>) {
    let Some(derivative) = calibration.derivative_checked(center) else {
        return (value, uncertainty);
    };
    let calibrated_value = derivative.abs() * value;
    let uncertainty = uncertainty.map(|uncertainty| {
        let derivative_uncertainty =
            (2.0 * center * calibration.a.uncertainty).hypot(calibration.b.uncertainty);
        (derivative * uncertainty).hypot(value * derivative_uncertainty)
    });
    if calibrated_value.is_finite() {
        (
            calibrated_value,
            uncertainty.filter(|value| value.is_finite()),
        )
    } else {
        (value, uncertainty)
    }
}

fn range_drag_ui(
    ui: &mut egui::Ui,
    minimum: &mut f64,
    guess: &mut f64,
    maximum: &mut f64,
    speed: f64,
    hard_minimum: f64,
    hard_maximum: f64,
) -> [bool; 3] {
    let mut changed = [false; 3];
    ui.horizontal(|ui| {
        changed[0] = ui
            .add(
                egui::DragValue::new(minimum)
                    .speed(speed)
                    .range(hard_minimum..=*guess),
            )
            .on_hover_text("Initial lower bound; drag horizontally or type a value")
            .changed();
        ui.label("≤");
        changed[1] = ui
            .add(
                egui::DragValue::new(guess)
                    .speed(speed)
                    .range(*minimum..=*maximum),
            )
            .on_hover_text("Initial solver value; drag horizontally or type a value")
            .changed();
        ui.label("≤");
        changed[2] = ui
            .add(
                egui::DragValue::new(maximum)
                    .speed(speed)
                    .range(*guess..=hard_maximum),
            )
            .on_hover_text("Initial upper bound; drag horizontally or type a value")
            .changed();
    });
    changed
}

#[expect(
    clippy::too_many_arguments,
    reason = "the three linked values require their calibration display context"
)]
fn calibrated_center_range_drag_ui(
    ui: &mut egui::Ui,
    minimum: &mut f64,
    guess: &mut f64,
    maximum: &mut f64,
    speed: f64,
    raw_domain: (f64, f64),
    calibration: &Calibration,
    decimals: usize,
) -> [bool; 3] {
    macro_rules! drag {
        ($value:expr, $range:expr) => {{
            let formatter = calibration.clone();
            let parser = calibration.clone();
            egui::DragValue::new($value)
                .speed(speed)
                .range($range)
                .custom_formatter(move |raw, _| {
                    formatter.calibrate_checked(raw).map_or_else(
                        || raw.to_string(),
                        |calibrated| format!("{calibrated:.decimals$}"),
                    )
                })
                .custom_parser(move |text| {
                    text.parse::<f64>()
                        .ok()
                        .and_then(|calibrated| parser.invert_in_range(calibrated, raw_domain))
                })
        }};
    }
    let mut changed = [false; 3];
    ui.horizontal(|ui| {
        changed[0] = ui
            .add(drag!(minimum, raw_domain.0..=*guess))
            .on_hover_text("Calibrated lower bound; drag horizontally or type a value")
            .changed();
        ui.label("≤");
        changed[1] = ui
            .add(drag!(guess, *minimum..=*maximum))
            .on_hover_text("Calibrated initial center; drag horizontally or type a value")
            .changed();
        ui.label("≤");
        changed[2] = ui
            .add(drag!(maximum, *guess..=raw_domain.1))
            .on_hover_text("Calibrated upper bound; drag horizontally or type a value")
            .changed();
    });
    changed
}

#[expect(
    clippy::too_many_arguments,
    reason = "the three linked values require their calibration display context"
)]
fn calibrated_fwhm_range_drag_ui(
    ui: &mut egui::Ui,
    minimum: &mut f64,
    guess: &mut f64,
    maximum: &mut f64,
    speed: f64,
    hard_minimum: f64,
    hard_maximum: f64,
    center: f64,
    calibration: &Calibration,
    decimals: usize,
) -> Option<[bool; 3]> {
    let scale = calibration.derivative_checked(center)?.abs();
    if !scale.is_finite() || scale <= f64::EPSILON {
        return None;
    }
    macro_rules! drag {
        ($value:expr, $range:expr) => {
            egui::DragValue::new($value)
                .speed(speed)
                .range($range)
                .custom_formatter(move |raw, _| format!("{:.decimals$}", raw * scale))
                .custom_parser(move |text| {
                    text.parse::<f64>()
                        .ok()
                        .map(|calibrated| calibrated / scale)
                })
        };
    }
    let mut changed = [false; 3];
    ui.horizontal(|ui| {
        changed[0] = ui
            .add(drag!(minimum, hard_minimum..=*guess))
            .on_hover_text("Calibrated FWHM lower bound; drag horizontally or type a value")
            .changed();
        ui.label("≤");
        changed[1] = ui
            .add(drag!(guess, *minimum..=*maximum))
            .on_hover_text("Calibrated initial FWHM; drag horizontally or type a value")
            .changed();
        ui.label("≤");
        changed[2] = ui
            .add(drag!(maximum, *guess..=hard_maximum))
            .on_hover_text("Calibrated FWHM upper bound; drag horizontally or type a value")
            .changed();
    });
    Some(changed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub enum SortCol {
    Fit,
    Peak,
    Mean,
    Fwhm,
    Height,
    Area,
    Amplitude,
    Sigma,
    Uuid,
    Energy,
}

#[derive(Debug, Clone, Copy, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct SortState {
    pub col: SortCol,
    pub asc: bool,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct Fits {
    pub temp_fit: Option<Fitter>,
    /// Invalidates worker results whenever the temporary fit is replaced or removed.
    #[serde(skip)]
    pub(crate) temp_fit_revision: u64,
    pub stored_fits: Vec<Fitter>,
    pub settings: FitSettings,
    pub calibration: Calibration,
    pub sort_state: SortState,
    #[serde(default)]
    pub palette: FitPaletteDefaults,
    #[serde(skip)]
    pub pending_modify_fit: Option<usize>,
    #[serde(skip)]
    pub pending_refit_all: bool,
}

impl Default for Fits {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SortState {
    fn default() -> Self {
        Self {
            col: SortCol::Fit,
            asc: true,
        }
    }
}

impl Fits {
    fn ensure_extension_if_missing(mut path: PathBuf, extension: &str) -> PathBuf {
        if path.extension().is_none() {
            path.set_extension(extension);
        }
        path
    }

    pub fn new() -> Self {
        Self {
            temp_fit: None,
            temp_fit_revision: 0,
            stored_fits: Vec::new(),
            settings: FitSettings::default(),
            calibration: Calibration::default(),
            sort_state: SortState::default(),
            palette: FitPaletteDefaults::default(),
            pending_modify_fit: None,
            pending_refit_all: false,
        }
    }

    pub fn take_pending_modify_fit(&mut self) -> Option<usize> {
        self.pending_modify_fit.take()
    }

    pub(crate) fn replace_temp_fit(&mut self, fit: Option<Fitter>) {
        self.temp_fit_revision = self.temp_fit_revision.wrapping_add(1);
        self.temp_fit = fit;
        self.apply_visibility_settings();
    }

    pub fn take_pending_refit_all(&mut self) -> bool {
        std::mem::take(&mut self.pending_refit_all)
    }

    pub fn store_temp_fit(&mut self) {
        if !self.temp_fit_is_storable() {
            return;
        }
        self.store_temp_fit_unchecked();
    }

    pub fn temp_fit_is_storable(&self) -> bool {
        self.temp_fit.as_ref().is_some_and(|fit| {
            fit.last_fit_error.is_none() && matches!(fit.fit_result, Some(FitResult::Gaussian(_)))
        })
    }

    fn store_temp_fit_unchecked(&mut self) {
        if let Some(mut temp_fit) = self.temp_fit.take() {
            self.temp_fit_revision = self.temp_fit_revision.wrapping_add(1);
            temp_fit.compact_display_data();
            temp_fit.apply_line_defaults(
                &self.palette.stored_background,
                &self.palette.stored_composition,
                &self.palette.stored_decomposition,
            );

            // remove Temp Fit from name
            let name = temp_fit.name.clone();
            let name = name.replace("Temp Fit", &format!("fit {}", self.stored_fits.len()));

            temp_fit.set_name(name);

            self.stored_fits.push(temp_fit);
            self.apply_visibility_settings();
        }
    }

    pub fn set_log(&mut self, log_y: bool, log_x: bool) {
        if let Some(temp_fit) = &mut self.temp_fit {
            temp_fit.set_log(log_y, log_x);
        }

        for fit in &mut self.stored_fits {
            fit.set_log(log_y, log_x);
        }
    }

    pub fn style_temporary_fit(&self, fit: &mut Fitter) {
        fit.apply_line_defaults(
            &self.palette.temporary_background,
            &self.palette.temporary_composition,
            &self.palette.temporary_decomposition,
        );
    }

    pub fn apply_generation_defaults(
        &mut self,
        defaults: &FitSettings,
        palette: &FitPaletteDefaults,
    ) {
        let calibrated = self.settings.calibrated;
        let calibration_mode = self.settings.calibration_mode;
        self.settings = defaults.clone();
        self.settings.calibrated = calibrated;
        self.settings.calibration_mode = calibration_mode;
        self.palette = palette.clone();

        if let Some(temp_fit) = &mut self.temp_fit {
            temp_fit.apply_line_defaults(
                &palette.temporary_background,
                &palette.temporary_composition,
                &palette.temporary_decomposition,
            );
        }
        for fit in &mut self.stored_fits {
            fit.apply_line_defaults(
                &palette.stored_background,
                &palette.stored_composition,
                &palette.stored_decomposition,
            );
        }
        self.apply_visibility_settings();
    }

    pub fn set_stored_fits_background_color(&mut self, color: egui::Color32) {
        for fit in &mut self.stored_fits {
            fit.background_line.color = color;
        }
    }

    pub fn set_stored_fits_composition_color(&mut self, color: egui::Color32) {
        for fit in &mut self.stored_fits {
            fit.composition_line.color = color;
        }
    }

    pub fn set_stored_fits_decomposition_color(&mut self, color: egui::Color32) {
        for fit in &mut self.stored_fits {
            for line in &mut fit.decomposition_lines {
                line.color = color;
            }
        }
    }

    pub fn update_visibility(&mut self) {
        if let Some(temp_fit) = &mut self.temp_fit {
            temp_fit.show_decomposition(
                self.settings.show_temp_fit && self.settings.show_decomposition,
            );
            temp_fit
                .show_composition(self.settings.show_temp_fit && self.settings.show_composition);
            temp_fit.show_background(self.settings.show_temp_fit && self.settings.show_background);
        }

        for fit in &mut self.stored_fits {
            fit.show_decomposition(self.settings.show_decomposition);
            fit.show_composition(self.settings.show_composition);
            fit.show_background(self.settings.show_background);
        }
    }

    pub fn apply_visibility_settings(&mut self) {
        self.update_visibility();
    }

    fn save_to_file(&self) {
        if let Some(path) = FileDialog::new()
            .add_filter("JSON", &["json"])
            .set_file_name("spectrix_fits.json")
            .save_file()
        {
            let path = Self::ensure_extension_if_missing(path, "json");
            let file = File::create(path);
            match file {
                Ok(mut file) => match serde_json::to_string(self) {
                    Ok(json) => {
                        if let Err(e) = file.write_all(json.as_bytes()) {
                            log::error!("Failed to write fits file: {e}");
                        }
                    }
                    Err(e) => {
                        log::error!("Failed to serialize fits: {e}");
                    }
                },
                Err(e) => {
                    log::error!("Error creating file: {e:?}");
                }
            }
        }
    }

    fn sync_calibration_values(&mut self) {
        for fit in &mut self.stored_fits {
            fit.sync_calibration_values(&self.calibration);
        }

        if let Some(temp_fit) = &mut self.temp_fit {
            temp_fit.sync_calibration_values(&self.calibration);
        }
    }

    fn apply_loaded_fits(&mut self, mut loaded_fits: Self) {
        for fit in &mut loaded_fits.stored_fits {
            fit.compact_display_data();
        }
        if let Some(temp_fit) = &mut loaded_fits.temp_fit {
            temp_fit.compact_display_data();
        }
        self.settings = loaded_fits.settings;
        self.calibration = loaded_fits.calibration;
        self.sort_state = loaded_fits.sort_state;
        self.stored_fits.extend(loaded_fits.stored_fits);
        self.replace_temp_fit(loaded_fits.temp_fit);
        self.sync_calibration_values();
        self.apply_visibility_settings();
    }

    fn load_from_file(&mut self) {
        if let Some(path) = FileDialog::new().add_filter("JSON", &["json"]).pick_file() {
            match std::fs::read_to_string(&path) {
                Ok(contents) => match serde_json::from_str::<Self>(&contents) {
                    Ok(loaded_fits) => {
                        self.apply_loaded_fits(loaded_fits);
                    }
                    Err(e) => {
                        log::error!("Failed to deserialize fits from {}: {e}", path.display());
                    }
                },
                Err(e) => {
                    log::error!("Error reading file: {e:?}");
                }
            }
        }
    }

    pub fn calibrate_stored_fits(&mut self, linear: bool, quadratic: bool) {
        let mut mean = Vec::new();
        let mut energy = Vec::new();

        for fit in &mut self.stored_fits {
            // get the energy valye from the
            let fit_result = fit.fit_result.clone();

            if let Some(FitResult::Gaussian(gauss)) = fit_result {
                let cal_data = gauss.get_calibration_data();
                for (fit_mean, _fit_mean_unc, fit_energy, _fit_energy_unc) in cal_data {
                    mean.push(fit_mean);
                    energy.push(fit_energy);
                }
            }
        }

        let unique_mean_count = unique_value_count(&mean);

        if linear {
            // make sure there are at least 2 points to fit a linear
            if mean.len() < 2 || energy.len() < 2 || unique_mean_count < 2 {
                log::error!("Not enough points to fit a linear. Need at least 2 points.");
                return;
            }

            if let Some(candidate) = exact_linear_calibration(&mean, &energy) {
                self.calibration = candidate;
            } else {
                // Fit a linear model when more than two points are available.
                let mut fitter = LinearFitter::new(Data {
                    x: mean.clone(),
                    y: energy.clone(),
                });

                match fitter.fit() {
                    Ok(_) => {
                        let candidate = Calibration {
                            a: crate::fitter::common::Value {
                                value: 0.0,
                                uncertainty: 0.0,
                            },
                            b: crate::fitter::common::Value {
                                value: fitter.paramaters.slope.value.unwrap_or(1.0),
                                uncertainty: fitter.paramaters.slope.uncertainty.unwrap_or(0.0),
                            },
                            c: crate::fitter::common::Value {
                                value: fitter.paramaters.intercept.value.unwrap_or(0.0),
                                uncertainty: fitter.paramaters.intercept.uncertainty.unwrap_or(0.0),
                            },
                            cov: None,
                            uncertainty_available: true,
                        };

                        if !candidate.coefficients_are_finite() {
                            log::error!("Calibration fit produced invalid linear coefficients.");
                            return;
                        }

                        self.calibration = candidate;
                    }
                    Err(e) => {
                        log::error!("Calibration fit failed: {e:?}");
                    }
                }
            }
        }

        if quadratic {
            // make sure there are at least 3 points to fit a quadratic
            if mean.len() < 3 || energy.len() < 3 || unique_mean_count < 3 {
                log::error!(
                    "Not enough points to fit a quadratic. Need at least 3 distinct points."
                );
                return;
            }

            let mut fitter = quadratic::QuadraticFitter::new(Data {
                x: mean.clone(),
                y: energy.clone(),
            });

            match fitter.fit() {
                Ok(_) => {
                    let candidate = Calibration {
                        a: crate::fitter::common::Value {
                            value: fitter.paramaters.a.value.unwrap_or(0.0),
                            uncertainty: fitter.paramaters.a.uncertainty.unwrap_or(0.0),
                        },
                        b: crate::fitter::common::Value {
                            value: fitter.paramaters.b.value.unwrap_or(1.0),
                            uncertainty: fitter.paramaters.b.uncertainty.unwrap_or(0.0),
                        },
                        c: crate::fitter::common::Value {
                            value: fitter.paramaters.c.value.unwrap_or(0.0),
                            uncertainty: fitter.paramaters.c.uncertainty.unwrap_or(0.0),
                        },
                        cov: fitter.covar,
                        uncertainty_available: true,
                    };

                    if !candidate.coefficients_are_finite() {
                        log::error!("Calibration fit produced invalid quadratic coefficients.");
                        return;
                    }

                    let x_min = mean.iter().copied().fold(f64::INFINITY, f64::min);
                    let x_max = mean.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    if !candidate.is_display_safe_on((x_min, x_max)) {
                        log::error!(
                            "Quadratic calibration is not monotonic over the fitted calibration points. Check the assigned energies and try again."
                        );
                        return;
                    }

                    self.calibration = candidate;
                }
                Err(e) => {
                    log::error!("Calibration fit failed: {e:?}");
                }
            }
        }

        if !self.calibration.coefficients_are_finite() {
            log::error!("Calibration contains invalid coefficients and was not applied.");
            return;
        }

        // Apply the calibration to all stored fits and temp fit
        for fit in &mut self.stored_fits {
            fit.calibrate(&self.calibration);
        }

        if let Some(temp_fit) = &mut self.temp_fit {
            temp_fit.calibrate(&self.calibration);
        }
    }

    fn calibration_plot_points(&self) -> Vec<CalibrationPlotPoint> {
        let mut points = Vec::new();

        for fit in &self.stored_fits {
            if let Some(FitResult::Gaussian(gauss)) = &fit.fit_result {
                points.extend(gauss.fit_result.iter().enumerate().filter_map(
                    |(peak_index, params)| {
                        let (
                            Some(energy),
                            Some(energy_uncertainty),
                            Some(mean),
                            Some(mean_uncertainty),
                        ) = (
                            params.energy.value,
                            params.energy.uncertainty,
                            params.mean.value,
                            params.mean.uncertainty,
                        )
                        else {
                            return None;
                        };

                        (energy != -1.0
                            && mean.is_finite()
                            && mean_uncertainty.is_finite()
                            && energy.is_finite()
                            && energy_uncertainty.is_finite())
                        .then(|| CalibrationPlotPoint {
                            mean,
                            mean_uncertainty: mean_uncertainty.max(0.0),
                            energy,
                            energy_uncertainty: energy_uncertainty.max(0.0),
                            uuid: params.uuid,
                            fit_name: fit.name.clone(),
                            peak_index,
                        })
                    },
                ));
            }
        }

        points.sort_by(|a, b| a.mean.total_cmp(&b.mean));
        points
    }

    fn calibration_curve_bounds(points: &[CalibrationPlotPoint]) -> ((f64, f64), (f64, f64)) {
        let x_min = points
            .iter()
            .map(|point| point.mean - point.mean_uncertainty.max(0.0))
            .fold(f64::INFINITY, f64::min);
        let x_max = points
            .iter()
            .map(|point| point.mean + point.mean_uncertainty.max(0.0))
            .fold(f64::NEG_INFINITY, f64::max);

        let y_min = points
            .iter()
            .map(|point| point.energy - point.energy_uncertainty.max(0.0))
            .fold(f64::INFINITY, f64::min);
        let y_max = points
            .iter()
            .map(|point| point.energy + point.energy_uncertainty.max(0.0))
            .fold(f64::NEG_INFINITY, f64::max);

        (padded_range(x_min, x_max), padded_range(y_min, y_max))
    }

    fn calibration_fit_series(
        &self,
        x_bounds: (f64, f64),
        samples: usize,
    ) -> (Vec<[f64; 2]>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let sample_count = samples.max(2);
        let step = (x_bounds.1 - x_bounds.0) / (sample_count.saturating_sub(1) as f64);

        let mut line_points = Vec::with_capacity(sample_count);
        let mut band_x = Vec::with_capacity(sample_count);
        let mut band_lower = Vec::with_capacity(sample_count);
        let mut band_upper = Vec::with_capacity(sample_count);

        for idx in 0..sample_count {
            let x = if idx + 1 == sample_count {
                x_bounds.1
            } else {
                x_bounds.0 + step * idx as f64
            };
            let Some(y) = self.calibration.calibrate_checked(x) else {
                continue;
            };
            let Some(uncertainty) = self.calibration.curve_uncertainty_checked(x) else {
                continue;
            };

            let lower = y - uncertainty;
            let upper = y + uncertainty;
            if !lower.is_finite() || !upper.is_finite() {
                continue;
            }

            line_points.push([x, y]);
            band_x.push(x);
            band_lower.push(lower);
            band_upper.push(upper);
        }

        (line_points, band_x, band_lower, band_upper)
    }

    fn energy_calibration_plots_ui(&self, ui: &mut egui::Ui, histogram_range: (f64, f64)) {
        let points = self.calibration_plot_points();
        if points.is_empty() {
            return;
        }

        let plot_width = ui.available_width().max(1.0);
        let point_color = Color32::from_rgb(210, 90, 90);
        let fit_color = Color32::from_rgb(70, 140, 235);
        let calibration_plot_id = ui.id().with("energy_calibration_plot");
        let calibration_band_id = ui.id().with("energy_calibration_fit_band");
        let calibration_line_id = ui.id().with("energy_calibration_fit_line");
        let residual_plot_id = ui.id().with("energy_calibration_residuals_plot");

        let (point_x_bounds, y_bounds) = Self::calibration_curve_bounds(&points);
        let fit_x_bounds = if histogram_range.0.is_finite()
            && histogram_range.1.is_finite()
            && histogram_range.0 != histogram_range.1
        {
            padded_range(
                histogram_range.0.min(histogram_range.1),
                histogram_range.0.max(histogram_range.1),
            )
        } else {
            point_x_bounds
        };
        let (fit_line, band_x, band_lower, band_upper) =
            self.calibration_fit_series(fit_x_bounds, 256);

        let x_span = (fit_x_bounds.1 - fit_x_bounds.0).abs().max(1.0);
        let y_span = (y_bounds.1 - y_bounds.0).abs().max(1.0);
        let x_cap_half_width = x_span * 0.01;
        let y_cap_half_height = y_span * 0.015;
        let uuid_label_offset = y_span * 0.02;

        ui.label("Energy Calibration");
        ui.label(format!(
            "Calibration curve shown over histogram range [{:.2}, {:.2}]",
            histogram_range.0.min(histogram_range.1),
            histogram_range.0.max(histogram_range.1)
        ));
        if !self.calibration.is_display_safe_on(histogram_range)
            && let Some(turning_point) = self.calibration.turning_point()
            && turning_point >= histogram_range.0.min(histogram_range.1)
            && turning_point <= histogram_range.0.max(histogram_range.1)
        {
            ui.colored_label(
                Color32::from_rgb(200, 120, 0),
                format!(
                    "Calibration turns over near x = {turning_point:.2} inside the histogram range."
                ),
            );
        }

        Plot::new(calibration_plot_id)
            .width(plot_width)
            .height(240.0)
            .show_x(true)
            .show_y(true)
            .allow_zoom(false)
            .allow_scroll(false)
            .allow_drag(false)
            .allow_boxed_zoom(false)
            .x_axis_label("Fit Mean")
            .y_axis_label("Energy")
            .label_formatter(|hover| {
                let (name, value) = match hover {
                    egui_plot::HoverPosition::NearDataPoint {
                        plot_name,
                        position,
                        ..
                    } => (*plot_name, *position),
                    egui_plot::HoverPosition::Elsewhere { position } => ("", *position),
                };
                if name.is_empty() {
                    Some(format!("x: {:.3}\ny: {:.3}", value.x, value.y))
                } else {
                    Some(name.to_owned())
                }
            })
            .show(ui, |plot_ui| {
                if band_x.len() >= 2 {
                    plot_ui.add(
                        FilledArea::new("Calibration Fit Band", &band_x, &band_lower, &band_upper)
                            .allow_hover(false)
                            .fill_color(fit_color.linear_multiply(0.18))
                            .id(calibration_band_id),
                    );
                }

                if fit_line.len() >= 2 {
                    plot_ui.line(
                        Line::new("Calibration Fit", fit_line.clone())
                            .allow_hover(false)
                            .color(fit_color)
                            .width(2.0)
                            .id(calibration_line_id),
                    );
                }

                for point in &points {
                    let uuid_line = if point.uuid != 0 {
                        format!("UUID: {}", point.uuid)
                    } else {
                        "UUID: —".to_owned()
                    };
                    let hover_label = format!(
                        "{}\nPeak: {}\n{}\nMean: {}\nEnergy: {}",
                        point.fit_name,
                        point.peak_index,
                        uuid_line,
                        fmt_value_uncertainty(point.mean, point.mean_uncertainty, 3),
                        fmt_value_uncertainty(point.energy, point.energy_uncertainty, 3),
                    );

                    plot_ui.points(
                        Points::new(hover_label, vec![[point.mean, point.energy]])
                            .shape(MarkerShape::Circle)
                            .color(point_color)
                            .filled(true)
                            .radius(4.0),
                    );
                }

                for (index, point) in points.iter().enumerate() {
                    draw_cross_error_bars(
                        plot_ui,
                        CrossErrorBars {
                            id_prefix: "energy_calibration",
                            index,
                            x: point.mean,
                            y: point.energy,
                            x_uncertainty: point.mean_uncertainty,
                            y_uncertainty: point.energy_uncertainty,
                            x_cap_half_width,
                            y_cap_half_height,
                            color: point_color,
                        },
                    );
                }

                for (index, point) in points.iter().enumerate() {
                    if point.uuid == 0 {
                        continue;
                    }

                    plot_ui.text(
                        Text::new(
                            format!("energy_calibration_uuid_{index}"),
                            [
                                point.mean,
                                point.energy
                                    + point.energy_uncertainty.max(0.0)
                                    + uuid_label_offset,
                            ]
                            .into(),
                            point.uuid.to_string(),
                        )
                        .anchor(Align2::CENTER_BOTTOM)
                        .color(point_color)
                        .allow_hover(false),
                    );
                }
            });

        ui.add_space(8.0);

        let residual_data: Vec<(CalibrationPlotPoint, [f64; 2], f64, f64, f64)> = points
            .iter()
            .filter_map(|point| {
                let fit_value = self.calibration.calibrate_checked(point.mean)?;
                let derivative = self.calibration.derivative_checked(point.mean)?;
                let residual = point.energy - fit_value;
                let residual_uncertainty = point
                    .energy_uncertainty
                    .hypot(derivative * point.mean_uncertainty);

                (residual.is_finite() && residual_uncertainty.is_finite()).then_some((
                    point.clone(),
                    [point.mean, residual],
                    point.mean_uncertainty,
                    residual_uncertainty,
                    fit_value,
                ))
            })
            .collect();

        if residual_data.is_empty() {
            ui.separator();
            return;
        }

        let residual_y_min = residual_data
            .iter()
            .map(|(_, point, _, residual_uncertainty, _)| point[1] - residual_uncertainty.max(0.0))
            .fold(f64::INFINITY, f64::min);
        let residual_y_max = residual_data
            .iter()
            .map(|(_, point, _, residual_uncertainty, _)| point[1] + residual_uncertainty.max(0.0))
            .fold(f64::NEG_INFINITY, f64::max);
        let residual_y_bounds = padded_range(residual_y_min, residual_y_max);
        let residual_y_span = (residual_y_bounds.1 - residual_y_bounds.0).abs().max(1.0);
        let residual_y_cap_half_height = residual_y_span * 0.015;

        Plot::new(residual_plot_id)
            .width(plot_width)
            .height(150.0)
            .show_x(true)
            .show_y(true)
            .allow_zoom(false)
            .allow_scroll(false)
            .allow_drag(false)
            .allow_boxed_zoom(false)
            .x_axis_label("Fit Mean")
            .y_axis_label("Residual")
            .label_formatter(|hover| {
                let (name, value) = match hover {
                    egui_plot::HoverPosition::NearDataPoint {
                        plot_name,
                        position,
                        ..
                    } => (*plot_name, *position),
                    egui_plot::HoverPosition::Elsewhere { position } => ("", *position),
                };
                if name.is_empty() {
                    Some(format!("x: {:.3}\ny: {:.3}", value.x, value.y))
                } else {
                    Some(name.to_owned())
                }
            })
            .show(ui, |plot_ui| {
                draw_plot_segment(
                    plot_ui,
                    ("energy_calibration_residuals", "zero_line"),
                    vec![[fit_x_bounds.0, 0.0], [fit_x_bounds.1, 0.0]],
                    Color32::GRAY,
                    1.0,
                );

                for (source_point, point, _, residual_uncertainty, fit_value) in &residual_data {
                    let uuid_line = if source_point.uuid != 0 {
                        format!("UUID: {}", source_point.uuid)
                    } else {
                        "UUID: —".to_owned()
                    };
                    let hover_label = format!(
                        "{}\nPeak: {}\n{}\nMean: {}\nResidual: {}\nEnergy: {}\nFit Energy: {:.3}",
                        source_point.fit_name,
                        source_point.peak_index,
                        uuid_line,
                        fmt_value_uncertainty(source_point.mean, source_point.mean_uncertainty, 3),
                        fmt_value_uncertainty(point[1], *residual_uncertainty, 3),
                        fmt_value_uncertainty(
                            source_point.energy,
                            source_point.energy_uncertainty,
                            3
                        ),
                        fit_value,
                    );

                    plot_ui.points(
                        Points::new(hover_label, vec![*point])
                            .shape(MarkerShape::Circle)
                            .color(point_color)
                            .filled(true)
                            .radius(4.0),
                    );
                }

                for (index, (_, point, mean_uncertainty, residual_uncertainty, _)) in
                    residual_data.iter().enumerate()
                {
                    draw_cross_error_bars(
                        plot_ui,
                        CrossErrorBars {
                            id_prefix: "energy_calibration_residuals",
                            index,
                            x: point[0],
                            y: point[1],
                            x_uncertainty: *mean_uncertainty,
                            y_uncertainty: *residual_uncertainty,
                            x_cap_half_width,
                            y_cap_half_height: residual_y_cap_half_height,
                            color: point_color,
                        },
                    );
                }
            });

        ui.separator();
    }

    pub fn save_and_load_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {

            if !self.stored_fits.is_empty() {
                if ui
                        .button("Refit")
                        .on_hover_text(
                            "Refit stored fits on current data in a worker. Results replace the stored collection together; current temporary markers and fits are preserved. Editing the inputs while it runs discards the outdated results.",
                        )
                        .clicked()
                {
                    self.pending_refit_all = true;
                }

                ui.separator();
            }

            if ui
                .button("Save Fits")
                .on_hover_text(
                    "Save Spectrix fits as .json for restoring and continuing work inside Spectrix.\n\
                    Best option for normal Spectrix workflows.",
                )
                .clicked()
            {
                self.save_to_file();
            }

            ui.separator();

            if ui
                .button("Load Fits")
                .on_hover_text(
                    "Load Spectrix .json fits saved with 'Save Fits'.\n\
                    Use this to restore fits for continued work in Spectrix.",
                )
                .clicked()
            {
                self.load_from_file();
            }
        });
    }

    pub fn remove_temp_fits(&mut self) {
        self.replace_temp_fit(None);
    }

    pub fn draw(
        &mut self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        histogram_bins: &[u64],
        histogram_range: (f64, f64),
        histogram_bin_width: f64,
        calibration: Option<&Calibration>,
    ) {
        let calibrated = calibration.is_some();
        let uuid_draw_options = UuidDrawOptions {
            calibrate: calibrated,
            log_x: false,
            log_y: false,
            label_size: self.settings.uuid_label_size,
            label_lift: self.settings.uuid_label_lift,
            draw_label_guide: self.settings.uuid_label_guides,
        };
        let histogram_draw_context = HistogramDrawContext {
            bins: histogram_bins,
            range: histogram_range,
            bin_width: histogram_bin_width,
        };
        if self.settings.show_temp_fit
            && let Some(temp_fit) = &self.temp_fit
        {
            let visible = temp_fit.overlaps_visible_x(plot_ui, calibration);
            if visible {
                temp_fit.draw(plot_ui, calibration, self.settings.show_fit_lines_area);
            }

            if visible && let Some(FitResult::Gaussian(gauss)) = &temp_fit.fit_result {
                gauss.draw_uuid(
                    plot_ui,
                    UuidDrawOptions {
                        log_x: temp_fit.composition_line.log_x,
                        log_y: temp_fit.composition_line.log_y,
                        ..uuid_draw_options
                    },
                    histogram_draw_context,
                );
            }
        }

        for fit in &self.stored_fits {
            if !fit.overlaps_visible_x(plot_ui, calibration) {
                continue;
            }
            fit.draw(plot_ui, calibration, self.settings.show_fit_lines_area);

            // put the uuid above each peak if it is not 0
            if let Some(FitResult::Gaussian(gauss)) = &fit.fit_result {
                gauss.draw_uuid(
                    plot_ui,
                    UuidDrawOptions {
                        log_x: fit.composition_line.log_x,
                        log_y: fit.composition_line.log_y,
                        ..uuid_draw_options
                    },
                    histogram_draw_context,
                );
            }
        }
    }

    pub fn has_uuid_labels(&self) -> bool {
        let fit_has_uuid_labels = |fit: &Fitter| {
            if let Some(FitResult::Gaussian(gauss)) = &fit.fit_result {
                gauss.fit_result.iter().any(|params| params.uuid != 0)
            } else {
                false
            }
        };

        (self.settings.show_temp_fit && self.temp_fit.as_ref().is_some_and(fit_has_uuid_labels))
            || self.stored_fits.iter().any(fit_has_uuid_labels)
    }

    fn manual_peak_table_ui(
        &self,
        ui: &mut egui::Ui,
        markers: &mut FitMarkers,
        bin_width: f64,
        equal_sigma: bool,
    ) -> bool {
        markers.ensure_peak_ids();
        let snapshot = self
            .temp_fit
            .as_ref()
            .and_then(|fit| match &fit.fit_result {
                Some(FitResult::Gaussian(gaussian)) => gaussian.native_result.as_ref(),
                None => None,
            })
            .map(|result| {
                let initial = result
                    .peak_seeds
                    .iter()
                    .enumerate()
                    .map(|(index, seed)| {
                        let definition = |suffix: &str| {
                            result
                                .initial_parameters
                                .iter()
                                .find(|parameter| parameter.name == format!("g{index}_{suffix}"))
                        };
                        let center = definition("center");
                        let sigma = definition("sigma");
                        let height = definition("height");
                        let center_bounds =
                            center.map_or([seed.center, seed.center], |parameter| {
                                [
                                    bound_value(parameter.bounds.lower, seed.center),
                                    bound_value(parameter.bounds.upper, seed.center),
                                ]
                            });
                        let sigma_bounds = sigma.map_or([seed.sigma, seed.sigma], |parameter| {
                            [
                                bound_value(parameter.bounds.lower, seed.sigma),
                                bound_value(parameter.bounds.upper, seed.sigma),
                            ]
                        });
                        let response = spectrix_fitting::evaluate_manual_peak(
                            ManualPeakSeed {
                                center: seed.center,
                                sigma: seed.sigma,
                                amplitude: 1.0,
                            },
                            seed.center,
                            bin_width,
                        )
                        .unwrap_or(0.0);
                        let net_height =
                            height.map_or(seed.amplitude * response, |value| value.initial);
                        let height_bounds = height.map_or([net_height, net_height], |parameter| {
                            [
                                bound_value(parameter.bounds.lower, net_height),
                                bound_value(parameter.bounds.upper, net_height),
                            ]
                        });
                        ManualInitialTableRow {
                            id: markers
                                .peak_markers
                                .get(index)
                                .map_or(index as u64 + 1, |guess| guess.id),
                            seed: *seed,
                            bounds: ManualPeakBounds {
                                center: center_bounds,
                                sigma: sigma_bounds,
                                net_height: height_bounds,
                            },
                            net_height,
                            source: "Fit input".to_owned(),
                            valid: true,
                        }
                    })
                    .collect::<Vec<_>>();
                let fitted = (0..result.peak_seeds.len())
                    .filter_map(|index| {
                        let estimate =
                            |suffix: &str| {
                                result.fit.parameters.iter().find(|parameter| {
                                    parameter.name == format!("g{index}_{suffix}")
                                })
                            };
                        let pair = |suffix: &str| {
                            estimate(suffix)
                                .map(|parameter| (parameter.value, parameter.standard_error))
                                .unwrap_or((f64::NAN, None))
                        };
                        estimate("center")?;
                        Some(ManualTempTableRow {
                            center: pair("center"),
                            fwhm: pair("fwhm"),
                            height: pair("height"),
                            area: pair("area"),
                            amplitude: pair("amplitude"),
                            sigma: pair("sigma"),
                        })
                    })
                    .collect::<Vec<_>>();
                (initial, fitted)
            });

        if markers.peak_markers.is_empty()
            && snapshot
                .as_ref()
                .is_none_or(|(initial, _)| initial.is_empty())
        {
            return false;
        }

        ui.separator();
        ui.horizontal_wrapped(|ui| {
            ui.strong("Initial / Temporary Peak Parameters");
            if self.settings.calibrated {
                ui.label("Initial center and FWHM values are shown in calibrated units.");
            }
        });
        if let Some(error) = &markers.estimate_error {
            ui.colored_label(Color32::YELLOW, error);
        }

        let mut changed = false;
        let mut reset_all = false;
        let mut reestimate = Vec::new();
        let mut shared_fwhm = None;
        let mut shared_fwhm_bounds = None;
        let show_initial_parameters = self.settings.show_initial_parameters;
        let calibration = self.settings.calibrated.then_some(&self.calibration);
        let raw_initial_decimals = decimal_places_for_resolution(bin_width);
        if snapshot.is_none() {
            reset_all = ui.button("Reset All").clicked();
        }

        egui::ScrollArea::horizontal()
            .id_salt("manual_peak_results_scroll")
            .show(ui, |ui| {
                egui::Grid::new("manual_peak_results_table")
                    .striped(true)
                    .show(ui, |ui| {
                        let group_headings = if show_initial_parameters {
                            vec![
                                "Peak",
                                "Initial",
                                "",
                                "",
                                "Fitted",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "Source / status",
                                "Options",
                            ]
                        } else {
                            vec![
                                "Peak",
                                "Fitted",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "Source / status",
                                "Options",
                            ]
                        };
                        for heading in group_headings {
                            ui.strong(heading);
                        }
                        ui.end_row();

                        let parameter_headings = if show_initial_parameters {
                            vec![
                                "",
                                "Center",
                                "FWHM",
                                "Height",
                                "Center",
                                "FWHM",
                                "Height",
                                "Area",
                                "Amplitude",
                                "Sigma",
                                "",
                                "",
                            ]
                        } else {
                            vec![
                                "",
                                "Center",
                                "FWHM",
                                "Height",
                                "Area",
                                "Amplitude",
                                "Sigma",
                                "",
                                "",
                            ]
                        };
                        for heading in parameter_headings {
                            ui.strong(heading);
                        }
                        ui.end_row();

                        if let Some((initial_rows, fitted_rows)) = &snapshot {
                            for (index, initial) in initial_rows.iter().enumerate() {
                                ui.label(format!("#{}", initial.id));
                                let fitted = fitted_rows.get(index);
                                if show_initial_parameters {
                                    let center_range = calibration
                                        .and_then(|calibration| {
                                            calibrated_center_range(
                                                calibration,
                                                initial.bounds.center[0],
                                                initial.seed.center,
                                                initial.bounds.center[1],
                                            )
                                        })
                                        .unwrap_or((
                                            initial.bounds.center[0],
                                            initial.seed.center,
                                            initial.bounds.center[1],
                                        ));
                                    let center_decimals = calibration
                                        .and_then(|calibration| {
                                            calibration
                                                .calibrate_checked(initial.seed.center + bin_width)
                                                .zip(
                                                    calibration
                                                        .calibrate_checked(initial.seed.center),
                                                )
                                                .map(|(next, current)| (next - current).abs())
                                        })
                                        .map(decimal_places_for_resolution)
                                        .unwrap_or(raw_initial_decimals);
                                    range_label(
                                        ui,
                                        center_range.0,
                                        center_range.1,
                                        center_range.2,
                                        center_decimals,
                                    );
                                    let fwhm_range = calibration
                                        .and_then(|calibration| {
                                            calibrated_fwhm_range(
                                                calibration,
                                                initial.seed.center,
                                                initial.bounds.sigma[0] * MANUAL_FWHM_FACTOR,
                                                initial.seed.sigma * MANUAL_FWHM_FACTOR,
                                                initial.bounds.sigma[1] * MANUAL_FWHM_FACTOR,
                                            )
                                        })
                                        .unwrap_or((
                                            initial.bounds.sigma[0] * MANUAL_FWHM_FACTOR,
                                            initial.seed.sigma * MANUAL_FWHM_FACTOR,
                                            initial.bounds.sigma[1] * MANUAL_FWHM_FACTOR,
                                        ));
                                    let fwhm_decimals = calibration
                                        .and_then(|calibration| {
                                            calibration
                                                .derivative_checked(initial.seed.center)
                                                .map(|derivative| derivative.abs() * bin_width)
                                        })
                                        .map(decimal_places_for_resolution)
                                        .unwrap_or(raw_initial_decimals);
                                    range_label(
                                        ui,
                                        fwhm_range.0,
                                        fwhm_range.1,
                                        fwhm_range.2,
                                        fwhm_decimals,
                                    );
                                    range_label(
                                        ui,
                                        initial.bounds.net_height[0],
                                        initial.net_height,
                                        initial.bounds.net_height[1],
                                        0,
                                    );
                                }
                                if let Some(fitted) = fitted {
                                    let center = calibration.map_or(fitted.center, |calibration| {
                                        calibrated_center_measurement(
                                            calibration,
                                            fitted.center.0,
                                            fitted.center.1,
                                        )
                                    });
                                    let fwhm = calibration.map_or(fitted.fwhm, |calibration| {
                                        calibrated_width_measurement(
                                            calibration,
                                            fitted.center.0,
                                            fitted.fwhm.0,
                                            fitted.fwhm.1,
                                        )
                                    });
                                    let sigma = calibration.map_or(fitted.sigma, |calibration| {
                                        calibrated_width_measurement(
                                            calibration,
                                            fitted.center.0,
                                            fitted.sigma.0,
                                            fitted.sigma.1,
                                        )
                                    });
                                    for measurement in [
                                        center,
                                        fwhm,
                                        fitted.height,
                                        fitted.area,
                                        fitted.amplitude,
                                        sigma,
                                    ] {
                                        fit_measurement_label(
                                            ui,
                                            Some(measurement.0),
                                            measurement.1,
                                        );
                                    }
                                } else {
                                    for _ in 0..6 {
                                        ui.label("—");
                                    }
                                }
                                ui.colored_label(
                                    if initial.valid {
                                        Color32::GREEN
                                    } else {
                                        Color32::RED
                                    },
                                    &initial.source,
                                );
                                ui.label("—");
                                ui.end_row();
                            }
                        } else {
                            let region_positions = markers.get_region_marker_positions();
                            let (region_minimum, region_maximum) = if region_positions.len() == 2 {
                                (
                                    region_positions[0].min(region_positions[1]),
                                    region_positions[0].max(region_positions[1]),
                                )
                            } else {
                                (f64::NEG_INFINITY, f64::INFINITY)
                            };
                            let maximum_width =
                                if region_minimum.is_finite() && region_maximum.is_finite() {
                                    (region_maximum - region_minimum).max(bin_width)
                                } else {
                                    f64::INFINITY
                                };
                            for guess in &mut markers.peak_markers {
                                ui.label(format!("#{}", guess.id));

                                if show_initial_parameters {
                                    let mut center_minimum =
                                        guess.center_min.min(guess.center.x_value);
                                    let mut center = guess.center.x_value;
                                    let mut center_maximum =
                                        guess.center_max.max(guess.center.x_value);
                                    center_minimum = center_minimum.max(region_minimum);
                                    center_maximum = center_maximum.min(region_maximum);
                                    let center_calibrated_decimals =
                                        calibration.map_or(raw_initial_decimals, |calibration| {
                                            decimal_places_for_resolution(
                                                calibration
                                                    .calibrate_checked(center + bin_width)
                                                    .zip(calibration.calibrate_checked(center))
                                                    .map_or(bin_width, |(next, current)| {
                                                        (next - current).abs()
                                                    }),
                                            )
                                        });
                                    let center_changes = match calibration {
                                        Some(calibration) => calibrated_center_range_drag_ui(
                                            ui,
                                            &mut center_minimum,
                                            &mut center,
                                            &mut center_maximum,
                                            bin_width.max(1.0e-9) * 0.1,
                                            (region_minimum, region_maximum),
                                            calibration,
                                            center_calibrated_decimals,
                                        ),
                                        None => range_drag_ui(
                                            ui,
                                            &mut center_minimum,
                                            &mut center,
                                            &mut center_maximum,
                                            bin_width.max(1.0e-9) * 0.1,
                                            region_minimum,
                                            region_maximum,
                                        ),
                                    };
                                    if center_changes.iter().any(|value| *value) {
                                        guess.center_min = center_minimum;
                                        guess.center.x_value = center;
                                        guess.center_max = center_maximum;
                                        guess.hold_current_seed();
                                        changed = true;
                                    }

                                    let mut fwhm_minimum = guess.fwhm_min.min(guess.fwhm);
                                    let mut fwhm = guess.fwhm;
                                    let mut fwhm_maximum = guess.fwhm_max.max(guess.fwhm);
                                    let fwhm_changes = calibration
                                        .and_then(|calibration| {
                                            calibrated_fwhm_range_drag_ui(
                                                ui,
                                                &mut fwhm_minimum,
                                                &mut fwhm,
                                                &mut fwhm_maximum,
                                                bin_width.max(1.0e-9) * 0.05,
                                                (0.5 * bin_width).max(f64::EPSILON),
                                                maximum_width,
                                                guess.center.x_value,
                                                calibration,
                                                decimal_places_for_resolution(
                                                    calibration
                                                        .derivative_checked(guess.center.x_value)
                                                        .map_or(bin_width, |derivative| {
                                                            derivative.abs() * bin_width
                                                        }),
                                                ),
                                            )
                                        })
                                        .unwrap_or_else(|| {
                                            range_drag_ui(
                                                ui,
                                                &mut fwhm_minimum,
                                                &mut fwhm,
                                                &mut fwhm_maximum,
                                                bin_width.max(1.0e-9) * 0.05,
                                                (0.5 * bin_width).max(f64::EPSILON),
                                                maximum_width,
                                            )
                                        });
                                    if fwhm_changes[1] {
                                        guess.set_fwhm_preserving_height(fwhm, bin_width);
                                        shared_fwhm = equal_sigma.then_some(fwhm);
                                    }
                                    if fwhm_changes[0] || fwhm_changes[2] {
                                        guess.fwhm_min = fwhm_minimum;
                                        guess.fwhm_max = fwhm_maximum;
                                        shared_fwhm_bounds =
                                            equal_sigma.then_some([fwhm_minimum, fwhm_maximum]);
                                    }
                                    if fwhm_changes.iter().any(|value| *value) {
                                        guess.hold_current_seed();
                                        changed = true;
                                    }

                                    let mut height_minimum =
                                        guess.net_height_min.min(guess.net_height);
                                    let mut height = guess.net_height;
                                    let mut height_maximum =
                                        guess.net_height_max.max(guess.net_height);
                                    let height_speed = height.abs().max(1.0) * 0.01;
                                    let height_changes = range_drag_ui(
                                        ui,
                                        &mut height_minimum,
                                        &mut height,
                                        &mut height_maximum,
                                        height_speed,
                                        0.0,
                                        f64::INFINITY,
                                    );
                                    if height_changes[1] {
                                        guess.set_net_height(height, bin_width);
                                    }
                                    if height_changes[0] || height_changes[2] {
                                        guess.net_height_min = height_minimum;
                                        guess.net_height_max = height_maximum;
                                    }
                                    if height_changes.iter().any(|value| *value) {
                                        guess.hold_current_seed();
                                        changed = true;
                                    }
                                }

                                for _ in 0..6 {
                                    ui.label("—");
                                }
                                guess.valid = guess.center.x_value.is_finite()
                                    && guess.fwhm.is_finite()
                                    && guess.fwhm > 0.0
                                    && guess.amplitude.is_finite()
                                    && guess.amplitude > 0.0
                                    && guess.bounds_valid();
                                ui.colored_label(
                                    if guess.valid {
                                        Color32::GREEN
                                    } else {
                                        Color32::RED
                                    },
                                    format!(
                                        "W:{} H:{} B:{} · {}",
                                        guess.width_source.label(),
                                        guess.amplitude_source.label(),
                                        guess.bounds_source.label(),
                                        if guess.valid { "valid" } else { "invalid" },
                                    ),
                                );
                                if ui.button("Re-estimate").clicked() {
                                    reestimate.push(guess.id);
                                }
                                ui.end_row();
                            }
                        }
                    });
            });

        if let Some(fwhm) = shared_fwhm {
            for guess in &mut markers.peak_markers {
                guess.set_fwhm_preserving_height(fwhm, bin_width);
                guess.hold_current_seed();
            }
        }
        if let Some([minimum, maximum]) = shared_fwhm_bounds {
            for guess in &mut markers.peak_markers {
                guess.fwhm_min = minimum;
                guess.fwhm_max = maximum;
                guess.hold_current_seed();
            }
        }
        if reset_all {
            for guess in &mut markers.peak_markers {
                guess.reset_estimates();
            }
            changed = true;
        }
        if !reestimate.is_empty() {
            for guess in &mut markers.peak_markers {
                if reestimate.contains(&guess.id) {
                    guess.reset_estimates();
                }
            }
            changed = true;
        }
        if changed {
            markers
                .peak_markers
                .sort_by(|left, right| left.center.x_value.total_cmp(&right.center.x_value));
            markers.estimate_signature = 0;
        }
        changed
    }

    pub fn fit_stats_grid_ui(&mut self, ui: &mut egui::Ui) {
        // Current Initial/Temp rows are rendered together above. This table retains
        // the sortable stored-fit workflow.
        if self.stored_fits.is_empty() {
            return;
        }

        // --- helper row & collectors (local to this fn) ---
        #[derive(Clone)]
        struct Row {
            fit_idx: Option<usize>, // None => Temp
            peak: usize,            // peak index
            mean: (f64, f64),       // (val, unc)
            fwhm: (f64, f64),
            height: (f64, f64),
            area: (f64, f64),
            amplitude: (f64, f64),
            sigma: (f64, f64),
            uuid: usize,
            energy: (f64, f64),
        }

        let pick = |v: Option<f64>, u: Option<f64>| -> (f64, f64) {
            (v.unwrap_or(f64::NAN), u.unwrap_or(f64::NAN))
        };

        let pick_cal = |raw: (Option<f64>, Option<f64>),
                        cal: (Option<f64>, Option<f64>),
                        use_cal: bool|
         -> (f64, f64) {
            if use_cal {
                pick(cal.0, cal.1)
            } else {
                pick(raw.0, raw.1)
            }
        };

        let mut rows: Vec<Row> = Vec::new();
        let calibrated = self.settings.calibrated;

        // Stored fits rows
        for (fi, fit) in self.stored_fits.iter().enumerate() {
            if let Some(super::main_fitter::FitResult::Gaussian(g)) = &fit.fit_result {
                for (i, p) in g.fit_result.iter().enumerate() {
                    rows.push(Row {
                        fit_idx: Some(fi),
                        peak: i,
                        mean: pick_cal(
                            (p.mean.value, p.mean.uncertainty),
                            (p.mean.calibrated_value, p.mean.calibrated_uncertainty),
                            calibrated,
                        ),
                        fwhm: pick_cal(
                            (p.fwhm.value, p.fwhm.uncertainty),
                            (p.fwhm.calibrated_value, p.fwhm.calibrated_uncertainty),
                            calibrated,
                        ),
                        height: pick_cal(
                            (p.height.value, p.height.uncertainty),
                            (p.height.calibrated_value, p.height.calibrated_uncertainty),
                            calibrated,
                        ),
                        area: pick_cal(
                            (p.area.value, p.area.uncertainty),
                            (p.area.calibrated_value, p.area.calibrated_uncertainty),
                            calibrated,
                        ),
                        amplitude: pick_cal(
                            (p.amplitude.value, p.amplitude.uncertainty),
                            (
                                p.amplitude.calibrated_value,
                                p.amplitude.calibrated_uncertainty,
                            ),
                            calibrated,
                        ),
                        sigma: pick_cal(
                            (p.sigma.value, p.sigma.uncertainty),
                            (p.sigma.calibrated_value, p.sigma.calibrated_uncertainty),
                            calibrated,
                        ),
                        uuid: p.uuid,
                        energy: pick(p.energy.value, p.energy.uncertainty),
                    });
                }
            }
        }

        // local snapshot (read-only this frame) + click stash
        let current = self.sort_state;
        let mut new_state = current;

        // NEW: stash a pending deletion
        let mut to_remove: Option<usize> = None;
        let mut to_modify: Option<usize> = None;

        // NEW: sorting key helper
        let key = |r: &Row, col: SortCol| -> f64 {
            let nan_hi = |v: f64| if v.is_nan() { f64::INFINITY } else { v };
            match col {
                SortCol::Fit => r.fit_idx.map(|x| x as f64).unwrap_or(-1.0),
                SortCol::Peak => r.peak as f64,
                SortCol::Mean => nan_hi(r.mean.0),
                SortCol::Fwhm => nan_hi(r.fwhm.0),
                SortCol::Height => nan_hi(r.height.0),
                SortCol::Area => nan_hi(r.area.0),
                SortCol::Amplitude => nan_hi(r.amplitude.0),
                SortCol::Sigma => nan_hi(r.sigma.0),
                SortCol::Uuid => r.uuid as f64,
                SortCol::Energy => nan_hi(r.energy.0),
            }
        };

        let sort_rows = |rows: &mut [Row], state: SortState| {
            rows.sort_by(|a, b| {
                let ka = key(a, state.col);
                let kb = key(b, state.col);
                let ord = ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal);
                if state.asc { ord } else { ord.reverse() }
            });
        };

        let mut uuid_updates: Vec<(Option<usize>, usize, usize)> = Vec::new(); // (fit_idx, peak, new_uuid)
        let mut energy_updates: Vec<(Option<usize>, usize, f64, f64)> = Vec::new(); // (fit_idx, peak, energy, unc)

        ui.horizontal(|ui| {
            ui.strong("Stored Fits");
            if ui
                .button("\u{1F4CB}")
                .on_hover_text(
                    "Copy the currently displayed fit table to the clipboard as comma-delimited text, including values and uncertainties.",
                )
                .clicked()
            {
                let csv_number = |value: f64| {
                    if value.is_finite() {
                        value.to_string()
                    } else {
                        String::new()
                    }
                };

                let mut csv_lines = Vec::with_capacity(rows.len() + 1);
                csv_lines.push(
                    "fit,peak,mean,mean_uncertainty,fwhm,fwhm_uncertainty,height,height_uncertainty,area,area_uncertainty,amplitude,amplitude_uncertainty,sigma,sigma_uncertainty,uuid,energy,energy_uncertainty".to_owned(),
                );

                for row in &rows {
                    let fit_label = row
                        .fit_idx
                        .map_or_else(|| "Temp".to_owned(), |fit_idx| fit_idx.to_string());
                    csv_lines.push(format!(
                        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                        fit_label,
                        row.peak,
                        csv_number(row.mean.0),
                        csv_number(row.mean.1),
                        csv_number(row.fwhm.0),
                        csv_number(row.fwhm.1),
                        csv_number(row.height.0),
                        csv_number(row.height.1),
                        csv_number(row.area.0),
                        csv_number(row.area.1),
                        csv_number(row.amplitude.0),
                        csv_number(row.amplitude.1),
                        csv_number(row.sigma.0),
                        csv_number(row.sigma.1),
                        row.uuid,
                        csv_number(row.energy.0),
                        csv_number(row.energy.1),
                    ));
                }

                ui.ctx().copy_text(csv_lines.join("\n"));
            }
        });

        sort_rows(&mut rows, current);
        TableBuilder::new(ui)
            .id_salt("fit_params_table_compact")
            .striped(true)
            .vscroll(true)
            .min_scrolled_height(120.0)
            .max_scroll_height(480.0)
            // Size each column from its header and visible cells. Unlike fixed widths, these
            // stay compact for ordinary fits while still expanding for longer numeric values.
            .column(Column::auto())
            .column(Column::auto())
            .columns(Column::auto(), 6)
            .column(Column::auto())
            .column(Column::auto())
            .column(Column::auto())
            .header(24.0, |mut header| {
                let mut add_header =
                    |header: &mut egui_extras::TableRow<'_, '_>, label: &str, col: SortCol| {
                        header.col(|ui| {
                            let arrow = if current.col == col {
                                if current.asc { " ⬆" } else { " ⬇" }
                            } else {
                                ""
                            };
                            if ui.button(format!("{label}{arrow}")).clicked() {
                                new_state = if current.col == col {
                                    SortState {
                                        col,
                                        asc: !current.asc,
                                    }
                                } else {
                                    SortState { col, asc: true }
                                };
                            }
                        });
                    };
                add_header(&mut header, "Fit #", SortCol::Fit);
                add_header(&mut header, "Peak #", SortCol::Peak);
                add_header(&mut header, "Mean", SortCol::Mean);
                add_header(&mut header, "FWHM", SortCol::Fwhm);
                add_header(&mut header, "Height", SortCol::Height);
                add_header(&mut header, "Area", SortCol::Area);
                add_header(&mut header, "Amplitude", SortCol::Amplitude);
                add_header(&mut header, "Sigma", SortCol::Sigma);
                add_header(&mut header, "UUID", SortCol::Uuid);
                add_header(&mut header, "Energy", SortCol::Energy);
                header.col(|ui| {
                    ui.label("Options");
                });
            })
            .body(|body| {
                body.rows(24.0, rows.len(), |mut row| {
                    let r = &rows[row.index()];
                    row.col(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(r.fit_idx.map_or_else(
                                || "Temp".to_owned(),
                                |fit_index| fit_index.to_string(),
                            ));
                            if let Some(fit_index) = r.fit_idx
                                && ui.small_button("X").clicked()
                            {
                                to_remove = Some(fit_index);
                            }
                        });
                    });
                    row.col(|ui| {
                        ui.label(r.peak.to_string());
                    });
                    for measurement in [r.mean, r.fwhm, r.height, r.area, r.amplitude, r.sigma] {
                        row.col(|ui| {
                            fit_measurement_label(ui, Some(measurement.0), Some(measurement.1));
                        });
                    }
                    row.col(|ui| {
                        let mut uuid_edit = r.uuid;
                        if ui
                            .add(
                                egui::DragValue::new(&mut uuid_edit)
                                    .speed(1)
                                    .update_while_editing(false),
                            )
                            .changed()
                        {
                            uuid_updates.push((r.fit_idx, r.peak, uuid_edit));
                        }
                    });
                    row.col(|ui| {
                        let mut energy = if r.energy.0.is_finite() {
                            r.energy.0
                        } else {
                            0.0
                        };
                        let mut uncertainty = if r.energy.1.is_finite() {
                            r.energy.1
                        } else {
                            0.0
                        };
                        let mut changed = false;
                        ui.horizontal(|ui| {
                            changed |= ui
                                .add(
                                    egui::DragValue::new(&mut energy)
                                        .speed(0.1)
                                        .update_while_editing(false),
                                )
                                .changed();
                            ui.label("±");
                            changed |= ui
                                .add(
                                    egui::DragValue::new(&mut uncertainty)
                                        .speed(0.1)
                                        .update_while_editing(false),
                                )
                                .changed();
                        });
                        if changed {
                            energy_updates.push((r.fit_idx, r.peak, energy, uncertainty));
                        }
                    });
                    row.col(|ui| {
                        if r.peak != 0 {
                            ui.label("—");
                            return;
                        }
                        if let Some(fit_index) = r.fit_idx {
                            if matches!(
                                self.stored_fits[fit_index].fit_result,
                                Some(super::main_fitter::FitResult::Gaussian(_))
                            ) {
                                ui.menu_button("Options", |ui| {
                                    if ui.button("Modify").clicked() {
                                        to_modify = Some(fit_index);
                                        ui.close();
                                    }
                                    ui.menu_button("Fit Report", |ui| {
                                        egui::ScrollArea::vertical().show(ui, |ui| {
                                            ui.horizontal_wrapped(|ui| {
                                                ui.label(
                                                    self.stored_fits[fit_index].get_fit_report(),
                                                );
                                            });
                                        });
                                    });
                                });
                            } else {
                                ui.label("—");
                            }
                        } else if let Some(temp) = &self.temp_fit {
                            if matches!(
                                temp.fit_result,
                                Some(super::main_fitter::FitResult::Gaussian(_))
                            ) {
                                ui.menu_button("Options", |ui| {
                                    ui.menu_button("Fit Report", |ui| {
                                        egui::ScrollArea::vertical().show(ui, |ui| {
                                            ui.horizontal_wrapped(|ui| {
                                                ui.label(temp.get_fit_report());
                                            });
                                        });
                                    });
                                });
                            } else {
                                ui.label("—");
                            }
                        } else {
                            ui.label("—");
                        }
                    });
                });
            });

        self.sort_state = new_state;

        // Apply UUID changes
        for (fit_idx, peak, new_uuid) in uuid_updates {
            match fit_idx {
                Some(i) => {
                    if let Some(super::main_fitter::FitResult::Gaussian(g)) =
                        &mut self.stored_fits[i].fit_result
                        && let Err(e) = g.update_uuid_for_peak(peak, new_uuid)
                    {
                        eprintln!("UUID update failed: {e}");
                    }
                }
                None => {
                    if let Some(temp) = &mut self.temp_fit
                        && let Some(super::main_fitter::FitResult::Gaussian(g)) =
                            &mut temp.fit_result
                        && let Err(e) = g.update_uuid_for_peak(peak, new_uuid)
                    {
                        eprintln!("UUID update failed: {e}");
                    }
                }
            }
        }
        // Apply Energy changes
        for (fit_idx, peak, e, du) in energy_updates {
            match fit_idx {
                Some(i) => {
                    if let Some(super::main_fitter::FitResult::Gaussian(g)) =
                        &mut self.stored_fits[i].fit_result
                        && let Err(e) = g.update_energy_for_peak(peak, e, du)
                    {
                        eprintln!("Energy update failed: {e}");
                    }
                }
                None => {
                    if let Some(temp) = &mut self.temp_fit
                        && let Some(super::main_fitter::FitResult::Gaussian(g)) =
                            &mut temp.fit_result
                        && let Err(e) = g.update_energy_for_peak(peak, e, du)
                    {
                        eprintln!("Energy update failed: {e}");
                    }
                }
            }
        }

        // apply deletion (single fit index)
        if let Some(i) = to_remove
            && i < self.stored_fits.len()
        {
            self.stored_fits.remove(i);
        }

        self.pending_modify_fit = to_modify;
    }

    fn fit_quality_ui(&self, ui: &mut egui::Ui) {
        let Some(temp) = &self.temp_fit else {
            return;
        };
        if let Some(error) = &temp.last_fit_error {
            ui.colored_label(Color32::RED, error);
        }
        let Some(FitResult::Gaussian(gaussian)) = &temp.fit_result else {
            return;
        };
        let Some(result) = &gaussian.native_result else {
            return;
        };
        let statistics = &result.fit.statistics;
        ui.horizontal_wrapped(|ui| {
            ui.label(format!(
                "Objective: {:?}  ·  widths: {}  ·  termination: {}",
                statistics.objective,
                if gaussian.fit_settings.equal_stdev {
                    "Shared"
                } else {
                    "Independent"
                },
                result.fit.termination.reason,
            )).on_hover_text("Convergence checks local stationarity. Inspect residuals, the background, and fit-quality warnings to assess whether the model describes the data.");
        });
        ui.label(format!(
            "Objective: {} to {:.5}  ·  improvement: {}  ·  RMSE: {:.5}  ·  R²: {}",
            statistics
                .initial_objective
                .map_or_else(|| "—".to_owned(), |value| format!("{value:.5}")),
            statistics.final_objective,
            statistics
                .objective_improvement
                .map_or_else(|| "—".to_owned(), |value| format!("{:.2}%", 100.0 * value),),
            statistics.rmse,
            statistics
                .r_squared
                .map_or_else(|| "—".to_owned(), |value| format!("{value:.5}")),
        ));
        if statistics.objective == ObjectiveKind::PoissonDeviance {
            ui.label(format!(
                "Deviance / reduced: {} / {}  ·  Pearson / reduced: {} / {}  ·  p: {}",
                statistics
                    .deviance
                    .map_or_else(|| "—".to_owned(), |value| format!("{value:.5}")),
                statistics
                    .reduced_deviance
                    .map_or_else(|| "—".to_owned(), |value| format!("{value:.5}")),
                statistics
                    .pearson_chi_square
                    .map_or_else(|| "—".to_owned(), |value| format!("{value:.5}")),
                statistics
                    .reduced_pearson_chi_square
                    .map_or_else(|| "—".to_owned(), |value| format!("{value:.5}")),
                statistics
                    .goodness_of_fit_p_value
                    .map_or_else(|| "—".to_owned(), |value| format!("{value:.4e}")),
            ));
        }
        ui.label(format!(
            "BIC: {}  ·  covariance: {}  ·  evaluations: {}",
            statistics
                .bic
                .map_or_else(|| "—".to_owned(), |value| format!("{value:.3}")),
            if result.fit.covariance.is_some() {
                "available"
            } else {
                "unavailable"
            },
            statistics.evaluations,
        ));
        let (color, label) = match result.quality_status {
            FitQualityStatus::Good => (Color32::GREEN, "Good"),
            FitQualityStatus::Review => (Color32::YELLOW, "Review"),
            FitQualityStatus::Poor => (Color32::from_rgb(230, 130, 20), "Poor"),
            FitQualityStatus::Failed => (Color32::RED, "Failed"),
        };
        ui.colored_label(color, format!("Advisory fit quality: {label}"));
        for issue in &result.quality_issues {
            let message = match issue {
                FitQualityIssue::FailedConvergence { reason } => {
                    format!("Optimizer did not converge: {reason}")
                }
                FitQualityIssue::MissingCovariance => {
                    "Parameter covariance is unavailable.".to_owned()
                }
                FitQualityIssue::NonFiniteResult => {
                    "The returned parameters or curve contain a non-finite value.".to_owned()
                }
                FitQualityIssue::ActiveBounds { parameters } => {
                    format!("Active bounds: {}", parameters.join(", "))
                }
                FitQualityIssue::NonpositivePrediction => {
                    "The Poisson model has an invalid expected count.".to_owned()
                }
                FitQualityIssue::ObjectiveWorsened {
                    initial,
                    final_value,
                } => format!("Objective worsened from {initial:.5} to {final_value:.5}."),
                FitQualityIssue::PoorGoodnessOfFit { p_value } => {
                    format!("Poisson goodness-of-fit p-value is {p_value:.4e}.")
                }
                FitQualityIssue::UnmodeledResidualPeaks { positions } => {
                    format!("Possible unmarked residual peaks near {positions:?}.")
                }
            };
            ui.label(format!("• {message}"));
        }
    }

    fn fit_panel_contents_ui(
        &mut self,
        ui: &mut egui::Ui,
        histogram_range: (f64, f64),
        markers: &mut FitMarkers,
        bin_width: f64,
    ) -> bool {
        ui.horizontal(|ui| {
            ui.heading("Fit Panel");
            ui.separator();
            ui.checkbox(&mut self.settings.fit_panel_popout, "Pop Out")
                .on_hover_text("Open the fit panel in a separate native window when supported.");
        });

        let storable = self.temp_fit_is_storable();
        let mut store = false;
        let mut manual_changed = false;
        ui.group(|ui| {
            self.fit_quality_ui(ui);
            if storable {
                store = ui
                    .button("Store Fit")
                    .on_hover_text(
                        "Store the completed temporary fit. Quality messages are advisory.",
                    )
                    .clicked();
            }
            manual_changed |=
                self.manual_peak_table_ui(ui, markers, bin_width, self.settings.equal_stddev);
        });
        if store {
            self.store_temp_fit();
            markers.clear_peak_markers();
            markers.preview_background.clear();
            markers.estimate_error = None;
        }

        self.save_and_load_ui(ui);

        let background_parameters_are_current = self
            .temp_fit
            .as_ref()
            .and_then(|fit| fit.background_result.as_ref())
            .is_some_and(|result| self.settings.background_parameters_match(result));
        let manual_background_available = self
            .temp_fit
            .as_ref()
            .is_some_and(|fit| fit.background_was_fit_manually && fit.background_result.is_some());
        self.settings.ui(
            ui,
            background_parameters_are_current,
            manual_background_available,
            !markers.background_markers.is_empty(),
        );

        self.visuals_ui(ui);
        self.calubration_ui(ui);

        self.fit_stats_grid_ui(ui);
        self.energy_calibration_plots_ui(ui, histogram_range);

        ui.add_space(10.0);
        manual_changed
    }

    fn visuals_ui(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Visuals", |ui| {
            let mut visibility_changed = false;
            ui.horizontal_wrapped(|ui| {
                visibility_changed |= ui
                    .checkbox(&mut self.settings.show_decomposition, "Decomposition")
                    .on_hover_text("Show the individual Gaussian peak curves.")
                    .changed();
                visibility_changed |= ui
                    .checkbox(&mut self.settings.show_composition, "Composition")
                    .on_hover_text("Show the combined Gaussian and background curve.")
                    .changed();
                visibility_changed |= ui
                    .checkbox(&mut self.settings.show_background, "Background")
                    .on_hover_text("Show the fitted background curve.")
                    .changed();
                visibility_changed |= ui
                    .checkbox(&mut self.settings.show_fit_lines_area, "1σ uncertainty")
                    .on_hover_text(
                        "Draw the covariance-based, Student-t-scaled total-fit 1σ uncertainty band.",
                    )
                    .changed();
                visibility_changed |= ui
                    .checkbox(&mut self.settings.show_temp_fit, "Temporary fit")
                    .on_hover_text(
                        "Show the active, un-stored fit. Stored fits remain visible when this is off.",
                    )
                    .changed();
            });
            if visibility_changed {
                self.apply_visibility_settings();
            }

            ui.separator();
            ui.label("UUID labels");
            ui.horizontal_wrapped(|ui| {
                ui.add(egui::Slider::new(&mut self.settings.uuid_label_size, 8.0..=32.0).text("Size"))
                    .on_hover_text(
                        "Adjust the UUID label size drawn above the fitted composition peaks.",
                    );
                ui.add(egui::Slider::new(&mut self.settings.uuid_label_lift, 0.0..=3.0).text("Lift"))
                    .on_hover_text(
                        "Move UUID labels closer to or farther above their reference height.",
                    );
                ui.checkbox(&mut self.settings.uuid_label_guides, "Guide")
                    .on_hover_text(
                        "Draw a dashed vertical guide from the UUID label to its reference height.",
                    );
            });
        });
        ui.separator();
    }

    fn displayed_calibration_mode(&self) -> CalibrationMode {
        if self.settings.calibration_mode == CalibrationMode::None && self.settings.calibrated {
            // Workspaces saved before the mode existed used editable coefficients.
            CalibrationMode::Manual
        } else {
            self.settings.calibration_mode
        }
    }

    fn calibration_coefficient_label(&self, name: &str, value: &super::common::Value) -> String {
        if self.calibration.uncertainty_available {
            format!("{name} = {:.6} ± {:.6}", value.value, value.uncertainty)
        } else {
            format!("{name} = {:.6} (uncertainty unavailable)", value.value)
        }
    }

    pub fn calubration_ui(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Calibration", |ui| {
            let previous_mode = self.displayed_calibration_mode();
            let mut mode = previous_mode;
            ui.horizontal_wrapped(|ui| {
                egui::ComboBox::from_label("Mode")
                    .selected_text(mode.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut mode, CalibrationMode::None, "None")
                            .on_hover_text("Use raw histogram-bin coordinates.");
                        ui.selectable_value(&mut mode, CalibrationMode::Linear, "Linear")
                            .on_hover_text("Fit a line from assigned energies in stored fits.");
                        ui.selectable_value(&mut mode, CalibrationMode::Quadratic, "Quadratic")
                            .on_hover_text("Fit a quadratic from assigned energies in stored fits.");
                        ui.selectable_value(&mut mode, CalibrationMode::Manual, "Manual")
                            .on_hover_text("Enter the calibration coefficients directly.");
                    });

                if mode != previous_mode {
                    self.settings.calibration_mode = mode;
                    self.settings.calibrated = mode != CalibrationMode::None;
                    match mode {
                        CalibrationMode::Linear => self.calibrate_stored_fits(true, false),
                        CalibrationMode::Quadratic => self.calibrate_stored_fits(false, true),
                        CalibrationMode::None | CalibrationMode::Manual => {}
                    }
                }

                match mode {
                    CalibrationMode::None => {}
                    CalibrationMode::Linear | CalibrationMode::Quadratic => {
                        ui.label(self.calibration_coefficient_label("a", &self.calibration.a));
                        ui.label(self.calibration_coefficient_label("b", &self.calibration.b));
                        ui.label(self.calibration_coefficient_label("c", &self.calibration.c));
                        if ui
                            .button("Recalculate")
                            .on_hover_text(
                                "Re-fit this calibration using the assigned energies in stored fits.",
                            )
                            .clicked()
                        {
                            if mode == CalibrationMode::Linear {
                                self.calibrate_stored_fits(true, false);
                            } else {
                                self.calibrate_stored_fits(false, true);
                            }
                        }
                    }
                    CalibrationMode::Manual => {
                        if self.calibration.ui(ui) {
                            self.calibration.cov = None;
                        }
                        if ui
                            .button("Apply manual calibration")
                            .on_hover_text(
                                "Apply the entered coefficients to temporary and stored fits.",
                            )
                            .clicked()
                        {
                            self.calibrate_stored_fits(false, false);
                        }
                    }
                }
            });
        });

        ui.separator();
    }

    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        hist_name: &str,
        histogram_range: (f64, f64),
        markers: &mut FitMarkers,
        bin_width: f64,
    ) -> bool {
        let mut manual_changed = false;
        if self.settings.show_fit_stats {
            let title = format!("Fit Panel: {hist_name}");
            let scroll_id = format!("fit_panel_scroll_{hist_name}");
            let mut open = self.settings.show_fit_stats;

            if self.settings.fit_panel_popout {
                let viewport_id =
                    egui::ViewportId::from_hash_of((ui.id(), hist_name, "fit_panel_viewport"));
                let viewport_builder = egui::ViewportBuilder::default()
                    .with_title(title)
                    .with_inner_size([960.0, 720.0])
                    .with_min_inner_size([520.0, 360.0]);

                ui.ctx()
                    .show_viewport_immediate(viewport_id, viewport_builder, |ui, _class| {
                        if ui.ctx().input(|input| input.viewport().close_requested()) {
                            open = false;
                            return;
                        }

                        egui::CentralPanel::default().show(ui, |ui| {
                            egui::ScrollArea::both()
                                .id_salt(scroll_id.as_str())
                                .show(ui, |ui| {
                                    manual_changed |= self.fit_panel_contents_ui(
                                        ui,
                                        histogram_range,
                                        markers,
                                        bin_width,
                                    );
                                });
                        });
                    });
            } else {
                egui::Window::new(title)
                    .open(&mut open)
                    .show(ui.ctx(), |ui| {
                        egui::ScrollArea::both()
                            .id_salt(scroll_id.as_str())
                            .show(ui, |ui| {
                                manual_changed |= self.fit_panel_contents_ui(
                                    ui,
                                    histogram_range,
                                    markers,
                                    bin_width,
                                );
                            });
                    });
            }

            self.settings.show_fit_stats = open;
        }
        manual_changed
    }

    pub fn fit_context_menu_ui(
        &mut self,
        ui: &mut egui::Ui,
        histogram_range: (f64, f64),
        markers: &mut FitMarkers,
        bin_width: f64,
    ) -> bool {
        let mut manual_changed = false;
        egui::ScrollArea::both()
            .id_salt(ui.id().with("fit_context_menu_scroll"))
            .max_height(450.0)
            .show(ui, |ui| {
                manual_changed |=
                    self.fit_panel_contents_ui(ui, histogram_range, markers, bin_width);
            });
        manual_changed
    }

    pub fn sync_uuid(&mut self, uuid_map: &[FitUUID]) {
        // UUID -> (energy, uncertainty)
        let lut: HashMap<usize, (f64, f64)> = uuid_map
            .iter()
            .map(|m| (m.uuid, (m.energy.0, m.energy.1)))
            .collect();

        // helper to apply to a single fitter
        let apply = |fitter: &mut Fitter| {
            if let Some(FitResult::Gaussian(gauss)) = &mut fitter.fit_result {
                // iterate by index so we can also call &mut methods on `gauss`
                for peak_idx in 0..gauss.fit_result.len() {
                    let uuid = gauss.fit_result[peak_idx].uuid;
                    if let Some(&(e, de)) = lut.get(&uuid)
                        && let Err(err) = gauss.update_energy_for_peak(peak_idx, e, de)
                    {
                        log::error!(
                            "Failed to update energy for UUID {uuid} (peak {peak_idx}): {err:?}"
                        );
                    }
                }
            }
        };

        // Stored fits
        for fitter in &mut self.stored_fits {
            apply(fitter);
        }

        // Temp fit
        if let Some(ref mut temp) = self.temp_fit {
            apply(temp);
        }
    }

    // pub fn
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitter::common::Value;
    use crate::fitter::models::gaussian::{GaussianFitter, GaussianParameters};
    use spectrix_fitting::BackgroundCoupling;

    fn calibration(a: f64, b: f64, c: f64) -> Calibration {
        Calibration {
            a: Value {
                value: a,
                uncertainty: 0.0,
            },
            b: Value {
                value: b,
                uncertainty: 0.0,
            },
            c: Value {
                value: c,
                uncertainty: 0.0,
            },
            cov: None,
            uncertainty_available: true,
        }
    }

    #[test]
    fn exact_two_point_linear_calibration_has_no_uncertainty() {
        let calibration = exact_linear_calibration(&[10.0, 20.0], &[100.0, 250.0])
            .expect("two distinct points define a line");
        assert_eq!(calibration.b.value, 15.0);
        assert_eq!(calibration.c.value, -50.0);
        assert!(calibration.cov.is_none());
        assert!(!calibration.uncertainty_available);
        assert!(calibration.curve_uncertainty_checked(15.0).is_none());
    }

    #[test]
    fn exact_linear_calibration_rejects_duplicate_positions() {
        assert!(exact_linear_calibration(&[10.0, 10.0], &[100.0, 250.0]).is_none());
    }

    #[test]
    fn apply_loaded_fits_restores_saved_calibration_and_fit_settings() {
        let mut current = Fits::new();
        let mut loaded = Fits::new();
        loaded.settings.calibrated = true;
        loaded.settings.show_fit_stats = true;
        loaded.settings.background_coupling = BackgroundCoupling::PrefitJoint;
        loaded.calibration = calibration(0.02, 1.5, 8.0);
        loaded.sort_state = SortState {
            col: SortCol::Energy,
            asc: false,
        };

        current.apply_loaded_fits(loaded);

        assert!(current.settings.calibrated);
        assert!(current.settings.show_fit_stats);
        assert_eq!(
            current.settings.background_coupling,
            BackgroundCoupling::PrefitJoint
        );
        assert_eq!(current.calibration.a.value, 0.02);
        assert_eq!(current.calibration.b.value, 1.5);
        assert_eq!(current.calibration.c.value, 8.0);
        assert_eq!(current.sort_state.col, SortCol::Energy);
        assert!(!current.sort_state.asc);
    }

    #[test]
    fn apply_loaded_fits_refreshes_live_calibrated_parameter_values() {
        let mut gaussian = GaussianFitter::default();
        let mut params = GaussianParameters::new(
            (100.0, 1.0),
            (4.0, 0.2),
            (1.0, 0.1),
            (2.355, 0.2),
            (250.0, 5.0),
        );
        params.energy.value = Some(42.0);
        params.energy.uncertainty = Some(0.3);
        gaussian.fit_result.push(params);

        let fitter = Fitter {
            fit_result: Some(FitResult::Gaussian(gaussian)),
            ..Fitter::default()
        };

        let mut loaded = Fits::new();
        loaded.settings.calibrated = true;
        loaded.calibration = calibration(0.5, 2.0, 10.0);
        loaded.stored_fits.push(fitter);

        let mut current = Fits::new();
        current.apply_loaded_fits(loaded);

        let Some(FitResult::Gaussian(gaussian)) = &current.stored_fits[0].fit_result else {
            panic!("loaded fit should remain Gaussian");
        };
        let params = &gaussian.fit_result[0];

        assert_eq!(current.stored_fits[0].calibration.a.value, 0.5);
        assert_eq!(params.mean.calibrated_value, Some(26.0));
        assert_eq!(params.sigma.calibrated_value, Some(6.0));
        assert_eq!(params.energy.calibrated_value, Some(42.0));
        assert_eq!(params.area.calibrated_value, Some(250.0));
    }

    #[test]
    fn apply_loaded_fits_compacts_legacy_dense_display_curves() {
        let point_count = 20_000;
        let peak_index = 11_111;
        let points = (0..point_count)
            .map(|index| [index as f64, if index == peak_index { 100.0 } else { 0.0 }])
            .collect::<Vec<_>>();
        let mut gaussian = GaussianFitter {
            fit_points: points.clone(),
            uncertainty_band: crate::egui_plot_stuff::egui_filled_area::EguiFilledArea::new(
                points.iter().map(|point| point[0]).collect(),
                vec![0.0; point_count],
                points.iter().map(|point| point[1] + 1.0).collect(),
            ),
            ..GaussianFitter::default()
        };
        let peak = GaussianParameters {
            fit_points: points.clone(),
            ..GaussianParameters::default()
        };
        gaussian.fit_result.push(peak);

        let mut fitter = Fitter {
            fit_result: Some(FitResult::Gaussian(gaussian)),
            composition_line: crate::egui_plot_stuff::egui_line::EguiLine::new_with_points(
                points.clone(),
            ),
            decomposition_lines: vec![
                crate::egui_plot_stuff::egui_line::EguiLine::new_with_points(points),
            ],
            ..Fitter::default()
        };
        fitter.name = "dense legacy fit".to_owned();

        let mut loaded = Fits::new();
        loaded.stored_fits.push(fitter);
        let mut current = Fits::new();
        current.apply_loaded_fits(loaded);

        let compacted = &current.stored_fits[0];
        let Some(FitResult::Gaussian(gaussian)) = &compacted.fit_result else {
            panic!("loaded fit should remain Gaussian");
        };
        assert!(gaussian.fit_points.len() <= 2048);
        assert!(gaussian.fit_result[0].fit_points.len() <= 512);
        assert!(gaussian.uncertainty_band.xs.len() <= 512);
        assert!(compacted.composition_line.points.len() <= 2048);
        assert!(compacted.decomposition_lines[0].points.len() <= 512);
        assert!(
            compacted.decomposition_lines[0]
                .points
                .iter()
                .any(|point| point[1] == 100.0)
        );
    }
}
