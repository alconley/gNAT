use crate::defaults::{
    Histogram1DDefaults, Histogram2DDefaults, LineDefaults, MarkerStyleDefaults, SpectrixDefaults,
    search_matches,
};
use crate::egui_plot_stuff::egui_plot_settings::EguiPlotSettings;
use crate::egui_plot_stuff::line_style::SerializableLineStyle;
use crate::histoer::histo2d::colormaps::ColorMap;
use crate::util::processer::FileSortKey;

fn section_matches(query: &str, title: &str, keywords: &[&str]) -> bool {
    search_matches(query, title, keywords)
}

fn section(
    ui: &mut egui::Ui,
    title: &str,
    query: &str,
    visible: bool,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    if !visible {
        return;
    }
    egui::CollapsingHeader::new(title)
        .default_open(!query.trim().is_empty())
        .show(ui, add_contents);
}

fn reset_row<T: Default>(ui: &mut egui::Ui, value: &mut T) {
    if ui.small_button("Reset section").clicked() {
        *value = T::default();
    }
    ui.separator();
}

fn line_style_ui(ui: &mut egui::Ui, style: &mut SerializableLineStyle, length: &mut f32) {
    ui.horizontal_wrapped(|ui| {
        ui.radio_value(style, SerializableLineStyle::Solid, "Solid");
        ui.radio_value(style, SerializableLineStyle::Dotted, "Dotted");
        ui.radio_value(style, SerializableLineStyle::Dashed, "Dashed");
        ui.add(
            egui::DragValue::new(length)
                .speed(1.0)
                .range(0.0..=f32::INFINITY)
                .prefix("Length: "),
        );
    });
}

fn line_defaults_ui(
    ui: &mut egui::Ui,
    line: &mut LineDefaults,
    query: &str,
    mut theme_colors: Option<(&mut egui::Color32, &mut egui::Color32)>,
) {
    if search_matches(
        query,
        "Line visibility and legend",
        &["draw", "highlighted"],
    ) {
        ui.checkbox(&mut line.draw, "Draw");
        ui.checkbox(&mut line.name_in_legend, "Name in legend");
        ui.checkbox(&mut line.highlighted, "Highlighted");
    }
    let color_matches = if theme_colors.is_some() {
        search_matches(
            query,
            "Histogram color light theme dark theme",
            &["automatic color", "line color", "rgb", "stroke"],
        )
    } else {
        search_matches(query, "Line color", &["rgb", "stroke"])
    };
    if color_matches {
        ui.horizontal(|ui| {
            if let Some((light, dark)) = theme_colors.take() {
                ui.label("Light theme");
                ui.color_edit_button_srgba(light);
                ui.label("Dark theme");
                ui.color_edit_button_srgba(dark);
            } else {
                ui.label("Color");
                ui.color_edit_button_srgba(&mut line.color);
            }
        });
    }
    if search_matches(query, "Line width", &["stroke", "thickness"]) {
        ui.add(egui::Slider::new(&mut line.width, 0.0..=10.0).text("Width"));
    }
    if search_matches(query, "Line style", &["solid", "dotted", "dashed"]) {
        line_style_ui(ui, &mut line.style, &mut line.style_length);
    }
    if search_matches(query, "Line fill", &["area", "alpha", "reference"]) {
        ui.checkbox(&mut line.reference_fill, "Fill under line");
        if line.reference_fill {
            ui.add(egui::DragValue::new(&mut line.fill).prefix("Reference: "));
            ui.add(egui::Slider::new(&mut line.fill_alpha, 0.0..=1.0).text("Fill alpha"));
        }
    }
}

fn marker_style_ui(ui: &mut egui::Ui, label: &str, marker: &mut MarkerStyleDefaults) {
    ui.group(|ui| {
        ui.label(egui::RichText::new(label).strong());
        ui.horizontal(|ui| {
            ui.label("Color");
            ui.color_edit_button_srgba(&mut marker.color);
        });
        ui.add(egui::Slider::new(&mut marker.width, 0.0..=10.0).text("Width"));
        ui.add(egui::Slider::new(&mut marker.midpoint_radius, 0.0..=12.0).text("Handle radius"));
        line_style_ui(ui, &mut marker.style, &mut marker.style_length);
    });
}

fn plot_defaults_ui(ui: &mut egui::Ui, plot: &mut EguiPlotSettings, query: &str) {
    if search_matches(query, "Legend", &["plot key"]) {
        ui.checkbox(&mut plot.legend, "Legend");
    }
    if search_matches(query, "Log axes", &["log x", "log y", "logarithmic"]) {
        ui.checkbox(&mut plot.log_x, "Log X");
        ui.checkbox(&mut plot.log_y, "Log Y");
    }
    if search_matches(query, "Axis values", &["show x", "show y", "center axes"]) {
        ui.checkbox(&mut plot.show_x_value, "Show X value");
        ui.checkbox(&mut plot.show_y_value, "Show Y value");
        ui.checkbox(&mut plot.center_x_axis, "Center X axis");
        ui.checkbox(&mut plot.center_y_axis, "Center Y axis");
    }
    if search_matches(
        query,
        "Plot interaction",
        &["zoom", "drag", "scroll", "double click"],
    ) {
        ui.checkbox(&mut plot.allow_zoom, "Allow zoom");
        ui.checkbox(&mut plot.allow_boxed_zoom, "Allow boxed zoom");
        ui.checkbox(&mut plot.allow_drag, "Allow drag");
        ui.checkbox(&mut plot.allow_scroll, "Allow scroll");
        ui.checkbox(
            &mut plot.allow_double_click_reset,
            "Allow double-click reset",
        );
        ui.checkbox(&mut plot.limit_scrolling, "Limit scrolling to data");
    }
    if search_matches(
        query,
        "Grid and background",
        &["grid", "background", "sharp", "clamp"],
    ) {
        ui.checkbox(&mut plot.show_grid, "Show grid");
        ui.checkbox(&mut plot.sharp_grid_lines, "Sharp grid lines");
        ui.checkbox(&mut plot.clamp_grid, "Clamp grid");
        ui.checkbox(&mut plot.show_background, "Show background");
    }
}

fn palette_ui(ui: &mut egui::Ui, palette: &mut [egui::Color32; 6]) {
    ui.label("Rotating cut palette");
    ui.horizontal_wrapped(|ui| {
        for color in palette {
            ui.color_edit_button_srgba(color);
        }
    });
}

fn histogram_1d_ui(ui: &mut egui::Ui, defaults: &mut Histogram1DDefaults, query: &str) {
    reset_row(ui, defaults);

    section(
        ui,
        "Histogram Line",
        query,
        section_matches(
            query,
            "Histogram Line",
            &["color", "width", "style", "fill"],
        ),
        |ui| {
            line_defaults_ui(
                ui,
                &mut defaults.line,
                query,
                Some((
                    &mut defaults.light_theme_color,
                    &mut defaults.dark_theme_color,
                )),
            );
        },
    );

    section(
        ui,
        "Plot",
        query,
        section_matches(
            query,
            "Plot",
            &["axis", "legend", "zoom", "grid", "statistics", "auto y"],
        ),
        |ui| {
            plot_defaults_ui(ui, &mut defaults.plot, query);
            if search_matches(query, "Statistics", &["stats", "mean", "counts"]) {
                ui.checkbox(&mut defaults.show_statistics, "Show statistics");
            }
            if search_matches(
                query,
                "Automatic Y range",
                &["auto y", "visible range", "multiplier"],
            ) {
                ui.checkbox(
                    &mut defaults.auto_fit_y_to_visible_range,
                    "Fit Y to visible X range",
                );
                ui.add(
                    egui::DragValue::new(&mut defaults.auto_fit_y_multiplier_linear)
                        .range(1.0..=f64::INFINITY)
                        .speed(0.01)
                        .prefix("Linear padding: "),
                );
                ui.add(
                    egui::DragValue::new(&mut defaults.auto_fit_y_multiplier_log)
                        .range(1.0..=f64::INFINITY)
                        .speed(0.01)
                        .prefix("Log padding: "),
                );
            }
        },
    );

    section(
        ui,
        "Markers",
        query,
        section_matches(
            query,
            "Markers",
            &["region", "peak", "background", "handle"],
        ),
        |ui| {
            marker_style_ui(ui, "Region markers", &mut defaults.markers.region);
            marker_style_ui(ui, "Peak markers", &mut defaults.markers.peak);
            marker_style_ui(ui, "Background markers", &mut defaults.markers.background);
            ui.add(
                egui::Slider::new(&mut defaults.markers.background_fill_alpha, 0.0..=1.0)
                    .text("Background fill alpha"),
            );
        },
    );

    section(
        ui,
        "Fits",
        query,
        section_matches(
            query,
            "Fits",
            &[
                "gaussian",
                "sigma",
                "initial estimates",
                "auto estimate",
                "show initial parameters",
                "background model",
                "composition",
                "decomposition",
                "uuid",
            ],
        ),
        |ui| {
            defaults.fit.ui(ui, false, false, false);
            for (label, line) in [
                (
                    "Temporary background",
                    &mut defaults.fit_palette.temporary_background,
                ),
                (
                    "Temporary composition",
                    &mut defaults.fit_palette.temporary_composition,
                ),
                (
                    "Temporary decomposition",
                    &mut defaults.fit_palette.temporary_decomposition,
                ),
                (
                    "Stored background",
                    &mut defaults.fit_palette.stored_background,
                ),
                (
                    "Stored composition",
                    &mut defaults.fit_palette.stored_composition,
                ),
                (
                    "Stored decomposition",
                    &mut defaults.fit_palette.stored_decomposition,
                ),
            ] {
                ui.collapsing(label, |ui| line_defaults_ui(ui, line, "", None));
            }
        },
    );

    section(
        ui,
        "Cuts",
        query,
        section_matches(
            query,
            "Cuts",
            &["gate", "palette", "fill", "inset", "active"],
        ),
        |ui| {
            palette_ui(ui, &mut defaults.cuts.palette);
            ui.checkbox(&mut defaults.cuts.active, "Active by default");
            ui.add(egui::Slider::new(&mut defaults.cuts.line_width, 0.0..=10.0).text("Line width"));
            line_style_ui(
                ui,
                &mut defaults.cuts.line_style,
                &mut defaults.cuts.style_length,
            );
            ui.add(
                egui::Slider::new(&mut defaults.cuts.midpoint_radius, 0.0..=12.0)
                    .text("Handle radius"),
            );
            ui.add(egui::Slider::new(&mut defaults.cuts.fill_alpha, 0.0..=1.0).text("Fill alpha"));
            ui.add(
                egui::Slider::new(&mut defaults.cuts.initial_inset_fraction, 0.0..=0.5)
                    .text("Initial inset"),
            );
        },
    );
}

fn histogram_2d_ui(ui: &mut egui::Ui, defaults: &mut Histogram2DDefaults, query: &str) {
    reset_row(ui, defaults);
    section(
        ui,
        "Plot",
        query,
        section_matches(
            query,
            "Plot",
            &["axis", "legend", "zoom", "grid", "statistics"],
        ),
        |ui| {
            plot_defaults_ui(ui, &mut defaults.plot, query);
            ui.checkbox(&mut defaults.show_statistics, "Show statistics");
        },
    );
    section(
        ui,
        "Heatmap",
        query,
        section_matches(
            query,
            "Heatmap",
            &[
                "colormap", "log z", "reverse", "range", "tint", "texture", "rotation",
            ],
        ),
        |ui| {
            ui.label("Colormap");
            for (map, label) in [
                (ColorMap::Viridis, "Viridis"),
                (ColorMap::Fast, "Fast"),
                (ColorMap::SmoothCoolWarm, "Smooth Cool Warm"),
                (ColorMap::BentCoolWarm, "Bent Cool Warm"),
                (ColorMap::Plasma, "Plasma"),
                (ColorMap::Blackbody, "Blackbody"),
                (ColorMap::Inferno, "Inferno"),
                (ColorMap::Kindlmann, "Kindlmann"),
                (ColorMap::ExtendedKindlmann, "Extended Kindlmann"),
                (ColorMap::Turbo, "Turbo"),
                (ColorMap::Jet, "Jet"),
            ] {
                ui.radio_value(&mut defaults.colormap, map, label);
            }
            ui.checkbox(&mut defaults.colormap_options.log_norm, "Log Z");
            ui.checkbox(&mut defaults.colormap_options.reverse, "Reverse colormap");
            ui.checkbox(
                &mut defaults.colormap_options.custom_display_range,
                "Custom Z range",
            );
            if defaults.colormap_options.custom_display_range {
                ui.add(
                    egui::DragValue::new(&mut defaults.colormap_options.display_min)
                        .prefix("Min: "),
                );
                ui.add(
                    egui::DragValue::new(&mut defaults.colormap_options.display_max)
                        .prefix("Max: "),
                );
                ui.checkbox(
                    &mut defaults.colormap_options.remove,
                    "Hide values outside range",
                );
            }
            ui.separator();
            ui.checkbox(&mut defaults.image.draw, "Draw image");
            ui.checkbox(&mut defaults.image.name_in_legend, "Name in legend");
            ui.checkbox(&mut defaults.image.highlighted, "Highlighted");
            ui.checkbox(&mut defaults.image.add_background, "Background color");
            if defaults.image.add_background {
                ui.color_edit_button_srgba(&mut defaults.image.background_color);
            }
            ui.horizontal(|ui| {
                ui.label("Tint");
                ui.color_edit_button_srgba(&mut defaults.image.tint);
            });
            ui.add(
                egui::Slider::new(&mut defaults.image.rotate_degrees, 0.0..=360.0).text("Rotation"),
            );
            ui.horizontal(|ui| {
                ui.label("Texture filtering");
                ui.radio_value(
                    &mut defaults.image.magnification,
                    egui::TextureFilter::Nearest,
                    "Nearest",
                );
                ui.radio_value(
                    &mut defaults.image.magnification,
                    egui::TextureFilter::Linear,
                    "Linear",
                );
            });
            defaults.image.minification = defaults.image.magnification;
        },
    );
    section(
        ui,
        "Projections",
        query,
        section_matches(
            query,
            "Projections",
            &["x projection", "y projection", "color", "fill", "inset"],
        ),
        |ui| {
            ui.checkbox(
                &mut defaults.projections.add_x_projection,
                "Add X projection",
            );
            ui.checkbox(
                &mut defaults.projections.add_y_projection,
                "Add Y projection",
            );
            ui.horizontal(|ui| {
                ui.label("X color");
                ui.color_edit_button_srgba(&mut defaults.projections.x_color);
            });
            ui.horizontal(|ui| {
                ui.label("Y color");
                ui.color_edit_button_srgba(&mut defaults.projections.y_color);
            });
            ui.add(
                egui::Slider::new(&mut defaults.projections.line_width, 0.0..=10.0)
                    .text("Line width"),
            );
            line_style_ui(
                ui,
                &mut defaults.projections.line_style,
                &mut defaults.projections.style_length,
            );
            ui.add(
                egui::Slider::new(&mut defaults.projections.midpoint_radius, 0.0..=12.0)
                    .text("Handle radius"),
            );
            ui.add(
                egui::Slider::new(&mut defaults.projections.fill_alpha, 0.0..=1.0)
                    .text("Fill alpha"),
            );
            ui.add(
                egui::Slider::new(&mut defaults.projections.initial_inset_fraction, 0.0..=0.5)
                    .text("Initial inset"),
            );
        },
    );
    section(
        ui,
        "Cuts",
        query,
        section_matches(
            query,
            "Cuts",
            &["polygon", "gate", "palette", "fill", "dragging"],
        ),
        |ui| {
            palette_ui(ui, &mut defaults.cuts.palette);
            ui.checkbox(&mut defaults.cuts.active, "Active by default");
            ui.checkbox(&mut defaults.cuts.draw, "Draw polygon");
            ui.checkbox(&mut defaults.cuts.name_in_legend, "Name in legend");
            ui.checkbox(&mut defaults.cuts.interactive_dragging, "Drag vertices");
            ui.add(egui::Slider::new(&mut defaults.cuts.line_width, 0.0..=10.0).text("Line width"));
            line_style_ui(
                ui,
                &mut defaults.cuts.line_style,
                &mut defaults.cuts.style_length,
            );
            ui.horizontal(|ui| {
                ui.label("Fill color");
                ui.color_edit_button_srgba(&mut defaults.cuts.fill_color);
            });
        },
    );
}

pub fn show(ui: &mut egui::Ui, defaults: &mut SpectrixDefaults, search: &mut String) {
    ui.horizontal(|ui| {
        ui.heading("Defaults");
        if ui.small_button("Reset all").clicked() {
            *defaults = SpectrixDefaults::default();
        }
    });
    ui.label(
        egui::RichText::new(
            "Histogram defaults apply the next time histograms are generated or imported.",
        )
        .weak()
        .small(),
    );
    ui.horizontal(|ui| {
        ui.add(egui::TextEdit::singleline(search).hint_text("Search defaults…"));
        if !search.is_empty() && ui.small_button("Clear").clicked() {
            search.clear();
        }
    });
    ui.separator();

    let q = search.as_str();
    let show_1d = section_matches(
        q,
        "1D Histograms",
        &[
            "line width color style fill light dark",
            "fit gaussian sigma background composition decomposition uuid stored temporary",
            "marker region peak background handle",
            "manual peak guess height fwhm width position bounds uncertainty",
            "cut gate palette inset active",
            "plot axis legend zoom grid statistics stats auto y visible range",
        ],
    );
    let show_2d = section_matches(
        q,
        "2D Histograms",
        &[
            "heatmap colormap log z reverse range tint texture rotation",
            "projection x y color fill inset",
            "polygon cut gate palette dragging fill",
            "plot axis legend zoom grid statistics stats",
        ],
    );
    let show_layout = section_matches(
        q,
        "Histogrammer Layout",
        &["tile", "tab", "gap", "pane", "prune", "simplification"],
    );
    let show_general = section_matches(q, "General", &["memory", "ram"]);
    let show_ai = section_matches(q, "AI", &["ollama", "model", "url"]);
    let show_sessions = section_matches(
        q,
        "New Sessions",
        &["files panel", "histogram script", "separate", "sort"],
    );

    egui::ScrollArea::vertical().show(ui, |ui| {
        section(ui, "1D Histograms", q, show_1d, |ui| {
            histogram_1d_ui(ui, &mut defaults.histogram_1d, q);
        });
        section(ui, "2D Histograms", q, show_2d, |ui| {
            histogram_2d_ui(ui, &mut defaults.histogram_2d, q);
        });
        section(ui, "Histogrammer Layout", q, show_layout, |ui| {
            reset_row(ui, &mut defaults.histogrammer);
            ui.add(
                egui::Slider::new(&mut defaults.histogrammer.tab_bar_height, 0.0..=100.0)
                    .text("Tab bar height"),
            );
            ui.add(
                egui::Slider::new(&mut defaults.histogrammer.gap_width, 0.0..=20.0)
                    .text("Gap width"),
            );
            ui.add(
                egui::DragValue::new(&mut defaults.histogrammer.min_size)
                    .range(0.0..=f32::INFINITY)
                    .prefix("Minimum pane size: "),
            );
            ui.checkbox(
                &mut defaults.histogrammer.preview_dragged_panes,
                "Preview dragged panes",
            );
            ui.separator();
            ui.checkbox(
                &mut defaults.histogrammer.prune_empty_tabs,
                "Prune empty tabs",
            );
            ui.checkbox(
                &mut defaults.histogrammer.prune_empty_containers,
                "Prune empty containers",
            );
            ui.checkbox(
                &mut defaults.histogrammer.prune_single_child_tabs,
                "Prune single-child tabs",
            );
            ui.checkbox(
                &mut defaults.histogrammer.prune_single_child_containers,
                "Prune single-child containers",
            );
            ui.checkbox(
                &mut defaults.histogrammer.all_panes_must_have_tabs,
                "All panes must have tabs",
            );
            ui.checkbox(
                &mut defaults.histogrammer.join_nested_linear_containers,
                "Join nested linear containers",
            );
            ui.checkbox(
                &mut defaults.histogrammer.flatten_tabs_in_tabs,
                "Flatten nested tabs",
            );
        });
        section(ui, "General", q, show_general, |ui| {
            reset_row(ui, &mut defaults.general);
            ui.add(
                egui::DragValue::new(&mut defaults.general.estimated_memory_gb)
                    .range(0.1..=f64::INFINITY)
                    .suffix(" GB")
                    .prefix("Memory budget: "),
            );
        });
        section(ui, "AI", q, show_ai, |ui| {
            reset_row(ui, &mut defaults.ai);
            ui.label("Ollama URL");
            ui.text_edit_singleline(&mut defaults.ai.base_url);
            ui.label("Model");
            ui.text_edit_singleline(&mut defaults.ai.model);
        });
        section(ui, "New Sessions", q, show_sessions, |ui| {
            reset_row(ui, &mut defaults.new_sessions);
            ui.checkbox(
                &mut defaults.new_sessions.files_panel_open,
                "Files panel open",
            );
            ui.checkbox(
                &mut defaults.new_sessions.histogram_script_open,
                "Histogram Script panel open",
            );
            ui.checkbox(&mut defaults.new_sessions.ai_open, "AI panel open");
            ui.checkbox(
                &mut defaults.new_sessions.calculate_histograms_separately,
                "Calculate/Get histograms separately",
            );
            ui.horizontal(|ui| {
                ui.label("File sort");
                ui.radio_value(
                    &mut defaults.new_sessions.file_sort.key,
                    FileSortKey::Name,
                    "Name",
                );
                ui.radio_value(
                    &mut defaults.new_sessions.file_sort.key,
                    FileSortKey::Size,
                    "Size",
                );
                ui.radio_value(
                    &mut defaults.new_sessions.file_sort.key,
                    FileSortKey::Modified,
                    "Time",
                );
            });
            ui.checkbox(
                &mut defaults.new_sessions.file_sort.ascending,
                "Ascending sort",
            );
        });
    });

    if !q.trim().is_empty()
        && !(show_1d || show_2d || show_layout || show_general || show_ai || show_sessions)
    {
        ui.label("No defaults match this search.");
    }
}

#[cfg(test)]
mod tests {
    use super::section_matches;

    #[test]
    fn section_search_routes_aliases() {
        assert!(section_matches("ollama", "AI", &["ollama", "model"]));
        assert!(section_matches("fwhm", "1D Histograms", &["peak", "fwhm"]));
        assert!(!section_matches("polygon", "General", &["memory"]));
    }
}
