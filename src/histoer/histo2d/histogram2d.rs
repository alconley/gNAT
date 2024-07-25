use fnv::FnvHashMap;

use crate::egui_plot_stuff::egui_image::EguiImage;

use super::plot_settings::PlotSettings;

#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub struct Histogram2D {
    pub name: String,
    pub bins: Bins,
    pub range: Range,
    pub plot_settings: PlotSettings,
    pub image: EguiImage,
    pub backup_bins: Option<Bins>,
}

impl Histogram2D {
    // Create a new 2D Histogram with specified ranges and number of bins for each axis
    pub fn new(name: &str, bins: (usize, usize), range: ((f64, f64), (f64, f64))) -> Self {
        Histogram2D {
            name: name.to_string(),
            bins: Bins {
                x: bins.0,
                x_width: (range.0 .1 - range.0 .0) / bins.0 as f64,
                y: bins.1,
                y_width: (range.1 .1 - range.1 .0) / bins.1 as f64,
                counts: FnvHashMap::default(),
                min_count: u64::MAX,
                max_count: u64::MIN,
            },
            range: Range {
                x: Value {
                    min: range.0 .0,
                    max: range.0 .1,
                },
                y: Value {
                    min: range.1 .0,
                    max: range.1 .1,
                },
            },
            plot_settings: PlotSettings::default(),
            image: EguiImage::heatmap(
                name.to_string(),
                [range.0 .0, range.0 .1],
                [range.1 .0, range.1 .1],
            ),
            backup_bins: None,
        }
    }

    // Add a value to the histogram
    pub fn fill(&mut self, x_value: f64, y_value: f64) {
        if x_value >= self.range.x.min
            && x_value < self.range.x.max
            && y_value >= self.range.y.min
            && y_value < self.range.y.max
        {
            let x_index = ((x_value - self.range.x.min) / self.bins.x_width) as usize;
            let y_index = ((y_value - self.range.y.min) / self.bins.y_width) as usize;
            let count = self.bins.counts.entry((x_index, y_index)).or_insert(0);
            *count += 1;

            self.bins.min_count = self.bins.min_count.min(*count);
            self.bins.max_count = self.bins.max_count.max(*count);
        }
    }

    // get the bin index for a given x value
    pub fn get_bin_index_x(&self, x: f64) -> Option<usize> {
        if x < self.range.x.min || x > self.range.x.max {
            return None;
        }

        let bin_index: usize = ((x - self.range.x.min) / self.bins.x_width).floor() as usize;

        Some(bin_index)
    }

    // get the bin index for a given y value
    pub fn get_bin_index_y(&self, y: f64) -> Option<usize> {
        if y < self.range.y.min || y > self.range.y.max {
            return None;
        }

        let bin_index: usize = ((y - self.range.y.min) / self.bins.y_width).floor() as usize;

        Some(bin_index)
    }

    // Convert histogram data to a ColorImage
    fn data_2_image(&self) -> egui::ColorImage {
        let width = ((self.range.x.max - self.range.x.min) / self.bins.x_width) as usize; // number of pixels in x direction
        let height = ((self.range.y.max - self.range.y.min) / self.bins.y_width) as usize; // number of pixels in y direction

        // The pixels, row by row, from top to bottom. Each pixel is a Color32.
        let mut pixels = Vec::with_capacity(width * height);

        let colormap_options = self.plot_settings.colormap_options;

        for y in 0..height {
            for x in 0..width {
                let count = self
                    .bins
                    .counts
                    .get(&(x, height - y - 1))
                    .cloned()
                    .unwrap_or(0);
                let color = self.plot_settings.colormap.color(
                    count,
                    self.bins.min_count,
                    self.bins.max_count,
                    colormap_options,
                );
                pixels.push(color);
            }
        }

        // Create the ColorImage with the specified width and height and pixel data
        egui::ColorImage {
            size: [width, height],
            pixels,
        }
    }

    // Recalculate the image and replace the existing texture
    fn calculate_image(&mut self, ui: &mut egui::Ui) {
        self.image.texture = None;
        let color_image = self.data_2_image();
        self.image.get_texture(ui, color_image);
    }

    // Draw the histogram on the plot
    fn draw(&mut self, plot_ui: &mut egui_plot::PlotUi) {
        self.show_stats(plot_ui);

        let heatmap_image = self.image.get_plot_image_from_texture();

        if let Some(image) = heatmap_image {
            self.image.draw(plot_ui, image);
        }
        // self.image.draw(plot_ui, plot_image);
        self.plot_settings.cursor_position = plot_ui.pointer_coordinate();
        self.plot_settings.draw(plot_ui);
    }

    // Render the histogram using egui_plot
    pub fn render(&mut self, ui: &mut egui::Ui) {
        // Recalculate the image if the settings have changed, like the colormap
        if self.plot_settings.recalculate_image {
            self.calculate_image(ui);
            self.plot_settings.recalculate_image = false;
        }

        let mut plot = egui_plot::Plot::new(self.name.clone());
        plot = self.plot_settings.egui_settings.apply_to_plot(plot);

        if self.image.texture.is_none() {
            self.calculate_image(ui);
        }

        self.check_projections();
        self.plot_settings.projections.show(ui);

        let plot_response = plot.show(ui, |plot_ui| {
            self.draw(plot_ui);
        });

        plot_response.response.context_menu(|ui| {
            self.context_menu(ui);
        });

        self.plot_settings.interactive_response(&plot_response);
    }
}

#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub struct Bins {
    pub x: usize,
    pub x_width: f64,
    pub y: usize,
    pub y_width: f64,
    pub counts: FnvHashMap<(usize, usize), u64>, // uses a hash map to store the histogram data (zero overhead for empty bins)
    pub min_count: u64,
    pub max_count: u64,
}

#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub struct Value {
    pub min: f64,
    pub max: f64,
}

#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub struct Range {
    pub x: Value,
    pub y: Value,
}
