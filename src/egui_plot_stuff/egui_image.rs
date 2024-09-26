use egui::{
    pos2, Color32, ColorImage, DragValue, ImageData, Rect, TextureFilter, TextureHandle,
    TextureOptions, TextureWrapMode, Ui,
};
use egui_plot::{PlotImage, PlotPoint, PlotUi};

#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub struct EguiImage {
    pub draw: bool,
    pub name_in_legend: bool,
    pub name: String,
    pub highlighted: bool,
    pub uv: Rect,
    pub add_background: bool,
    pub bg_color: Color32,
    pub tint: Color32,
    pub rotate: f64,
    pub image_center: [f64; 2],
    pub image_width: f32,
    pub image_height: f32,

    #[serde(skip)]
    pub texture: Option<TextureHandle>,
    pub texture_options: TextureOptions,
}

impl Default for EguiImage {
    fn default() -> Self {
        EguiImage {
            draw: true,
            name_in_legend: false,
            name: "Image".to_string(),
            highlighted: false,
            uv: Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
            add_background: false,
            bg_color: Color32::TRANSPARENT,
            tint: Color32::WHITE,
            rotate: 0.0,

            // color_image: None,
            image_center: [0.0, 0.0],
            image_width: 0.0,
            image_height: 0.0,

            texture: None,
            texture_options: egui::TextureOptions {
                magnification: egui::TextureFilter::Nearest,
                minification: egui::TextureFilter::Nearest,
                wrap_mode: egui::TextureWrapMode::ClampToEdge,
                mipmap_mode: None,
            },
        }
    }
}

impl EguiImage {
    pub fn heatmap(name: String, range_x: [f64; 2], range_y: [f64; 2]) -> Self {
        let image = EguiImage::default();

        EguiImage {
            name,
            image_width: (range_x[1] as f32 - range_x[0] as f32),
            image_height: (range_y[1] as f32 - range_y[0] as f32),
            image_center: [
                range_x[0] + (range_x[1] - range_x[0]) / 2.0,
                range_y[0] + (range_y[1] - range_y[0]) / 2.0,
            ],
            ..image
        }
    }

    // Convert ColorImage to ImageData (Byte array)
    fn to_image_data(&self, color_image: ColorImage) -> ImageData {
        let width = color_image.size[0];
        let height = color_image.size[1];
        let mut rgba_data = Vec::with_capacity(width * height * 4);
        for pixel in color_image.pixels.iter() {
            rgba_data.extend_from_slice(&pixel.to_array());
        }

        ImageData::Color(ColorImage::from_rgba_unmultiplied([width, height], &rgba_data).into())
    }

    // Get the center of the image
    fn get_image_center(&self) -> PlotPoint {
        PlotPoint::new(self.image_center[0], self.image_center[1])
    }

    // Get the size of the image
    fn get_image_size(&self) -> egui::Vec2 {
        egui::Vec2::new(self.image_width, self.image_height)
    }

    pub fn get_plot_image_from_texture(&mut self) -> Option<egui_plot::PlotImage> {
        if let Some(texture) = &self.texture {
            Some(PlotImage::new(
                texture,
                self.get_image_center(),
                self.get_image_size(),
            ))
        } else {
            log::warn!("Failed to get texture for image: {}", self.name);
            None
        }
    }

    pub fn get_texture(&mut self, ui: &mut egui::Ui, color_image: ColorImage) {
        if self.texture.is_none() {
            let image_data = self.to_image_data(color_image);
            self.texture = Some(ui.ctx().load_texture(
                self.name.clone(),
                image_data,
                self.texture_options,
            ));
        }
    }

    pub fn draw(&self, plot_ui: &mut PlotUi, mut image: PlotImage) {
        image = image
            .highlight(self.highlighted)
            .uv(self.uv)
            .bg_fill(self.bg_color)
            .tint(self.tint)
            .rotate(self.rotate * std::f64::consts::PI / 180.0);

        if self.name_in_legend {
            image = image.name(self.name.clone());
        }

        if self.draw {
            plot_ui.image(image);
        } else {
            log::warn!("Drawing is disabled for image: {}", self.name);
        }
    }

    pub fn menu_button(&mut self, ui: &mut Ui) {
        ui.menu_button(format!("{} Image", self.name), |ui| {
            ui.label(self.name.to_string());

            ui.vertical(|ui| {
                ui.checkbox(&mut self.draw, "Draw");

                ui.checkbox(&mut self.name_in_legend, "Name in Legend")
                    .on_hover_text("Show in legend");
                ui.checkbox(&mut self.highlighted, "Highlighted");

                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.add_background, "Add Background Color");
                    if self.add_background {
                        ui.color_edit_button_srgba(&mut self.bg_color);
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Tint Color (Default: White)");
                    ui.color_edit_button_srgba(&mut self.tint);
                });

                ui.add(
                    DragValue::new(&mut self.rotate)
                        .prefix("Rotate: ")
                        .suffix("°")
                        .range(0.0..=360.0)
                        .speed(0.1),
                );
            });

            ui.separator();

            ui.label("Texture Options");
            ui.horizontal(|ui| {
                ui.label("Magnification");
                ui.radio_value(
                    &mut self.texture_options.magnification,
                    TextureFilter::Nearest,
                    "Nearest",
                );
                ui.radio_value(
                    &mut self.texture_options.magnification,
                    TextureFilter::Linear,
                    "Linear",
                );
            });

            ui.horizontal(|ui| {
                ui.label("Minification");
                ui.radio_value(
                    &mut self.texture_options.minification,
                    TextureFilter::Nearest,
                    "Nearest",
                );
                ui.radio_value(
                    &mut self.texture_options.minification,
                    TextureFilter::Linear,
                    "Linear",
                );
            });

            ui.horizontal(|ui| {
                ui.label("Wrap Mode");
                ui.radio_value(
                    &mut self.texture_options.wrap_mode,
                    TextureWrapMode::ClampToEdge,
                    "ClampToEdge",
                );
                ui.radio_value(
                    &mut self.texture_options.wrap_mode,
                    TextureWrapMode::Repeat,
                    "Repeat",
                );
                ui.radio_value(
                    &mut self.texture_options.wrap_mode,
                    TextureWrapMode::MirroredRepeat,
                    "MirroredRepeat",
                );
            });

            // combo box between None, TextureFilter::Nearest, TextureFilter::Linear
            egui::ComboBox::from_id_salt("mipmap_mode")
                .selected_text(format!("{:?}", self.texture_options.mipmap_mode))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.texture_options.mipmap_mode, None, "None");
                    ui.selectable_value(
                        &mut self.texture_options.mipmap_mode,
                        Some(TextureFilter::Nearest),
                        "Nearest",
                    );
                    ui.selectable_value(
                        &mut self.texture_options.mipmap_mode,
                        Some(TextureFilter::Linear),
                        "Linear",
                    );
                });

            ui.separator();

            ui.label("Image Information");
            ui.label(format!("Image Width: {}", self.image_width));
            ui.label(format!("Image Height: {}", self.image_height));
            ui.label(format!(
                "Image Center: ({}, {})",
                self.image_center[0], self.image_center[1]
            ));
        });
    }
}
