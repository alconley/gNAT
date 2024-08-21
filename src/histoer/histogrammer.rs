use super::histo1d::histogram1d::Histogram;
use super::histo2d::histogram2d::Histogram2D;
use super::pane::Pane;
use super::tree::TreeBehavior;
use egui_tiles::TileId;
use polars::prelude::*;
use std::thread::JoinHandle;

use std::sync::{Arc, Mutex};

use std::collections::HashMap;

pub enum ContainerType {
    Grid,
    Tabs,
    Vertical,
    Horizontal,
}

#[derive(serde::Deserialize, serde::Serialize)]
pub struct Histogrammer {
    pub tree: egui_tiles::Tree<Pane>,
    pub behavior: TreeBehavior,
    #[serde(skip)]
    pub handles: Vec<JoinHandle<()>>, // Multiple thread handles
    pub grid_histogram_map: HashMap<TileId, Vec<TileId>>, // Map grid IDs to histogram IDs
}

impl Default for Histogrammer {
    fn default() -> Self {
        Self {
            tree: egui_tiles::Tree::empty("Empty tree"),
            behavior: Default::default(),
            handles: vec![],
            grid_histogram_map: HashMap::new(),
        }
    }
}

impl Histogrammer {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        // Check and join finished threads
        self.check_and_join_finished_threads();

        self.tree.ui(&mut self.behavior, ui);
    }

    pub fn check_and_join_finished_threads(&mut self) {
        // Only proceed if there are threads to check
        if self.handles.is_empty() {
            return;
        }

        let mut finished_indices = Vec::new();

        // First, identify all the threads that have finished
        for (i, handle) in self.handles.iter().enumerate() {
            if handle.is_finished() {
                finished_indices.push(i);
            }
        }

        // Then, remove and join the finished threads
        for &i in finished_indices.iter().rev() {
            let handle = self.handles.swap_remove(i);
            match handle.join() {
                Ok(_) => log::info!("A thread completed successfully."),
                Err(e) => log::error!("A thread encountered an error: {:?}", e),
            }
        }
    }

    pub fn side_panel_ui(&mut self, ui: &mut egui::Ui) {
        self.behavior.ui(ui);

        ui.separator();

        if let Some(root) = self.tree.root() {
            if ui.button("Reorganize").clicked() {
                self.reorganize();
            }

            tree_ui(ui, &mut self.behavior, &mut self.tree.tiles, root);
        }
    }

    pub fn create_grid(&mut self, tab_name: String) -> egui_tiles::TileId {
        // Create a new grid container
        let grid = egui_tiles::Grid::new(vec![]);
        let grid_container = egui_tiles::Container::Grid(grid);
        let grid_id = self.tree.tiles.insert_new(grid_container.into());

        // Create a new tab and place the grid inside it
        let tab = egui_tiles::Tabs::new(vec![grid_id]);
        let tab_id =
            self.tree
                .tiles
                .insert_new(egui_tiles::Tile::Container(egui_tiles::Container::Tabs(
                    tab,
                )));

        // Set the tab name in the behavior's tile_map
        self.behavior.set_tile_tab_mapping(grid_id, tab_name);

        // If the tree is empty, set this new tab as the root
        if self.tree.is_empty() {
            self.tree.root = Some(tab_id);
        } else if let Some(root_id) = self.tree.root {
            // Access the container at the root
            if let Some(egui_tiles::Tile::Container(egui_tiles::Container::Tabs(tabs))) =
                self.tree.tiles.get_mut(root_id)
            {
                // Add the new tab to the existing tab container
                tabs.add_child(tab_id);
            } else {
                // If the root is not a tabs container, create a new tabs container
                let new_tabs = egui_tiles::Tabs::new(vec![root_id, tab_id]);
                let new_root_id = self.tree.tiles.insert_new(egui_tiles::Tile::Container(
                    egui_tiles::Container::Tabs(new_tabs),
                ));
                self.tree.root = Some(new_root_id);
            }
        }

        grid_id
    }

    pub fn reorganize(&mut self) {
        // Iterate over each container (either a grid or a tab) in the map
        for (container_id, histogram_ids) in &self.grid_histogram_map {
            // Iterate over the histograms that should be in this container
            for (index, &histogram_id) in histogram_ids.iter().enumerate() {
                if self.tree.tiles.get(histogram_id).is_some() {
                    // Move each histogram to its proper position within the container
                    self.tree
                        .move_tile_to_container(histogram_id, *container_id, index, true);
                }
            }
        }
    }

    fn get_or_create_other_grid(&mut self) -> TileId {
        // Search for an existing "Other" grid
        for (grid_id, tile) in self.tree.tiles.iter() {
            if let egui_tiles::Tile::Container(egui_tiles::Container::Grid(_)) = tile {
                if let Some(tab_name) = self.behavior.get_tab_name(grid_id) {
                    if tab_name == "Other" {
                        return *grid_id;
                    }
                }
            }
        }

        // If "Other" grid does not exist, create a new one
        self.create_grid("Other".to_string())
    }

    pub fn add_hist1d(&mut self, name: &str, bins: usize, range: (f64, f64), grid: Option<TileId>) {
        let mut pane_id_to_update = None;

        // Search for an existing histogram with the same name to update
        for (id, tile) in self.tree.tiles.iter_mut() {
            if let egui_tiles::Tile::Pane(Pane::Histogram(hist)) = tile {
                if hist.lock().unwrap().name == name {
                    hist.lock().unwrap().reset();
                    pane_id_to_update = Some(*id);
                    break;
                }
            }
        }

        // If no existing histogram was found, create a new one
        if pane_id_to_update.is_none() {
            let hist = Histogram::new(name, bins, range);
            let pane = Pane::Histogram(Arc::new(Mutex::new(Box::new(hist))));
            let pane_id = self.tree.tiles.insert_pane(pane);

            // Handle the case where grid is None by using or creating the "Other" grid
            let grid_id = if let Some(grid_id) = grid {
                grid_id
            } else {
                self.get_or_create_other_grid()
            };

            if let Some(egui_tiles::Tile::Container(egui_tiles::Container::Grid(grid))) =
                self.tree.tiles.get_mut(grid_id)
            {
                grid.add_child(pane_id);
                self.grid_histogram_map
                    .entry(grid_id)
                    .or_default()
                    .push(pane_id);
            } else {
                log::error!("Invalid grid ID provided");
            }
        }
    }

    pub fn fill_hist1d(&mut self, name: &str, lf: &LazyFrame, column_name: &str) -> bool {
        if let Some((_id, egui_tiles::Tile::Pane(Pane::Histogram(hist)))) =
            self.tree.tiles.iter_mut().find(|(_id, tile)| {
                if let egui_tiles::Tile::Pane(Pane::Histogram(hist)) = tile {
                    hist.lock().unwrap().name == name
                } else {
                    false
                }
            })
        {
            let hist = Arc::clone(hist); // Clone the Arc to share ownership
            let hist_range = hist.lock().unwrap().range; // Access the range safely
            let filter_expr = col(column_name)
                .gt(lit(hist_range.0))
                .and(col(column_name).lt(lit(hist_range.1)));

            let lf = lf.clone();
            let name = name.to_string();
            let column_name = column_name.to_string();

            log::info!(
                "Starting to fill histogram '{}' with data from column '{}'",
                name,
                column_name
            );

            // Spawn a new thread for the filling operation
            let handle = std::thread::spawn(move || {
                log::info!("Thread started for filling histogram '{}'", name);

                if let Ok(df) = lf
                    .select([col(&column_name)])
                    .filter(filter_expr.clone()) // Clone for logging purposes
                    .collect()
                {
                    log::info!("Data collected for histogram '{}'", name);

                    let series = df.column(&column_name).unwrap();
                    let values = series.f64().unwrap();
                    let total_steps = values.len();

                    log::info!(
                        "Histogram '{}' will be filled with {} values from column '{}'",
                        name,
                        total_steps,
                        column_name
                    );

                    for (i, value) in values.iter().enumerate() {
                        if let Some(v) = value {
                            let mut hist = hist.lock().unwrap(); // Lock the mutex to access the correct Histogram
                            hist.fill(v, i, total_steps); // Pass the progress to the fill method
                        }
                    }

                    log::info!("Completed filling histogram '{}'", name);

                    // Optionally: Set progress to None or trigger any final updates here
                    hist.lock().unwrap().plot_settings.progress = None;
                } else {
                    log::error!("Failed to collect LazyFrame for histogram '{}'", name);
                }
            });

            // Store the thread handle in the vector
            self.handles.push(handle);

            return true;
        }

        log::error!("Histogram '{}' not found in the tree", name);
        false
    }

    pub fn add_fill_hist1d(
        &mut self,
        name: &str,
        lf: &LazyFrame,
        column_name: &str,
        bins: usize,
        range: (f64, f64),
        grid: Option<TileId>,
    ) {
        self.add_hist1d(name, bins, range, grid); // Add the histogram.
        self.fill_hist1d(name, lf, column_name); // Fill it with data.
    }

    pub fn add_hist2d(
        &mut self,
        name: &str,
        bins: (usize, usize),
        range: ((f64, f64), (f64, f64)),
        grid: Option<TileId>,
    ) {
        let mut pane_id_to_update = None;

        // Search for an existing histogram with the same name to update
        for (id, tile) in self.tree.tiles.iter_mut() {
            if let egui_tiles::Tile::Pane(Pane::Histogram2D(hist)) = tile {
                if hist.lock().unwrap().name == name {
                    hist.lock().unwrap().reset();
                    pane_id_to_update = Some(*id);
                    break;
                }
            }
        }

        // If no existing histogram was found, create a new one
        if pane_id_to_update.is_none() {
            let hist = Histogram2D::new(name, bins, range);
            let pane = Pane::Histogram2D(Arc::new(Mutex::new(Box::new(hist))));
            let pane_id = self.tree.tiles.insert_pane(pane);

            // Handle the case where grid is None by using or creating the "Other" grid
            let grid_id = if let Some(grid_id) = grid {
                grid_id
            } else {
                self.get_or_create_other_grid()
            };

            if let Some(egui_tiles::Tile::Container(egui_tiles::Container::Grid(grid))) =
                self.tree.tiles.get_mut(grid_id)
            {
                grid.add_child(pane_id);
                self.grid_histogram_map
                    .entry(grid_id)
                    .or_default()
                    .push(pane_id);
            } else {
                log::error!("Invalid grid ID provided");
            }
        }
    }

    pub fn fill_hist2d(
        &mut self,
        name: &str,
        lf: &LazyFrame,
        x_column_name: &str,
        y_column_name: &str,
    ) -> bool {
        if let Some((_id, egui_tiles::Tile::Pane(Pane::Histogram2D(hist)))) =
            self.tree.tiles.iter_mut().find(|(_id, tile)| {
                if let egui_tiles::Tile::Pane(Pane::Histogram2D(hist)) = tile {
                    hist.lock().unwrap().name == name
                } else {
                    false
                }
            })
        {
            let hist = Arc::clone(hist); // Clone the Arc to share ownership
            let hist_range = hist.lock().unwrap().range.clone(); // Access the range safely
            let filter_expr = col(x_column_name)
                .gt(lit(hist_range.x.min))
                .and(col(x_column_name).lt(lit(hist_range.x.max)))
                .and(col(y_column_name).gt(lit(hist_range.y.min)))
                .and(col(y_column_name).lt(lit(hist_range.y.max)));

            let lf = lf.clone();
            let name = name.to_string();
            let x_column_name = x_column_name.to_string();
            let y_column_name = y_column_name.to_string();

            hist.lock().unwrap().plot_settings.cuts.x_column = x_column_name.clone();
            hist.lock().unwrap().plot_settings.cuts.y_column = y_column_name.clone();

            log::info!(
                "Starting to fill 2D histogram '{}' with data from columns '{}' and '{}'",
                name,
                x_column_name,
                y_column_name
            );

            // Spawn a new thread for the filling operation
            let handle = std::thread::spawn(move || {
                log::info!("Thread started for filling 2D histogram '{}'", name);

                if let Ok(df) = lf
                    .select([col(&x_column_name), col(&y_column_name)])
                    .filter(filter_expr.clone()) // Clone for logging purposes
                    .collect()
                {
                    log::info!("Data collected for 2D histogram '{}'", name);

                    let x_values = df.column(&x_column_name).unwrap().f64().unwrap();
                    let y_values = df.column(&y_column_name).unwrap().f64().unwrap();
                    let total_steps = x_values.len();

                    log::info!(
                        "2D Histogram '{}' will be filled with {} value pairs from columns '{}' and '{}'",
                        name,
                        total_steps,
                        x_column_name,
                        y_column_name
                    );

                    for (i, (x_value, y_value)) in x_values.iter().zip(y_values.iter()).enumerate()
                    {
                        if let (Some(x), Some(y)) = (x_value, y_value) {
                            let mut hist = hist.lock().unwrap(); // Lock the mutex to access the correct Histogram2D
                            hist.fill(x, y, i, total_steps); // Pass the progress to the fill method
                        }
                    }

                    log::info!("Completed filling 2D histogram '{}'", name);

                    // Optionally: Set progress to None or trigger any final updates here
                    hist.lock().unwrap().plot_settings.progress = None;
                } else {
                    log::error!("Failed to collect LazyFrame for 2D histogram '{}'", name);
                }
            });

            // Store the thread handle in the vector
            self.handles.push(handle);

            return true;
        }

        log::error!("2D Histogram '{}' not found in the tree", name);
        false
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_fill_hist2d(
        &mut self,
        name: &str,
        lf: &LazyFrame,
        x_column_name: &str,
        y_column_name: &str,
        bins: (usize, usize),
        range: ((f64, f64), (f64, f64)),
        grid: Option<TileId>,
    ) {
        self.add_hist2d(name, bins, range, grid); // Add the histogram.
        self.fill_hist2d(name, lf, x_column_name, y_column_name); // Fill it with data.
    }
}

fn tree_ui(
    ui: &mut egui::Ui,
    behavior: &mut dyn egui_tiles::Behavior<Pane>,
    tiles: &mut egui_tiles::Tiles<Pane>,
    tile_id: egui_tiles::TileId,
) {
    // Get the name BEFORE we remove the tile below!
    let text = format!(
        "{} - {tile_id:?}",
        behavior.tab_title_for_tile(tiles, tile_id).text()
    );

    // Temporarily remove the tile to circumvent the borrowchecker
    let Some(mut tile) = tiles.remove(tile_id) else {
        log::debug!("Missing tile {tile_id:?}");
        return;
    };

    egui::collapsing_header::CollapsingState::load_with_default_open(
        ui.ctx(),
        egui::Id::new((tile_id, "tree")),
        false,
    )
    .show_header(ui, |ui| {
        ui.label(text);
        let mut visible = tiles.is_visible(tile_id);
        ui.checkbox(&mut visible, "Visible");
        tiles.set_visible(tile_id, visible);
    })
    .body(|ui| match &mut tile {
        egui_tiles::Tile::Pane(_) => {}
        egui_tiles::Tile::Container(container) => {
            for &child in container.children() {
                tree_ui(ui, behavior, tiles, child);
            }
        }
    });

    // Put the tile back
    tiles.insert(tile_id, tile);
}
