#[cfg(not(target_arch = "wasm32"))]
mod app;
#[cfg(not(target_arch = "wasm32"))]
pub use app::GNATApp;

#[cfg(target_arch = "wasm32")]
mod app_web;
#[cfg(target_arch = "wasm32")]
pub use app_web::GNATApp;
