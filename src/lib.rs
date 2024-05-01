#![warn(clippy::all, rust_2018_idioms)]

#[cfg(not(target_arch = "wasm32"))]
mod app;
#[cfg(not(target_arch = "wasm32"))]
mod cutter;
#[cfg(not(target_arch = "wasm32"))]
mod fitter;
#[cfg(not(target_arch = "wasm32"))]
pub mod histoer;
#[cfg(not(target_arch = "wasm32"))]
mod lazyframer;
#[cfg(not(target_arch = "wasm32"))]
pub mod processer;
#[cfg(not(target_arch = "wasm32"))]
pub mod workspacer;
#[cfg(not(target_arch = "wasm32"))]
pub use app::MUCApp;

pub mod pane;
pub mod tree;

#[cfg(target_arch = "wasm32")]
mod app_web;
#[cfg(target_arch = "wasm32")]
pub use app_web::MUCApp;
