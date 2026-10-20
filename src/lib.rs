pub mod config;
pub mod db;
pub mod image_loader;
pub mod indexer;
pub mod model;
pub mod paths;

#[cfg(windows)]
pub mod service;
#[cfg(windows)]
pub mod ui;
