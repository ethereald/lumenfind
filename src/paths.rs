use std::path::PathBuf;

use anyhow::{Context, Result};
use directories::ProjectDirs;

pub const APP_NAME: &str = "LumenFind";

#[derive(Clone, Debug)]
pub struct AppPaths {
    pub data_dir: PathBuf,
    pub config_file: PathBuf,
    pub database_file: PathBuf,
    pub model_dir: PathBuf,
    pub log_dir: PathBuf,
}

impl AppPaths {
    pub fn user() -> Result<Self> {
        let project = ProjectDirs::from("com", "LumenFind", APP_NAME)
            .context("Windows did not provide a local application-data folder")?;
        Ok(Self::under(project.data_local_dir().to_path_buf()))
    }

    pub fn service() -> Self {
        let base = std::env::var_os("PROGRAMDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"C:\ProgramData"));
        Self::under(base.join(APP_NAME))
    }

    pub fn under(data_dir: PathBuf) -> Self {
        Self {
            config_file: data_dir.join("config.json"),
            database_file: data_dir.join("photos.db"),
            model_dir: data_dir.join("models").join("embeddinggemma-2-q4"),
            log_dir: data_dir.join("logs"),
            data_dir,
        }
    }

    pub fn create(&self) -> Result<()> {
        std::fs::create_dir_all(&self.data_dir)?;
        std::fs::create_dir_all(&self.model_dir)?;
        std::fs::create_dir_all(&self.log_dir)?;
        Ok(())
    }
}
