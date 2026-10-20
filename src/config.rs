use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::paths::AppPaths;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub photo_roots: Vec<PathBuf>,
    pub result_limit: usize,
    pub scan_interval_minutes: u64,
    pub start_hidden: bool,
    pub indexing_paused: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            photo_roots: Vec::new(),
            result_limit: 24,
            scan_interval_minutes: 10,
            start_hidden: true,
            indexing_paused: false,
        }
    }
}

impl Config {
    pub fn load(paths: &AppPaths) -> Result<Self> {
        if !paths.config_file.exists() {
            return Ok(Self::default());
        }
        let bytes = fs::read(&paths.config_file)
            .with_context(|| format!("could not read {}", paths.config_file.display()))?;
        serde_json::from_slice(&bytes).context("configuration is not valid JSON")
    }

    pub fn save(&self, paths: &AppPaths) -> Result<()> {
        paths.create()?;
        let temporary = paths.config_file.with_extension("json.tmp");
        fs::write(&temporary, serde_json::to_vec_pretty(self)?)?;
        if fs::rename(&temporary, &paths.config_file).is_err() {
            // Windows does not replace an existing destination with rename.
            fs::copy(&temporary, &paths.config_file)?;
            fs::remove_file(&temporary)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn updates_an_existing_configuration_file() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = AppPaths::under(temporary.path().to_path_buf());
        let mut config = Config::default();
        config.save(&paths).unwrap();
        config.indexing_paused = true;
        config.save(&paths).unwrap();
        assert!(Config::load(&paths).unwrap().indexing_paused);
    }
}
