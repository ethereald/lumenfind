#![cfg_attr(windows, windows_subsystem = "windows")]

use anyhow::Result;
use lumenfind::{config::Config, paths::AppPaths};

fn main() -> Result<()> {
    let paths = AppPaths::user()?;
    paths.create()?;
    let config = Config::load(&paths)?;
    let visible = !config.start_hidden
        || config.photo_roots.is_empty()
        || !lumenfind::model::model_is_ready(&paths.model_dir);
    #[cfg(windows)]
    return lumenfind::ui::run(paths, config, visible);
    #[cfg(not(windows))]
    anyhow::bail!("LumenFind currently supports Windows only");
}
