use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use anyhow::{Context, Result};
use walkdir::WalkDir;

use crate::{
    db::{PhotoDb, PhotoRecord},
    model::EmbeddingModel,
};

#[derive(Clone, Debug, Default)]
pub struct ScanSummary {
    pub total: usize,
    pub processed: usize,
    pub discovered: usize,
    pub indexed: usize,
    pub unchanged: usize,
    pub removed: usize,
    pub failed: usize,
    pub paused: bool,
}

pub fn scan(
    roots: &[PathBuf],
    database: &PhotoDb,
    model: &mut EmbeddingModel,
    mut is_paused: impl FnMut() -> bool,
    mut progress: impl FnMut(&Path, &ScanSummary),
) -> Result<ScanSummary> {
    let mut summary = ScanSummary::default();
    let mut seen = HashSet::new();
    let available_roots = roots
        .iter()
        .filter(|root| {
            let available = root.is_dir();
            if !available {
                tracing::warn!(root = %root.display(), "configured photo folder is unavailable");
            }
            available
        })
        .cloned()
        .collect::<Vec<_>>();
    for root in &available_roots {
        for entry in WalkDir::new(root)
            .follow_links(false)
            .into_iter()
            .filter_map(Result::ok)
        {
            if is_paused() {
                summary.paused = true;
                return Ok(summary);
            }
            if entry.file_type().is_file() && is_supported_photo(entry.path()) {
                summary.total += 1;
            }
        }
    }
    for root in &available_roots {
        for entry in WalkDir::new(root)
            .follow_links(false)
            .into_iter()
            .filter_map(Result::ok)
        {
            if is_paused() {
                summary.paused = true;
                break;
            }
            if !entry.file_type().is_file() || !is_supported_photo(entry.path()) {
                continue;
            }
            let path = entry.path().to_path_buf();
            seen.insert(path.clone());
            summary.discovered += 1;
            let result = (|| -> Result<()> {
                let metadata = entry.metadata()?;
                let modified_ms = metadata
                    .modified()?
                    .duration_since(UNIX_EPOCH)
                    .context("photo modification time predates 1970")?
                    .as_millis() as i64;
                let byte_len = metadata.len() as i64;
                if database.is_current(&path, modified_ms, byte_len)? {
                    summary.unchanged += 1;
                    return Ok(());
                }
                let embedding = model.embed_image_path(&path)?;
                database.upsert(&PhotoRecord {
                    path: path.clone(),
                    modified_unix_ms: modified_ms,
                    byte_len,
                    embedding,
                })?;
                summary.indexed += 1;
                Ok(())
            })();
            if let Err(error) = result {
                summary.failed += 1;
                tracing::warn!(photo = %path.display(), %error, "could not index photo");
            }
            summary.processed += 1;
            progress(&path, &summary);
        }
        if summary.paused {
            break;
        }
    }
    // A partial scan has not visited every file. Removing unseen rows here would
    // incorrectly treat the unvisited portion as deleted.
    if !summary.paused {
        summary.removed = database.remove_missing(&available_roots, &seen)?;
    }
    Ok(summary)
}

pub fn is_supported_photo(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|value| value.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("jpg" | "jpeg" | "png" | "webp" | "bmp" | "gif" | "tif" | "tiff" | "heic" | "heif")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_photo_extensions_case_insensitively() {
        assert!(is_supported_photo(Path::new("holiday.JPEG")));
        assert!(is_supported_photo(Path::new("scan.tiff")));
        assert!(is_supported_photo(Path::new("IMG_8854.HEIC")));
        assert!(is_supported_photo(Path::new("image.heif")));
        assert!(!is_supported_photo(Path::new("notes.txt")));
    }
}
