#![cfg_attr(windows, windows_subsystem = "windows")]

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use eframe::egui::{self, Color32};
use lumenfind::paths::AppPaths;
use rusqlite::{Connection, OpenFlags};

#[derive(Clone)]
struct PhotoRow {
    path: PathBuf,
    modified: String,
    byte_len: i64,
    embedding: Vec<f32>,
    norm: f32,
}

#[derive(Default)]
struct Snapshot {
    schema: String,
    rows: Vec<PhotoRow>,
    source_bytes: i64,
    embedding_bytes: usize,
}

struct InspectorApp {
    database_path: PathBuf,
    snapshot: Snapshot,
    filter: String,
    selected: Option<usize>,
    error: Option<String>,
}

impl InspectorApp {
    fn new(database_path: PathBuf) -> Self {
        let mut app = Self {
            database_path,
            snapshot: Snapshot::default(),
            filter: String::new(),
            selected: None,
            error: None,
        };
        app.refresh();
        app
    }

    fn refresh(&mut self) {
        match load_snapshot(&self.database_path) {
            Ok(snapshot) => {
                self.snapshot = snapshot;
                self.selected = self
                    .selected
                    .filter(|&index| index < self.snapshot.rows.len());
                self.error = None;
            }
            Err(error) => self.error = Some(format!("{error:#}")),
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("LumenFind Index Inspector");
            if ui.button("Open database...").clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("SQLite database", &["db", "sqlite", "sqlite3"])
                    .set_directory(
                        self.database_path
                            .parent()
                            .unwrap_or_else(|| Path::new(".")),
                    )
                    .pick_file()
            {
                self.database_path = path;
                self.refresh();
            }
            if ui.button("Refresh").clicked() {
                self.refresh();
            }
        });
        ui.label(
            egui::RichText::new(self.database_path.display().to_string())
                .small()
                .color(Color32::GRAY),
        );
        if let Some(error) = &self.error {
            ui.colored_label(Color32::LIGHT_RED, error);
        } else {
            ui.horizontal(|ui| {
                ui.label(format!("{} photos", self.snapshot.rows.len()));
                ui.separator();
                ui.label(format!(
                    "{} source files",
                    format_bytes(self.snapshot.source_bytes.max(0) as u64)
                ));
                ui.separator();
                ui.label(format!(
                    "{} embeddings",
                    format_bytes(self.snapshot.embedding_bytes as u64)
                ));
            });
        }
        ui.add_space(6.0);
        ui.add_sized(
            [ui.available_width(), 34.0],
            egui::TextEdit::singleline(&mut self.filter)
                .hint_text("Filter by file name or path..."),
        );
    }

    fn record_table(&mut self, ui: &mut egui::Ui) {
        let filter = self.filter.to_ascii_lowercase();
        egui::ScrollArea::both()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::Grid::new("index-records")
                    .num_columns(5)
                    .striped(true)
                    .spacing([18.0, 6.0])
                    .show(ui, |ui| {
                        ui.strong("Photo path");
                        ui.strong("Modified");
                        ui.strong("File size");
                        ui.strong("Dimensions");
                        ui.strong("Vector norm");
                        ui.end_row();
                        for (index, row) in self.snapshot.rows.iter().enumerate() {
                            if !filter.is_empty()
                                && !row
                                    .path
                                    .to_string_lossy()
                                    .to_ascii_lowercase()
                                    .contains(&filter)
                            {
                                continue;
                            }
                            let selected = self.selected == Some(index);
                            if ui
                                .selectable_label(selected, row.path.display().to_string())
                                .clicked()
                            {
                                self.selected = Some(index);
                            }
                            ui.label(&row.modified);
                            ui.label(format_bytes(row.byte_len.max(0) as u64));
                            ui.label(row.embedding.len().to_string());
                            ui.label(format!("{:.6}", row.norm));
                            ui.end_row();
                        }
                    });
            });
    }

    fn details(&mut self, ui: &mut egui::Ui) {
        let Some(index) = self.selected else {
            ui.label("Select a row to inspect its embedding.");
            return;
        };
        let Some(row) = self.snapshot.rows.get(index) else {
            return;
        };
        ui.horizontal(|ui| {
            ui.strong("Selected record");
            if ui.button("Copy path").clicked() {
                ui.ctx().copy_text(row.path.display().to_string());
            }
            if ui.button("Open photo").clicked()
                && let Err(error) = opener::open(&row.path)
            {
                self.error = Some(format!("Could not open photo: {error}"));
            }
        });
        ui.label(row.path.display().to_string());
        ui.label(format!(
            "Modified: {} | Source: {} | Vector: {} x f32 ({} bytes) | Norm: {:.6}",
            row.modified,
            format_bytes(row.byte_len.max(0) as u64),
            row.embedding.len(),
            row.embedding.len() * size_of::<f32>(),
            row.norm
        ));
        ui.collapsing("Embedding values", |ui| {
            let values = row
                .embedding
                .iter()
                .enumerate()
                .map(|(index, value)| format!("{index:03}: {value:+.8}"))
                .collect::<Vec<_>>()
                .join("\n");
            egui::ScrollArea::vertical()
                .max_height(180.0)
                .show(ui, |ui| {
                    ui.monospace(values);
                });
        });
        ui.collapsing("Database schema", |ui| {
            ui.monospace(&self.snapshot.schema);
        });
    }
}

impl eframe::App for InspectorApp {
    fn ui(&mut self, root_ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(root_ui, |ui| {
            self.toolbar(ui);
            ui.separator();
            let available = (ui.available_height() - 220.0).max(180.0);
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), available),
                egui::Layout::top_down(egui::Align::LEFT),
                |ui| self.record_table(ui),
            );
            ui.separator();
            self.details(ui);
        });
    }
}

fn load_snapshot(path: &Path) -> Result<Snapshot> {
    if !path.is_file() {
        bail!("Database does not exist: {}", path.display());
    }
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .with_context(|| format!("Could not open {} read-only", path.display()))?;
    let schema: String = connection.query_row(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='photos'",
        [],
        |row| row.get(0),
    )?;
    let mut statement = connection.prepare(
        "SELECT path,
                datetime(modified_ms / 1000, 'unixepoch', 'localtime'),
                byte_len,
                embedding
         FROM photos ORDER BY path",
    )?;
    let mut rows = Vec::new();
    let mut source_bytes = 0_i64;
    let mut embedding_bytes = 0_usize;
    let records = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
            row.get::<_, Vec<u8>>(3)?,
        ))
    })?;
    for record in records {
        let (path, modified, byte_len, bytes) = record?;
        let (chunks, remainder) = bytes.as_chunks::<4>();
        if !remainder.is_empty() {
            bail!("{} has a malformed embedding", path);
        }
        let embedding = chunks
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect::<Vec<_>>();
        let norm = embedding
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        source_bytes += byte_len;
        embedding_bytes += bytes.len();
        rows.push(PhotoRow {
            path: path.into(),
            modified,
            byte_len,
            embedding,
            norm,
        });
    }
    Ok(Snapshot {
        schema,
        rows,
        source_bytes,
        embedding_bytes,
    })
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

fn main() -> Result<()> {
    let paths = AppPaths::user()?;
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("LumenFind Index Inspector")
            .with_inner_size([1180.0, 760.0])
            .with_min_inner_size([760.0, 520.0]),
        ..Default::default()
    };
    eframe::run_native(
        "LumenFind Index Inspector",
        options,
        Box::new(move |creation| {
            lumenfind::ui::configure_windows_fonts(&creation.egui_ctx);
            let mut visuals = egui::Visuals::dark();
            visuals.panel_fill = Color32::from_rgb(19, 20, 25);
            creation.egui_ctx.set_visuals(visuals);
            Ok(Box::new(InspectorApp::new(paths.database_file)))
        }),
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_file_sizes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1536), "1.50 KiB");
    }
}
