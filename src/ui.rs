use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    time::{Duration, Instant},
};

use anyhow::Result;
use eframe::egui::{
    self, Color32, ColorImage, Context, FontData, FontDefinitions, FontFamily, TextureHandle, Vec2,
};
use global_hotkey::{
    GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState,
    hotkey::{Code, HotKey, Modifiers},
};
use tray_icon::{
    Icon, TrayIcon, TrayIconBuilder,
    menu::{Menu, MenuEvent, MenuId, MenuItem},
};

use crate::{
    config::Config,
    db::{PhotoDb, SearchHit},
    image_loader::load_image,
    indexer::{self, ScanSummary},
    model::{EmbeddingModel, download_model, model_is_ready, required_download_bytes},
    paths::AppPaths,
};

enum WorkerCommand {
    Download,
    Index,
    Search(String),
    Stop,
}

enum WorkerEvent {
    Progress(String),
    IndexProgress {
        name: String,
        processed: usize,
        total: usize,
    },
    ModelReady,
    Indexed(ScanSummary, usize),
    Results(String, Vec<SearchHit>),
    Error(String),
}

pub fn run(paths: AppPaths, config: Config, initially_visible: bool) -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("LumenFind")
            .with_inner_size([760.0, 560.0])
            .with_min_inner_size([560.0, 360.0])
            .with_decorations(false)
            .with_taskbar(false)
            .with_resizable(true)
            .with_visible(initially_visible),
        ..Default::default()
    };
    eframe::run_native(
        "LumenFind",
        options,
        Box::new(move |creation| {
            configure_style(&creation.egui_ctx);
            Ok(Box::new(LumenApp::new(paths, config, initially_visible)?))
        }),
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))
}

struct LumenApp {
    paths: AppPaths,
    config: Config,
    query: String,
    submitted_query: String,
    query_changed_at: Instant,
    hits: Vec<SearchHit>,
    results_page: usize,
    thumbnails: HashMap<PathBuf, TextureHandle>,
    status: String,
    model_ready: bool,
    indexed_count: usize,
    index_progress: Option<(usize, usize)>,
    command_tx: Sender<WorkerCommand>,
    event_rx: Receiver<WorkerEvent>,
    hotkey_manager: GlobalHotKeyManager,
    hotkey: HotKey,
    _tray: TrayIcon,
    show_menu_id: MenuId,
    index_menu_id: MenuId,
    pause_menu_id: MenuId,
    index_menu_item: MenuItem,
    pause_menu_item: MenuItem,
    quit_menu_id: MenuId,
    indexing_paused: Arc<AtomicBool>,
    visible: bool,
    focus_query: bool,
    quitting: bool,
}

impl LumenApp {
    fn new(paths: AppPaths, config: Config, visible: bool) -> Result<Self> {
        let model_ready = model_is_ready(&paths.model_dir);
        let indexed_count = PhotoDb::open(&paths.database_file)?.count()?;
        let (command_tx, command_rx) = mpsc::channel();
        let (event_tx, event_rx) = mpsc::channel();
        let indexing_paused = Arc::new(AtomicBool::new(config.indexing_paused));
        spawn_worker(
            paths.clone(),
            config.clone(),
            Arc::clone(&indexing_paused),
            command_rx,
            event_tx,
        );

        let hotkey_manager = GlobalHotKeyManager::new()?;
        let hotkey = HotKey::new(Some(Modifiers::CONTROL | Modifiers::ALT), Code::Space);
        hotkey_manager.register(hotkey)?;

        let menu = Menu::new();
        let show_item = MenuItem::new("Show LumenFind", true, None);
        let index_item = MenuItem::new("Index now", !config.indexing_paused, None);
        let pause_item = MenuItem::new(
            if config.indexing_paused {
                "Resume indexing"
            } else {
                "Pause indexing"
            },
            true,
            None,
        );
        let quit_item = MenuItem::new("Quit", true, None);
        let show_menu_id = show_item.id().clone();
        let index_menu_id = index_item.id().clone();
        let pause_menu_id = pause_item.id().clone();
        let quit_menu_id = quit_item.id().clone();
        menu.append_items(&[&show_item, &index_item, &pause_item, &quit_item])?;
        let tray = TrayIconBuilder::new()
            .with_menu(Box::new(menu))
            .with_tooltip("LumenFind - Ctrl+Alt+Space")
            .with_icon(make_tray_icon()?)
            .build()?;

        Ok(Self {
            paths,
            config,
            query: String::new(),
            submitted_query: String::new(),
            query_changed_at: Instant::now(),
            hits: Vec::new(),
            results_page: 0,
            thumbnails: HashMap::new(),
            status: if model_ready {
                format!("{indexed_count} photos indexed")
            } else {
                "Model setup required".into()
            },
            model_ready,
            indexed_count,
            index_progress: None,
            command_tx,
            event_rx,
            hotkey_manager,
            hotkey,
            _tray: tray,
            show_menu_id,
            index_menu_id,
            pause_menu_id,
            index_menu_item: index_item,
            pause_menu_item: pause_item,
            quit_menu_id,
            indexing_paused,
            visible,
            focus_query: visible,
            quitting: false,
        })
    }

    fn set_visible(&mut self, ctx: &Context, visible: bool) {
        self.visible = visible;
        ctx.send_viewport_cmd(egui::ViewportCommand::Visible(visible));
        if visible {
            ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
            self.focus_query = true;
        }
    }

    fn receive_events(&mut self, ctx: &Context) {
        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                WorkerEvent::Progress(status) => self.status = status,
                WorkerEvent::IndexProgress {
                    name,
                    processed,
                    total,
                } => {
                    self.index_progress = Some((processed, total));
                    self.status = format!("Checking {name}");
                }
                WorkerEvent::ModelReady => {
                    self.model_ready = true;
                    self.status = "Model ready - choose a photo folder".into();
                }
                WorkerEvent::Indexed(summary, count) => {
                    self.index_progress = None;
                    self.indexed_count = count;
                    self.status = if summary.paused {
                        format!("Indexing paused | {count} photos remain searchable")
                    } else {
                        format!(
                            "{count} photos indexed | {} new/changed | {} unchanged | {} failed",
                            summary.indexed, summary.unchanged, summary.failed
                        )
                    };
                }
                WorkerEvent::Results(query, hits) if query == self.query.trim() => {
                    self.hits = hits;
                    self.results_page = 0;
                    self.thumbnails.clear();
                    self.status = format!(
                        "{} matches | {} photos indexed",
                        self.hits.len(),
                        self.indexed_count
                    );
                }
                WorkerEvent::Results(_, _) => {}
                WorkerEvent::Error(error) => self.status = format!("Error: {error}"),
            }
            ctx.request_repaint();
        }
    }

    fn handle_global_events(&mut self, ctx: &Context) {
        while let Ok(event) = GlobalHotKeyEvent::receiver().try_recv() {
            if event.id == self.hotkey.id() && event.state == HotKeyState::Pressed {
                self.set_visible(ctx, !self.visible);
            }
        }
        while let Ok(event) = MenuEvent::receiver().try_recv() {
            if event.id == self.show_menu_id {
                self.set_visible(ctx, true);
            } else if event.id == self.index_menu_id {
                let _ = self.command_tx.send(WorkerCommand::Index);
            } else if event.id == self.pause_menu_id {
                self.set_indexing_paused(!self.config.indexing_paused);
            } else if event.id == self.quit_menu_id {
                self.quitting = true;
                let _ = self.command_tx.send(WorkerCommand::Stop);
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
    }

    fn set_indexing_paused(&mut self, paused: bool) {
        let previous = self.config.indexing_paused;
        self.config.indexing_paused = paused;
        if let Err(error) = self.config.save(&self.paths) {
            self.config.indexing_paused = previous;
            self.status = format!("Error: {error}");
            return;
        }
        self.indexing_paused.store(paused, Ordering::Release);
        self.index_menu_item.set_enabled(!paused);
        self.pause_menu_item.set_text(if paused {
            "Resume indexing"
        } else {
            "Pause indexing"
        });
        if paused {
            self.index_progress = None;
            self.status = format!(
                "Indexing paused | {} photos remain searchable",
                self.indexed_count
            );
        } else {
            self.status = "Indexing resumed - checking for changes...".into();
            let _ = self.command_tx.send(WorkerCommand::Index);
        }
    }

    fn header(&mut self, ui: &mut egui::Ui, ctx: &Context) {
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("LF")
                    .size(18.0)
                    .color(Color32::from_rgb(116, 142, 255)),
            );
            ui.label(egui::RichText::new("LumenFind").size(18.0).strong());
            ui.add_space(8.0);
            ui.label(egui::RichText::new("Private photo search").color(Color32::from_gray(145)));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button("X")
                    .on_hover_text("Hide to notification area")
                    .clicked()
                {
                    self.set_visible(ctx, false);
                }
                if ui.button("_").clicked() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true));
                }
            });
        });
        ui.add_space(10.0);
        let response = ui.add_sized(
            [ui.available_width(), 46.0],
            egui::TextEdit::singleline(&mut self.query)
                .hint_text("Search photos...  e.g. \"red kayak on a lake\"")
                .font(egui::TextStyle::Heading)
                .margin(Vec2::new(14.0, 10.0)),
        );
        if self.focus_query {
            response.request_focus();
            self.focus_query = false;
        }
        if response.changed() {
            self.query_changed_at = Instant::now();
        }
        if ui.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.set_visible(ctx, false);
        }
    }

    fn setup_panel(&mut self, ui: &mut egui::Ui) {
        ui.add_space(22.0);
        ui.heading("Finish local setup");
        ui.label("LumenFind sends no photos or searches anywhere. The model and index remain on this PC.");
        ui.add_space(12.0);
        if !self.model_ready {
            let size_mib = required_download_bytes() / 1024 / 1024;
            if ui
                .button(format!("Download EmbeddingGemma 2 ({size_mib} MiB)"))
                .clicked()
            {
                self.status = "Starting model download...".into();
                let _ = self.command_tx.send(WorkerCommand::Download);
            }
        } else if ui.button("Add photo folder").clicked()
            && let Some(folder) = rfd::FileDialog::new().pick_folder()
            && !self.config.photo_roots.contains(&folder)
        {
            self.config.photo_roots.push(folder);
            if let Err(error) = self.config.save(&self.paths) {
                self.status = format!("Error: {error}");
            } else {
                let _ = self.command_tx.send(WorkerCommand::Index);
            }
        }
        if !self.config.photo_roots.is_empty() {
            ui.add_space(14.0);
            ui.label(egui::RichText::new("Indexed folders").strong());
            for root in self.config.photo_roots.clone() {
                ui.horizontal(|ui| {
                    ui.label(root.display().to_string());
                    if ui.small_button("Remove").clicked() {
                        self.config.photo_roots.retain(|value| value != &root);
                        let _ = self.config.save(&self.paths);
                    }
                });
            }
            ui.horizontal(|ui| {
                let label = if self.config.indexing_paused {
                    "Resume indexing"
                } else {
                    "Pause indexing"
                };
                if ui.button(label).clicked() {
                    self.set_indexing_paused(!self.config.indexing_paused);
                }
            });
            if ui
                .add_enabled(!self.config.indexing_paused, egui::Button::new("Index now"))
                .clicked()
            {
                let _ = self.command_tx.send(WorkerCommand::Index);
            }
        }
    }

    fn results(&mut self, ui: &mut egui::Ui, ctx: &Context) {
        let page_size = self.config.result_limit.max(1);
        let page_count = self.hits.len().div_ceil(page_size);
        self.results_page = self.results_page.min(page_count.saturating_sub(1));
        let start = self.results_page * page_size;
        let end = (start + page_size).min(self.hits.len());
        let hits = self.hits[start..end].to_vec();
        if page_count > 1 {
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(self.results_page > 0, egui::Button::new("Previous"))
                    .clicked()
                {
                    self.results_page -= 1;
                }
                ui.label(format!(
                    "Page {} of {} | results {}-{} of {}",
                    self.results_page + 1,
                    page_count,
                    start + 1,
                    end,
                    self.hits.len()
                ));
                if ui
                    .add_enabled(
                        self.results_page + 1 < page_count,
                        egui::Button::new("Next"),
                    )
                    .clicked()
                {
                    self.results_page += 1;
                }
            });
        }
        egui::ScrollArea::vertical().show(ui, |ui| {
            let columns = ((ui.available_width() / 170.0).floor() as usize).max(1);
            egui::Grid::new("photo-results")
                .num_columns(columns)
                .spacing([10.0, 12.0])
                .show(ui, |ui| {
                    for (index, hit) in hits.iter().enumerate() {
                        ui.vertical(|ui| {
                            let texture = self.thumbnail(ctx, &hit.path);
                            let clicked = if let Some(texture) = texture {
                                ui.add(
                                    egui::Button::image((texture.id(), Vec2::new(150.0, 110.0)))
                                        .frame(false),
                                )
                                .clicked()
                            } else {
                                ui.add_sized(
                                    [150.0, 110.0],
                                    egui::Button::new("Preview unavailable"),
                                )
                                .clicked()
                            };
                            if clicked && let Err(error) = opener::open(&hit.path) {
                                self.status = format!("Could not open photo: {error}");
                            }
                            let name = hit
                                .path
                                .file_name()
                                .and_then(|value| value.to_str())
                                .unwrap_or("Photo");
                            ui.add(
                                egui::Label::new(egui::RichText::new(name).size(12.0)).truncate(),
                            );
                            ui.label(
                                egui::RichText::new(format!("{:.0}%", hit.score * 100.0))
                                    .size(10.0)
                                    .color(Color32::GRAY),
                            );
                        });
                        if (index + 1) % columns == 0 {
                            ui.end_row();
                        }
                    }
                });
        });
    }

    fn thumbnail(&mut self, ctx: &Context, path: &Path) -> Option<&TextureHandle> {
        if !self.thumbnails.contains_key(path) {
            let image = match load_image(path) {
                Ok(image) => image,
                Err(error) => {
                    tracing::warn!(photo = %path.display(), %error, "could not load photo preview");
                    return None;
                }
            };
            let thumb = image.thumbnail(300, 220).to_rgba8();
            let size = [thumb.width() as usize, thumb.height() as usize];
            let texture = ctx.load_texture(
                path.to_string_lossy(),
                ColorImage::from_rgba_unmultiplied(size, thumb.as_raw()),
                egui::TextureOptions::LINEAR,
            );
            self.thumbnails.insert(path.to_path_buf(), texture);
        }
        self.thumbnails.get(path)
    }
}

impl eframe::App for LumenApp {
    fn logic(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        let _keep_alive = &self.hotkey_manager;
        self.handle_global_events(ctx);
        self.receive_events(ctx);
        ctx.send_viewport_cmd(egui::ViewportCommand::RequestPaintWhileHidden);
        ctx.request_repaint_after(Duration::from_millis(50));
        if ctx.input(|input| input.viewport().close_requested()) && !self.quitting {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.set_visible(ctx, false);
        }
        if !self.query.trim().is_empty()
            && self.query.trim() != self.submitted_query
            && self.query_changed_at.elapsed() >= Duration::from_millis(220)
            && self.model_ready
        {
            self.submitted_query = self.query.trim().to_owned();
            self.hits.clear();
            self.results_page = 0;
            self.thumbnails.clear();
            let _ = self
                .command_tx
                .send(WorkerCommand::Search(self.submitted_query.clone()));
            self.status = "Searching locally...".into();
        }
    }

    fn ui(&mut self, root_ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = root_ui.ctx().clone();
        egui::CentralPanel::default().show(root_ui, |ui| {
            self.header(ui, &ctx);
            ui.separator();
            if !self.model_ready
                || self.config.photo_roots.is_empty()
                || self.query.trim().is_empty()
            {
                self.setup_panel(ui);
            } else {
                self.results(ui, &ctx);
            }
            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(&self.status)
                            .size(11.0)
                            .color(Color32::GRAY),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new("Ctrl Alt Space")
                                .size(10.0)
                                .color(Color32::DARK_GRAY),
                        );
                    });
                });
                if let Some((processed, total)) = self.index_progress {
                    let fraction = if total == 0 {
                        0.0
                    } else {
                        processed as f32 / total as f32
                    };
                    ui.add(
                        egui::ProgressBar::new(fraction.clamp(0.0, 1.0))
                            .show_percentage()
                            .text(format!("Indexing {processed} / {total}")),
                    );
                }
            });
        });
        ctx.request_repaint_after(Duration::from_millis(50));
    }
}

impl Drop for LumenApp {
    fn drop(&mut self) {
        let _ = self.hotkey_manager.unregister(self.hotkey);
        let _ = self.command_tx.send(WorkerCommand::Stop);
    }
}

fn spawn_worker(
    paths: AppPaths,
    initial_config: Config,
    indexing_paused: Arc<AtomicBool>,
    rx: Receiver<WorkerCommand>,
    tx: Sender<WorkerEvent>,
) {
    std::thread::spawn(move || {
        let mut model: Option<EmbeddingModel> = None;
        if model_is_ready(&paths.model_dir) {
            match EmbeddingModel::load(&paths.model_dir) {
                Ok(value) => model = Some(value),
                Err(error) => {
                    let _ = tx.send(WorkerEvent::Error(error.to_string()));
                }
            }
        }
        let mut config = initial_config;
        while let Ok(command) = rx.recv() {
            if matches!(command, WorkerCommand::Stop) {
                break;
            }
            let result: Result<()> = (|| {
                match command {
                    WorkerCommand::Download => {
                        download_model(&paths.model_dir, |name, received, total| {
                            let percent = total
                                .map(|value| received as f64 / value as f64 * 100.0)
                                .unwrap_or(0.0);
                            let _ = tx.send(WorkerEvent::Progress(format!(
                                "Downloading {name} - {percent:.0}%"
                            )));
                        })?;
                        model = Some(EmbeddingModel::load(&paths.model_dir)?);
                        let _ = tx.send(WorkerEvent::ModelReady);
                    }
                    WorkerCommand::Index => {
                        config = Config::load(&paths)?;
                        if indexing_paused.load(Ordering::Acquire) || config.indexing_paused {
                            let _ = tx.send(WorkerEvent::Progress(
                                "Indexing paused - search remains available".into(),
                            ));
                            return Ok(());
                        }
                        let active = model
                            .as_mut()
                            .ok_or_else(|| anyhow::anyhow!("download the model first"))?;
                        let database = PhotoDb::open(&paths.database_file)?;
                        let _ = tx.send(WorkerEvent::Progress(
                            "Counting photos in selected folders...".into(),
                        ));
                        let summary = indexer::scan(
                            &config.photo_roots,
                            &database,
                            active,
                            || indexing_paused.load(Ordering::Acquire),
                            |path, progress| {
                                let name = path
                                    .file_name()
                                    .and_then(|value| value.to_str())
                                    .unwrap_or("photo");
                                let _ = tx.send(WorkerEvent::IndexProgress {
                                    name: name.to_owned(),
                                    processed: progress.processed,
                                    total: progress.total,
                                });
                            },
                        )?;
                        let _ = tx.send(WorkerEvent::Indexed(summary, database.count()?));
                    }
                    WorkerCommand::Search(query) => {
                        let active = model
                            .as_mut()
                            .ok_or_else(|| anyhow::anyhow!("download the model first"))?;
                        let embedding = active.embed_query(&query)?;
                        let hits =
                            PhotoDb::open(&paths.database_file)?.search(&embedding, usize::MAX)?;
                        let _ = tx.send(WorkerEvent::Results(query, hits));
                    }
                    WorkerCommand::Stop => unreachable!(),
                }
                Ok(())
            })();
            if let Err(error) = result {
                let _ = tx.send(WorkerEvent::Error(format!("{error:#}")));
            }
        }
    });
}

fn configure_style(ctx: &Context) {
    configure_windows_fonts(ctx);
    let mut visuals = egui::Visuals::dark();
    visuals.window_corner_radius = egui::CornerRadius::same(14);
    visuals.panel_fill = Color32::from_rgb(19, 20, 25);
    visuals.extreme_bg_color = Color32::from_rgb(29, 31, 39);
    visuals.selection.bg_fill = Color32::from_rgb(75, 96, 195);
    ctx.set_visuals(visuals);
}

pub fn configure_windows_fonts(ctx: &Context) {
    let windows_dir = std::env::var_os("WINDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
    let font_dir = windows_dir.join("Fonts");
    let candidates = [
        ("Segoe UI", "segoeui.ttf", true),
        ("Noto Sans SC", "NotoSansSC-VF.ttf", false),
        ("Microsoft YaHei", "msyh.ttc", false),
        ("Yu Gothic", "YuGothR.ttc", false),
        ("Malgun Gothic", "malgun.ttf", false),
        ("Segoe UI Emoji", "seguiemj.ttf", false),
    ];
    let mut fonts = FontDefinitions::default();
    for (name, file_name, primary) in candidates {
        let Ok(bytes) = std::fs::read(font_dir.join(file_name)) else {
            continue;
        };
        fonts
            .font_data
            .insert(name.to_owned(), Arc::new(FontData::from_owned(bytes)));
        for family in [FontFamily::Proportional, FontFamily::Monospace] {
            if let Some(chain) = fonts.families.get_mut(&family) {
                if primary && family == FontFamily::Proportional {
                    chain.insert(0, name.to_owned());
                } else {
                    chain.push(name.to_owned());
                }
            }
        }
    }
    ctx.set_fonts(fonts);
}

fn make_tray_icon() -> Result<Icon> {
    let size = 32_u32;
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - 15.5;
            let dy = y as f32 - 15.5;
            let radius = (dx * dx + dy * dy).sqrt();
            if radius < 13.5 {
                rgba.extend_from_slice(&[94, 119, 255, 255]);
            } else {
                rgba.extend_from_slice(&[0, 0, 0, 0]);
            }
        }
    }
    Ok(Icon::from_rgba(rgba, size, size)?)
}
