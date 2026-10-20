use anyhow::Result;
use clap::{Parser, Subcommand};
use lumenfind::{config::Config, paths::AppPaths};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "lumenfind",
    version,
    about = "Private offline semantic photo search"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run as a Windows Service (called by the Service Control Manager).
    Service {
        #[arg(long)]
        data_dir: PathBuf,
        #[arg(long)]
        model_dir: PathBuf,
    },
    /// Install and start the background indexer service (requires Administrator).
    InstallService,
    /// Stop and remove the background indexer service (requires Administrator).
    UninstallService,
    /// Index configured photo folders once without opening the UI.
    Index,
    /// Download and verify the offline EmbeddingGemma 2 model.
    DownloadModel,
    /// Run a text query and print matching local photo paths.
    Search { query: String },
    /// Pause all manual and service indexing while keeping search available.
    PauseIndexing,
    /// Resume indexing and allow the service to scan for changes.
    ResumeIndexing,
    /// Verify the local tokenizer, text model, and vision model end to end.
    Doctor,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(Command::Service {
        data_dir,
        model_dir,
    }) = &cli.command
    {
        let mut service_paths = AppPaths::under(data_dir.clone());
        service_paths.model_dir = model_dir.clone();
        service_paths.create()?;
        init_logging(&service_paths);
        #[cfg(windows)]
        return lumenfind::service::dispatch(data_dir.clone(), model_dir.clone());
        #[cfg(not(windows))]
        anyhow::bail!("LumenFind services currently support Windows only");
    }
    let paths = AppPaths::user()?;
    paths.create()?;
    init_logging(&paths);

    match cli.command {
        #[cfg(windows)]
        Some(Command::Service { .. }) => unreachable!(),
        #[cfg(windows)]
        Some(Command::InstallService) => lumenfind::service::install(),
        #[cfg(windows)]
        Some(Command::UninstallService) => lumenfind::service::uninstall(),
        Some(Command::Index) => index_once(paths),
        Some(Command::DownloadModel) => {
            lumenfind::model::download_model(&paths.model_dir, |name, received, total| {
                let percent = total
                    .map(|size| received as f64 / size as f64 * 100.0)
                    .unwrap_or(0.0);
                println!("{name}: {percent:.0}%");
            })
        }
        Some(Command::Search { query }) => search_once(paths, &query),
        Some(Command::PauseIndexing) => set_indexing_paused(paths, true),
        Some(Command::ResumeIndexing) => set_indexing_paused(paths, false),
        Some(Command::Doctor) => doctor(paths),
        None => {
            let config = Config::load(&paths)?;
            let visible = !config.start_hidden
                || config.photo_roots.is_empty()
                || !lumenfind::model::model_is_ready(&paths.model_dir);
            #[cfg(windows)]
            return lumenfind::ui::run(paths, config, visible);
            #[cfg(not(windows))]
            anyhow::bail!("LumenFind currently supports Windows only");
        }
    }
}

fn search_once(paths: AppPaths, query: &str) -> Result<()> {
    let config = Config::load(&paths)?;
    let mut model = lumenfind::model::EmbeddingModel::load(&paths.model_dir)?;
    let embedding = model.embed_query(query)?;
    for hit in lumenfind::db::PhotoDb::open(&paths.database_file)?
        .search(&embedding, config.result_limit)?
    {
        println!("{:.4}\t{}", hit.score, hit.path.display());
    }
    Ok(())
}

fn doctor(paths: AppPaths) -> Result<()> {
    let mut model = lumenfind::model::EmbeddingModel::load(&paths.model_dir)?;
    let text = model.embed_query("a solid blue square")?;
    let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        96,
        96,
        image::Rgb([20, 80, 220]),
    ));
    let vision = model.embed_image(&image)?;
    let similarity: f32 = text
        .iter()
        .zip(&vision)
        .map(|(left, right)| left * right)
        .sum();
    println!(
        "OK: text and vision pipelines returned 768D unit vectors (similarity {similarity:.4})"
    );
    Ok(())
}

fn index_once(paths: AppPaths) -> Result<()> {
    let config = Config::load(&paths)?;
    if config.indexing_paused {
        println!("Indexing is paused. Run `lumenfind resume-indexing` to resume.");
        return Ok(());
    }
    let database = lumenfind::db::PhotoDb::open(&paths.database_file)?;
    let mut model = lumenfind::model::EmbeddingModel::load(&paths.model_dir)?;
    let summary = lumenfind::indexer::scan(
        &config.photo_roots,
        &database,
        &mut model,
        || false,
        |path, done| {
            println!("{} ({})", path.display(), done.indexed);
        },
    )?;
    println!(
        "Indexed {}, unchanged {}, failed {}",
        summary.indexed, summary.unchanged, summary.failed
    );
    Ok(())
}

fn set_indexing_paused(paths: AppPaths, paused: bool) -> Result<()> {
    let mut config = Config::load(&paths)?;
    config.indexing_paused = paused;
    config.save(&paths)?;
    println!(
        "Indexing {}. Search remains available.",
        if paused { "paused" } else { "resumed" }
    );
    Ok(())
}

fn init_logging(paths: &AppPaths) {
    let file = tracing_appender::rolling::daily(&paths.log_dir, "lumenfind.log");
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(file)
        .with_ansi(false)
        .try_init();
}
