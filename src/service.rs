use std::{
    ffi::OsString,
    path::PathBuf,
    sync::{OnceLock, mpsc},
    time::{Duration, Instant},
};

use anyhow::{Result, bail};
use windows_service::{
    define_windows_service,
    service::{
        ServiceAccess, ServiceControl, ServiceControlAccept, ServiceErrorControl, ServiceExitCode,
        ServiceInfo, ServiceStartType, ServiceState, ServiceStatus, ServiceType,
    },
    service_control_handler::{self, ServiceControlHandlerResult},
    service_dispatcher,
    service_manager::{ServiceManager, ServiceManagerAccess},
};

use crate::{
    config::Config,
    db::PhotoDb,
    indexer,
    model::{EmbeddingModel, model_files},
    paths::AppPaths,
};

pub const SERVICE_NAME: &str = "LumenFindIndexer";
const SERVICE_DISPLAY_NAME: &str = "LumenFind Offline Photo Indexer";
static SERVICE_PATHS: OnceLock<AppPaths> = OnceLock::new();

define_windows_service!(ffi_service_main, service_main);

pub fn dispatch(data_dir: PathBuf, model_dir: PathBuf) -> Result<()> {
    let mut paths = AppPaths::under(data_dir);
    paths.model_dir = model_dir;
    let _ = SERVICE_PATHS.set(paths);
    service_dispatcher::start(SERVICE_NAME, ffi_service_main)?;
    Ok(())
}

fn service_main(_arguments: Vec<OsString>) {
    if let Err(error) = run_service() {
        tracing::error!(%error, "service stopped with an error");
    }
}

fn run_service() -> Result<()> {
    let (stop_sender, stop_receiver) = mpsc::channel();
    let status_handle =
        service_control_handler::register(SERVICE_NAME, move |control| match control {
            ServiceControl::Stop => {
                let _ = stop_sender.send(());
                ServiceControlHandlerResult::NoError
            }
            ServiceControl::Interrogate => ServiceControlHandlerResult::NoError,
            _ => ServiceControlHandlerResult::NotImplemented,
        })?;
    status_handle.set_service_status(status(ServiceState::Running, ServiceControlAccept::STOP))?;

    let paths = SERVICE_PATHS.get().cloned().unwrap_or(AppPaths::service());
    paths.create()?;
    loop {
        if let Err(error) = run_one_scan(&paths) {
            tracing::error!(%error, "background scan failed");
        }
        let interval = Config::load(&paths)
            .unwrap_or_default()
            .scan_interval_minutes
            .max(1);
        if stop_receiver
            .recv_timeout(Duration::from_secs(interval * 60))
            .is_ok()
        {
            break;
        }
    }
    status_handle
        .set_service_status(status(ServiceState::Stopped, ServiceControlAccept::empty()))?;
    Ok(())
}

fn status(state: ServiceState, controls: ServiceControlAccept) -> ServiceStatus {
    ServiceStatus {
        service_type: ServiceType::OWN_PROCESS,
        current_state: state,
        controls_accepted: controls,
        exit_code: ServiceExitCode::Win32(0),
        checkpoint: 0,
        wait_hint: Duration::ZERO,
        process_id: None,
    }
}

fn run_one_scan(paths: &AppPaths) -> Result<()> {
    let config = Config::load(paths)?;
    if config.photo_roots.is_empty() || config.indexing_paused {
        return Ok(());
    }
    let database = PhotoDb::open(&paths.database_file)?;
    let mut model = EmbeddingModel::load(&paths.model_dir)?;
    let mut last_pause_check = Instant::now() - Duration::from_secs(1);
    let mut paused = false;
    indexer::scan(
        &config.photo_roots,
        &database,
        &mut model,
        || {
            if last_pause_check.elapsed() >= Duration::from_millis(500) {
                paused = Config::load(paths)
                    .map(|current| current.indexing_paused)
                    .unwrap_or(false);
                last_pause_check = Instant::now();
            }
            paused
        },
        |_, _| {},
    )?;
    Ok(())
}

pub fn install() -> Result<()> {
    let source_executable = std::env::current_exe()?;
    let user_paths = AppPaths::user()?;
    if !user_paths.model_dir.join("model_q4.onnx").is_file() {
        bail!("download the model before installing the service");
    }
    let program_files = std::env::var_os("ProgramFiles")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Program Files"));
    let install_dir = program_files.join("LumenFind");
    let installed_model_dir = install_dir.join("models").join("embeddinggemma-2-q4");
    std::fs::create_dir_all(&installed_model_dir)?;
    let installed_executable = install_dir.join("lumenfind.exe");
    std::fs::copy(source_executable, &installed_executable)?;
    for source in model_files(&user_paths.model_dir) {
        let name = source
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("model file has no name"))?;
        std::fs::copy(&source, installed_model_dir.join(name))?;
    }

    let manager = ServiceManager::local_computer(
        None::<&str>,
        ServiceManagerAccess::CONNECT | ServiceManagerAccess::CREATE_SERVICE,
    )?;
    let info = ServiceInfo {
        name: OsString::from(SERVICE_NAME),
        display_name: OsString::from(SERVICE_DISPLAY_NAME),
        service_type: ServiceType::OWN_PROCESS,
        start_type: ServiceStartType::AutoStart,
        error_control: ServiceErrorControl::Normal,
        executable_path: installed_executable,
        launch_arguments: vec![
            OsString::from("service"),
            OsString::from("--data-dir"),
            user_paths.data_dir.as_os_str().to_owned(),
            OsString::from("--model-dir"),
            installed_model_dir.as_os_str().to_owned(),
        ],
        dependencies: vec![],
        account_name: None,
        account_password: None,
    };
    let service =
        manager.create_service(&info, ServiceAccess::START | ServiceAccess::CHANGE_CONFIG)?;
    service.set_description(
        "Indexes configured photo folders locally for LumenFind semantic search.",
    )?;
    service.start::<&OsString>(&[])?;
    Ok(())
}

pub fn uninstall() -> Result<()> {
    let manager = ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CONNECT)?;
    let service = manager.open_service(
        SERVICE_NAME,
        ServiceAccess::STOP | ServiceAccess::DELETE | ServiceAccess::QUERY_STATUS,
    )?;
    if service.query_status()?.current_state != ServiceState::Stopped {
        let _ = service.stop();
        for _ in 0..40 {
            if service.query_status()?.current_state == ServiceState::Stopped {
                break;
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }
    service.delete()?;
    drop(service);
    let program_files = std::env::var_os("ProgramFiles")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Program Files"));
    let install_dir = program_files.join("LumenFind");
    if install_dir.starts_with(&program_files)
        && install_dir
            .file_name()
            .is_some_and(|name| name == "LumenFind")
        && install_dir.is_dir()
    {
        std::fs::remove_dir_all(install_dir)?;
    }
    Ok(())
}
