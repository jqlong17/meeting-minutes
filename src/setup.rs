use crate::asr::model_store;
use crate::cli::{ConfigInitArgs, SetupArgs};
use crate::config;
use crate::error::{AppError, AppResult};
use crate::util::display_path;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_MODEL_BASE_URL: &str =
    "https://huggingface.co/DennisHuang648/SenseVoiceSmall-onnx/resolve/main";
const MODEL_FILES: [&str; 3] = ["model.onnx", "tokens.json", "am.mvn"];

pub async fn run_setup(args: &SetupArgs) -> AppResult<()> {
    let model_dir = args
        .model_dir
        .clone()
        .unwrap_or_else(config::default_model_dir);
    let config_path = args
        .config
        .clone()
        .unwrap_or_else(config::default_user_config_path);

    if !args.skip_model_download {
        ensure_model_assets(&model_dir, args.model_base_url.as_deref(), args.force).await?;
        println!("ASR 模型目录: {}", display_path(&model_dir));
    }

    ensure_model_service_config(&config_path, args.force)?;
    println!("模型服务配置: {}", display_path(&config_path));

    let ffmpeg_bin = ensure_ffmpeg(args.ffmpeg_bin.clone(), args.skip_ffmpeg_install)?;
    println!("ffmpeg: {}", display_path(&ffmpeg_bin));
    println!("setup 完成");

    Ok(())
}

pub fn run_config_init(args: &ConfigInitArgs) -> AppResult<()> {
    let config_path = args
        .path
        .clone()
        .unwrap_or_else(config::default_user_config_path);
    ensure_model_service_config(&config_path, args.force)?;
    println!("模型服务配置已初始化: {}", display_path(&config_path));
    Ok(())
}

pub async fn ensure_model_assets(
    model_dir: &Path,
    model_base_url: Option<&str>,
    force: bool,
) -> AppResult<()> {
    if model_dir_ready(model_dir) && !force {
        return Ok(());
    }

    fs::create_dir_all(model_dir).map_err(|error| {
        AppError::output(format!(
            "创建模型目录失败 {}: {error}",
            display_path(model_dir)
        ))
    })?;

    if !force {
        if let Some(existing_dir) = existing_model_source(model_dir) {
            copy_model_assets(&existing_dir, model_dir)?;
            if model_dir_ready(model_dir) {
                return Ok(());
            }
        }
    }

    let base_url = model_base_url
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| std::env::var("MEETING_MINUTES_MODEL_BASE_URL").ok())
        .unwrap_or_else(|| DEFAULT_MODEL_BASE_URL.to_string());

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|error| AppError::config(format!("初始化下载客户端失败: {error}")))?;

    for file_name in MODEL_FILES {
        let file_path = model_dir.join(file_name);
        if file_path.exists() && !force {
            continue;
        }

        let url = format!("{}/{}", base_url.trim_end_matches('/'), file_name);
        let bytes = client
            .get(&url)
            .send()
            .await
            .map_err(|error| AppError::config(format!("下载模型文件失败 {}: {error}", url)))?
            .error_for_status()
            .map_err(|error| AppError::config(format!("下载模型文件失败 {}: {error}", url)))?
            .bytes()
            .await
            .map_err(|error| AppError::config(format!("读取模型文件失败 {}: {error}", url)))?;

        fs::write(&file_path, &bytes).map_err(|error| {
            AppError::output(format!(
                "写入模型文件失败 {}: {error}",
                display_path(&file_path)
            ))
        })?;
    }

    if !model_dir_ready(model_dir) {
        return Err(AppError::config(format!(
            "模型下载完成后仍不可用: {}",
            display_path(model_dir)
        )));
    }

    Ok(())
}

pub fn ensure_model_service_config(config_path: &Path, force: bool) -> AppResult<()> {
    if config_path.exists() && !force {
        return Ok(());
    }

    let example_path = PathBuf::from("config/model-service.example.toml");
    if !example_path.exists() {
        return Err(AppError::config(format!(
            "缺少模型服务示例配置: {}",
            display_path(&example_path)
        )));
    }

    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            AppError::output(format!(
                "创建配置目录失败 {}: {error}",
                display_path(parent)
            ))
        })?;
    }

    fs::copy(&example_path, config_path).map_err(|error| {
        AppError::output(format!(
            "复制模型配置失败 {} -> {}: {error}",
            display_path(&example_path),
            display_path(config_path)
        ))
    })?;

    Ok(())
}

pub fn ensure_ffmpeg(
    ffmpeg_override: Option<PathBuf>,
    skip_ffmpeg_install: bool,
) -> AppResult<PathBuf> {
    if let Some(path) = config::resolve_ffmpeg_bin(ffmpeg_override.clone()) {
        return Ok(path);
    }

    if skip_ffmpeg_install {
        return Err(AppError::config(
            "未找到 ffmpeg。请安装 ffmpeg，或重新运行 setup 并允许自动安装。",
        ));
    }

    if cfg!(target_os = "macos") && command_exists("brew") {
        let status = Command::new("brew")
            .args(["install", "ffmpeg"])
            .status()
            .map_err(|error| AppError::config(format!("执行 brew install ffmpeg 失败: {error}")))?;

        if !status.success() {
            return Err(AppError::config(format!(
                "brew install ffmpeg 失败，退出码: {:?}",
                status.code()
            )));
        }

        if let Some(path) = config::resolve_ffmpeg_bin(ffmpeg_override) {
            return Ok(path);
        }
    }

    Err(AppError::config(
        "未找到 ffmpeg，且自动安装未成功。请手动安装 ffmpeg，或通过 --ffmpeg-bin 指定路径。",
    ))
}

fn model_dir_ready(model_dir: &Path) -> bool {
    let has_model =
        model_dir.join("model.onnx").exists() || model_dir.join("model_quant.onnx").exists();
    has_model && model_dir.join("tokens.json").exists() && model_dir.join("am.mvn").exists()
}

fn existing_model_source(target_dir: &Path) -> Option<PathBuf> {
    if let Ok(source_dir) = std::env::var("MEETING_MINUTES_MODEL_SOURCE_DIR") {
        let path = PathBuf::from(source_dir);
        if path != target_dir && model_dir_ready(&path) {
            return Some(path);
        }
    }

    model_store::discover_existing_model_dir().filter(|path| path != target_dir)
}

fn copy_model_assets(source_dir: &Path, target_dir: &Path) -> AppResult<()> {
    for file_name in MODEL_FILES {
        let source = source_dir.join(file_name);
        if !source.exists() {
            continue;
        }
        let target = target_dir.join(file_name);
        fs::copy(&source, &target).map_err(|error| {
            AppError::output(format!(
                "复制模型文件失败 {} -> {}: {error}",
                display_path(&source),
                display_path(&target)
            ))
        })?;
    }

    if source_dir.join("model_quant.onnx").exists() && !target_dir.join("model_quant.onnx").exists()
    {
        let source = source_dir.join("model_quant.onnx");
        let target = target_dir.join("model_quant.onnx");
        fs::copy(&source, &target).map_err(|error| {
            AppError::output(format!(
                "复制模型文件失败 {} -> {}: {error}",
                display_path(&source),
                display_path(&target)
            ))
        })?;
    }

    Ok(())
}

fn command_exists(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}
