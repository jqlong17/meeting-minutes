use crate::cli::RunArgs;
use crate::error::{AppError, AppResult};
use directories::ProjectDirs;
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub video_path: PathBuf,
    pub output_dir: PathBuf,
    pub model_dir: PathBuf,
    pub model_base_url: Option<String>,
    pub ffmpeg_bin: PathBuf,
    pub asr_chunk_duration_ms: u64,
    pub asr_threads: usize,
    pub output_wav: bool,
    pub output_asr: bool,
    pub output_asr_polished: bool,
    pub auto_skip_polish_for_long_input: bool,
    pub output_summary: bool,
    pub auto_download_model: bool,
    pub llm: Option<LlmConfig>,
}

#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

#[derive(Debug, Deserialize, Default)]
struct ModelServiceFile {
    default_provider: Option<String>,
    #[serde(default)]
    providers: HashMap<String, ProviderConfig>,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct ProviderConfig {
    #[serde(rename = "type")]
    provider_type: Option<String>,
    base_url: Option<String>,
    api_key: Option<String>,
    api_key_env: Option<String>,
    model: Option<String>,
}

impl AppConfig {
    pub fn from_run_args(args: &RunArgs) -> AppResult<Self> {
        let llm = resolve_llm_config_from_parts(
            args.config.clone(),
            args.api_base_url.clone(),
            args.api_key.clone(),
            args.api_model.clone(),
        )?;

        let ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg_bin.clone())
            .ok_or_else(|| AppError::config("未找到可用的 ffmpeg，可通过 --ffmpeg-bin 指定路径"))?;

        Ok(Self {
            video_path: args.video.clone(),
            output_dir: args.output_dir.clone().unwrap_or_else(default_output_dir),
            model_dir: args
                .model_dir
                .clone()
                .or_else(|| {
                    env::var("MEETING_MINUTES_MODEL_DIR")
                        .ok()
                        .map(PathBuf::from)
                })
                .unwrap_or_else(default_model_dir),
            model_base_url: args.model_base_url.clone(),
            ffmpeg_bin,
            asr_chunk_duration_ms: args.asr_chunk_seconds.saturating_mul(1000).max(1_000),
            asr_threads: args.asr_threads.max(1),
            output_wav: args.output_wav,
            output_asr: args.output_asr,
            output_asr_polished: args.output_asr_polished,
            auto_skip_polish_for_long_input: args.auto_skip_polish_for_long_input,
            output_summary: args.output_summary,
            auto_download_model: args.auto_download_model,
            llm,
        })
    }
}

pub fn resolve_llm_config_from_parts(
    config_path: Option<PathBuf>,
    api_base_url: Option<String>,
    api_key: Option<String>,
    api_model: Option<String>,
) -> AppResult<Option<LlmConfig>> {
    let explicit_model_service_path = config_path.or_else(|| {
        env::var("MEETING_MINUTES_MODEL_CONFIG")
            .ok()
            .map(PathBuf::from)
    });

    resolve_llm_config(
        api_base_url,
        api_key,
        api_model,
        explicit_model_service_path.as_deref(),
    )
}

pub fn default_user_config_path() -> PathBuf {
    if let Some(project_dirs) = ProjectDirs::from("io", "meeting-minutes", "MeetingMinutesCli") {
        return project_dirs.config_dir().join("model-service.toml");
    }

    if let Some(home) = env::var_os("HOME") {
        return PathBuf::from(home)
            .join(".meeting-minutes")
            .join("config")
            .join("model-service.toml");
    }

    PathBuf::from("config").join("model-service.toml")
}

fn read_model_service_file(path: &Path) -> AppResult<Option<ModelServiceFile>> {
    if !path.exists() {
        return Ok(None);
    }

    let content = std::fs::read_to_string(path).map_err(|error| {
        AppError::config(format!("读取模型配置失败 {}: {error}", path.display()))
    })?;

    let parsed = toml::from_str::<ModelServiceFile>(&content).map_err(|error| {
        AppError::config(format!("解析模型配置失败 {}: {error}", path.display()))
    })?;
    Ok(Some(parsed))
}

fn resolve_llm_config(
    api_base_url: Option<String>,
    api_key: Option<String>,
    api_model: Option<String>,
    explicit_path: Option<&Path>,
) -> AppResult<Option<LlmConfig>> {
    let cli_has_override = api_base_url.is_some() || api_key.is_some() || api_model.is_some();

    if cli_has_override {
        let base_url = api_base_url
            .ok_or_else(|| AppError::config("使用 CLI 覆盖 LLM 配置时必须提供 --api-base-url"))?;
        let api_key = api_key
            .ok_or_else(|| AppError::config("使用 CLI 覆盖 LLM 配置时必须提供 --api-key"))?;
        let model = api_model
            .ok_or_else(|| AppError::config("使用 CLI 覆盖 LLM 配置时必须提供 --api-model"))?;
        return Ok(Some(LlmConfig {
            base_url,
            api_key,
            model,
        }));
    }

    if let (Ok(base_url), Ok(api_key), Ok(model)) = (
        env::var("MEETING_MINUTES_API_BASE_URL"),
        env::var("MEETING_MINUTES_API_KEY"),
        env::var("MEETING_MINUTES_API_MODEL"),
    ) {
        return Ok(Some(LlmConfig {
            base_url,
            api_key,
            model,
        }));
    }

    for path in model_service_candidates(explicit_path) {
        let Some(file) = read_model_service_file(&path)? else {
            continue;
        };
        if let Some(config) = llm_from_file(&file)? {
            return Ok(Some(config));
        }
    }

    Ok(None)
}

fn model_service_candidates(explicit_path: Option<&Path>) -> Vec<PathBuf> {
    if let Some(path) = explicit_path {
        return vec![path.to_path_buf()];
    }

    vec![
        default_user_config_path(),
        PathBuf::from("config").join("model-service.toml"),
    ]
}

fn llm_from_file(file: &ModelServiceFile) -> AppResult<Option<LlmConfig>> {
    let Some(provider_name) = file.default_provider.as_deref() else {
        return Ok(None);
    };
    let Some(provider) = file.providers.get(provider_name) else {
        return Ok(None);
    };

    if let Some(config) = llm_from_provider(provider)? {
        return Ok(Some(config));
    }

    if let Some(provider) = file.providers.get("openai-compatible") {
        return llm_from_provider(provider);
    }

    Ok(None)
}

fn llm_from_provider(provider: &ProviderConfig) -> AppResult<Option<LlmConfig>> {
    if provider.provider_type.as_deref() != Some("openai-compatible") {
        return Ok(None);
    }

    let api_key = if let Some(raw) = provider.api_key.clone() {
        raw
    } else if let Some(env_key) = provider.api_key_env.as_deref() {
        env::var(env_key).map_err(|_| AppError::config(format!("环境变量未设置: {env_key}")))?
    } else {
        return Ok(None);
    };

    let Some(base_url) = provider.base_url.clone() else {
        return Ok(None);
    };
    let Some(model) = provider.model.clone() else {
        return Ok(None);
    };

    if api_key.trim().is_empty()
        || api_key.starts_with("sk-your-key")
        || base_url.contains("api.example.com")
    {
        return Ok(None);
    }

    Ok(Some(LlmConfig {
        base_url,
        api_key,
        model,
    }))
}

pub fn resolve_ffmpeg_bin(override_path: Option<PathBuf>) -> Option<PathBuf> {
    if let Some(path) = override_path {
        return path.is_file().then_some(path);
    }

    if let Ok(raw) = env::var("MEETING_MINUTES_FFMPEG_BIN") {
        let path = PathBuf::from(raw);
        if path.is_file() {
            return Some(path);
        }
    }

    [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
        "/Applications/VideoFusion-macOS.app/Contents/Resources/ffmpeg",
    ]
    .iter()
    .map(PathBuf::from)
    .find(|path| path.is_file())
}

pub fn default_output_dir() -> PathBuf {
    if let Some(home) = env::var_os("HOME") {
        return PathBuf::from(home).join("Downloads").join("会议纪要");
    }
    PathBuf::from("会议纪要")
}

pub fn default_model_dir() -> PathBuf {
    if let Some(project_dirs) = ProjectDirs::from("io", "meeting-minutes", "MeetingMinutesCli") {
        return project_dirs
            .data_dir()
            .join("models")
            .join("sensevoice-small");
    }

    if let Some(home) = env::var_os("HOME") {
        return PathBuf::from(home)
            .join(".meeting-minutes")
            .join("models")
            .join("sensevoice-small");
    }

    PathBuf::from("models").join("sensevoice-small")
}
