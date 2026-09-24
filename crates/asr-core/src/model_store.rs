use anyhow::{bail, Result};
use directories::ProjectDirs;
use std::path::{Path, PathBuf};

const APP_QUALIFIER: &str = "cn";
const APP_ORGANIZATION: &str = "Ruska";
const APP_NAME: &str = "QuickMeetingMinutesCli";
const MODEL_SUBDIR: &str = "sensevoice-small";
const MODEL_ENV_KEY: &str = "MEETING_MINUTES_MODEL_DIR";

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ModelStatus {
    pub model_path: PathBuf,
    pub model_exists: bool,
    pub onnx_exists: bool,
    pub tokens_exists: bool,
    pub ready: bool,
}

pub fn resolve_model_dir(model_override: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(path) = model_override {
        ensure_ready(&path)?;
        return Ok(path);
    }

    if let Ok(raw) = std::env::var(MODEL_ENV_KEY) {
        let path = PathBuf::from(raw);
        ensure_ready(&path)?;
        return Ok(path);
    }

    if let Some(path) = bundled_model_dir() {
        return Ok(path);
    }

    if let Some(path) = discover_existing_model_dir() {
        return Ok(path);
    }

    let path = default_model_dir()?;
    ensure_ready(&path)?;
    Ok(path)
}

pub fn default_model_dir() -> Result<PathBuf> {
    let project_dirs = ProjectDirs::from(APP_QUALIFIER, APP_ORGANIZATION, APP_NAME)
        .ok_or_else(|| anyhow::anyhow!("无法确定会议纪要 CLI 数据目录"))?;
    Ok(project_dirs.data_dir().join("models").join(MODEL_SUBDIR))
}

pub fn model_is_ready(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    let has_model = dir.join("model.onnx").exists() || dir.join("model_quant.onnx").exists();
    let has_tokens = dir.join("tokens.json").exists();
    has_model && has_tokens
}

pub fn inspect_model_dir(dir: PathBuf) -> ModelStatus {
    let model_exists = dir.exists();
    let onnx_exists = dir.join("model.onnx").exists() || dir.join("model_quant.onnx").exists();
    let tokens_exists = dir.join("tokens.json").exists();
    let ready = model_exists && onnx_exists && tokens_exists;

    ModelStatus {
        model_path: dir,
        model_exists,
        onnx_exists,
        tokens_exists,
        ready,
    }
}

fn ensure_ready(path: &Path) -> Result<()> {
    let status = inspect_model_dir(path.to_path_buf());
    if status.ready {
        return Ok(());
    }

    bail!(
        "ASR 模型目录不可用: {} (dir_exists={}, onnx_exists={}, tokens_exists={}). \
请设置 {} 或将模型放到默认目录。",
        path.display(),
        status.model_exists,
        status.onnx_exists,
        status.tokens_exists,
        MODEL_ENV_KEY
    );
}

fn bundled_model_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let app_dir = exe
        .ancestors()
        .find(|path| path.extension().and_then(|s| s.to_str()) == Some("app"))?;
    let resources = app_dir.join("Contents").join("Resources");
    for candidate in [
        resources.join(MODEL_SUBDIR),
        resources.join("models").join(MODEL_SUBDIR),
    ] {
        if model_is_ready(&candidate) {
            return Some(candidate);
        }
    }
    None
}

pub fn discover_existing_model_dir() -> Option<PathBuf> {
    let mut candidates = Vec::new();

    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        candidates.push(home.join("projects/models/sensevoice-small"));
        candidates.push(
            home.join("Library/Application Support/com.openflow.open-flow/models/sensevoice-small"),
        );
        candidates.push(home.join(
            "Library/Application Support/com.openflow.open-flow-mas-dev/models/sensevoice-small",
        ));
        candidates.push(PathBuf::from(
            "/Applications/Open Flow.app/Contents/Resources/models/sensevoice-small",
        ));
        candidates
            .push(home.join("Library/Application Support/Shandianshuo/models/sensevoice-small"));
    }

    candidates.into_iter().find(|path| model_is_ready(path))
}
