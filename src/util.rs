use crate::error::{AppError, AppResult};
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use time::{OffsetDateTime, UtcOffset};

pub fn ensure_absolute_file(path: &Path, label: &str) -> AppResult<()> {
    if !path.is_absolute() {
        return Err(AppError::input(format!(
            "{label}必须是绝对路径: {}",
            path.display()
        )));
    }
    if !path.exists() {
        return Err(AppError::input(format!(
            "{label}不存在: {}",
            path.display()
        )));
    }
    if !path.is_file() {
        return Err(AppError::input(format!(
            "{label}不是文件: {}",
            path.display()
        )));
    }
    Ok(())
}

pub fn ensure_directory(path: &Path, label: &str) -> AppResult<()> {
    if !path.exists() {
        std::fs::create_dir_all(path).map_err(|error| {
            AppError::output(format!("创建{label}失败 {}: {error}", path.display()))
        })?;
    }
    if !path.is_dir() {
        return Err(AppError::output(format!(
            "{label}不是目录: {}",
            path.display()
        )));
    }
    Ok(())
}

pub fn shanghai_now_for_dir() -> String {
    let offset = UtcOffset::from_hms(8, 0, 0).expect("valid offset");
    let now = OffsetDateTime::now_utc().to_offset(offset);
    format!(
        "{:04}-{:02}-{:02}-{:02}-{:02}-{:02}",
        now.year(),
        u8::from(now.month()),
        now.day(),
        now.hour(),
        now.minute(),
        now.second()
    )
}

pub fn display_path(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

pub fn join_if_exists(dir: &Path, file_name: &str) -> PathBuf {
    dir.join(file_name)
}

pub fn shanghai_offset() -> UtcOffset {
    UtcOffset::from_hms(8, 0, 0).expect("valid offset")
}

pub fn format_system_time(time: SystemTime) -> String {
    let offset = shanghai_offset();
    let datetime = OffsetDateTime::from(time).to_offset(offset);
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        datetime.year(),
        u8::from(datetime.month()),
        datetime.day(),
        datetime.hour(),
        datetime.minute(),
        datetime.second()
    )
}

pub fn format_duration_ms(duration_ms: u64) -> String {
    let total_seconds = duration_ms / 1000;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    let millis = duration_ms % 1000;
    format!("{hours:02}:{minutes:02}:{seconds:02}.{millis:03}")
}

pub fn format_file_size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;

    let bytes_f64 = bytes as f64;
    if bytes_f64 >= GB {
        format!("{:.2} GB", bytes_f64 / GB)
    } else if bytes_f64 >= MB {
        format!("{:.2} MB", bytes_f64 / MB)
    } else if bytes_f64 >= KB {
        format!("{:.2} KB", bytes_f64 / KB)
    } else {
        format!("{bytes} B")
    }
}
