use crate::error::{AppError, AppResult};
use crate::util::display_path;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
pub struct ExtractedAudio {
    pub path: PathBuf,
}

pub fn extract_audio(
    ffmpeg_bin: &Path,
    video_path: &Path,
    wav_path: &Path,
) -> AppResult<ExtractedAudio> {
    let output = Command::new(ffmpeg_bin)
        .arg("-y")
        .arg("-hide_banner")
        .arg("-i")
        .arg(video_path)
        .arg("-vn")
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg("16000")
        .arg("-c:a")
        .arg("pcm_s16le")
        .arg(wav_path)
        .output()
        .map_err(|error| {
            AppError::media(format!(
                "执行 ffmpeg 失败 {}: {error}",
                display_path(ffmpeg_bin)
            ))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(AppError::media(format!(
            "ffmpeg 抽取音频失败，退出码: {:?}，stderr: {}",
            output.status.code(),
            if stderr.is_empty() {
                "<empty>"
            } else {
                &stderr
            }
        )));
    }

    if !wav_path.exists() {
        return Err(AppError::media(format!(
            "音频抽取后未生成 wav 文件: {}",
            display_path(wav_path)
        )));
    }

    Ok(ExtractedAudio {
        path: wav_path.to_path_buf(),
    })
}

pub fn compress_wav_for_storage(
    ffmpeg_bin: &Path,
    source_wav: &Path,
    target_wav: &Path,
) -> AppResult<()> {
    let output = Command::new(ffmpeg_bin)
        .arg("-y")
        .arg("-hide_banner")
        .arg("-i")
        .arg(source_wav)
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg("16000")
        .arg("-c:a")
        .arg("adpcm_ima_wav")
        .arg(target_wav)
        .output()
        .map_err(|error| {
            AppError::media(format!(
                "执行 ffmpeg 压缩 wav 失败 {}: {error}",
                display_path(ffmpeg_bin)
            ))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(AppError::media(format!(
            "ffmpeg 压缩 wav 失败，退出码: {:?}，stderr: {}",
            output.status.code(),
            if stderr.is_empty() {
                "<empty>"
            } else {
                &stderr
            }
        )));
    }

    if !target_wav.exists() {
        return Err(AppError::media(format!(
            "压缩后未生成 wav 文件: {}",
            display_path(target_wav)
        )));
    }

    Ok(())
}

pub fn probe_media_duration_ms(ffmpeg_bin: &Path, media_path: &Path) -> Option<u64> {
    let ffprobe_bin = resolve_ffprobe_bin(ffmpeg_bin)?;
    let output = Command::new(ffprobe_bin)
        .arg("-v")
        .arg("error")
        .arg("-show_entries")
        .arg("format=duration")
        .arg("-of")
        .arg("default=noprint_wrappers=1:nokey=1")
        .arg(media_path)
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let seconds = value.parse::<f64>().ok()?;
    Some((seconds * 1000.0).round() as u64)
}

fn resolve_ffprobe_bin(ffmpeg_bin: &Path) -> Option<PathBuf> {
    if let Some(parent) = ffmpeg_bin.parent() {
        let sibling = parent.join("ffprobe");
        if sibling.is_file() {
            return Some(sibling);
        }
    }

    [
        "/opt/homebrew/bin/ffprobe",
        "/usr/local/bin/ffprobe",
        "/usr/bin/ffprobe",
    ]
    .iter()
    .map(PathBuf::from)
    .find(|path| path.is_file())
}
