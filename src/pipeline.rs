use crate::asr::AsrEngine;
use crate::cli::{PolishAsrArgs, SummarizeAsrArgs, TranscribeAudioArgs};
use crate::config::AppConfig;
use crate::config::resolve_llm_config_from_parts;
use crate::error::{AppError, AppResult};
use crate::llm::OpenAiCompatibleClient;
use crate::media::{compress_wav_for_storage, extract_audio, probe_media_duration_ms};
use crate::output::{prepare_run_paths, write_text};
use crate::setup;
use crate::summary::{
    build_asr_polish_prompt, build_prompt, normalize_polished_asr, normalize_summary,
};
use crate::util::{
    display_path, ensure_absolute_file, ensure_directory, format_duration_ms, format_file_size,
    format_system_time,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::time::SystemTime;

const AUTO_SKIP_POLISH_CHAR_THRESHOLD: usize = 8_000;
const AUTO_SKIP_POLISH_DURATION_MS_THRESHOLD: u64 = 30 * 60 * 1000;

pub async fn run(config: AppConfig) -> AppResult<()> {
    if config.auto_download_model {
        setup::ensure_model_assets(&config.model_dir, config.model_base_url.as_deref(), false)
            .await?;
    }
    validate(&config)?;
    let run_paths = prepare_run_paths(&config.output_dir)?;
    let started_at = SystemTime::now();
    let video_info = collect_file_info(
        &config.ffmpeg_bin,
        &config.video_path,
        Some(config.video_path.clone()),
    );
    let mut wav_info = FileLogInfo::missing();
    let mut asr_chars = None;
    let mut polished_asr_chars = None;
    let mut summary_chars = None;
    let mut extract_audio_runtime_ms = None;
    let mut asr_runtime_ms = None;
    let mut polish_runtime_ms = None;
    let mut summary_runtime_ms = None;
    let mut polish_skipped_reason = None;

    let outcome: AppResult<Option<String>> = async {
        let extract_started = Instant::now();
        let extracted = extract_audio(
            &config.ffmpeg_bin,
            &config.video_path,
            &run_paths.temp_wav_path,
        )?;
        extract_audio_runtime_ms = Some(extract_started.elapsed().as_millis() as u64);

        let mut asr_engine =
            AsrEngine::from_resolved_model_dir(Some(config.model_dir.clone()), config.asr_threads)
                .map_err(|error| AppError::asr(error.to_string()))?;
        asr_engine.warmup();
        let asr_started = Instant::now();
        let asr_output = asr_engine
            .transcribe_file_chunked(&extracted.path, config.asr_chunk_duration_ms)
            .map_err(|error| AppError::asr(error.to_string()))?;
        asr_runtime_ms = Some(asr_started.elapsed().as_millis() as u64);
        asr_chars = Some(asr_output.text.chars().count());

        if config.output_asr {
            write_text(&run_paths.asr_path, &asr_output.text)?;
        }

        let llm_client = if config.output_summary || config.output_asr_polished {
            let llm_config = config.llm.clone().ok_or_else(|| {
                AppError::config("未找到可用的大模型配置，无法生成会议纪要或 ASR 优化稿")
            })?;
            Some(OpenAiCompatibleClient::new(llm_config)?)
        } else {
            None
        };

        let should_skip_polish = config.output_asr_polished
            && config.auto_skip_polish_for_long_input
            && (asr_output.text.chars().count() > AUTO_SKIP_POLISH_CHAR_THRESHOLD
                || video_info
                    .duration_ms
                    .map(|duration_ms| duration_ms > AUTO_SKIP_POLISH_DURATION_MS_THRESHOLD)
                    .unwrap_or(false));

        let polished_asr_text = if should_skip_polish {
            polish_skipped_reason = Some(format!(
                "auto-skipped: asr_chars={}, video_duration_ms={}",
                asr_output.text.chars().count(),
                video_info.duration_ms.unwrap_or_default()
            ));
            println!("跳过 asr_优化：输入过长，默认直接使用原始 ASR 继续生成会议纪要");
            None
        } else if config.output_asr_polished {
            let client = llm_client
                .as_ref()
                .ok_or_else(|| AppError::config("未找到可用的大模型配置，无法生成 ASR 优化稿"))?;
            let polish_started = Instant::now();
            let raw_polished = client
                .polish_asr(&build_asr_polish_prompt(&asr_output.text))
                .await?;
            polish_runtime_ms = Some(polish_started.elapsed().as_millis() as u64);
            let normalized = normalize_polished_asr(&raw_polished, &asr_output.text)?;
            polished_asr_chars = Some(normalized.chars().count());
            write_text(&run_paths.asr_polished_path, &normalized)?;
            Some(normalized)
        } else {
            None
        };

        let summary_text = if config.output_summary {
            let client = llm_client
                .as_ref()
                .ok_or_else(|| AppError::config("未找到可用的大模型配置，无法生成会议纪要"))?;
            let summary_source = polished_asr_text.as_deref().unwrap_or(&asr_output.text);
            let summary_started = Instant::now();
            let raw_summary = client.summarize(&build_prompt(summary_source)).await?;
            summary_runtime_ms = Some(summary_started.elapsed().as_millis() as u64);
            let normalized = normalize_summary(&raw_summary, summary_source)?;
            summary_chars = Some(normalized.chars().count());
            write_text(&run_paths.summary_path, &normalized)?;
            Some(normalized)
        } else {
            None
        };

        if config.output_wav {
            compress_wav_for_storage(
                &config.ffmpeg_bin,
                &run_paths.temp_wav_path,
                &run_paths.wav_path,
            )?;
            wav_info = collect_file_info(
                &config.ffmpeg_bin,
                &run_paths.wav_path,
                Some(run_paths.wav_path.clone()),
            );
        } else {
            wav_info = collect_file_info(
                &config.ffmpeg_bin,
                &run_paths.temp_wav_path,
                Some(run_paths.temp_wav_path.clone()),
            );
        }

        let _ = std::fs::remove_file(&run_paths.temp_wav_path);

        println!("任务完成");
        println!("输出目录: {}", display_path(&run_paths.run_dir));
        if config.output_wav {
            println!("wav: {}", display_path(&run_paths.wav_path));
        }
        if config.output_asr {
            println!("asr: {}", display_path(&run_paths.asr_path));
        }
        if config.output_asr_polished {
            println!(
                "asr_polished: {}",
                display_path(&run_paths.asr_polished_path)
            );
        }
        if config.output_summary {
            println!("summary: {}", display_path(&run_paths.summary_path));
        }
        if let Some(polished_asr_text) = polished_asr_text {
            println!(
                "\n[ASR 优化稿预览]\n{}",
                preview_text(&polished_asr_text, 800)
            );
        }
        if let Some(summary_text) = &summary_text {
            println!("\n{}", summary_text.trim());
        }

        Ok(summary_text)
    }
    .await;

    let ended_at = SystemTime::now();
    let log_text = build_run_log(
        &config,
        &run_paths,
        started_at,
        ended_at,
        &video_info,
        &wav_info,
        asr_chars,
        polished_asr_chars,
        summary_chars,
        extract_audio_runtime_ms,
        asr_runtime_ms,
        polish_runtime_ms,
        summary_runtime_ms,
        polish_skipped_reason.as_deref(),
        outcome.as_ref().err(),
    );
    let _ = write_text(&run_paths.log_path, &log_text);

    outcome.map(|_| ())
}

pub async fn polish_asr(args: &PolishAsrArgs) -> AppResult<()> {
    ensure_absolute_file(&args.input, "ASR 文本")?;
    let output_path = args
        .output
        .clone()
        .unwrap_or_else(|| sibling_file(&args.input, "asr_优化.txt"));
    ensure_parent_dir(&output_path)?;

    let llm = resolve_llm_config_from_parts(
        args.config.clone(),
        args.api_base_url.clone(),
        args.api_key.clone(),
        args.api_model.clone(),
    )?
    .ok_or_else(|| AppError::config("未找到可用的大模型配置，无法生成 ASR 优化稿"))?;
    let client = OpenAiCompatibleClient::new(llm)?;

    let asr_text = fs::read_to_string(&args.input).map_err(|error| {
        AppError::input(format!(
            "读取 ASR 文本失败 {}: {error}",
            args.input.display()
        ))
    })?;
    let raw_polished = client
        .polish_asr(&build_asr_polish_prompt(&asr_text))
        .await?;
    let normalized = normalize_polished_asr(&raw_polished, &asr_text)?;
    write_text(&output_path, &normalized)?;

    println!("ASR 优化完成");
    println!("input: {}", display_path(&args.input));
    println!("output: {}", display_path(&output_path));
    println!("\n{}", preview_text(&normalized, 1200));
    Ok(())
}

pub async fn summarize_asr(args: &SummarizeAsrArgs) -> AppResult<()> {
    ensure_absolute_file(&args.input, "ASR 文本")?;
    let output_path = args
        .output
        .clone()
        .unwrap_or_else(|| sibling_file(&args.input, "会议纪要.txt"));
    ensure_parent_dir(&output_path)?;

    let llm = resolve_llm_config_from_parts(
        args.config.clone(),
        args.api_base_url.clone(),
        args.api_key.clone(),
        args.api_model.clone(),
    )?
    .ok_or_else(|| AppError::config("未找到可用的大模型配置，无法生成会议纪要"))?;
    let client = OpenAiCompatibleClient::new(llm)?;

    let asr_text = fs::read_to_string(&args.input).map_err(|error| {
        AppError::input(format!(
            "读取 ASR 文本失败 {}: {error}",
            args.input.display()
        ))
    })?;
    let raw_summary = client.summarize(&build_prompt(&asr_text)).await?;
    let normalized = normalize_summary(&raw_summary, &asr_text)?;
    write_text(&output_path, &normalized)?;

    println!("会议纪要生成完成");
    println!("input: {}", display_path(&args.input));
    println!("output: {}", display_path(&output_path));
    println!("\n{}", normalized.trim());
    Ok(())
}

pub fn transcribe_audio(args: &TranscribeAudioArgs) -> AppResult<()> {
    ensure_absolute_file(&args.audio, "音频文件")?;
    if let Some(output_path) = &args.output {
        ensure_parent_dir(output_path)?;
    }

    let model_dir = args
        .model_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("/Users/ruska/projects/models/sensevoice-small"));
    ensure_directory(
        model_dir.parent().unwrap_or(model_dir.as_path()),
        "模型根目录",
    )?;
    if !model_dir.exists() {
        return Err(AppError::config(format!(
            "ASR 模型目录不存在: {}",
            model_dir.display()
        )));
    }

    let mut asr_engine = AsrEngine::from_resolved_model_dir(Some(model_dir), args.asr_threads)
        .map_err(|error| AppError::asr(error.to_string()))?;
    asr_engine.warmup();
    let asr_output = asr_engine
        .transcribe_file_chunked(&args.audio, args.asr_chunk_seconds.max(1) * 1000)
        .map_err(|error| AppError::asr(error.to_string()))?;
    let json = serde_json::to_string_pretty(&asr_output)
        .map_err(|error| AppError::asr(error.to_string()))?;

    if let Some(output_path) = &args.output {
        write_text(output_path, &json)?;
    } else {
        println!("{json}");
    }

    Ok(())
}

pub fn scan_videos(dir: &Path) -> AppResult<()> {
    if !dir.is_absolute() {
        return Err(AppError::input(format!(
            "扫描目录必须是绝对路径: {}",
            dir.display()
        )));
    }
    if !dir.exists() {
        return Err(AppError::input(format!(
            "扫描目录不存在: {}",
            dir.display()
        )));
    }
    if !dir.is_dir() {
        return Err(AppError::input(format!(
            "扫描路径不是目录: {}",
            dir.display()
        )));
    }

    let mut videos = Vec::new();
    collect_videos(dir, &mut videos)?;
    videos.sort_by(|a, b| {
        b.sort_time
            .cmp(&a.sort_time)
            .then_with(|| a.path.cmp(&b.path))
    });

    if videos.is_empty() {
        println!("未发现视频文件");
        return Ok(());
    }

    println!("扫描目录: {}", display_path(dir));
    println!("视频数量: {}", videos.len());
    println!();

    for item in videos {
        println!(
            "[{}] {}",
            format_system_time(item.sort_time),
            item.path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
        );
        println!("{}", display_path(&item.path));
        println!();
    }

    Ok(())
}

fn validate(config: &AppConfig) -> AppResult<()> {
    ensure_absolute_file(&config.video_path, "视频文件")?;

    let ext = config
        .video_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if ext != "mp4" {
        return Err(AppError::input(format!(
            "当前仅支持 mp4 文件，收到: {}",
            config.video_path.display()
        )));
    }

    ensure_directory(&config.output_dir, "输出目录")?;
    ensure_directory(
        config
            .model_dir
            .parent()
            .unwrap_or(config.model_dir.as_path()),
        "模型根目录",
    )?;

    if !config.model_dir.exists() {
        return Err(AppError::config(format!(
            "ASR 模型目录不存在: {}",
            config.model_dir.display()
        )));
    }

    if !config.ffmpeg_bin.exists() {
        return Err(AppError::config(format!(
            "ffmpeg 不存在: {}",
            config.ffmpeg_bin.display()
        )));
    }

    if (config.output_summary || config.output_asr_polished) && config.llm.is_none() {
        return Err(AppError::config(
            "会议纪要或 ASR 优化稿输出默认开启，但当前未发现可用的大模型配置。请提供 --config 或 --api-* 参数，或者关闭相关输出后运行。",
        ));
    }

    Ok(())
}

fn preview_text(text: &str, limit: usize) -> String {
    let preview: String = text.chars().take(limit).collect();
    if text.chars().count() > limit {
        format!("{preview}...")
    } else {
        preview
    }
}

fn sibling_file(input: &Path, name: &str) -> PathBuf {
    input
        .parent()
        .map(|parent| parent.join(name))
        .unwrap_or_else(|| PathBuf::from(name))
}

fn ensure_parent_dir(path: &Path) -> AppResult<()> {
    if let Some(parent) = path.parent() {
        ensure_directory(parent, "输出目录")?;
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct FileLogInfo {
    exists: bool,
    path: Option<PathBuf>,
    size_bytes: Option<u64>,
    duration_ms: Option<u64>,
}

impl FileLogInfo {
    fn missing() -> Self {
        Self {
            exists: false,
            path: None,
            size_bytes: None,
            duration_ms: None,
        }
    }
}

fn collect_file_info(ffmpeg_bin: &Path, path: &Path, stored_path: Option<PathBuf>) -> FileLogInfo {
    if !path.exists() {
        return FileLogInfo {
            exists: false,
            path: stored_path,
            size_bytes: None,
            duration_ms: None,
        };
    }

    let size_bytes = fs::metadata(path).ok().map(|metadata| metadata.len());
    let duration_ms = probe_media_duration_ms(ffmpeg_bin, path);

    FileLogInfo {
        exists: true,
        path: stored_path,
        size_bytes,
        duration_ms,
    }
}

fn build_run_log(
    config: &AppConfig,
    run_paths: &crate::output::RunPaths,
    started_at: SystemTime,
    ended_at: SystemTime,
    video_info: &FileLogInfo,
    wav_info: &FileLogInfo,
    asr_chars: Option<usize>,
    polished_asr_chars: Option<usize>,
    summary_chars: Option<usize>,
    extract_audio_runtime_ms: Option<u64>,
    asr_runtime_ms: Option<u64>,
    polish_runtime_ms: Option<u64>,
    summary_runtime_ms: Option<u64>,
    polish_skipped_reason: Option<&str>,
    error: Option<&AppError>,
) -> String {
    let status = if error.is_some() { "failed" } else { "success" };
    let started = format_system_time(started_at);
    let ended = format_system_time(ended_at);
    let total_runtime_ms = ended_at
        .duration_since(started_at)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default();

    let llm_base_url = config
        .llm
        .as_ref()
        .map(|llm| llm.base_url.as_str())
        .unwrap_or("disabled");
    let llm_model = config
        .llm
        .as_ref()
        .map(|llm| llm.model.as_str())
        .unwrap_or("disabled");

    let total_text_chars =
        asr_chars.unwrap_or(0) + polished_asr_chars.unwrap_or(0) + summary_chars.unwrap_or(0);

    format!(
        "status: {status}\n\
started_at: {started}\n\
ended_at: {ended}\n\
total_runtime: {}\n\
run_dir: {}\n\
log_path: {}\n\
\n\
video.path: {}\n\
video.exists: {}\n\
video.size_bytes: {}\n\
video.size_pretty: {}\n\
video.duration_ms: {}\n\
video.duration_pretty: {}\n\
\n\
wav.path: {}\n\
wav.exists: {}\n\
wav.size_bytes: {}\n\
wav.size_pretty: {}\n\
wav.duration_ms: {}\n\
wav.duration_pretty: {}\n\
\n\
asr.path: {}\n\
asr.output_enabled: {}\n\
asr.char_count: {}\n\
\n\
asr_polished.path: {}\n\
asr_polished.output_enabled: {}\n\
asr_polished.char_count: {}\n\
asr_polished.skipped_reason: {}\n\
\n\
summary.path: {}\n\
summary.output_enabled: {}\n\
summary.char_count: {}\n\
\n\
text.total_char_count: {}\n\
\n\
stage.extract_audio_runtime_ms: {}\n\
stage.extract_audio_runtime_pretty: {}\n\
stage.asr_runtime_ms: {}\n\
stage.asr_runtime_pretty: {}\n\
stage.polish_runtime_ms: {}\n\
stage.polish_runtime_pretty: {}\n\
stage.summary_runtime_ms: {}\n\
stage.summary_runtime_pretty: {}\n\
\n\
video_input_path: {}\n\
ffmpeg_bin: {}\n\
asr_model_dir: {}\n\
asr_chunk_duration_ms: {}\n\
asr_chunk_duration_pretty: {}\n\
asr_threads: {}\n\
llm.base_url: {}\n\
llm.model: {}\n\
output_wav: {}\n\
output_asr: {}\n\
output_asr_polished: {}\n\
output_summary: {}\n\
\n\
error: {}\n",
        format_duration_ms(total_runtime_ms),
        display_path(&run_paths.run_dir),
        display_path(&run_paths.log_path),
        display_optional_path(video_info.path.as_ref()),
        video_info.exists,
        display_optional_u64(video_info.size_bytes),
        display_optional_size(video_info.size_bytes),
        display_optional_u64(video_info.duration_ms),
        display_optional_duration(video_info.duration_ms),
        display_optional_path(wav_info.path.as_ref()),
        wav_info.exists,
        display_optional_u64(wav_info.size_bytes),
        display_optional_size(wav_info.size_bytes),
        display_optional_u64(wav_info.duration_ms),
        display_optional_duration(wav_info.duration_ms),
        display_path(&run_paths.asr_path),
        config.output_asr,
        display_optional_usize(asr_chars),
        display_path(&run_paths.asr_polished_path),
        config.output_asr_polished,
        display_optional_usize(polished_asr_chars),
        polish_skipped_reason.unwrap_or("none"),
        display_path(&run_paths.summary_path),
        config.output_summary,
        display_optional_usize(summary_chars),
        total_text_chars,
        display_optional_u64(extract_audio_runtime_ms),
        display_optional_duration(extract_audio_runtime_ms),
        display_optional_u64(asr_runtime_ms),
        display_optional_duration(asr_runtime_ms),
        display_optional_u64(polish_runtime_ms),
        display_optional_duration(polish_runtime_ms),
        display_optional_u64(summary_runtime_ms),
        display_optional_duration(summary_runtime_ms),
        display_path(&config.video_path),
        display_path(&config.ffmpeg_bin),
        display_path(&config.model_dir),
        config.asr_chunk_duration_ms,
        format_duration_ms(config.asr_chunk_duration_ms),
        config.asr_threads,
        llm_base_url,
        llm_model,
        config.output_wav,
        config.output_asr,
        config.output_asr_polished,
        config.output_summary,
        error
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string()),
    )
}

fn display_optional_path(path: Option<&PathBuf>) -> String {
    path.map(|value| display_path(value))
        .unwrap_or_else(|| "unknown".to_string())
}

fn display_optional_u64(value: Option<u64>) -> String {
    value
        .map(|raw| raw.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn display_optional_usize(value: Option<usize>) -> String {
    value
        .map(|raw| raw.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn display_optional_duration(value: Option<u64>) -> String {
    value
        .map(format_duration_ms)
        .unwrap_or_else(|| "unknown".to_string())
}

fn display_optional_size(value: Option<u64>) -> String {
    value
        .map(format_file_size)
        .unwrap_or_else(|| "unknown".to_string())
}

#[derive(Debug)]
struct VideoEntry {
    path: PathBuf,
    sort_time: SystemTime,
}

fn collect_videos(dir: &Path, videos: &mut Vec<VideoEntry>) -> AppResult<()> {
    for entry in fs::read_dir(dir)
        .map_err(|error| AppError::input(format!("读取目录失败 {}: {error}", dir.display())))?
    {
        let entry = entry.map_err(|error| {
            AppError::input(format!("读取目录项失败 {}: {error}", dir.display()))
        })?;
        let path = entry.path();
        let file_type = entry.file_type().map_err(|error| {
            AppError::input(format!("读取文件类型失败 {}: {error}", path.display()))
        })?;

        if file_type.is_dir() {
            collect_videos(&path, videos)?;
            continue;
        }

        if !file_type.is_file() || !is_video_file(&path) {
            continue;
        }

        let metadata = entry.metadata().map_err(|error| {
            AppError::input(format!("读取文件元信息失败 {}: {error}", path.display()))
        })?;
        let sort_time = metadata
            .created()
            .or_else(|_| metadata.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);

        videos.push(VideoEntry { path, sort_time });
    }

    Ok(())
}

fn is_video_file(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "mp4" | "mov" | "m4v" | "mkv" | "avi" | "flv" | "wmv" | "webm"
    )
}
