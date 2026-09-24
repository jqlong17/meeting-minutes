use anyhow::{bail, Context, Result};
use asr_core::LanguageHint;
use asr_core::{
    AsrEngine, AsrOutput, DiarizationEngine, DiarizationOptions, DiarizationOutput, SpeechInterval,
    TranscribeOptions, VadEngine,
};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const SERVICE_CAPABILITIES: &[&str] = &["transcribe", "detect_vad", "rich_transcript", "diarize"];

#[derive(Debug, Parser)]
#[command(name = "meetloom-asr", about = "Meetloom local ASR service")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Long-running JSON-lines service over a Unix domain socket.
    Serve(ServeArgs),
    /// One-shot transcription for benchmarks and debugging.
    Transcribe(TranscribeArgs),
}

#[derive(Debug, Parser)]
struct ServeArgs {
    #[arg(long, value_name = "PATH")]
    socket: Option<PathBuf>,

    #[arg(long, default_value_t = 2, value_name = "N")]
    default_threads: usize,
}

#[derive(Debug, Parser)]
struct TranscribeArgs {
    #[arg(long, value_name = "ABSOLUTE_AUDIO_PATH")]
    audio: PathBuf,

    #[arg(long, value_name = "FILE")]
    output: Option<PathBuf>,

    #[arg(long, value_name = "DIR")]
    model_dir: PathBuf,

    #[arg(long, default_value_t = 30, value_name = "SECONDS")]
    chunk_seconds: u64,

    #[arg(long, default_value_t = 2, value_name = "N")]
    threads: usize,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ServiceRequest {
    Ping {
        id: String,
    },
    Transcribe {
        id: String,
        audio_path: PathBuf,
        model_dir: PathBuf,
        #[serde(default = "default_chunk_seconds")]
        chunk_seconds: u64,
        #[serde(default = "default_threads")]
        threads: usize,
        #[serde(default = "default_language")]
        language: String,
        #[serde(default = "default_use_itn")]
        use_itn: bool,
    },
    DetectVad {
        id: String,
        audio_path: PathBuf,
        vad_model_dir: PathBuf,
        #[serde(default = "default_threads")]
        threads: usize,
    },
    Diarize {
        id: String,
        audio_path: PathBuf,
        embedding_model_dir: PathBuf,
        segments: Vec<VadSegment>,
        #[serde(default = "default_min_speakers")]
        min_speakers: usize,
        #[serde(default = "default_max_speakers")]
        max_speakers: usize,
        #[serde(default = "default_threads")]
        threads: usize,
    },
    Shutdown,
}

fn default_language() -> String {
    "auto".to_string()
}

fn default_use_itn() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VadSegment {
    start_ms: u64,
    end_ms: u64,
}

#[derive(Debug, Serialize)]
struct VadOutput {
    segments: Vec<VadSegment>,
}

fn default_chunk_seconds() -> u64 {
    30
}

fn default_threads() -> usize {
    2
}

fn default_min_speakers() -> usize {
    1
}

fn default_max_speakers() -> usize {
    8
}

#[derive(Debug, Serialize)]
struct ServiceResponse {
    id: Option<String>,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    version: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<AsrOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vad: Option<VadOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    diarization: Option<DiarizationOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    capabilities: Option<Vec<&'static str>>,
}

#[derive(Debug, Serialize)]
struct ServiceProgress {
    id: Option<String>,
    status: &'static str,
    #[serde(rename = "type")]
    kind: &'static str,
    stage: &'static str,
    progress: f64,
    percent: u8,
    chunk_index: usize,
    processed_samples: usize,
}

struct ServerState {
    default_threads: usize,
    engines: Mutex<HashMap<String, Arc<Mutex<AsrEngine>>>>,
    vad_engines: Mutex<HashMap<String, Arc<Mutex<VadEngine>>>>,
    diarization_engines: Mutex<HashMap<String, Arc<Mutex<DiarizationEngine>>>>,
}

impl ServerState {
    fn new(default_threads: usize) -> Self {
        Self {
            default_threads,
            engines: Mutex::new(HashMap::new()),
            vad_engines: Mutex::new(HashMap::new()),
            diarization_engines: Mutex::new(HashMap::new()),
        }
    }

    fn engine_for(&self, model_dir: &Path, threads: usize) -> Result<Arc<Mutex<AsrEngine>>> {
        let key = format!("{}:{}", model_dir.display(), threads.max(1));
        let mut engines = self
            .engines
            .lock()
            .map_err(|_| anyhow::anyhow!("engine cache lock poisoned"))?;

        if let Some(existing) = engines.get(&key) {
            return Ok(existing.clone());
        }

        let mut engine = AsrEngine::from_resolved_model_dir(Some(model_dir.to_path_buf()), threads)
            .with_context(|| format!("failed to load ASR model from {}", model_dir.display()))?;
        engine.warmup();
        let wrapped = Arc::new(Mutex::new(engine));
        engines.insert(key, wrapped.clone());
        Ok(wrapped)
    }

    fn vad_engine_for(&self, model_dir: &Path, threads: usize) -> Result<Arc<Mutex<VadEngine>>> {
        let key = format!("{}:{}", model_dir.display(), threads.max(1));
        let mut engines = self
            .vad_engines
            .lock()
            .map_err(|_| anyhow::anyhow!("vad engine cache lock poisoned"))?;

        if let Some(existing) = engines.get(&key) {
            return Ok(existing.clone());
        }

        let engine = VadEngine::new(model_dir, threads.max(1))
            .with_context(|| format!("failed to load VAD model from {}", model_dir.display()))?;
        let wrapped = Arc::new(Mutex::new(engine));
        engines.insert(key, wrapped.clone());
        Ok(wrapped)
    }

    fn diarization_engine_for(
        &self,
        model_dir: &Path,
        threads: usize,
    ) -> Result<Arc<Mutex<DiarizationEngine>>> {
        let key = format!("{}:{}", model_dir.display(), threads.max(1));
        let mut engines = self
            .diarization_engines
            .lock()
            .map_err(|_| anyhow::anyhow!("diarization engine cache lock poisoned"))?;

        if let Some(existing) = engines.get(&key) {
            return Ok(existing.clone());
        }

        let engine = DiarizationEngine::new(model_dir, threads.max(1)).with_context(|| {
            format!(
                "failed to load diarization model from {}",
                model_dir.display()
            )
        })?;
        let wrapped = Arc::new(Mutex::new(engine));
        engines.insert(key, wrapped.clone());
        Ok(wrapped)
    }
}

fn default_socket_path() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").context("HOME is not set")?;
    // Keep under macOS sun_path limit (104 bytes including NUL).
    Ok(PathBuf::from(home).join("Library/Group Containers/group.com.ruskaapps.Meetloom/asr.sock"))
}

fn prepare_socket(path: &Path) -> Result<UnixListener> {
    let path_len = path.as_os_str().len();
    if path_len >= 104 {
        bail!(
            "socket path too long for Unix domain socket ({} bytes, max 103): {}",
            path_len,
            path.display()
        );
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if path.exists() {
        let _ = fs::remove_file(path);
    }
    UnixListener::bind(path).with_context(|| format!("failed to bind {}", path.display()))
}

fn main() {
    if let Err(error) = run() {
        eprintln!("meetloom-asr error: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(false)
        .with_max_level(tracing::Level::INFO)
        .compact()
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => run_serve(args),
        Command::Transcribe(args) => run_transcribe(args),
    }
}

fn run_serve(args: ServeArgs) -> Result<()> {
    let socket_path = args
        .socket
        .clone()
        .unwrap_or_else(|| default_socket_path().expect("default socket path"));
    let listener = prepare_socket(&socket_path)?;
    let pid_path = socket_path.with_extension("pid");
    fs::write(&pid_path, std::process::id().to_string())?;

    tracing::info!(
        "meetloom-asr serve listening on {} (pid {})",
        socket_path.display(),
        std::process::id()
    );

    let state = Arc::new(ServerState::new(args.default_threads));

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let state = state.clone();
                thread::spawn(move || {
                    if let Err(error) = handle_client(stream, state) {
                        tracing::warn!("client session ended with error: {error:#}");
                    }
                });
            }
            Err(error) => tracing::warn!("accept failed: {error}"),
        }
    }

    Ok(())
}

fn handle_client(stream: UnixStream, state: Arc<ServerState>) -> Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;

    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<ServiceRequest>(trimmed) {
            Ok(ServiceRequest::Transcribe {
                id,
                audio_path,
                model_dir,
                chunk_seconds,
                threads,
                language,
                use_itn,
            }) => {
                let progress_id = id.clone();
                let mut progress_error: Option<String> = None;
                let response = process_transcribe_request(
                    id,
                    audio_path,
                    model_dir,
                    chunk_seconds,
                    threads,
                    language,
                    use_itn,
                    &state,
                    |chunk_index, processed_samples, progress| {
                        if progress_error.is_some() {
                            return;
                        }
                        let clamped = progress.clamp(0.0, 1.0);
                        let update = ServiceProgress {
                            id: Some(progress_id.clone()),
                            status: "progress",
                            kind: "progress",
                            stage: "asr",
                            progress: clamped,
                            percent: (clamped * 100.0).round().clamp(0.0, 100.0) as u8,
                            chunk_index,
                            processed_samples,
                        };
                        if let Err(error) = write_json_line(&mut writer, &update) {
                            progress_error = Some(error.to_string());
                        }
                    },
                );
                match (response, progress_error) {
                    (_, Some(error)) => error_response(Some(progress_id), error),
                    (Ok(response), None) => response,
                    (Err(error), None) => error_response(None, error.to_string()),
                }
            }
            Ok(request) => match process_request(request, &state) {
                Ok(response) => response,
                Err(error) => error_response(None, error.to_string()),
            },
            Err(error) => ServiceResponse {
                id: None,
                status: "error",
                version: None,
                output: None,
                vad: None,
                diarization: None,
                error: Some(format!("invalid request JSON: {error}")),
                capabilities: None,
            },
        };

        write_json_line(&mut writer, &response)?;
    }

    Ok(())
}

fn write_json_line<T: Serialize>(writer: &mut UnixStream, value: &T) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn process_request(request: ServiceRequest, state: &ServerState) -> Result<ServiceResponse> {
    match request {
        ServiceRequest::Ping { id } => Ok(ServiceResponse {
            id: Some(id),
            status: "ok",
            version: Some(VERSION),
            output: None,
            vad: None,
            diarization: None,
            error: None,
            capabilities: Some(SERVICE_CAPABILITIES.to_vec()),
        }),
        ServiceRequest::Shutdown => {
            tracing::info!("shutdown requested");
            std::process::exit(0);
        }
        ServiceRequest::Transcribe {
            id,
            audio_path,
            model_dir,
            chunk_seconds,
            threads,
            language,
            use_itn,
        } => process_transcribe_request(
            id,
            audio_path,
            model_dir,
            chunk_seconds,
            threads,
            language,
            use_itn,
            state,
            |_, _, _| {},
        ),
        ServiceRequest::DetectVad {
            id,
            audio_path,
            vad_model_dir,
            threads,
        } => {
            if !audio_path.is_absolute() {
                return Ok(error_response(
                    Some(id),
                    format!("audio_path must be absolute: {}", audio_path.display()),
                ));
            }
            if !vad_model_dir.is_absolute() {
                return Ok(error_response(
                    Some(id),
                    format!(
                        "vad_model_dir must be absolute: {}",
                        vad_model_dir.display()
                    ),
                ));
            }

            let engine = match state.vad_engine_for(&vad_model_dir, threads.max(1)) {
                Ok(engine) => engine,
                Err(error) => return Ok(error_response(Some(id), error.to_string())),
            };

            let segments = {
                let mut engine = engine
                    .lock()
                    .map_err(|_| anyhow::anyhow!("vad engine lock poisoned"))?;
                engine.detect_segments(&audio_path)
            };

            match segments {
                Ok(segments) => Ok(ServiceResponse {
                    id: Some(id),
                    status: "ok",
                    version: None,
                    output: None,
                    vad: Some(VadOutput {
                        segments: segments
                            .into_iter()
                            .map(|(start_ms, end_ms)| VadSegment { start_ms, end_ms })
                            .collect(),
                    }),
                    diarization: None,
                    error: None,
                    capabilities: None,
                }),
                Err(error) => Ok(error_response(Some(id), error.to_string())),
            }
        }
        ServiceRequest::Diarize {
            id,
            audio_path,
            embedding_model_dir,
            segments,
            min_speakers,
            max_speakers,
            threads,
        } => {
            if !audio_path.is_absolute() {
                return Ok(error_response(
                    Some(id),
                    format!("audio_path must be absolute: {}", audio_path.display()),
                ));
            }
            if !embedding_model_dir.is_absolute() {
                return Ok(error_response(
                    Some(id),
                    format!(
                        "embedding_model_dir must be absolute: {}",
                        embedding_model_dir.display()
                    ),
                ));
            }

            let engine = match state.diarization_engine_for(&embedding_model_dir, threads.max(1)) {
                Ok(engine) => engine,
                Err(error) => return Ok(error_response(Some(id), error.to_string())),
            };

            let speech_segments: Vec<SpeechInterval> = segments
                .into_iter()
                .map(|segment| SpeechInterval {
                    start_ms: segment.start_ms,
                    end_ms: segment.end_ms,
                })
                .collect();

            let options = DiarizationOptions {
                min_speakers: min_speakers.max(1),
                max_speakers: max_speakers.max(min_speakers.max(1)),
                min_segment_ms: 500,
            };

            let output = {
                let mut engine = engine
                    .lock()
                    .map_err(|_| anyhow::anyhow!("diarization engine lock poisoned"))?;
                engine.diarize_file(&audio_path, &speech_segments, options)
            };

            match output {
                Ok(diarization) => Ok(ServiceResponse {
                    id: Some(id),
                    status: "ok",
                    version: None,
                    output: None,
                    vad: None,
                    diarization: Some(diarization),
                    error: None,
                    capabilities: None,
                }),
                Err(error) => Ok(error_response(Some(id), error.to_string())),
            }
        }
    }
}

fn process_transcribe_request(
    id: String,
    audio_path: PathBuf,
    model_dir: PathBuf,
    chunk_seconds: u64,
    threads: usize,
    language: String,
    use_itn: bool,
    state: &ServerState,
    mut on_progress: impl FnMut(usize, usize, f64),
) -> Result<ServiceResponse> {
    if !audio_path.is_absolute() {
        return Ok(error_response(
            Some(id),
            format!("audio_path must be absolute: {}", audio_path.display()),
        ));
    }
    if !model_dir.is_absolute() {
        return Ok(error_response(
            Some(id),
            format!("model_dir must be absolute: {}", model_dir.display()),
        ));
    }

    let engine = match state.engine_for(&model_dir, threads.max(1)) {
        Ok(engine) => engine,
        Err(error) => {
            return Ok(error_response(Some(id), error.to_string()));
        }
    };

    let options = TranscribeOptions {
        language: LanguageHint::parse(&language),
        use_itn,
    };

    let output = {
        let mut engine = engine
            .lock()
            .map_err(|_| anyhow::anyhow!("engine lock poisoned"))?;
        engine.transcribe_file_chunked_with_options_and_progress(
            &audio_path,
            chunk_seconds.max(1) * 1000,
            options,
            |chunk_index, processed_samples, progress| {
                on_progress(chunk_index, processed_samples, progress)
            },
        )
    };

    match output {
        Ok(output) => Ok(ServiceResponse {
            id: Some(id),
            status: "ok",
            version: None,
            output: Some(output),
            vad: None,
            diarization: None,
            error: None,
            capabilities: None,
        }),
        Err(error) => Ok(error_response(Some(id), error.to_string())),
    }
}

fn error_response(id: Option<String>, message: String) -> ServiceResponse {
    ServiceResponse {
        id,
        status: "error",
        version: None,
        output: None,
        vad: None,
        diarization: None,
        error: Some(message),
        capabilities: None,
    }
}

fn run_transcribe(args: TranscribeArgs) -> Result<()> {
    if !args.audio.is_absolute() {
        bail!("audio path must be absolute");
    }
    if !args.model_dir.is_absolute() {
        bail!("model_dir must be absolute");
    }

    let mut engine = AsrEngine::from_resolved_model_dir(Some(args.model_dir), args.threads.max(1))?;
    engine.warmup();
    let output = engine.transcribe_file_chunked(&args.audio, args.chunk_seconds.max(1) * 1000)?;
    let json = serde_json::to_string_pretty(&output)?;

    if let Some(output_path) = args.output {
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(output_path, json)?;
    } else {
        println!("{json}");
    }

    Ok(())
}
