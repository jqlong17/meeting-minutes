use clap::{ArgAction, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "meeting-minutes")]
#[command(about = "本地视频会议纪要 CLI")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Run(RunArgs),
    TranscribeAudio(TranscribeAudioArgs),
    Scan(ScanArgs),
    PolishAsr(PolishAsrArgs),
    SummarizeAsr(SummarizeAsrArgs),
    Setup(SetupArgs),
    Config(ConfigArgs),
}

#[derive(Debug, Clone, Parser)]
pub struct RunArgs {
    #[arg(long, value_name = "ABSOLUTE_VIDEO_PATH")]
    pub video: PathBuf,

    #[arg(long, value_name = "DIR")]
    pub output_dir: Option<PathBuf>,

    #[arg(long, value_name = "DIR")]
    pub model_dir: Option<PathBuf>,

    #[arg(long, value_name = "URL")]
    pub model_base_url: Option<String>,

    #[arg(long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    #[arg(long, value_name = "PATH")]
    pub ffmpeg_bin: Option<PathBuf>,

    #[arg(long, value_name = "URL")]
    pub api_base_url: Option<String>,

    #[arg(long, value_name = "KEY")]
    pub api_key: Option<String>,

    #[arg(long, value_name = "MODEL")]
    pub api_model: Option<String>,

    #[arg(long, default_value_t = 45, value_name = "SECONDS")]
    pub asr_chunk_seconds: u64,

    #[arg(long, default_value_t = 2, value_name = "N")]
    pub asr_threads: usize,

    #[arg(
        long,
        default_value_t = true,
        action = ArgAction::Set,
        value_parser = clap::value_parser!(bool)
    )]
    pub auto_download_model: bool,

    #[arg(
        long,
        default_value_t = true,
        action = ArgAction::Set,
        value_parser = clap::value_parser!(bool)
    )]
    pub output_wav: bool,

    #[arg(
        long,
        default_value_t = true,
        action = ArgAction::Set,
        value_parser = clap::value_parser!(bool)
    )]
    pub output_asr: bool,

    #[arg(
        long,
        default_value_t = true,
        action = ArgAction::Set,
        value_parser = clap::value_parser!(bool)
    )]
    pub output_asr_polished: bool,

    #[arg(
        long,
        default_value_t = true,
        action = ArgAction::Set,
        value_parser = clap::value_parser!(bool)
    )]
    pub auto_skip_polish_for_long_input: bool,

    #[arg(
        long,
        default_value_t = true,
        action = ArgAction::Set,
        value_parser = clap::value_parser!(bool)
    )]
    pub output_summary: bool,
}

#[derive(Debug, Clone, Parser)]
pub struct TranscribeAudioArgs {
    #[arg(long, value_name = "ABSOLUTE_AUDIO_PATH")]
    pub audio: PathBuf,

    #[arg(long, value_name = "FILE")]
    pub output: Option<PathBuf>,

    #[arg(long, value_name = "DIR")]
    pub model_dir: Option<PathBuf>,

    #[arg(long, default_value_t = 30, value_name = "SECONDS")]
    pub asr_chunk_seconds: u64,

    #[arg(long, default_value_t = 2, value_name = "N")]
    pub asr_threads: usize,
}

#[derive(Debug, Clone, Parser)]
pub struct ScanArgs {
    #[arg(long, value_name = "ABSOLUTE_DIR_PATH")]
    pub dir: PathBuf,
}

#[derive(Debug, Clone, Parser)]
pub struct PolishAsrArgs {
    #[arg(long, value_name = "ABSOLUTE_ASR_FILE")]
    pub input: PathBuf,

    #[arg(long, value_name = "FILE")]
    pub output: Option<PathBuf>,

    #[arg(long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    #[arg(long, value_name = "URL")]
    pub api_base_url: Option<String>,

    #[arg(long, value_name = "KEY")]
    pub api_key: Option<String>,

    #[arg(long, value_name = "MODEL")]
    pub api_model: Option<String>,
}

#[derive(Debug, Clone, Parser)]
pub struct SummarizeAsrArgs {
    #[arg(long, value_name = "ABSOLUTE_ASR_FILE")]
    pub input: PathBuf,

    #[arg(long, value_name = "FILE")]
    pub output: Option<PathBuf>,

    #[arg(long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    #[arg(long, value_name = "URL")]
    pub api_base_url: Option<String>,

    #[arg(long, value_name = "KEY")]
    pub api_key: Option<String>,

    #[arg(long, value_name = "MODEL")]
    pub api_model: Option<String>,
}

#[derive(Debug, Clone, Parser)]
pub struct SetupArgs {
    #[arg(long, value_name = "DIR")]
    pub model_dir: Option<PathBuf>,

    #[arg(long, value_name = "URL")]
    pub model_base_url: Option<String>,

    #[arg(long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    #[arg(long, value_name = "PATH")]
    pub ffmpeg_bin: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    pub skip_model_download: bool,

    #[arg(long, default_value_t = false)]
    pub skip_ffmpeg_install: bool,

    #[arg(long, default_value_t = false)]
    pub force: bool,
}

#[derive(Debug, Clone, Parser)]
pub struct ConfigArgs {
    #[command(subcommand)]
    pub command: ConfigCommand,
}

#[derive(Debug, Clone, Subcommand)]
pub enum ConfigCommand {
    Init(ConfigInitArgs),
}

#[derive(Debug, Clone, Parser)]
pub struct ConfigInitArgs {
    #[arg(long, value_name = "FILE")]
    pub path: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    pub force: bool,
}
