mod asr;
mod cli;
mod config;
mod error;
mod llm;
mod media;
mod output;
mod pipeline;
mod setup;
mod summary;
mod util;

use crate::cli::Cli;
use crate::error::AppResult;
use clap::Parser;

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("错误: {error}");
        std::process::exit(1);
    }
}

async fn run() -> AppResult<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(false)
        .with_max_level(tracing::Level::WARN)
        .compact()
        .init();

    let cli = Cli::parse();
    match cli.command {
        cli::Command::Run(args) => {
            let config = config::AppConfig::from_run_args(&args)?;
            pipeline::run(config).await
        }
        cli::Command::TranscribeAudio(args) => pipeline::transcribe_audio(&args),
        cli::Command::Scan(args) => pipeline::scan_videos(&args.dir),
        cli::Command::PolishAsr(args) => pipeline::polish_asr(&args).await,
        cli::Command::SummarizeAsr(args) => pipeline::summarize_asr(&args).await,
        cli::Command::Setup(args) => setup::run_setup(&args).await,
        cli::Command::Config(args) => match args.command {
            cli::ConfigCommand::Init(init_args) => setup::run_config_init(&init_args),
        },
    }
}
