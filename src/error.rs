use thiserror::Error;

pub type AppResult<T> = Result<T, AppError>;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("输入错误: {0}")]
    Input(String),

    #[error("配置错误: {0}")]
    Config(String),

    #[error("媒体处理错误: {0}")]
    Media(String),

    #[error("ASR 错误: {0}")]
    Asr(String),

    #[error("大模型服务错误: {0}")]
    Llm(String),

    #[error("输出写入错误: {0}")]
    Output(String),

    #[error("I/O 错误: {0}")]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl AppError {
    pub fn input(message: impl Into<String>) -> Self {
        Self::Input(message.into())
    }

    pub fn config(message: impl Into<String>) -> Self {
        Self::Config(message.into())
    }

    pub fn media(message: impl Into<String>) -> Self {
        Self::Media(message.into())
    }

    pub fn asr(message: impl Into<String>) -> Self {
        Self::Asr(message.into())
    }

    pub fn llm(message: impl Into<String>) -> Self {
        Self::Llm(message.into())
    }

    pub fn output(message: impl Into<String>) -> Self {
        Self::Output(message.into())
    }
}
