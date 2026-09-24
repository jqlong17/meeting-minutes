use crate::error::{AppError, AppResult};
use crate::util::{join_if_exists, shanghai_now_for_dir};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct RunPaths {
    pub run_dir: PathBuf,
    pub temp_wav_path: PathBuf,
    pub wav_path: PathBuf,
    pub asr_path: PathBuf,
    pub asr_polished_path: PathBuf,
    pub summary_path: PathBuf,
    pub log_path: PathBuf,
}

pub fn prepare_run_paths(output_dir: &Path) -> AppResult<RunPaths> {
    fs::create_dir_all(output_dir).map_err(|error| {
        AppError::output(format!(
            "创建输出根目录失败 {}: {error}",
            output_dir.display()
        ))
    })?;

    let run_dir = next_run_dir(output_dir);
    fs::create_dir_all(&run_dir).map_err(|error| {
        AppError::output(format!(
            "创建任务输出目录失败 {}: {error}",
            run_dir.display()
        ))
    })?;

    Ok(RunPaths {
        temp_wav_path: join_if_exists(&run_dir, "audio.asr.wav"),
        wav_path: join_if_exists(&run_dir, "audio.wav"),
        asr_path: join_if_exists(&run_dir, "asr.txt"),
        asr_polished_path: join_if_exists(&run_dir, "asr_优化.txt"),
        summary_path: join_if_exists(&run_dir, "会议纪要.txt"),
        log_path: join_if_exists(&run_dir, "run.log"),
        run_dir,
    })
}

fn next_run_dir(output_dir: &Path) -> PathBuf {
    let base = shanghai_now_for_dir();
    let candidate = output_dir.join(&base);
    if !candidate.exists() {
        return candidate;
    }

    for suffix in 1..=999 {
        let candidate = output_dir.join(format!("{base}-{:02}", suffix));
        if !candidate.exists() {
            return candidate;
        }
    }

    output_dir.join(format!("{base}-overflow"))
}

pub fn write_text(path: &Path, content: &str) -> AppResult<()> {
    fs::write(path, content)
        .map_err(|error| AppError::output(format!("写入文件失败 {}: {error}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::next_run_dir;
    use std::fs;
    use std::path::Path;

    #[test]
    fn run_dir_uses_timestamp_style() {
        let temp_dir =
            std::env::temp_dir().join(format!("meeting-minutes-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let run_dir = next_run_dir(Path::new(&temp_dir));
        let name = run_dir.file_name().unwrap().to_string_lossy().to_string();

        assert!(name.len() >= 19);
        assert_eq!(&name[4..5], "-");
        assert_eq!(&name[7..8], "-");
        assert_eq!(&name[10..11], "-");
        assert_eq!(&name[13..14], "-");
        assert_eq!(&name[16..17], "-");

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn run_dir_appends_suffix_when_timestamp_exists() {
        let temp_dir = std::env::temp_dir().join(format!(
            "meeting-minutes-test-collision-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let first = next_run_dir(Path::new(&temp_dir));
        fs::create_dir_all(&first).unwrap();
        let second = next_run_dir(Path::new(&temp_dir));

        assert_ne!(first, second);
        let second_name = second.file_name().unwrap().to_string_lossy().to_string();
        assert!(second_name.len() > 19);

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
