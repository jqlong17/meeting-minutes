use asr_core::VadEngine;
use std::path::Path;

const MODEL_DIR: &str = "/Users/ruska/projects/models/fsmn-vad-onnx";
const TEST_WAV: &str = "/Users/ruska/projects/models/fsmn-vad-onnx/asr_example.wav";

#[test]
fn vad_detects_asr_example_segment() {
    let model_dir = Path::new(MODEL_DIR);
    let wav_path = Path::new(TEST_WAV);
    if !model_dir.exists() || !wav_path.exists() {
        eprintln!("跳过 VAD 集成测试：模型或测试音频不存在");
        return;
    }

    let mut engine = VadEngine::new(model_dir, 2).expect("VadEngine 初始化失败");
    let segments = engine.detect_segments(wav_path).expect("VAD 分段检测失败");

    assert!(
        !segments.is_empty(),
        "期望至少检测到一个语音段，实际: {segments:?}"
    );

    eprintln!("VAD segments: {segments:?}");

    let (start_ms, end_ms) = segments[0];
    assert!(
        (500..=700).contains(&start_ms),
        "起始时间应接近 610ms，实际: {start_ms}"
    );
    assert!(
        (5400..=5600).contains(&end_ms),
        "结束时间应接近 5530ms，实际: {end_ms}"
    );
}
