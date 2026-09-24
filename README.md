# meeting-minutes

本地把会议视频或音频转成文字，再写成会议纪要。音频不上传。模型权重不在这个仓库里。

## 这条链路用什么

| 步骤 | 能力 | 来源 |
| --- | --- | --- |
| 视频抽音频 | ffmpeg | 本机安装，`brew install ffmpeg` |
| 语音转文字 | SenseVoice-Small，ONNX | 首次运行从 Hugging Face 下载 |
| 推理 | ONNX Runtime（Rust `ort`） | 编进 `meeting-minutes` |
| 会议纪要正文 | 当前 Cursor 对话里的模型 | 不由这个 CLI 调用云端模型 |

转写模型文件：

- `model.onnx`
- `tokens.json`
- `am.mvn`

下载前缀：

```text
https://huggingface.co/DennisHuang648/SenseVoiceSmall-onnx/resolve/main
```

默认放到：

```text
~/Library/Application Support/io.meeting-minutes.MeetingMinutesCli/models/sensevoice-small
```

## 给 Cursor 用

用 Cursor 打开本仓库。对助手说「写会议纪要」，并给出视频、音频或 txt/md 的路径。

助手按 `.cursor/skills/meeting-minutes/SKILL.md` 执行：

1. 没有 ffmpeg 就安装。
2. 没有模型就运行 `meeting-minutes setup --skip-ffmpeg-install`，从上面的地址下载到模型目录。
3. 转写。
4. 按固定模块写 `会议纪要.md`。

原文 `asr.txt` 和纪要默认写到 `~/Downloads/会议纪要/<文件名>/`。指定了文件夹就用指定的文件夹。

## 命令行

```bash
cargo build --release
./target/release/meeting-minutes setup --skip-ffmpeg-install

./target/release/meeting-minutes run \
  --video "/绝对路径/会议.mp4" \
  --output-dir "$HOME/Downloads/会议纪要/会议"
```

音频：

```bash
./target/release/meeting-minutes transcribe-audio \
  --audio "/绝对路径/会议.wav" \
  --output "$HOME/Downloads/会议纪要/会议/asr.txt"
```

`run` 默认会抽出 wav 并写出 `asr.txt`。纪要正文请用 Cursor skill 来写，不要依赖 CLI 里可选的云端摘要。

## 不包含

- 不提交 `model.onnx` 和其他模型权重。
- 不提交会议原文、纪要和 `~/Downloads` 里的结果。
- 不提交 API Key。本仓库的纪要不需要配置云端模型。
