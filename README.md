# meeting-minutes

本地把会议视频或音频转成文字，再写成会议纪要。音频不上传。模型权重不在这个仓库里。

## 使用方法

### 用智能体

1. 克隆本仓库，用 Cursor、Codex 或 Workbuddy 打开。
2. 准备一份材料，可以是视频、音频，或已经转好的 txt、md。
3. 对智能体说「写会议纪要」，并给出文件的绝对路径。要换输出位置时，同时说存放文件夹。

示例：

```text
写会议纪要 /Users/me/Downloads/需求评审.mp4
```

```text
写会议纪要 /Users/me/Downloads/需求评审.mp4，结果放到 /Users/me/Documents/会议
```

智能体第一次运行会检查本机 `ffmpeg` 和 SenseVoice 模型。没有 ffmpeg 时安装；没有模型时执行 `meeting-minutes setup --skip-ffmpeg-install`，从 Hugging Face 下载到本机模型目录，不会把权重写进仓库。

支持的材料：

| 类型 | 扩展名 |
| --- | --- |
| 视频 | mp4、mov、m4v、mkv、avi、flv、wmv、webm |
| 音频 | wav、mp3、m4a、aac、flac、ogg |
| 文本 | txt、md |

没指定文件夹时，结果在：

```text
~/Downloads/会议纪要/<原文件名>/
```

这个目录里有：

- `asr.txt`：转写原文。txt、md 输入没有这一步。
- `audio.wav`：从视频抽出的音频。纯音频或文本输入不一定有。
- `会议纪要.md`：背景、目标、参会人角色、会议内容、核心关注、结论、待办。

材料里没说的会写成「材料未提及」。靠上下文补上的会标「（推测）」。

### 只用命令行转写

纪要正文仍建议交给智能体写。命令行只负责抽出音频并转成文字。

```bash
git clone https://github.com/jqlong17/meeting-minutes.git
cd meeting-minutes
brew install ffmpeg
cargo build --release
./target/release/meeting-minutes setup --skip-ffmpeg-install

./target/release/meeting-minutes run \
  --video "/绝对路径/需求评审.mp4" \
  --output-dir "$HOME/Downloads/会议纪要/需求评审"
```

已有音频时：

```bash
./target/release/meeting-minutes transcribe-audio \
  --audio "/绝对路径/需求评审.wav" \
  --output "$HOME/Downloads/会议纪要/需求评审/asr.txt"
```

## 这条链路用什么

| 步骤 | 能力 | 来源 |
| --- | --- | --- |
| 视频抽音频 | ffmpeg | 本机安装，`brew install ffmpeg` |
| 语音转文字 | SenseVoice-Small，ONNX | 首次运行从 Hugging Face 下载 |
| 推理 | ONNX Runtime（Rust `ort`） | 编进 `meeting-minutes` |
| 会议纪要正文 | 当前智能体里的模型 | Cursor、Codex、Workbuddy 或其他能跑本地命令的智能体。不由这个 CLI 调用云端模型 |

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

## 支持的智能体

这条链路不绑定某一个对话产品。智能体只要能阅读本仓库的 skill，并在本机执行命令，就可以完成抽音频、下载模型和转写。纪要正文由该智能体自己的模型来写。

已按各自的 skill 目录放好同一份说明：

| 智能体 | 本仓库中的 skill |
| --- | --- |
| Cursor | `.cursor/skills/meeting-minutes/SKILL.md` |
| Codex | `.codex/skills/meeting-minutes/SKILL.md` |
| Workbuddy | `.workbuddy/skills/meeting-minutes/SKILL.md` |

其他智能体读 `skills/meeting-minutes/SKILL.md`。这份是正文，上面三处与它一致。

用其中任一智能体打开本仓库，说「写会议纪要」，并给出视频、音频或 txt/md 的路径。智能体按 skill 执行：

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

`run` 默认会抽出 wav 并写出 `asr.txt`。纪要正文由智能体按 skill 来写，不要依赖 CLI 里可选的云端摘要。

## 不包含

- 不提交 `model.onnx` 和其他模型权重。
- 不提交会议原文、纪要和 `~/Downloads` 里的结果。
- 不提交 API Key。本仓库的纪要不需要配置云端模型。
