# meeting-minutes

把一场会的视频、录音或文字稿，整理成会议纪要。

用 Cursor、Codex 或 Workbuddy 打开本仓库，把文件路径发给智能体，说「写会议纪要」。具体说法见下面的使用方法。

## 使用方法

用 Cursor、Codex 或 Workbuddy 打开本仓库，把文件的绝对路径发给智能体，并说要写会议纪要。

视频：

```text
写会议纪要 /Users/me/Downloads/周三评审.mp4
```

音频：

```text
把这个录音整理成会议纪要 /Users/me/Downloads/周三评审.m4a
```

已有转写或笔记（txt、md）：

```text
根据这份文字写会议纪要 /Users/me/Downloads/周三评审.txt
```

结果要放到指定文件夹时，把目录一起说出来：

```text
写会议纪要 /Users/me/Downloads/周三评审.mp4
输出到 /Users/me/Documents/会议记录
```

没说输出目录时，原文 `asr.txt` 和 `会议纪要.md` 写到 `~/Downloads/会议纪要/<文件名>/`。

智能体会自己抽音频、下载转写模型、转写，再按会议背景、目标、参会人角色、会议内容、核心关注、结论、待办来写。不需要先手动跑命令。

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

音频只在本机转写，不会上传。模型文件第一次使用时再下载，不放在这个仓库里。

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
