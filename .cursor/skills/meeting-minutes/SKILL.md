---
name: meeting-minutes
description: >-
  把会议视频、音频或 txt/md 整理成会议纪要。视频用 ffmpeg 抽音频，再用本机 SenseVoice-Small ONNX 转写。
  Use when 用户说会议纪要、会议分析、会议总结、整理这场会、转写会议录音或会议视频。
---

# 会议纪要

输出用中文。先判定材料格式，再转成文本，最后按固定模块写纪要。没有依据的判断标成「推测」，不写成事实。

纪要正文由当前对话里的模型来写。转写不把音频上传到云端。

## 1. 判定格式

| 格式 | 扩展名 | 下一步 |
| --- | --- | --- |
| 视频 | mp4、mov、m4v、mkv、avi、flv、wmv、webm | ffmpeg 抽音频，再转写 |
| 音频 | wav、mp3、m4a、aac、flac、ogg | 直接转写 |
| 纯文本 | txt、md | 直接读，不转写 |

一次只处理用户点名的文件。

## 2. 工具和模型

- 抽音频：本机 `ffmpeg`。没有就 `brew install ffmpeg`。
- 转写：本仓库的 `meeting-minutes`。引擎是 ONNX Runtime，模型是 SenseVoice-Small ONNX。
- 模型不在仓库里。缺失时由本 skill 下载，不要把 `model.onnx` 提交进 git。

模型文件：

- `model.onnx`
- `tokens.json`
- `am.mvn`

下载地址前缀：

`https://huggingface.co/DennisHuang648/SenseVoiceSmall-onnx/resolve/main`

放到：

`~/Library/Application Support/io.meeting-minutes.MeetingMinutesCli/models/sensevoice-small`

二进制按这个顺序找：

1. `meeting-minutes`（已在 PATH 中）
2. 本仓库 `target/release/meeting-minutes`

都没有时，在仓库根目录执行 `cargo build --release`。

模型目录缺上面三个文件时，先下载再转写：

```bash
meeting-minutes setup --skip-ffmpeg-install
```

`setup` 会从上面的地址把三个文件写到模型目录。下载失败就停，不要用空文本编纪要。

## 3. 转写

输出目录：用户指定的文件夹。没指定时用 `~/Downloads/会议纪要/<原文件名去掉扩展名>/`。转写原文和纪要都放在这个目录。

视频：

```bash
meeting-minutes run \
  --video "<绝对路径>" \
  --output-dir "<输出目录>" \
  --output-asr true \
  --output-asr-polished false \
  --output-summary false \
  --output-wav true
```

音频：

```bash
meeting-minutes transcribe-audio \
  --audio "<绝对路径>" \
  --output "<输出目录>/asr.txt"
```

读 `asr.txt`。转写失败就停。纯文本直接读全文。

不要使用 CLI 自带的短摘要。纪要按下面的模块写。

## 4. 纪要模块

每一节都要有内容。材料里没有时写「材料未提及」，有推断时在该条末尾标「（推测）」。

1. **会议背景**
2. **会议目标**
3. **参会人角色**：对不上姓名的按发言角色写，不编人名。
4. **会议内容**：能分出议题就按模块，否则按时间线。
5. **核心关注**
6. **结论**：没拍板的不写进结论。
7. **待办**：没有负责人或时间就空着，不补。

## 5. 落盘

把纪要写到输出目录的 `会议纪要.md`。回复里先给纪要正文，再给这个目录的路径。
