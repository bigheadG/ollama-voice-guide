# Ollama Voice Loop

繁體中文教學與可執行程式：在 Windows、macOS 或 Linux 安裝 Ollama，使用文字或麥克風提問，並朗讀模型回覆。

## 快速開始

1. 安裝 [Ollama](https://ollama.com/download)。
2. 下載模型：`ollama pull qwen3:4b`
3. 下載本倉庫的 `downloads/ollama_voice_portable.py`。
4. 文字與語音選擇模式：

```bash
python3 ollama_voice_portable.py --input select
```

Windows 可使用 `python` 代替 `python3`。

## 麥克風套件

Windows／macOS：

```bash
python -m pip install numpy sounddevice faster-whisper
```

Debian／Ubuntu：

```bash
sudo apt update && sudo apt install -y python3-pip libportaudio2 espeak-ng
python3 -m pip install numpy sounddevice faster-whisper
```

macOS 另提供原生版本 `downloads/ollama_say_zh_tw.py`，使用 Apple Speech 与 `say`，不需要 Whisper Python 套件。

## 隱私

Ollama 模型在本機執行。跨平台麥克風版本使用本機 faster-whisper。首次使用 Whisper 模型時需要下載模型檔案。

## 授權

範例程式碼採 MIT License。
