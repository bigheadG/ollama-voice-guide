#!/usr/bin/env python3
"""Cross-platform Ollama voice chat for Windows, macOS, and Linux.

Text mode uses only Python's standard library. Microphone mode additionally
needs numpy, sounddevice, and faster-whisper. Replies are spoken with macOS
`say`, Windows SAPI, or Linux `espeak-ng`.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request


SYSTEM_PROMPT = (
    "你是一位親切、耐心的助理。請使用自然、適合朗讀的繁體中文回答。"
    "答案要簡潔，不要輸出思考過程，也不要使用 Markdown 表格。"
)
STOP = object()
_WHISPER_MODELS: dict[str, object] = {}


def clean_for_speech(text: str) -> str:
    text = re.sub(r"```.*?```", " 省略程式碼。 ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"https?://\S+", " 網頁連結 ", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)
    return re.sub(r"[*_~>|]|\s+", " ", text).strip()


class ThinkingFilter:
    def __init__(self, hidden_prefix: bool = False) -> None:
        self.buffer = ""
        self.in_think = hidden_prefix
        self.hidden_prefix = hidden_prefix

    def feed(self, chunk: str, final: bool = False) -> str:
        self.buffer += chunk
        output: list[str] = []
        while self.buffer:
            if self.in_think:
                end = self.buffer.lower().find("</think>")
                if end < 0:
                    if final:
                        if self.hidden_prefix:
                            output.append(self.buffer)
                        self.buffer = ""
                    elif not self.hidden_prefix:
                        self.buffer = self.buffer[-7:]
                    break
                self.buffer = self.buffer[end + 8 :]
                self.in_think = self.hidden_prefix = False
            else:
                start = self.buffer.lower().find("<think>")
                if start >= 0:
                    output.append(self.buffer[:start])
                    self.buffer = self.buffer[start + 7 :]
                    self.in_think = True
                    continue
                if final:
                    output.append(self.buffer)
                    self.buffer = ""
                else:
                    keep = min(6, len(self.buffer))
                    if len(self.buffer) > keep:
                        output.append(self.buffer[:-keep])
                        self.buffer = self.buffer[-keep:]
                break
        return "".join(output)


class SentenceBuffer:
    END = re.compile(r"(?<=[。！？!?；;：:\n])")

    def __init__(self, max_chars: int = 90) -> None:
        self.pending = ""
        self.max_chars = max_chars

    def feed(self, text: str, final: bool = False) -> list[str]:
        self.pending += text
        pieces = self.END.split(self.pending)
        self.pending = pieces.pop() if pieces else ""
        ready = [piece for piece in pieces if piece.strip()]
        while len(self.pending) >= self.max_chars:
            cut = max(self.pending.rfind("，", 0, self.max_chars), self.pending.rfind(",", 0, self.max_chars))
            cut = self.max_chars if cut < self.max_chars // 3 else cut + 1
            ready.append(self.pending[:cut])
            self.pending = self.pending[cut:]
        if final and self.pending.strip():
            ready.append(self.pending)
            self.pending = ""
        return ready


def speak_command(text: str, rate: int) -> list[str]:
    system = platform.system()
    if system == "Darwin":
        return ["say", "-r", str(rate), text]
    if system == "Windows":
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        ps = (
            f"$t=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded}'));"
            "Add-Type -AssemblyName System.Speech;"
            "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            f"$s.Rate={max(-10, min(10, round((rate - 175) / 12)))};$s.Speak($t)"
        )
        shell = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
        return [shell, "-NoProfile", "-Command", ps]
    if shutil.which("espeak-ng"):
        return ["espeak-ng", "-v", "zh", "-s", str(rate), text]
    raise RuntimeError("Linux 找不到 espeak-ng。請先用套件管理器安裝 espeak-ng。")


class Speaker:
    def __init__(self, rate: int, enabled: bool) -> None:
        self.rate = rate
        self.enabled = enabled
        self.items: queue.Queue[object] = queue.Queue()
        self.error: str | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def speak(self, text: str) -> None:
        spoken = clean_for_speech(text)
        if spoken:
            self.items.put(spoken)

    def _run(self) -> None:
        while True:
            item = self.items.get()
            try:
                if item is STOP:
                    return
                if self.enabled:
                    try:
                        subprocess.run(speak_command(str(item), self.rate), check=False)
                    except (FileNotFoundError, RuntimeError) as error:
                        self.error = str(error)
            finally:
                self.items.task_done()

    def wait(self) -> None:
        self.items.join()
        if self.error:
            print(f"朗讀警告：{self.error}", file=sys.stderr)
            self.error = None

    def finish(self) -> None:
        self.items.put(STOP)
        self.thread.join()


def transcribe_microphone(model_size: str, language: str) -> str:
    try:
        import numpy as np
        import sounddevice as sd
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError(
            "麥克風模式需要額外套件。請執行："
            f"{sys.executable} -m pip install numpy sounddevice faster-whisper"
        ) from error

    frames: list[object] = []
    sample_rate = 16000

    def callback(indata, frame_count, time_info, status):
        del frame_count, time_info
        if status:
            print(f"\n錄音警告：{status}", file=sys.stderr)
        frames.append(indata.copy())

    print("🎤 正在聆聽…說完後按 Enter。", flush=True)
    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=callback):
            input()
    except Exception as error:
        raise RuntimeError(f"無法使用麥克風：{error}") from error
    if not frames:
        raise RuntimeError("沒有錄到聲音。請檢查系統麥克風權限與輸入裝置。")

    audio = np.concatenate(frames, axis=0).reshape(-1)
    if model_size not in _WHISPER_MODELS:
        print(f"正在載入 Whisper {model_size}（首次使用會下載模型）…", flush=True)
        _WHISPER_MODELS[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    whisper = _WHISPER_MODELS[model_size]
    segments, _ = whisper.transcribe(audio, language=language, vad_filter=True)
    result = "".join(segment.text for segment in segments).strip()
    if not result:
        raise RuntimeError("沒有識別到語音，請靠近麥克風再試一次。")
    return result


def ollama_stream(host: str, model: str, messages: list[dict[str, str]], timeout: float):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": True,
        "think": False,
        "options": {"temperature": 0.6},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat", data=body,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                if not raw_line.strip():
                    continue
                data = json.loads(raw_line)
                if data.get("error"):
                    raise RuntimeError(data["error"])
                content = (data.get("message") or {}).get("content", "")
                if content:
                    yield content
    except urllib.error.URLError as error:
        raise RuntimeError(f"無法連線到 Ollama（{host}）。請先啟動 Ollama。") from error


def answer(prompt: str, messages: list[dict[str, str]], args, speaker: Speaker) -> None:
    messages.append({"role": "user", "content": prompt})
    thinking = ThinkingFilter(args.model.casefold().startswith("qwen3"))
    sentences = SentenceBuffer()
    complete: list[str] = []
    print("助理：", end="", flush=True)
    for raw in ollama_stream(args.host, args.model, messages, args.timeout):
        visible = thinking.feed(raw)
        if visible:
            complete.append(visible)
            print(visible, end="", flush=True)
            for sentence in sentences.feed(visible):
                speaker.speak(sentence)
    tail = thinking.feed("", final=True)
    if tail:
        complete.append(tail)
        print(tail, end="", flush=True)
        for sentence in sentences.feed(tail):
            speaker.speak(sentence)
    for sentence in sentences.feed("", final=True):
        speaker.speak(sentence)
    print("\n")
    messages.append({"role": "assistant", "content": "".join(complete).strip()})


def parse_args():
    parser = argparse.ArgumentParser(description="Windows/macOS/Linux Ollama 語音助理")
    parser.add_argument("prompt", nargs="*")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--input", choices=("text", "mic", "select"), default="select")
    parser.add_argument("--rate", type=int, default=175)
    parser.add_argument("--whisper-model", default="small", choices=("tiny", "base", "small", "medium"))
    parser.add_argument("--language", default="zh")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--no-speak", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    speaker = Speaker(args.rate, not args.no_speak)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    try:
        if args.prompt:
            answer(" ".join(args.prompt), messages, args, speaker)
            return 0
        print(f"平台：{platform.system()}｜模型：{args.model}｜輸入：{args.input}\n")
        while True:
            if args.input == "select":
                choice = input("[1] 文字  [2] 麥克風  [q] 結束：").strip().casefold()
                if choice in {"q", "quit", "/quit", "/exit"}:
                    break
                if choice in {"1", "t", "text"}:
                    prompt = input("你（文字）：").strip()
                elif choice in {"2", "m", "mic"}:
                    speaker.wait()
                    prompt = transcribe_microphone(args.whisper_model, args.language)
                    print(f"你（語音）：{prompt}")
                else:
                    continue
            elif args.input == "mic":
                action = input("按 Enter 開始說話（q 結束）：").strip().casefold()
                if action in {"q", "quit", "/quit", "/exit"}:
                    break
                speaker.wait()
                prompt = transcribe_microphone(args.whisper_model, args.language)
                print(f"你（語音）：{prompt}")
            else:
                prompt = input("你：").strip()
            if prompt.casefold() in {"q", "quit", "/quit", "/exit"}:
                break
            if prompt.casefold() == "/clear":
                messages[:] = [{"role": "system", "content": SYSTEM_PROMPT}]
                print("對話記憶已清除。\n")
                continue
            if prompt:
                answer(prompt, messages, args, speaker)
    except (KeyboardInterrupt, EOFError):
        print("\n已停止。")
    except (RuntimeError, json.JSONDecodeError) as error:
        print(f"\n錯誤：{error}", file=sys.stderr)
        return 1
    finally:
        speaker.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
