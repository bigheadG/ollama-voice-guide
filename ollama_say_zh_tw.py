#!/usr/bin/env python3
"""Chat with a local Ollama model and speak replies with macOS `say`.

Uses only the Python standard library.  Text is printed as it arrives, while
complete sentences are put on a separate speech queue for sequential playback.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request


DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:4b"
SYSTEM_PROMPT = (
    "你是一位親切、耐心的助理。請一律使用自然、適合朗讀的繁體中文回答。"
    "答案要簡潔，避免 Markdown 表格、程式碼區塊、網址和不必要的特殊符號。"
    "不要輸出思考過程，只回答最後答案。"
)
STOP = object()
APP_DIR = Path(__file__).resolve().parent
MIC_SOURCE = APP_DIR / "mic_transcribe.swift"
MIC_INFO = APP_DIR / "mic_info.plist"
MIC_BINARY = APP_DIR / "mic_transcribe"


def available_voices() -> list[tuple[str, str]]:
    """Return (voice name, locale) pairs reported by macOS say."""
    try:
        result = subprocess.run(
            ["say", "-v", "?"], check=True, capture_output=True, text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    voices: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        match = re.match(r"^(.+?)\s+([a-z]{2}_[A-Z]{2})\s+#", line)
        if match:
            voices.append((match.group(1).strip(), match.group(2)))
    return voices


def choose_voice(requested: str | None) -> str | None:
    voices = available_voices()
    if requested:
        exact = [name for name, _ in voices if name.casefold() == requested.casefold()]
        if not exact:
            choices = ", ".join(name for name, locale in voices if locale == "zh_TW")
            raise SystemExit(
                f"找不到聲音 {requested!r}。可用的台灣中文聲音：{choices or '無'}"
            )
        return exact[0]

    # Prefer a Taiwan Mandarin voice. Meijia is widely available on recent macOS.
    taiwan = [name for name, locale in voices if locale == "zh_TW"]
    for preferred in ("Meijia", "Mei-Jia", "Eddy (中文（台灣）)"):
        for name in taiwan:
            if name.casefold() == preferred.casefold():
                return name
    return taiwan[0] if taiwan else None


def clean_for_speech(text: str) -> str:
    """Make model output friendlier to a speech synthesizer."""
    text = re.sub(r"```.*?```", " 省略程式碼。 ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"https?://\S+", " 網頁連結 ", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)
    text = re.sub(r"[*_~>|]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class ThinkingFilter:
    """Remove Qwen thinking text, including tags split across stream chunks.

    Some Ollama/Qwen3 combinations omit the opening ``<think>`` token but keep
    ``</think>``.  In that compatibility mode, hold the prefix until the closing
    tag appears.  If it never appears, the prefix was an ordinary answer and is
    released at end-of-stream.
    """

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
                        # No closing tag means `think: false` worked and this was
                        # a normal answer, rather than hidden reasoning.
                        if self.hidden_prefix:
                            output.append(self.buffer)
                        self.buffer = ""
                    else:
                        if not self.hidden_prefix:
                            self.buffer = self.buffer[-7:]
                    break
                self.buffer = self.buffer[end + len("</think>") :]
                self.in_think = False
                self.hidden_prefix = False
            else:
                start = self.buffer.lower().find("<think>")
                if start >= 0:
                    output.append(self.buffer[:start])
                    self.buffer = self.buffer[start + len("<think>") :]
                    self.in_think = True
                    continue
                if final:
                    output.append(self.buffer)
                    self.buffer = ""
                else:
                    # Keep enough text to detect a tag divided between chunks.
                    keep = min(6, len(self.buffer))
                    if len(self.buffer) > keep:
                        output.append(self.buffer[:-keep])
                        self.buffer = self.buffer[-keep:]
                break
        return "".join(output)


class SentenceBuffer:
    """Accumulate streamed text and emit chunks that sound natural."""

    END = re.compile(r"(?<=[。！？!?；;：:\n])")

    def __init__(self, max_chars: int = 90) -> None:
        self.pending = ""
        self.max_chars = max_chars

    def feed(self, text: str, final: bool = False) -> list[str]:
        self.pending += text
        pieces = self.END.split(self.pending)
        self.pending = pieces.pop() if pieces else ""
        ready = [piece for piece in pieces if piece.strip()]

        # If the model writes a very long sentence, split at a Chinese comma.
        while len(self.pending) >= self.max_chars:
            cut = max(
                self.pending.rfind("，", 0, self.max_chars),
                self.pending.rfind(",", 0, self.max_chars),
            )
            if cut < self.max_chars // 3:
                cut = self.max_chars
            else:
                cut += 1
            ready.append(self.pending[:cut])
            self.pending = self.pending[cut:]

        if final and self.pending.strip():
            ready.append(self.pending)
            self.pending = ""
        return ready


class Speaker:
    def __init__(self, voice: str | None, rate: int, enabled: bool = True) -> None:
        self.voice = voice
        self.rate = rate
        self.enabled = enabled
        self.items: queue.Queue[object] = queue.Queue()
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
                    command = ["say", "-r", str(self.rate)]
                    if self.voice:
                        command += ["-v", self.voice]
                    subprocess.run(command + [str(item)], check=False)
            finally:
                self.items.task_done()

    def finish(self) -> None:
        self.items.put(STOP)
        self.thread.join()

    def wait(self) -> None:
        """Wait until queued speech is finished before opening the microphone."""
        self.items.join()


def ensure_mic_helper() -> Path:
    """Compile the small macOS Speech helper when missing or out of date."""
    if sys.platform != "darwin":
        raise RuntimeError("麥克風語音輸入目前只支援 macOS。")
    if not MIC_SOURCE.exists() or not MIC_INFO.exists():
        raise RuntimeError("找不到 mic_transcribe.swift 或 mic_info.plist。")

    newest_source = max(MIC_SOURCE.stat().st_mtime, MIC_INFO.stat().st_mtime)
    if MIC_BINARY.exists() and MIC_BINARY.stat().st_mtime >= newest_source:
        return MIC_BINARY

    print("第一次使用麥克風，正在準備 macOS 語音識別工具…", flush=True)
    cache_dir = Path(tempfile.gettempdir()) / "ollama-say-swift-cache"
    cache_dir.mkdir(exist_ok=True)
    command = [
        "swiftc",
        str(MIC_SOURCE),
        "-o",
        str(MIC_BINARY),
        "-Xlinker",
        "-sectcreate",
        "-Xlinker",
        "__TEXT",
        "-Xlinker",
        "__info_plist",
        "-Xlinker",
        str(MIC_INFO),
    ]
    try:
        compile_env = os.environ.copy()
        compile_env["CLANG_MODULE_CACHE_PATH"] = str(cache_dir)
        compile_env["SWIFT_MODULECACHE_PATH"] = str(cache_dir)
        result = subprocess.run(
            command, capture_output=True, text=True, env=compile_env
        )
    except FileNotFoundError as error:
        raise RuntimeError("找不到 swiftc；请先安装 Apple Command Line Tools。") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"无法建立麦克风工具：{detail}")
    return MIC_BINARY


def transcribe_microphone(locale: str) -> str:
    """Record one utterance and return the macOS Speech transcription."""
    helper = ensure_mic_helper()
    process = subprocess.Popen(
        [str(helper), "--locale", locale],
        stdin=None,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
    )
    output, _ = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            "麦克风输入失败。请检查上方信息以及系统的麦克风、语音识别权限。"
        )
    prompt = output.strip()
    if not prompt:
        raise RuntimeError("没有识别到语音，请再试一次。")
    return prompt


def ollama_stream(
    host: str, model: str, messages: list[dict[str, str]], timeout: float
):
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": True,
            # Supported by current Ollama; older releases simply ignore it.
            "think": False,
            "options": {"temperature": 0.6},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                if not raw_line.strip():
                    continue
                data = json.loads(raw_line)
                if data.get("error"):
                    raise RuntimeError(data["error"])
                message = data.get("message") or {}
                # Intentionally ignore message["thinking"].
                content = message.get("content", "")
                if content:
                    yield content
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"無法連線到 Ollama（{host}）。請先開啟 Ollama app，或執行 `ollama serve`。"
        ) from error


def answer_once(
    prompt: str,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
    speaker: Speaker,
) -> str:
    messages.append({"role": "user", "content": prompt})
    # Qwen3 releases bundled with some Ollama versions hide the opening think
    # tag but expose the closing one. Other models use normal tagged output.
    thinking = ThinkingFilter(hidden_prefix=args.model.casefold().startswith("qwen3"))
    sentences = SentenceBuffer(args.chunk_chars)
    complete: list[str] = []

    print("助理：", end="", flush=True)
    for raw in ollama_stream(args.host, args.model, messages, args.timeout):
        visible = thinking.feed(raw)
        if not visible:
            continue
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
    print("\n", flush=True)

    reply = "".join(complete).strip()
    messages.append({"role": "assistant", "content": reply})
    return reply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 Ollama qwen3:4b 對話，並以 macOS say 朗讀繁體中文回答。"
    )
    parser.add_argument("prompt", nargs="*", help="單次提問；不填則進入連續對話")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--voice", help="macOS say 聲音名稱（預設自動選 zh_TW）")
    parser.add_argument("--rate", type=int, default=175, help="朗讀速度，預設 175")
    parser.add_argument(
        "--chunk-chars", type=int, default=90, help="無句號時的最長朗讀段落"
    )
    parser.add_argument("--timeout", type=float, default=300, help="連線逾時秒數")
    parser.add_argument("--no-speak", action="store_true", help="只串流文字，不朗讀")
    parser.add_argument("--mic", action="store_true", help="使用麥克風連續輸入問題")
    parser.add_argument(
        "--input",
        choices=("text", "mic", "select"),
        default="text",
        help="輸入模式：text、mic 或每輪選擇 select",
    )
    parser.add_argument("--mic-locale", default="zh-TW", help="語音識別語言，預設 zh-TW")
    parser.add_argument("--list-voices", action="store_true", help="列出台灣中文聲音")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_voices:
        matches = [(name, locale) for name, locale in available_voices() if locale == "zh_TW"]
        if not matches:
            print("找不到 zh_TW 聲音。請到系統設定下載台灣中文聲音。")
            return 1
        for name, locale in matches:
            print(f"{name}\t{locale}")
        return 0

    voice = None if args.no_speak else choose_voice(args.voice)
    if not args.no_speak:
        if voice:
            print(f"語音：{voice}（速度 {args.rate}）")
        else:
            print("警告：找不到台灣中文聲音，將使用 macOS 預設聲音。", file=sys.stderr)

    speaker = Speaker(voice, args.rate, enabled=not args.no_speak)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    input_mode = "mic" if args.mic else args.input
    try:
        if args.prompt:
            answer_once(" ".join(args.prompt), messages, args, speaker)
        else:
            if input_mode == "mic":
                print("麦克风模式：按 Enter 开始说话，输入 q 可结束。\n")
            elif input_mode == "select":
                print("选择模式：每一轮可选择文字或麦克风输入。\n")
            else:
                print(
                    "输入问题后按 Enter；输入 /mic 可说话，/quit 结束，"
                    "/clear 清除对话记忆。\n"
                )
            while True:
                try:
                    if input_mode == "mic":
                        action = input("按 Enter 开始说话（q 结束）：").strip()
                        if action.casefold() in {"q", "quit", "/quit", "/exit"}:
                            break
                        speaker.wait()
                        prompt = transcribe_microphone(args.mic_locale)
                        print(f"你（语音）：{prompt}")
                    elif input_mode == "select":
                        choice = input(
                            "选择输入方式：[1] 文字  [2] 麦克风  [q] 结束："
                        ).strip().casefold()
                        if choice in {"q", "quit", "/quit", "/exit"}:
                            break
                        if choice in {"1", "t", "text", "文字"}:
                            prompt = input("你（文字）：").strip()
                        elif choice in {"2", "m", "mic", "麦克风", "麥克風"}:
                            speaker.wait()
                            prompt = transcribe_microphone(args.mic_locale)
                            print(f"你（语音）：{prompt}")
                        else:
                            print("请输入 1、2 或 q。\n")
                            continue
                    else:
                        prompt = input("你：").strip()
                        if prompt.casefold() == "/mic":
                            speaker.wait()
                            prompt = transcribe_microphone(args.mic_locale)
                            print(f"你（语音）：{prompt}")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not prompt:
                    continue
                if prompt.casefold() in {"/quit", "/exit", "quit", "exit"}:
                    break
                if prompt.casefold() == "/clear":
                    messages[:] = [{"role": "system", "content": SYSTEM_PROMPT}]
                    print("對話記憶已清除。\n")
                    continue
                answer_once(prompt, messages, args, speaker)
    except KeyboardInterrupt:
        print("\n已停止。", file=sys.stderr)
        return 130
    except (RuntimeError, json.JSONDecodeError) as error:
        print(f"\n錯誤：{error}", file=sys.stderr)
        return 1
    finally:
        speaker.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
