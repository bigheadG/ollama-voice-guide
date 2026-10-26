import AVFoundation
import Foundation
import Speech

func fail(_ message: String, code: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data(("錯誤：\(message)\n").utf8))
    exit(code)
}

if CommandLine.arguments.contains("--help") {
    print("使用 macOS Speech 將麥克風語音轉為文字。開始後按 Enter 停止錄音。")
    exit(0)
}

var localeIdentifier = "zh-TW"
if let index = CommandLine.arguments.firstIndex(of: "--locale"),
   CommandLine.arguments.indices.contains(index + 1) {
    localeIdentifier = CommandLine.arguments[index + 1]
}

let speechPermission = DispatchSemaphore(value: 0)
var speechStatus: SFSpeechRecognizerAuthorizationStatus = .notDetermined
SFSpeechRecognizer.requestAuthorization { status in
    speechStatus = status
    speechPermission.signal()
}
if speechPermission.wait(timeout: .now() + 30) == .timedOut {
    fail("等待語音識別權限逾時。")
}
guard speechStatus == .authorized else {
    fail("未取得語音識別權限。請到「系統設定 → 隱私權與安全性 → 語音識別」允許 Terminal 或 Ollama Mic Input。", code: 2)
}

let microphonePermission = DispatchSemaphore(value: 0)
var microphoneAllowed = false
AVCaptureDevice.requestAccess(for: .audio) { allowed in
    microphoneAllowed = allowed
    microphonePermission.signal()
}
if microphonePermission.wait(timeout: .now() + 30) == .timedOut {
    fail("等待麥克風權限逾時。")
}
guard microphoneAllowed else {
    fail("未取得麥克風權限。請到「系統設定 → 隱私權與安全性 → 麥克風」允許 Terminal 或 Ollama Mic Input。", code: 2)
}

guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: localeIdentifier)) else {
    fail("無法建立 \(localeIdentifier) 語音識別器。")
}
guard recognizer.isAvailable else {
    fail("macOS 語音識別目前不可用。請確認網路或已下載的聽寫語言。")
}

let engine = AVAudioEngine()
let request = SFSpeechAudioBufferRecognitionRequest()
request.shouldReportPartialResults = true
// Let macOS choose on-device or Apple Speech service. Forcing on-device mode
// produces no transcript when the zh-TW dictation assets are not downloaded.
request.requiresOnDeviceRecognition = false

let lock = NSLock()
var transcript = ""
var recognitionError = ""
var lastDisplayedTranscript = ""
let finalResult = DispatchSemaphore(value: 0)
var finalWasSignalled = false

let task = recognizer.recognitionTask(with: request) { result, error in
    lock.lock()
    if let result = result {
        transcript = result.bestTranscription.formattedString
        if transcript != lastDisplayedTranscript {
            lastDisplayedTranscript = transcript
            FileHandle.standardError.write(Data(("\r📝 \(transcript)   ").utf8))
        }
    }
    if let error = error {
        recognitionError = error.localizedDescription
    }
    let shouldFinish = result?.isFinal == true || error != nil
    if shouldFinish && !finalWasSignalled {
        finalWasSignalled = true
        finalResult.signal()
    }
    lock.unlock()
}

let input = engine.inputNode
let format = input.outputFormat(forBus: 0)
guard format.sampleRate > 0 && format.channelCount > 0 else {
    task.cancel()
    fail("找不到可用的麥克風輸入。")
}

input.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
    request.append(buffer)
}

do {
    engine.prepare()
    try engine.start()
} catch {
    input.removeTap(onBus: 0)
    task.cancel()
    fail("無法啟動麥克風：\(error.localizedDescription)")
}

FileHandle.standardError.write(
    Data("🎤 正在聆聽…說完後按 Enter。\n".utf8)
)

// Keep the main run loop alive while readLine waits on a background thread.
// SFSpeechRecognizer may deliver callbacks on the main run loop on macOS.
let enterPressed = DispatchSemaphore(value: 0)
DispatchQueue.global(qos: .userInitiated).async {
    _ = readLine()
    enterPressed.signal()
}
while enterPressed.wait(timeout: .now()) == .timedOut {
    RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.05))
}

engine.stop()
input.removeTap(onBus: 0)
request.endAudio()

let finishDeadline = Date(timeIntervalSinceNow: 4.0)
while Date() < finishDeadline {
    if finalResult.wait(timeout: .now()) == .success {
        break
    }
    RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.05))
}

lock.lock()
let resultText = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
let errorText = recognitionError
lock.unlock()
task.cancel()
FileHandle.standardError.write(Data("\n".utf8))

guard !resultText.isEmpty else {
    if !errorText.isEmpty {
        fail("語音識別失敗：\(errorText)", code: 3)
    }
    fail("沒有識別到語音。請確認系統輸入裝置有音量，再靠近麥克風重試。", code: 3)
}
print(resultText)
