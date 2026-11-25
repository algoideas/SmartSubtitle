import { fileIo as fs } from '@kit.CoreFileKit';
import worker, { ThreadWorkerGlobalScope, MessageEvents } from '@ohos.worker';
import { startEngine, feedAudio, stopEngine, getLastError } from 'libentry.so';

// SubtitleWorker: 承载耗时的原生语音识别任务(模型加载 + ONNX 推理),
// 避免阻塞 UI 线程。主线程通过 postMessage 发送 init/pcm/stop 消息,
// 识别结果通过 postMessage 回传。
//
// 另支持 envcheck(环境检查): 读取沙箱中的预制 0.wav, 按实时节奏喂入
// 引擎, 结果经既有回调回传显示在字幕区; 完成后停止引擎。
const workerPort: ThreadWorkerGlobalScope = worker.workerPort;

// 按实时节奏喂入整个 16kHz 单声道 wav (跳过 RIFF 头), 喂完后停引擎
function feedWavAndStop(buffer: ArrayBuffer): void {
  const u8 = new Uint8Array(buffer);
  let dataOff = -1;
  let dataLen = 0;
  // 扫描 "data" 块 (跳过可变长度的 LIST 等扩展头)
  for (let i = 0; i < 200 && i + 8 < u8.length; i++) {
    if (u8[i] === 0x64 && u8[i + 1] === 0x61 && u8[i + 2] === 0x74 &&
      u8[i + 3] === 0x61) {
      dataOff = i + 8;
      dataLen = new DataView(buffer).getUint32(i + 4, true);
      break;
    }
  }
  if (dataOff < 0 || dataLen <= 0) {
    stopEngine();
    workerPort.postMessage({ type: 'envcheck-done', ok: false, error: 'wav data chunk not found' });
    return;
  }
  if (dataOff + dataLen > u8.length) {
    dataLen = u8.length - dataOff;
  }
  // 1600 样本(100ms)/块, 按 100ms 间隔喂入
  const CHUNK_BYTES = 3200;
  const INTERVAL_MS = 100;
  let off = dataOff;
  const timer = setInterval(() => {
    if (off >= dataOff + dataLen) {
      clearInterval(timer);
      // 喂完后排水 2 秒, 让引擎完成断句再停止
      setTimeout(() => {
        stopEngine();
        workerPort.postMessage({ type: 'envcheck-done', ok: true, error: '' });
      }, 2000);
      return;
    }
    const end = Math.min(off + CHUNK_BYTES, dataOff + dataLen);
    feedAudio(buffer.slice(off, end));
    off = end;
  }, INTERVAL_MS);
}

workerPort.onmessage = (e: MessageEvents) => {
  const msg = e.data as Record<string, string | boolean | ArrayBuffer>;
  const type = msg.type as string;

  if (type === 'init') {
    const cb = (text: string, isFinal: boolean) => {
      workerPort.postMessage({ type: 'result', text: text, isFinal: isFinal });
    };
    const ok = startEngine(msg.backend as string, msg.modelDir as string, cb);
    workerPort.postMessage({ type: 'started', ok: ok, error: getLastError() });
    return;
  }

  if (type === 'pcm') {
    feedAudio(msg.buffer as ArrayBuffer);
    return;
  }

  if (type === 'stop') {
    stopEngine();
    workerPort.postMessage({ type: 'stopped' });
    return;
  }

  if (type === 'envcheck') {
    const cb = (text: string, isFinal: boolean) => {
      workerPort.postMessage({ type: 'result', text: text, isFinal: isFinal });
    };
    const ok = startEngine(msg.backend as string, msg.modelDir as string, cb);
    if (!ok) {
      workerPort.postMessage({ type: 'envcheck-done', ok: false, error: getLastError() });
      return;
    }
    try {
      const path = msg.wavPath as string;
      const stat = fs.statSync(path);
      const buffer = new ArrayBuffer(stat.size);
      const file = fs.openSync(path, fs.OpenMode.READ_ONLY);
      fs.readSync(file.fd, buffer);
      fs.closeSync(file);
      feedWavAndStop(buffer);
    } catch (err) {
      stopEngine();
      workerPort.postMessage({ type: 'envcheck-done', ok: false, error: JSON.stringify(err) });
    }
    return;
  }
};
