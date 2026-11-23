import worker, { ThreadWorkerGlobalScope, MessageEvents } from '@ohos.worker';
import { startEngine, feedAudio, stopEngine, getLastError } from 'libentry.so';

// SubtitleWorker: 承载耗时的原生语音识别任务(模型加载 + ONNX 推理),
// 避免阻塞 UI 线程。主线程通过 postMessage 发送 init/pcm/stop 消息,
// 识别结果通过 postMessage 回传。
const workerPort: ThreadWorkerGlobalScope = worker.workerPort;

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
};
