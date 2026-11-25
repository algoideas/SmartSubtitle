export type SubtitleListener = (text: string, isFinal: boolean) => void;

/**
 * Start the subtitle engine.
 *
 * @param backend "cpu" (sherpa-onnx, any aarch64 OpenHarmony device)
 *                or "npu" (RKNN runtime, RK3588).
 * @param modelDir directory that holds the converted models and tokens.txt
 * @param callback invoked on the calling thread for each partial/final result
 * @returns true when the engine initialised successfully
 */
export function startEngine(backend: string, modelDir: string,
  callback: (text: string, isFinal: boolean) => void): boolean;

/** Stop the engine and release native resources. */
export function stopEngine(): void;

/**
 * Feed 16 kHz mono signed-16-bit PCM captured from the microphone.
 * @param pcm raw PCM chunk inside an ArrayBuffer
 */
export function feedAudio(pcm: ArrayBuffer): void;

/** Return the last backend error (empty string when none). */
export function getLastError(): string;

/** Preload the RKNN runtime (idempotent; call once from the main thread). */
export function preloadRuntime(): void;
