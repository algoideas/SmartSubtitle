#!/usr/bin/env python3
"""同一特征块上并排跑 float / int8 编码器, 对比 alpha 分布,
判定 int8 编码器文件量化质量 vs 板卡运行期."""
import numpy as np
import wave
import onnxruntime as ort
import onnx
import sys

MODEL_DIR = "third_party/models/sherpa-onnx-streaming-paraformer-bilingual-zh-en"
WAV = "entry/src/main/resources/rawfile/0.wav"

SR = 16000
FL, FS, NFFT, NMEL = 400, 160, 512, 80


def povey(n):
    a = np.arange(n)
    return np.power(0.5 - 0.5 * np.cos(2 * np.pi * a / (n - 1)), 0.85)


def mel_banks():
    low, high = 20.0, SR / 2.0
    ml, mh = 1127.0 * np.log(1 + low / 700), 1127.0 * np.log(1 + high / 700)
    pts = np.linspace(ml, mh, NMEL + 2)
    freqs = 700.0 * (np.exp(pts / 1127.0) - 1)
    bins = np.floor((NFFT + 1) * freqs / SR).astype(int)
    mat = np.zeros((NMEL, NFFT // 2 + 1))
    for b in range(NMEL):
        l, c, r = bins[b], bins[b + 1], bins[b + 2]
        if c == l:
            c = l + 1
        if r == c:
            r = c + 1
        for k in range(l, c):
            mat[b, k] = (k - l) / float(c - l)
        for k in range(c, r):
            mat[b, k] = (r - k) / float(r - c)
    return mat


MEL = mel_banks()
WIN = povey(FL)


def fbank(x16):
    x = x16.astype(np.float32)
    n = 1 + (len(x) - FL) // FS
    frames = np.stack([x[i * FS:i * FS + FL] for i in range(n)])
    frames = frames - frames.mean(axis=1, keepdims=True)
    p = np.copy(frames)
    p[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]
    p[:, 0] = frames[:, 0] - 0.97 * frames[:, 0]
    p *= WIN
    spec = np.abs(np.fft.rfft(p, NFFT)) ** 2
    mel = spec @ MEL.T
    mel = np.log(np.maximum(mel, 1e-15))
    return mel.astype(np.float32)


w = wave.open(WAV, "rb")
pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
mel = fbank(pcm)

m = onnx.load(MODEL_DIR + "/encoder.onnx", load_external_data=False)
meta = {kv.key: kv.value for kv in m.metadata_props}
neg_mean = np.array([float(x) for x in meta["neg_mean"].split(",")], dtype=np.float32)
inv = np.array([float(x) for x in meta["inv_stddev"].split(",")], dtype=np.float32)
inv *= np.sqrt(512.0)
LFR_WS, LFR_SHIFT, LEFT, RIGHT = 7, 6, 5, 3
K_SCALE = -0.03301197265941284
d280 = np.arange(280, dtype=np.float32)
base = np.exp(d280 * K_SCALE)


def cmvn(v):
    return (v - neg_mean) * inv


def pe(v, t_offset):
    T = len(v)
    out = np.empty_like(v)
    for t in range(T):
        off = (t + 1 + t_offset) * base
        out[t, :280] = np.sin(off)
        out[t, 280:] = np.cos(off)
    return v + out


so = ort.SessionOptions()
so.log_severity_level = 3
enc_f = ort.InferenceSession(MODEL_DIR + "/encoder.onnx", so,
                             providers=["CPUExecutionProvider"])
enc_i = ort.InferenceSession(MODEL_DIR + "/encoder.int8.onnx", so,
                             providers=["CPUExecutionProvider"])

feat_cache = np.zeros((LEFT + RIGHT, 560), dtype=np.float32)
processed = 0
mel_ext = np.vstack([mel, np.zeros((480, 80), dtype=np.float32)])
chunk_i = 0
while processed + 61 <= mel_ext.shape[0] and chunk_i < 12:
    fr = mel_ext[processed:processed + 61]
    processed_before = processed
    processed += 60
    n_lfr = (61 - LFR_WS) // LFR_SHIFT + 1
    cur = np.stack([fr[i * LFR_SHIFT:i * LFR_SHIFT + LFR_WS].flatten()
                    for i in range(n_lfr)]).astype(np.float32)
    cur = cmvn(cur)
    cur = pe(cur, processed_before // LFR_SHIFT)
    win = np.vstack([feat_cache, cur])
    feat_cache[:] = win[-(LEFT + RIGHT):]
    feed = {"speech": win[None].astype(np.float32),
            "speech_lengths": np.array([18], dtype=np.int32)}
    af = enc_f.run(["alphas"], feed)[0][0].copy()
    ai = enc_i.run(["alphas"], feed)[0][0].copy()
    afm = af[LEFT:-RIGHT].sum()
    aim = ai[LEFT:-RIGHT].sum()
    print("chunk %2d float_mid=%.3f int8_mid=%.3f" % (chunk_i, afm, aim))
    print("   float: [%s]" % ",".join("%.3f" % x for x in np.round(af, 3)))
    print("   int8 : [%s]" % ",".join("%.3f" % x for x in np.round(ai, 3)))
    chunk_i += 1
