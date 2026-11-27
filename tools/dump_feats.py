#!/usr/bin/env python3
"""导出主机各块 pre-CMVN dim0 均值 + 完整 18x560 特征块, 供与板卡 dump 逐块对比."""
import os
import numpy as np
import wave
import onnx

MODEL_DIR = "third_party/models/sherpa-onnx-streaming-paraformer-bilingual-zh-en"
WAV = "entry/src/main/resources/rawfile/0.wav"
OUT = "/tmp/host_feat"
os.makedirs(OUT, exist_ok=True)

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


feat_cache = np.zeros((LEFT + RIGHT, 560), dtype=np.float32)
processed = 0
mel_ext = np.vstack([mel, np.zeros((480, 80), dtype=np.float32)])
chunk_i = 0
while processed + 61 <= mel_ext.shape[0] and chunk_i < 32:
    fr = mel_ext[processed:processed + 61]
    processed_before = processed
    processed += 60
    n_lfr = (61 - LFR_WS) // LFR_SHIFT + 1
    cur = np.stack([fr[i * LFR_SHIFT:i * LFR_SHIFT + LFR_WS].flatten()
                    for i in range(n_lfr)]).astype(np.float32)  # [10,560]
    pre_dim0 = float(cur[:, 0].mean())
    cur = cmvn(cur)
    cur = pe(cur, processed_before // LFR_SHIFT)
    win = np.vstack([feat_cache, cur])
    feat_cache[:] = win[-(LEFT + RIGHT):]
    win.astype(np.float32).tofile("%s/%03d.bin" % (OUT, chunk_i))
    print("chunk %2d pre_dim0=%.3f" % (chunk_i, pre_dim0))
    chunk_i += 1
print("dumped %d blocks to %s" % (chunk_i, OUT))
