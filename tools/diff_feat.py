#!/usr/bin/env python3
"""逐元素对比: numpy 特征 vs 板卡 dump 特征 (chunk 1), 定位特征链分歧点."""
import numpy as np
import wave
import onnx

MODEL_DIR = "third_party/models/sherpa-onnx-streaming-paraformer-bilingual-zh-en"
SR, FL, FS, NFFT, NMEL = 16000, 400, 160, 512, 80


def povey(n):
    a = np.arange(n)
    return np.power(0.5 - 0.5 * np.cos(2 * np.pi * a / (n - 1)), 0.85)


def mel_banks():
    low, high = 20.0, SR / 2.0
    ml = 1127.0 * np.log(1 + low / 700)
    mh = 1127.0 * np.log(1 + high / 700)
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

w = wave.open(MODEL_DIR + "/test_wavs/0.wav", "rb")
pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
x = pcm.astype(np.float32)

# chunk 1 = fbank 帧 60..120 (每块取 61 帧, 步进 60)
fr = np.stack([x[i * FS + 60 * FS: i * FS + 60 * FS + FL]
               for i in range(61)])
fr = fr - fr.mean(axis=1, keepdims=True)
p = np.copy(fr)
p[:, 1:] = fr[:, 1:] - 0.97 * fr[:, :-1]
p[:, 0] = fr[:, 0] - 0.97 * fr[:, 0]
p *= WIN
spec = np.abs(np.fft.rfft(p, NFFT)) ** 2
mel = np.log(np.maximum(spec @ MEL.T, 1e-15)).astype(np.float32)

# LFR
outs = [mel[i * 6: i * 6 + 7].flatten() for i in range(10)]
cur = np.stack(outs)
cache_abs = mel[12:60].flatten()  # chunk0 的 abs LFR 帧 2..9 简化缓存
# 严格按 sherpa: chunk0 窗口=[8 zeros|abs0..9], 缓存=窗口末 8 帧=abs2..9
cache = np.vstack([np.zeros((8, 560), np.float32),
                   np.stack([mel[i * 6:i * 6 + 7].flatten()
                             for i in range(10)])])[-8:]
win_np = np.vstack([cache, cur])

m = onnx.load(MODEL_DIR + "/encoder.onnx", load_external_data=False)
meta = {kv.key: kv.value for kv in m.metadata_props}
neg_mean = np.array([float(x) for x in meta["neg_mean"].split(",")],
                    dtype=np.float32)
inv = np.array([float(x) for x in meta["inv_stddev"].split(",")],
               dtype=np.float32) * np.sqrt(512.0)
cmvn_np = (win_np - neg_mean) * inv

# PE
t_off = 10
pe = np.zeros_like(cmvn_np)
for t in range(18):
    off = t + 1 + t_off
    d = np.arange(280)
    it = off * np.exp(d * (-0.03301197265941284))
    pe[t, :280] = np.sin(it)
    pe[t, 280:] = np.cos(it)
win_np_full = cmvn_np + pe

board = np.fromfile("calib_fixed/001.bin", dtype=np.float32).reshape(18, 560)
d = board - win_np_full
print("board vs numpy 全窗: max_diff %.4f mean_diff %.4f"
      % (np.abs(d).max(), d.mean()))
# 去掉 PE 后比 (纯 CMVN 特征)
bn = board - pe
dn = bn - cmvn_np
print("去 PE 后:            max_diff %.4f mean_diff %.4f"
      % (np.abs(dn).max(), dn.mean()))
# 只比当前 10 帧 (窗口后 10 行)
print("当前帧区:            max_diff %.4f" % np.abs(dn[8:]).max())
print("缓存帧区:            max_diff %.4f" % np.abs(dn[:8]).max())
