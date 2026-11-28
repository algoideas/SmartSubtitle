#!/usr/bin/env python3
"""用修复后的管线生成 int8 量化校准数据.

旧 int8 模型 (11/26 转换) 使用了 LoadWav bug 时期板卡 dump 的校准数据,
所有层量化 scale 错误 (板上实测 enc 幅度偏移 1.7~7.3 倍)。本脚本用主机
已验证的特征管线 (与 tools/host_ref2.py 流式分块循环完全一致) 从模型目录
测试 wav 生成 [1,18,560] float32 校准样本, 并写出 dataset 列表供
third_party/rknn/convert_paraformer_encoder.py --dataset 使用。

用法:
    python3 tools/gen_calib.py [输出 dataset 列表]
"""
import os
import sys
import wave

import numpy as np
import onnx

MODEL_DIR = ("third_party/models/"
             "sherpa-onnx-streaming-paraformer-bilingual-zh-en")
WAVS = [MODEL_DIR + "/test_wavs/%d.wav" % i for i in range(4)]
WAVS.append("entry/src/main/resources/rawfile/0.wav")
OUT = sys.argv[1] if len(sys.argv) > 1 else "calib_clean/calib_clean.txt"

# ---- kaldi fbank (与 host_ref2.py 一致) ----
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


# ---- 元数据 (与参考一致) ----
m = onnx.load(MODEL_DIR + "/encoder.onnx", load_external_data=False)
meta = {kv.key: kv.value for kv in m.metadata_props}
neg_mean = np.array([float(x) for x in meta["neg_mean"].split(",")],
                    dtype=np.float32)
inv = np.array([float(x) for x in meta["inv_stddev"].split(",")],
               dtype=np.float32)
inv *= np.sqrt(512.0)
LFR_WS, LFR_SHIFT = 7, 6
LEFT, RIGHT = 5, 3
K_SCALE = -0.03301197265941284


def cmvn(v):
    return (v - neg_mean) * inv


def pe(v, t_offset):
    T = len(v)
    d = np.arange(280, dtype=np.float32)
    base = np.exp(d * K_SCALE)
    out = np.empty_like(v)
    for t in range(T):
        off = (t + 1 + t_offset) * base
        out[t, :280] = np.sin(off)
        out[t, 280:] = np.cos(off)
    return v + out


def streaming_blocks(pcm):
    """与 host_ref2.py 完全一致的流式分块循环, 返回每个 18×560 窗口."""
    mel = fbank(pcm)
    mel_ext = np.vstack([mel, np.zeros((480, 80), dtype=np.float32)])
    feat_cache = np.zeros((LEFT + RIGHT, 560), dtype=np.float32)
    processed = 0
    blocks = []
    while processed + 61 <= mel_ext.shape[0]:
        fr = mel_ext[processed:processed + 61]
        pbefore = processed
        processed += 60
        n_lfr = (61 - LFR_WS) // LFR_SHIFT + 1
        cur = np.stack([fr[i * LFR_SHIFT:i * LFR_SHIFT + LFR_WS].flatten()
                        for i in range(n_lfr)]).astype(np.float32)
        cur = cmvn(cur)
        cur = pe(cur, pbefore // LFR_SHIFT)
        win = np.vstack([feat_cache, cur])
        feat_cache[:] = win[-(LEFT + RIGHT):]
        blocks.append(win)
    return blocks


def main():
    out_dir = os.path.abspath(os.path.dirname(OUT))
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    idx = 0
    maxabs = 0.0
    for wpath in WAVS:
        w = wave.open(wpath, "rb")
        assert w.getframerate() == SR and w.getnchannels() == 1, wpath
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        n = 0
        # 6 个分块相位 (每 10ms 一帧), 增加样本量
        for phase in range(0, 960, 160):
            if phase >= len(pcm) - SR:
                break
            for win in streaming_blocks(pcm[phase:]):
                a = np.ascontiguousarray(win[None].astype(np.float32))
                fn = os.path.join(out_dir, "%04d.bin.npy" % idx)
                np.save(fn, a)
                paths.append(fn)
                idx += 1
                n += 1
                maxabs = max(maxabs, float(np.abs(a).max()))
        print("%s: %d blocks" % (os.path.basename(wpath), n))
    out_path = os.path.abspath(OUT)
    with open(out_path, "w") as f:
        f.write("\n".join(paths) + "\n")
    print("共 %d 校准样本 -> %s, max|feat|=%.2f" % (len(paths), out_path,
                                                    maxabs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
