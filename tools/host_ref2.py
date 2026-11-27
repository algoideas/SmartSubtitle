#!/usr/bin/env python3
"""与 vendored 参考 (online-recognizer-paraformer-impl.h DecodeStream) 逐行一致的
numpy 参考管线: kaldi fbank -> LFR -> [CMVN(-neg_mean, 含√512) + PE 仅作用于 10 个新帧]
-> 前置 8 帧上下文 -> 编码器 -> 首5/末3 清零 -> CIF(threshold=1.0, 跨块 integrate)
-> 解码器(卷积状态跨块持久). 用途: 主机确定性验证模型与管线逻辑."""
import sys
import numpy as np
import wave
import onnxruntime as ort
import onnx

MODEL_DIR = "third_party/models/sherpa-onnx-streaming-paraformer-bilingual-zh-en"
WAV = sys.argv[1] if len(sys.argv) > 1 else "entry/src/main/resources/rawfile/0.wav"

# ---- kaldi fbank (povey, preemph .97, 80 mel, ln, 25ms/10ms) ----
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
assert w.getframerate() == SR and w.getnchannels() == 1
pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
mel = fbank(pcm)
nfr_total = mel.shape[0]
print("wav %d samples %.2fs, fbank %d frames" % (len(pcm), len(pcm) / SR, nfr_total))

# ---- 元数据 (与参考 OnlineParaformerModel::Load 一致) ----
m = onnx.load(MODEL_DIR + "/encoder.onnx", load_external_data=False)
meta = {kv.key: kv.value for kv in m.metadata_props}
neg_mean = np.array([float(x) for x in meta["neg_mean"].split(",")], dtype=np.float32)
inv = np.array([float(x) for x in meta["inv_stddev"].split(",")], dtype=np.float32)
inv *= np.sqrt(512.0)  # 参考: inv_stddev *= sqrt(encoder_output_size)
LFR_WS, LFR_SHIFT = 7, 6
LEFT, RIGHT = 5, 3
K_SCALE = -0.03301197265941284


def cmvn(v):  # [T,560] -> [T,560]; 参考公式 (x - neg_mean) * inv_stddev
    return (v - neg_mean) * inv


def pe(v, t_offset):  # 参考: offset = t + 1 + t_offset, 前半 sin 后半 cos
    T = len(v)
    d = np.arange(280, dtype=np.float32)
    base = np.exp(d * K_SCALE)
    out = np.empty_like(v)
    for t in range(T):
        off = (t + 1 + t_offset) * base
        out[t, :280] = np.sin(off)
        out[t, 280:] = np.cos(off)
    return v + out


so = ort.SessionOptions()
so.log_severity_level = 3
enc_sess = ort.InferenceSession(MODEL_DIR + "/encoder.onnx", so,
                                providers=["CPUExecutionProvider"])
dec_sess = ort.InferenceSession(MODEL_DIR + "/decoder.int8.onnx", so,
                                providers=["CPUExecutionProvider"])

tokens = {}
for line in open(MODEL_DIR + "/tokens.txt", encoding="utf-8"):
    a = line.rstrip("\n").split(" ")
    if len(a) == 2:
        tokens[int(a[1])] = a[0]


def to_text(ids):
    s = ""
    for i in ids:
        t = tokens.get(i, "?")
        if t.endswith("@@"):
            s += t[:-2]
        else:
            s += t + " "
    return s


# ---- 状态 (与参考 per-stream 状态一致) ----
feat_cache = np.zeros((LEFT + RIGHT, 560), dtype=np.float32)  # 首块补零
initial_hidden = np.zeros(512, dtype=np.float32)
alpha_cache = np.zeros(1, dtype=np.float32)
dec_cache = {n: np.zeros((1, 512, 10), dtype=np.float32)
             for n in ["in_cache_%d" % i for i in range(16)]}

utter_ids = []
processed = 0  # 已处理原始 fbank 帧数 (参考 GetNumProcessedFrames)
chunk_i = 0
# 尾部静音: 与 C++ 析构注入等价 (6 个 0.5s 块 = 480 帧补零)
mel_ext = np.vstack([mel, np.zeros((480, 80), dtype=np.float32)])
nfr_total = mel_ext.shape[0]

while processed + 61 <= nfr_total:
    fr = mel_ext[processed:processed + 61]
    processed_before = processed
    processed += 60  # chunk_size - 1

    # LFR: 61 -> 10
    n_lfr = (61 - LFR_WS) // LFR_SHIFT + 1
    cur = np.stack([fr[i * LFR_SHIFT:i * LFR_SHIFT + LFR_WS].flatten()
                    for i in range(n_lfr)]).astype(np.float32)  # [10,560]

    # 参考顺序: 先对 10 个新帧 CMVN + PE
    cur = cmvn(cur)
    cur = pe(cur, processed_before // LFR_SHIFT)

    # 前置上下文 -> 18 帧
    win = np.vstack([feat_cache, cur])
    # 缓存 = 整窗末 8 帧 (参考 234 行)
    feat_cache[:] = win[-(LEFT + RIGHT):]

    feed = {"speech": win[None].astype(np.float32),
            "speech_lengths": np.array([18], dtype=np.int32)}
    enc, enc_len, alphas = enc_sess.run(["enc", "enc_len", "alphas"], feed)
    alphas = alphas[0].astype(np.float32).copy()
    alphas[:LEFT] = 0.0
    alphas[-RIGHT:] = 0.0

    # CIF (参考 292-314 行逐行)
    threshold = 1.0
    integrate = float(alpha_cache[0])
    acoustic = []
    for i in range(18):
        a = float(alphas[i])
        if integrate + a < threshold:
            integrate += a
            initial_hidden += enc[0, i] * a
            continue
        initial_hidden += enc[0, i] * (threshold - integrate)
        acoustic.append(initial_hidden.copy())
        integrate += a - threshold
        initial_hidden[:] = enc[0, i] * integrate
    alpha_cache[0] = integrate

    n_fire = len(acoustic)
    am = alphas[LEFT:-RIGHT].sum()
    print("chunk %2d fire=%d middle_alpha_sum=%.3f alpha=[%s]" % (
        chunk_i, n_fire, am,
        ",".join("%.3f" % x for x in np.round(alphas, 3))))

    if acoustic:
        xs = np.stack(acoustic).astype(np.float32)
        feed = {"enc": enc, "enc_len": np.array([18], dtype=np.int32),
                "acoustic_embeds": xs[None],
                "acoustic_embeds_len": np.array([len(xs)], dtype=np.int32)}
        feed.update(dec_cache)
        outs = dec_sess.run(
            ["logits", "sample_ids"] + ["out_cache_%d" % i for i in range(16)], feed)
        sample_ids = outs[1]
        for i in range(16):
            dec_cache["in_cache_%d" % i] = outs[2 + i]
        ids = [int(t) for t in sample_ids.flatten()]
        utter_ids.extend(ids)
        print("  text: %s" % to_text(utter_ids[-len(ids):]))
    chunk_i += 1

print("FINAL: %s" % to_text(utter_ids))
