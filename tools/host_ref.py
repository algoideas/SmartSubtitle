#!/usr/bin/env python3
"""主机端 numpy fbank + float 编码器 + CIF + 解码器, 对 0.wav �几个块做权威解码.
用于判定: 板卡特征链问题 vs 模型/音频本身问题."""
import sys
import numpy as np
import wave
import onnxruntime as ort

MODEL_DIR = "third_party/models/sherpa-onnx-streaming-paraformer-bilingual-zh-en"
WAV = sys.argv[1] if len(sys.argv) > 1 else MODEL_DIR + "/test_wavs/0.wav"
NCHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else 5

# ---- kaldi fbank (knf 默认: povey, preemph .97, 80 mel, ln, 25ms/10ms) ----
SR = 16000
FL = 400          # 25ms
FS = 160          # 10ms
NFFT = 512
NMEL = 80


def povey(n):
    a = np.arange(n)
    return np.power(0.5 - 0.5 * np.cos(2 * np.pi * a / (n - 1)), 0.85)


def mel_banks():
    # kaldi mel: 1127*ln(1+f/700), low_freq=20, high_freq=nyquist
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
    # 补齐到与 knf/snip_edges 相同的帧数
    frames = np.stack([x[i * FS:i * FS + FL] for i in range(n)])
    # remove dc
    frames = frames - frames.mean(axis=1, keepdims=True)
    # preemph (保留首样本)
    p = np.copy(frames)
    p[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]
    p[:, 0] = frames[:, 0] - 0.97 * frames[:, 0]
    p *= WIN
    # FFT -> power
    spec = np.abs(np.fft.rfft(p, NFFT)) ** 2
    mel = spec @ MEL.T
    mel = np.log(np.maximum(mel, 1e-15))
    return mel.astype(np.float32)  # [T, 80]


# ---- 读 wav ----
w = wave.open(WAV, "rb")
assert w.getframerate() == SR and w.getnchannels() == 1
pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
print("wav %d samples %.2fs" % (len(pcm), len(pcm) / SR))

mel = fbank(pcm)
print("fbank: shape", mel.shape, "mean %.3f" % mel.mean())

# ---- 元数据 ----
import onnx
m = onnx.load(MODEL_DIR + "/encoder.onnx", load_external_data=False)
meta = {kv.key: kv.value for kv in m.metadata_props}
neg_mean = np.array([float(x) for x in meta["neg_mean"].split(",")],
                    dtype=np.float32)
inv = np.array([float(x) for x in meta["inv_stddev"].split(",")],
               dtype=np.float32) * np.sqrt(512.0)
lfr_ws, lfr_shift = 7, 6

so = ort.SessionOptions()
so.log_severity_level = 3
enc_sess = ort.InferenceSession(MODEL_DIR + "/encoder.onnx", so,
                                providers=["CPUExecutionProvider"])
ein = [i.name for i in enc_sess.get_inputs()]
eout = [o.name for o in enc_sess.get_outputs()]

# tokens
tokens = {}
for line in open(MODEL_DIR + "/tokens.txt", encoding="utf-8"):
    a = line.rstrip("\n").split(" ")
    if len(a) == 2:
        tokens[int(a[1])] = a[0]
# 简单 BPE 还原: @@ 直接拼接, 否则空格
def to_text(ids):
    s = ""
    for i in ids:
        t = tokens.get(i, "?")
        if t.endswith("@@"):
            s += t[:-2]
        else:
            s += t + " "
    return s

cache = np.zeros((8, 560), dtype=np.float32)
raw_tail = None       # LFR 原始帧余量
processed = 0
raw_buf = pcm
pos = 0
cmvn_state = None     # 简化: 直接用元数据 (训练统计) 做 CMVN
nfr_total = len(pcm) // FS
chunk_i = 0
utter = ""
while pos + 61 <= nfr_total and chunk_i < NCHUNK:
    fr = mel[pos:pos + 61]
    pos += 60
    # LFR
    n_lfr = (61 - lfr_ws) // lfr_shift + 1
    outs = []
    for i in range(n_lfr):
        seg = fr[i * lfr_shift:i * lfr_shift + lfr_ws].flatten()
        outs.append(seg)
    cur = np.stack(outs)  # [10,560]
    win = np.vstack([cache, cur])  # [18,560]
    # CMVN (元数据)
    win = (win - neg_mean) * inv
    # PE
    t_off = (chunk_i * 60) // lfr_shift
    pe = np.zeros_like(win)
    for t in range(18):
        off = t + 1 + t_off
        d = np.arange(280)
        inv_t = off * np.exp(d * kScale if False else d * (-0.03301197265941284))
        pe[t, :280] = np.sin(inv_t)
        pe[t, 280:] = np.cos(inv_t)
    win = win + pe
    feed = {}
    for n in ein:
        feed[n] = np.array([18], dtype=np.int32) if "length" in n else \
            win[None].astype(np.float32)
    res = enc_sess.run(eout, feed)
    om = dict(zip(eout, res))
    ek = [k for k in om if k.startswith("enc")][0]
    ak = [k for k in om if "alpha" in om[k][:5] or "alpha" in k][0]
    enc, alphas = om[ek], om[ak]
    print("chunk %d alpha:" % chunk_i, np.round(alphas.flatten(), 3))
    # CIF
    al = alphas.copy().flatten()
    al[:5] = 0
    al[15:] = 0
    integ = np.zeros(512, dtype=np.float32)
    wsum = 0.0
    fired = []
    for t in range(18):
        w = float(al[t])
        integ += enc[0, t] * w
        wsum += w
        if wsum >= 1.0:
            fired.append(integ / wsum)
            integ[:] = 0
            wsum = 0.0
    cache = win[-8:]
    # 解码器: enc 上下文 + 卷积状态
    dsess = ort.InferenceSession(MODEL_DIR + "/decoder.int8.onnx", so,
                                 providers=["CPUExecutionProvider"])
    dnm = [i.name for i in dsess.get_inputs()]
    don = [o.name for o in dsess.get_outputs()]
    dec_cache = {n: np.zeros((1, 512, 10), dtype=np.float32)
                 for n in dnm if n.startswith("in_cache")}
    if fired:
        xs = np.stack(fired).astype(np.float32)  # [n,512]
        feed = {"enc": enc.astype(np.float32),
                "enc_len": np.array([18], dtype=np.int32),
                "acoustic_embeds": xs[None],
                "acoustic_embeds_len": np.array([len(xs)], dtype=np.int32)}
        feed.update(dec_cache)
        r = dsess.run(don, feed)
        omm = dict(zip(don, r))
        ids = np.array(omm[[k for k in omm if "id" in k][0]]).flatten()
        for tid in ids:
            utter += to_text([int(tid)])
        dec_cache = {n: omm[n.replace("in_cache", "out_cache")]
                     for n in dec_cache}
    print("chunk %d text: %s" % (chunk_i - 1, utter))
