#!/usr/bin/env python3
"""验证 "NPU 只跑 encoder(enc), CIF 外置到 CPU(float32)" 方案.

对 0.wav 逐块 (与 host_ref2.py 完全相同的特征管线) 对比:
  A: encoder.onnx(float) 内置 alphas            — 金标准
  B: encoder.int8.onnx 内置 alphas              — 当前 CPU 路径
  C: 外置 CIF(从 ONNX 反向提取的子图, ORT float32) @ float enc — 验证提取正确性
  D: 外置 CIF @ int8 enc                        — 方案预期质量(上界)
并用 D 的 alphas 跑完整 CIF 集成 + int8 解码器, 输出最终文本.

预期: C≈A (提取正确); D 与 A 接近; 最终文本正确.
"""
import sys
import numpy as np
import wave
import onnxruntime as ort
import onnx
from onnx import helper

MODEL_DIR = "third_party/models/sherpa-onnx-streaming-paraformer-bilingual-zh-en"
WAV = sys.argv[1] if len(sys.argv) > 1 else "entry/src/main/resources/rawfile/0.wav"

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


w = wave.open(WAV, "rb")
assert w.getframerate() == SR and w.getnchannels() == 1
pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
mel = fbank(pcm)
print("wav %d samples %.2fs, fbank %d frames" % (len(pcm), len(pcm) / SR, mel.shape[0]))

# ---- 元数据 ----
m = onnx.load(MODEL_DIR + "/encoder.onnx", load_external_data=False)
meta = {kv.key: kv.value for kv in m.metadata_props}
neg_mean = np.array([float(x) for x in meta["neg_mean"].split(",")], dtype=np.float32)
inv = np.array([float(x) for x in meta["inv_stddev"].split(",")], dtype=np.float32)
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


# ---- 反向 BFS 提取 alphas 生产链子图 ----
def build_cif_subgraph(model):
    # 停止点 = 原图输入 ∪ 原图输出(enc/enc_len), 防止 BFS 越过 enc 钻入编码器主干
    stop_tensors = ({i.name for i in model.graph.input} |
                    {o.name for o in model.graph.output})
    stop_tensors.discard("alphas")
    pred = {}
    for n in model.graph.node:
        for o in n.output:
            pred[o] = n
    init_map = {t.name: t for t in model.graph.initializer}
    needed, inits, used = set(), [], set()
    reached_inputs = set()
    queue = ["alphas"]
    while queue:
        t = queue.pop()
        if t in stop_tensors:
            reached_inputs.add(t)
            continue
        n = pred.get(t)
        if n is None:
            if t in init_map and t not in used:
                inits.append(init_map[t])
                used.add(t)
            continue
        if n.name in needed:
            continue
        needed.add(n.name)
        for i in n.input:
            if i:
                queue.append(i)
    sel = [n for n in model.graph.node if n.name in needed]
    shape_of = {"enc": [1, 18, 512], "speech_lengths": [1], "speech": [1, 18, 560]}
    dtype_of = {"speech_lengths": onnx.TensorProto.INT32}
    ins = [helper.make_tensor_value_info(
        name, dtype_of.get(name, onnx.TensorProto.FLOAT), shape_of[name])
        for name in sorted(reached_inputs)]
    g = helper.make_graph(sel, "cif_subgraph", ins,
                          [helper.make_tensor_value_info(
                              "alphas", onnx.TensorProto.FLOAT, [1, 18])],
                          inits)
    sm = helper.make_model(g, opset_imports=[helper.make_opsetid("", 14)])
    sm.ir_version = 9  # 主机 ORT 1.16 最高支持 IR 9
    onnx.checker.check_model(sm)
    return sm, sorted(reached_inputs)


cif_model, cif_ins = build_cif_subgraph(m)
cif_bytes = cif_model.SerializeToString()
open("/tmp/cif_subgraph.onnx", "wb").write(cif_bytes)
print("CIF 子图: %d 节点, %d 初始化器, 输入=%s, %.1f KB" % (
    len(cif_model.graph.node), len(cif_model.graph.initializer), cif_ins,
    len(cif_bytes) / 1024))

# ---- 会话 ----
so = ort.SessionOptions()
so.log_severity_level = 3
enc_f = ort.InferenceSession(MODEL_DIR + "/encoder.onnx", so,
                             providers=["CPUExecutionProvider"])
enc_i = ort.InferenceSession(MODEL_DIR + "/encoder.int8.onnx", so,
                             providers=["CPUExecutionProvider"])
dec = ort.InferenceSession(MODEL_DIR + "/decoder.int8.onnx", so,
                           providers=["CPUExecutionProvider"])
cif_s = ort.InferenceSession(cif_bytes, so, providers=["CPUExecutionProvider"])

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


def run_cif(enc_arr):
    feed = {"enc": enc_arr, "speech_lengths": np.array([18], dtype=np.int32)}
    if "speech" in cif_ins:
        feed["speech"] = np.zeros((1, 18, 560), dtype=np.float32)
    return cif_s.run(["alphas"], feed)[0][0].astype(np.float32)


# ---- 状态 (与 host_ref2.py 一致) ----
feat_cache = np.zeros((LEFT + RIGHT, 560), dtype=np.float32)
initial_hidden = np.zeros(512, dtype=np.float32)
alpha_cache = np.zeros(1, dtype=np.float32)
dec_cache = {n: np.zeros((1, 512, 10), dtype=np.float32)
             for n in ["in_cache_%d" % i for i in range(16)]}

mel_ext = np.vstack([mel, np.zeros((480, 80), dtype=np.float32)])
nfr_total = mel_ext.shape[0]
utter_ids = []
processed = 0
chunk_i = 0
diff_c_max, diff_d_max, diff_b_max = 0.0, 0.0, 0.0

while processed + 61 <= nfr_total:
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

    feed_f = {"speech": win[None].astype(np.float32),
              "speech_lengths": np.array([18], dtype=np.int32)}
    enc_fv, _, al_f = enc_f.run(["enc", "enc_len", "alphas"], feed_f)
    enc_iv, _, al_b = enc_i.run(["enc", "enc_len", "alphas"], feed_f)
    al_c = run_cif(enc_fv)
    al_d = run_cif(enc_iv)
    diff_c_max = max(diff_c_max, float(np.abs(al_c - al_f[0]).max()))
    diff_d_max = max(diff_d_max, float(np.abs(al_d - al_f[0]).max()))
    diff_b_max = max(diff_b_max, float(np.abs(al_b[0] - al_f[0]).max()))
    if chunk_i < 3 or chunk_i % 5 == 0:
        print("chunk %2d | A=%.3f B=%.3f C=%.3f D=%.3f | C-A max=%.2e D-A max=%.2e" % (
            chunk_i, al_f[0].sum(), al_b[0].sum(), al_c.sum(), al_d.sum(),
            np.abs(al_c - al_f[0]).max(), np.abs(al_d - al_f[0]).max()))

    # ---- 用 D (外置 CIF @ int8 enc) 跑 CIF 集成 + 解码 ----
    al = al_d.copy()
    al[:LEFT] = 0.0
    al[-RIGHT:] = 0.0
    threshold = 1.0
    integrate = float(alpha_cache[0])
    acoustic = []
    for i in range(18):
        a = float(al[i])
        if integrate + a < threshold:
            integrate += a
            initial_hidden += enc_iv[0, i] * a
            continue
        initial_hidden += enc_iv[0, i] * (threshold - integrate)
        acoustic.append(initial_hidden.copy())
        integrate += a - threshold
        initial_hidden[:] = enc_iv[0, i] * integrate
    alpha_cache[0] = integrate

    if acoustic:
        xs = np.stack(acoustic).astype(np.float32)
        feed = {"enc": enc_iv, "enc_len": np.array([18], dtype=np.int32),
                "acoustic_embeds": xs[None],
                "acoustic_embeds_len": np.array([len(xs)], dtype=np.int32)}
        feed.update(dec_cache)
        outs = dec.run(
            ["logits", "sample_ids"] + ["out_cache_%d" % i for i in range(16)], feed)
        sample_ids = outs[1]
        for i in range(16):
            dec_cache["in_cache_%d" % i] = outs[2 + i]
        ids = [int(t) for t in sample_ids.flatten()]
        utter_ids.extend(ids)
        if chunk_i < 6 or chunk_i % 4 == 0:
            print("  [D] fire=%d text: %s" % (len(ids), to_text(ids)))
    chunk_i += 1

print()
print("max|C-A| = %.2e  (外置 CIF 提取 vs 金标准)" % diff_c_max)
print("max|B-A| = %.2e  (int8 内置 CIF vs 金标准)" % diff_b_max)
print("max|D-A| = %.2e  (外置 CIF @ int8 enc vs 金标准)" % diff_d_max)
print("FINAL(D): %s" % to_text(utter_ids))
