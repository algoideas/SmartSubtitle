#!/usr/bin/env python3
"""主机端参考管线: 用 onnxruntime 复现 NPU 后端的 编码器+CIF+解码器 流程,
对单块 [1,18,560] 特征给出权威输出, 用于定位识别质量问题."""
import sys
import numpy as np
import onnxruntime as ort

MODEL_DIR = "third_party/models/sherpa-onnx-streaming-paraformer-bilingual-zh-en"
CHUNK = sys.argv[1] if len(sys.argv) > 1 else "calib_fixed/000.bin"

frames = np.fromfile(CHUNK, dtype=np.float32).reshape(1, 18, 560)
print("feat: shape", frames.shape, "mean %.3f std %.3f min %.3f max %.3f"
      % (frames.mean(), frames.std(), frames.min(), frames.max()))

so = ort.SessionOptions()
so.log_severity_level = 3


def run_enc(path):
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    outs = [o.name for o in sess.get_outputs()]
    feed = {}
    for n in names:
        if "length" in n:
            feed[n] = np.array([18], dtype=np.int32)
        else:
            feed[n] = frames
    res = sess.run(outs, feed)
    return dict(zip(outs, res))


enc_f = run_enc(MODEL_DIR + "/encoder.onnx")
enc_i = run_enc(MODEL_DIR + "/encoder.int8.onnx")

# 找 enc 与 alphas 输出名
enc_key = [k for k in enc_f if k.startswith("enc")][0]
alpha_key = [k for k in enc_f if "alpha" in k][0]
encf, af = enc_f[enc_key], enc_f[alpha_key]
enci, ai = enc_i[enc_key], enc_i[alpha_key]
print("float enc: shape", encf.shape, "abs_sum %.1f max %.3f" % (np.abs(encf).sum(), np.abs(encf).max()))
print("int8  enc: shape", enci.shape, "abs_sum %.1f" % np.abs(enci).sum())
print("enc max_diff(float vs int8): %.4f" % np.abs(encf - enci).max())
print("alpha float:", np.round(af.flatten(), 3))
print("alpha int8 :", np.round(ai.flatten(), 3))

# CIF (对齐 NPU 后端: 左 5 右 3 置零, 阈值 1.0 反向累计)
for tag, enc, alphas in (("float", encf, af), ("int8", enci, ai)):
    alphas = alphas.copy()
    alphas[0, :5] = 0
    alphas[0, 18 - 3:] = 0
    T = enc.shape[1]
    intergrated = np.zeros(enc.shape[2], dtype=np.float32)
    fired = []
    w_sum = 0.0
    for t in range(T):
        w = float(alphas[0, t])
        intergrated += enc[0, t] * w
        w_sum += w
        if w_sum >= 1.0:
            fired.append(intergrated.copy() / w_sum)
            intergrated[:] = 0
            w_sum = 0.0
    if not fired:
        print(tag, "CIF: 无 token")
        continue
    xs = np.stack(fired)[:, None, :]  # [n,1,512]
    tokens = {}
    for line in open(MODEL_DIR + "/tokens.txt", encoding="utf-8"):
        parts = line.rstrip("\n").split(" ")
        if len(parts) == 2:
            tokens[int(parts[1])] = parts[0]
    # 解码器
    dsess = ort.InferenceSession(MODEL_DIR + "/decoder.int8.onnx", so,
                                 providers=["CPUExecutionProvider"])
    dins = dsess.get_inputs()
    print(tag, "decoder inputs:", [(i.name, i.shape) for i in dins])
    cache = np.zeros((16, 2, 1, 512), dtype=np.float32)
    text = ""
    xname = [i.name for i in dins if "cache" not in i.name][0]
    cname = [i.name for i in dins if "cache" in i.name][0]
    onames = [o.name for o in dsess.get_outputs()]
    for x in xs:
        res = dsess.run(onames, {xname: x.astype(np.float32), cname: cache})
        omap = dict(zip(onames, res))
        ids = omap.get("sample_ids") or omap[[k for k in omap if "id" in k][0]]
        tid = int(np.array(ids).flatten()[0])
        text += tokens.get(tid, "(%d)" % tid)
        cache = omap[[k for k in omap if "cache" in k][0]]
    print(tag, "CIF fired:", len(fired), "text:", text)
