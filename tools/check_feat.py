#!/usr/bin/env python3
"""从板卡 dump 的 post-CMVN+PE 特征反推 pre-CMVN 特征, 逐维对比模型元数据."""
import sys
import numpy as np

MODEL_DIR = "third_party/models/sherpa-onnx-streaming-paraformer-bilingual-zh-en"
CHUNK = sys.argv[1] if len(sys.argv) > 1 else "calib_fixed/000.bin"
FEAT_DIM = 560
N = 18

# 读取模型元数据
import onnx
m = onnx.load(MODEL_DIR + "/encoder.onnx", load_external_data=False)
meta = {kv.key: kv.value for kv in m.metadata_props}
neg_mean = np.array([float(x) for x in meta["neg_mean"].split(",")])
inv_stddev = np.array([float(x) for x in meta["inv_stddev"].split(",")])
print("meta neg_mean[0:5] =", neg_mean[:5])

x = np.fromfile(CHUNK, dtype=np.float32).reshape(N, FEAT_DIM)

# 还原 PE (与 NPU 后端公式一致): offset = t + 1 + t_offset, t_offset=0 (首块)
kScale = -0.03301197265941284
pe = np.zeros((N, FEAT_DIM), dtype=np.float32)
for t in range(N):
    offset = t + 1
    for d in range(FEAT_DIM // 2):
        inv = offset * np.exp(d * kScale)
        pe[t, d] = np.sin(inv)
        pe[t, d + FEAT_DIM // 2] = np.cos(inv)
x_nope = x - pe

# 逆 CMVN: x = (v + neg_mean) * inv_stddev_scaled(=sqrt(512)*inv_stddev)
inv_scaled = inv_stddev * np.sqrt(512.0)
v = x_nope / inv_scaled[None, :] - neg_mean[None, :]
print("还原 pre-CMVN: mean %.3f std %.3f" % (v.mean(), v.std()))

# 还原 LFR: 逐帧 80 维 (t 窗口对应第 1..7 帧)
fb = v.reshape(N, 7, 80)
per_dim = fb.reshape(-1, 80).mean(axis=0)  # 各帧位置混合后的 80 维均值
mm = -neg_mean.reshape(7, 80).mean(axis=0)  # 元数据 7 帧平均 -> 80 维期望均值
print("80维均值(还原 vs 期望):")
for k in range(0, 80, 8):
    print("  dim%02d: got %7.3f  expect %7.3f  diff %+.3f"
          % (k, per_dim[k], mm[k], per_dim[k] - mm[k]))
