#!/usr/bin/env python3
"""自动混合量化转换: int8 + 量化误差大的层自动保 fp16.

背景: 纯 int8 转换的编码器在 RK3588 NPU 上 enc 输出量级系统性错乱
(1.6~8.6 倍, 逐块漂移), 导致 NPU 识别乱码; 纯 float(do_quantization=False)
构建的模型在 NPU 上输出 NaN。自动混合量化 (quantized_hybrid_level=1)
让 toolkit 按逐算子量化余弦相似度自动把敏感层改为 fp16。

用法:
    python3 tools/auto_hybrid_convert.py [输出 .rknn] [cos 阈值]
"""
import sys

from rknn.api import RKNN

SIM = ("third_party/models/"
       "sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.onnx.sim.onnx")
DATASET = "calib_clean/calib_clean.txt"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/paraformer-encoder-autohybrid.rknn"
COS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.98

rknn = RKNN(verbose=True)
print("-- config: auto hybrid, cos_thresh=%.2f" % COS)
rknn.config(
    target_platform="rk3588",
    quantized_method="channel",
    quantized_algorithm="normal",
    optimization_level=3,
    quantized_hybrid_level=1,
    auto_hybrid_cos_thresh=COS,
)
print("-- load_onnx (sim, speech [1,18,560])")
ret = rknn.load_onnx(model=SIM, inputs=["speech"],
                     input_size_list=[[1, 18, 560]])
assert ret == 0, "load_onnx failed: %d" % ret
print("-- build (int8 + auto hybrid fp16)")
ret = rknn.build(do_quantization=True, dataset=DATASET)
assert ret == 0, "build failed: %d" % ret
print("-- export:", OUT)
ret = rknn.export_rknn(OUT)
assert ret == 0, "export failed: %d" % ret
print("-- done:", OUT)
rknn.release()
