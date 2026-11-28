#!/usr/bin/env python3
"""RKNN 编码器转换实验脚本: 自动混合量化 / 关闭 SDPA 融合等开关组合.

背景: 纯 int8 与自动混合量化在 RK3588 NPU 上 enc 输出量级系统性错乱
(1.2~8.6 倍, 逐块漂移), 纯 float 构建输出 NaN。本脚本提供单变量对照开关:
  --no-hybrid     关闭自动混合量化 (纯 int8)
  --no-flash      config(enable_flash_attention=False), 禁用 SDPA 融合
  --algo NAME     quantized_algorithm: normal(默认)/mmse/kld
  --onnx PATH     自定义输入 ONNX (默认全编码器 sim 图)
  --dtype NAME    quantized_dtype: w8a8(默认)/w8a16/w16a16i_dfp/w4a16 (平台相关)
  --float         纯 float 构建 (do_quantization=False)
用法:
    python3 tools/rknn_experiment.py [输出.rknn] [cos阈值] [--no-hybrid]
                                     [--no-flash] [--algo mmse] [--onnx PATH]
                                     [--dtype w16a16i_dfp] [--float]
"""
import sys

from rknn.api import RKNN

SIM = ("third_party/models/"
       "sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.onnx.sim.onnx")
DATASET = "calib_clean/calib_clean.txt"


def main():
    args = sys.argv[1:]
    no_hybrid = "--no-hybrid" in args
    no_flash = "--no-flash" in args
    is_float = "--float" in args
    if is_float:
        args.remove("--float")
    sim = SIM
    algo = "normal"
    dtype = None
    for key in ("--onnx", "--algo", "--dtype"):
        if key in args:
            i = args.index(key)
            val = args[i + 1]
            del args[i:i + 2]
            if key == "--onnx":
                sim = val
            elif key == "--algo":
                algo = val
            else:
                dtype = val
    pos = [a for a in args if not a.startswith("--")]
    out = pos[0] if pos else "/tmp/paraformer-encoder-exp.rknn"
    cos = float(pos[1]) if len(pos) > 1 else 0.98

    cfg = dict(
        target_platform="rk3588",
        quantized_method="channel",
        quantized_algorithm=algo,
        optimization_level=3,
    )
    if not no_hybrid:
        cfg["quantized_hybrid_level"] = 1
        cfg["auto_hybrid_cos_thresh"] = cos
    if no_flash:
        cfg["enable_flash_attention"] = False
    if dtype:
        cfg["quantized_dtype"] = dtype
    print("-- config:", cfg)

    rknn = RKNN(verbose=True)
    rknn.config(**cfg)
    ret = rknn.load_onnx(model=sim, inputs=["speech"],
                         input_size_list=[[1, 18, 560]])
    assert ret == 0, "load_onnx failed: %d" % ret
    print("-- build (do_quantization=%s)" % (not is_float))
    ret = rknn.build(do_quantization=not is_float, dataset=DATASET)
    assert ret == 0, "build failed: %d" % ret
    print("-- export:", out)
    ret = rknn.export_rknn(out)
    assert ret == 0, "export failed: %d" % ret
    print("-- done:", out)
    rknn.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
