#!/usr/bin/env python3
"""RKNN 手工两步混合量化 (toolkit 2.3.2, rk3588).

背景: auto-hybrid (quantized_hybrid_level) 的逐算子余弦存在明显空隙
(提升层 <0.90, 保留层 >0.90), 降阈值 0.98→0.90 选层数零差异(306),
无法修复 int8 逐块误差累积 (2 块 cos 0.997 / 25 块 cos 0.95 / 50 块饱和).
本脚本走 step1 生成 quantization.cfg, 用 custom_quantize_layers 把
指定层 (默认全部 int8 层) 改为 float16, 再用全新 RKNN 实例 step2 + 导出.

注意: 2.3.2 的 step2 data_input 必须传 step1 生成的 .data pickle 文件
(传数据集 txt 会报 _pickle.UnpicklingError: invalid load key, '/')。

用法:
    python3 tools/hybrid_fp16.py <输出.rknn> [float16 层键正则 | all] [--reuse]
                                  [--names 层名文件]
    --reuse  复用已存在的 step1 产物 (.model/.data/.cfg), 跳过 step1
    --names  从文件读取精确层名清单 (每行一个), 与 cfg 键取交集
例:
    python3 tools/hybrid_fp16.py /tmp/enc-fp16all.rknn all --reuse
    python3 tools/hybrid_fp16.py /tmp/enc-306.rknn all --reuse --names /tmp/ah306_names.txt
"""
import glob
import os
import re
import sys

from rknn.api import RKNN

SIM = ("third_party/models/"
       "sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.onnx.sim.onnx")
DATASET = "calib_clean/calib_clean.txt"


def main():
    raw = sys.argv[1:]
    reuse = "--reuse" in raw
    names_file = None
    if "--names" in raw:
        i = raw.index("--names")
        names_file = raw[i + 1]
        del raw[i:i + 2]
    args = [a for a in raw if a != "--reuse"]
    out = args[0] if args else "/tmp/enc-fp16all.rknn"
    filt = args[1] if len(args) > 1 else "all"

    # ---- step1: 生成 .model/.data/.quantization.cfg ----
    cfg = sorted(glob.glob("*.quantization.cfg"))[-1]
    if reuse:
        assert os.path.exists(cfg), "--reuse 但未找到已有 cfg"
        print("-- reuse step1 产物:", cfg)
    else:
        r1 = RKNN(verbose=False)
        r1.config(target_platform="rk3588", quantized_method="channel",
                  quantized_algorithm="normal", optimization_level=3)
        ret = r1.load_onnx(model=SIM, inputs=["speech"],
                           input_size_list=[[1, 18, 560]])
        assert ret == 0, "load_onnx failed"
        ret = r1.hybrid_quantization_step1(dataset=DATASET)
        assert ret == 0, "step1 failed"
        r1.release()
        cfg = sorted(glob.glob("*.quantization.cfg"))[-1]
    print("-- cfg:", cfg)
    data = cfg.replace(".quantization.cfg", ".data")  # step2 用 pickle 数据

    # ---- 编辑 cfg: 匹配过滤器的 int8 层加入 custom_quantize_layers ----
    text = open(cfg).read()
    lines = text.splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines)
                 if l.startswith("quantize_parameters:"))
    body0 = "".join(lines[start:])  # 去掉既有的 custom 块 (幂等)
    keys, dt = [], {}
    cur = None
    for line in body0.splitlines():
        m = re.match(r"^    (\S+):\s*$", line)
        if m:
            cur = m.group(1)
            keys.append(cur)
            continue
        m = re.match(r"^        dtype: (\S+)", line)
        if m and cur:
            dt[cur] = m.group(1)
    n_int8 = len([k for k in keys if dt.get(k) == "int8"])
    if names_file:
        want = [l.strip() for l in open(names_file) if l.strip()]
        missing = [n for n in want if dt.get(n) != "int8"]
        if missing:
            print("-- 警告: %d 个层名不在 cfg int8 键中, 跳过: %s"
                  % (len(missing), missing[:3]))
        sel = [n for n in want if dt.get(n) == "int8"]
        print("-- float16 层 (names 文件): %d / %d" % (len(sel), len(want)))
    else:
        pat = None if filt == "all" else re.compile(filt)
        # custom_quantize_layers 只接受激活张量名(斜杠式); 点式 torch 权重名
        # (encoder.model....weight/bias) 会被 apply_hybrid_cfg 拒绝;
        # CIF 分支张量置 fp16 会让 QuantOptimizer._p_adjust_tanh_sigmoid
        # 抛 KeyError (CIF 很小, 且部署模型 alphas 走 CPU 外置 cif.onnx)
        sel = [k for k in keys if dt.get(k) == "int8" and "/" in k
               and "/cif" not in k
               and (pat is None or pat.search(k))]
        print("-- float16 层: %d / %d int8 (共 %d 键)" % (len(sel), n_int8,
                                                          len(keys)))
    assert sel, "无层匹配过滤器: %s" % filt
    body = "custom_quantize_layers:\n" + "".join(
        "    %s: float16\n" % k for k in sel)
    open(cfg, "w").write(body + body0)

    # ---- step2: 全新 RKNN 实例 (绝不 load_onnx) ----
    # 官方配方: model_input=step1 的 .model 临时模型, data_input=.data
    model_tmp = cfg.replace(".quantization.cfg", ".model")
    assert os.path.exists(model_tmp), "缺少 step1 产物: %s" % model_tmp
    r2 = RKNN(verbose=False)
    r2.config(target_platform="rk3588", quantized_method="channel",
              quantized_algorithm="normal", optimization_level=3)
    ret = r2.hybrid_quantization_step2(
        model_input=model_tmp, data_input=data,
        model_quantization_cfg=cfg)
    print("-- step2 ret:", ret)
    assert ret == 0, "step2 failed"
    ret = r2.export_rknn(out)
    assert ret == 0, "export failed"
    print("-- done:", out)
    r2.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
