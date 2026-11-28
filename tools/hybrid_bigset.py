#!/usr/bin/env python3
"""step2 大规模混合量化: 自动剔除非法层名循环 (toolkit 2.3.2, rk3588).

集合: cfg 中全部 int8 斜杠式激活键, 排除:
  - CIF 分支 (/cif*) — QuantOptimizer 对 fp16 CIF 报 KeyError,
    且部署模型 alphas 走 CPU 外置 cif.onnx;
  - 已知转换器内部临时名 (cvt4sdpa / 2sdpa)。
step2 若报 "Invalid operands name 'X'" 则自动剔除 X 重试, 直至构建成功.

前提: step1 产物 (.model/.data/.quantization.cfg) 已存在 (hybrid_fp16.py --reuse 同源).
用法: python3 tools/hybrid_bigset.py <输出.rknn>
"""
import glob
import re
import sys

from rknn.api import RKNN

CFG = sorted(glob.glob("*.quantization.cfg"))[-1]
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/enc-fp16big.rknn"
EXCL = ("/cif", "cvt4sdpa", "2sdpa")
MAX_ITER = 60


def parse_cfg():
    lines = open(CFG).read().splitlines()
    start = next(i for i, l in enumerate(lines)
                 if l.startswith("quantize_parameters:"))
    keys, dt, cur = [], {}, None
    for line in lines[start:]:
        m = re.match(r"^    (\S+):\s*$", line)
        if m:
            cur = m.group(1)
            keys.append(cur)
            continue
        m = re.match(r"^        dtype: (\S+)", line)
        if m and cur:
            dt[cur] = m.group(1)
    return keys, dt


def write_cfg(sel):
    lines = open(CFG).read().splitlines()
    start = next(i for i, l in enumerate(lines)
                 if l.startswith("quantize_parameters:"))
    body = "custom_quantize_layers:\n" + "".join(
        "    %s: float16\n" % k for k in sel)
    open(CFG, "w").write(body + "\n".join(lines[start:]) + "\n")


def main():
    model_tmp = CFG.replace(".quantization.cfg", ".model")
    data = CFG.replace(".quantization.cfg", ".data")
    keys, dt = parse_cfg()
    sel = [k for k in keys if dt.get(k) == "int8" and "/" in k
           and not any(e in k for e in EXCL)]
    print("-- 初始集合: %d 层" % len(sel), flush=True)

    for i in range(MAX_ITER):
        write_cfg(sel)
        r = RKNN(verbose=False)
        r.config(target_platform="rk3588", quantized_method="channel",
                 quantized_algorithm="normal", optimization_level=3)
        try:
            ret = r.hybrid_quantization_step2(
                model_input=model_tmp, data_input=data,
                model_quantization_cfg=CFG)
            if ret != 0:
                raise RuntimeError("step2 ret=%d" % ret)
            ret = r.export_rknn(OUT)
            assert ret == 0, "export failed"
            print("-- done: %s (fp16 层 %d, 剔除 %d 个非法名)"
                  % (OUT, len(sel), i), flush=True)
            return 0
        except ValueError as e:
            m = re.search(r"Invalid operands name '([^']+)'", str(e))
            if not m:
                raise
            bad = m.group(1)
            print("-- iter %d: 剔除非法名 %s (剩 %d)"
                  % (i, bad, len(sel) - 1), flush=True)
            sel = [k for k in sel if k != bad]
        except KeyError as e:
            print("-- 优化器 KeyError: %s — 集合规模/组合触发, 需人工调整"
                  % e, flush=True)
            return 2
        finally:
            try:
                r.release()
            except Exception:
                pass
    print("gave up: 非法名超过 %d 个" % MAX_ITER, flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
