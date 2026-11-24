#!/usr/bin/env python3
"""将流式 Paraformer 编码器转换为 RKNN 模型 (RK3588 NPU).

在 x86 Linux 主机上运行 (rknn-toolkit2 不依赖 NPU 硬件):

    pip3 install rknn-toolkit2==v2.1.0
    python3 convert_paraformer_encoder.py \
        --onnx <导出的编码器 onnx> --output paraformer-encoder.rknn

输出 paraformer-encoder.rknn 放入模型目录 (与 tokens.txt 同级),
应用 NPU 后端启动时自动加载。
"""
import argparse
import sys

try:
    from rknn.api import RKNN
except ImportError:
    print("请先安装 rknn-toolkit2: pip3 install rknn-toolkit2==v2.1.0", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a streaming Paraformer encoder ONNX to RKNN (RK3588)")
    parser.add_argument("--onnx", required=True,
                        help="编码器 ONNX 文件 (float 版, 非 int8)")
    parser.add_argument("--output", default="paraformer-encoder.rknn",
                        help="输出 .rknn 文件名 (默认 paraformer-encoder.rknn)")
    parser.add_argument("--dataset", default="",
                        help="量化校准样本列表 (可选; 提供后执行 int8 量化)")
    args = parser.parse_args()

    rknn = RKNN(verbose=False)

    print("-- 配置 target_platform=rk3588")
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[1, 1, 1]],
        target_platform="rk3588",
        quantized_method="channel",
        optimization_level=3,
    )

    print("-- 加载 ONNX:", args.onnx)
    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        print("load_onnx 失败, ret =", ret, file=sys.stderr)
        return 1

    print("-- 构建模型" + (" (int8 量化)" if args.dataset else " (float)"))
    ret = rknn.build(do_quantization=bool(args.dataset),
                     dataset=args.dataset if args.dataset else None)
    if ret != 0:
        print("build 失败, ret =", ret, file=sys.stderr)
        return 1

    print("-- 导出:", args.output)
    ret = rknn.export_rknn(args.output)
    if ret != 0:
        print("export_rknn 失败, ret =", ret, file=sys.stderr)
        return 1

    print("-- 完成:", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
