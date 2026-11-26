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
    parser.add_argument("--frames", type=int, default=48,
                        help="固定窗口帧数 (默认 48 帧 = 480ms @10ms 帧)")
    parser.add_argument("--feat-dim", type=int, default=560,
                        help="编码器输入特征维度 (默认 560)")
    parser.add_argument("--dataset", default="",
                        help="量化校准样本列表 (可选; 提供后执行 int8 量化)")
    parser.add_argument("--algorithm", default="normal",
                        choices=["normal", "mmse", "kl_divergence"],
                        help="int8 量化算法 (默认 normal)")
    args = parser.parse_args()

    rknn = RKNN(verbose=False)

    print("-- 配置 target_platform=rk3588")
    # 输入特征已在上游归一化, 不再设置 mean/std
    # (RKNN 要求 mean/std 长度与输入帧数一致, 非图像模型不适用)
    rknn.config(
        target_platform="rk3588",
        quantized_method="channel",
        quantized_algorithm=args.algorithm,
        optimization_level=3,
    )

    print("-- 加载 ONNX (固定输入 [1, %d, %d]): %s"
          % (args.frames, args.feat_dim, args.onnx))
    # RKNN 不支持动态 shape, 预处理三步:
    #   1. 固定 speech [1, T, 560] / speech_lengths [1]
    #   2. 将 speech_lengths 固化为 INT32 初始化器(值 T)并从图输入移除,
    #      使 make_pad_mask 的 Range 可被常量折叠
    #   3. onnxsim 常量折叠, 得到仅含 speech 输入的静态图
    import onnx
    import onnxsim
    import numpy as np
    model = onnx.load(args.onnx)
    for inp in model.graph.input:
        dims = inp.type.tensor_type.shape.dim
        if inp.name == "speech":
            dims[0].dim_value = 1
            dims[1].dim_value = args.frames
        elif inp.name == "speech_lengths":
            dims[0].dim_value = 1
    inputs_to_keep = [i for i in model.graph.input
                      if i.name != "speech_lengths"]
    del model.graph.input[:]
    model.graph.input.extend(inputs_to_keep)
    init = onnx.helper.make_tensor("speech_lengths",
                                   onnx.TensorProto.INT32, [1],
                                   [args.frames])
    model.graph.initializer.append(init)

    model_sim, ok = onnxsim.simplify(model)
    if not ok:
        print("onnxsim.simplify 返回失败, 继续使用原模型", file=sys.stderr)
        model_sim = model
    simplified = args.onnx + ".sim.onnx"
    onnx.save(model_sim, simplified)
    print("-- 已生成简化模型:", simplified)

    ret = rknn.load_onnx(
        model=simplified,
        inputs=["speech"],
        input_size_list=[[1, args.frames, args.feat_dim]],
    )
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
