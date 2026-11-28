#!/usr/bin/env python3
"""从编码器 ONNX 截取"尾部"子图: 编码器块 N 之后的部分.

与 make_trunc_model.py 互补: 截断模型取前缀 (speech → 块 N 残差),
尾部模型取后缀 (块 N 残差 → 'enc' 最终 hidden states)。两者配合实现
NPU 浅层 + CPU 深层拼接: RKNN 前段 (int8, 浅层保真度高) 的输出经
fp32 精确交接给 ORT 尾部 (与 CPU 全量路径同源), 绕开 RK3588 NPU 上
深层残差流动态范围超 int8 表征的缺陷 (见 README NPU 章节)。

源图选择决定算子形态:
  - encoder.int8.onnx (默认): int8 (DQL 形态, 无 Shape/Gather 动态形状
    计算), RK3588 板载 ORT 实测可加载; 尾部 int8 误差与 CPU 全量 int8
    路径同源
  - encoder.onnx.sim.onnx: fp32 sim 图 (含 192 组 Shape/Gather 动态形状
    计算, 板载 ORT 1.14 加载即崩溃, 不可用于板端)

节点保留 = 从目标输入前向可达 ∩ 产 'enc' 所需 (后向必需), 避免把消费
源图其他输入 (如 speech_lengths) 的 CIF 分支带进尾部图造成悬空输入。

用法:
    python3 tools/make_tail_model.py <输出.onnx> [目标输入张量名] [源图.onnx]
"""
import sys

import numpy as np
import onnx
from onnx import helper

DEFAULT_IN = "/encoder/encoders/encoders.24/Add_1_output_0"
OUT_TENSOR = "enc"
DEFAULT_SRC = ("third_party/models/"
               "sherpa-onnx-streaming-paraformer-bilingual-zh-en/"
               "encoder.int8.onnx")


def main():
    out_path = sys.argv[1]
    target_in = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_IN
    src = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_SRC

    m = onnx.load(src)
    graph = m.graph
    nodes = graph.node
    init_names = {i.name for i in graph.initializer}
    graph_in_names = {i.name for i in graph.input}

    producers = {}
    for n in nodes:
        for o in n.output:
            producers[o] = n
    consumers = {}
    for n in nodes:
        for i in n.input:
            if i:
                consumers.setdefault(i, []).append(n)

    # 后向闭包: 保留 'enc' 路径上全部节点 (从 OUT_TENSOR 沿输入回溯)。
    # 边界: target_in (前段输出, 不带走前段节点)、initializer (单独收集)、
    # 源图输入 (如 speech_lengths → 声明为尾部额外输入, C++ 侧喂入)。
    # 不引入"前向可达"交集: 只消费常量的节点 (如注意力形状 Unsqueeze) 不在
    # 前向可达集内, 但被路径节点消费, 交集会漏掉它们导致悬空输入
    extra_inputs = set()  # 'enc' 路径依赖的源图输入 (如 speech_lengths)
    kept = set()
    seen = set()
    frontier = [OUT_TENSOR]
    while frontier:
        t = frontier.pop()
        if t == target_in or t in init_names or t in seen:
            continue
        seen.add(t)
        n = producers.get(t)
        if n is None:
            assert t in graph_in_names, "unexpected external input: " + t
            extra_inputs.add(t)
            continue
        kept.add(id(n))
        for i in n.input:
            if i:
                frontier.append(i)

    kept_nodes = [n for n in nodes if id(n) in kept]
    # 尾部图第一输入统一命名 'mid': C++ 侧硬编码输入名喂入 (板端 ORT 1.14
    # 的 GetInputNameAllocated 为坏调用, 实测 SEGV_MAPERR), 同时避免
    # 以 '/' 开头的图输入名
    for n in kept_nodes:
        for idx, v in enumerate(list(n.input)):
            if v == target_in:
                n.input[idx] = "mid"
    used_inits = set()
    for n in kept_nodes:
        for inp in n.input:
            if inp in init_names:
                used_inits.add(inp)
    inits = [i for i in graph.initializer if i.name in used_inits]

    inputs_v = [helper.make_tensor_value_info("mid", onnx.TensorProto.FLOAT,
                                              [1, 18, 512])]
    for i in graph.input:
        if i.name in extra_inputs:
            inputs_v.append(i)  # 保留源图元素类型/形状 (动态维按动态处理)
    new_graph = helper.make_graph(
        kept_nodes,
        "tail_encoder",
        inputs_v,
        [helper.make_tensor_value_info(OUT_TENSOR, onnx.TensorProto.FLOAT,
                                       [1, 18, 512])],
        inits,
    )
    new_m = helper.make_model(new_graph, opset_imports=[helper.make_opsetid("", 14)])
    new_m.ir_version = 9
    onnx.checker.check_model(new_m, full_check=False)
    onnx.save(new_m, out_path)
    print("尾部模型 (%s): %d 节点, %d initializers -> %s" %
          (src.split('/')[-1], len(kept_nodes), len(inits), out_path))

    # ORT 一致性校验: 全图(speech[,speech_lengths]) → (enc_ref, mid);
    # 尾部(mid) → enc_tail; 断言一致 (同输入下 DQL 位精确, 容差留 1e-4)
    import onnxruntime as ort
    m2 = onnx.load(src)
    m2.graph.output.append(
        helper.make_tensor_value_info(target_in, onnx.TensorProto.FLOAT,
                                      [1, 18, 512]))
    full_tmp = out_path + ".fullcheck.onnx"
    onnx.save(m2, full_tmp)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess_full = ort.InferenceSession(full_tmp, so,
                                     providers=["CPUExecutionProvider"])
    sess_tail = ort.InferenceSession(out_path, so,
                                     providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 18, 560)).astype(np.float32)
    full_inputs = {"speech": x}
    if "speech_lengths" in {i.name for i in m2.graph.input}:
        full_inputs["speech_lengths"] = np.array([18], dtype=np.int32)
    enc_ref, mid = sess_full.run([OUT_TENSOR, target_in], full_inputs)
    tail_inputs = {"mid": mid}
    for name in extra_inputs:
        tail_inputs[name] = np.array([18], dtype=np.int32)
    enc_tail = sess_tail.run(None, tail_inputs)[0]
    diff = float(np.abs(enc_ref - enc_tail).max())
    print("ORT 一致性 (full vs tail(front)): max|diff|=%.3e" % diff)
    assert diff < 1e-4, "尾部模型与全模型输出不一致!"
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
