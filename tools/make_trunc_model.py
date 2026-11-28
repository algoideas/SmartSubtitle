#!/usr/bin/env python3
"""从 encoder.onnx.sim.onnx 截取出前 N 个编码器块子图.

目的: 在 RK3588 NPU 上二分定位"enc 特定帧整帧 ±127 饱和"缺陷首次
出现的层范围。截断模型输出 512 维 (encoders.0 残差后), 与 CLI 的
enc_dim=512 假设兼容, 可直接复用现有上板 dump 流程。

用法:
    python3 tools/make_trunc_model.py <输出.onnx> [目标输出张量名]
默认目标: /encoder/encoders/encoders.0/Add_1_output_0 (第 2 个块, 512 维)
"""
import sys

import numpy as np
import onnx
from onnx import helper

SIM = ("third_party/models/"
       "sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.onnx.sim.onnx")
DEFAULT_OUT = "/encoder/encoders/encoders.0/Add_1_output_0"


def main():
    out_path = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT

    m = onnx.load(SIM)
    graph = m.graph
    nodes = graph.node

    # 反向 BFS: 从目标张量收集可达节点
    producers = {}
    for n in nodes:
        for o in n.output:
            producers[o] = n
    init_names = {i.name for i in graph.initializer}
    graph_in_names = {i.name for i in graph.input}

    keep = set()
    frontier = [target]
    while frontier:
        t = frontier.pop()
        if t in keep:
            continue
        keep.add(t)
        n = producers.get(t)
        if n is None:
            continue
        for inp in n.input:
            if inp and inp not in init_names and inp not in keep:
                frontier.append(inp)
        for o in n.output:
            keep.add(o)

    kept_nodes = [n for n in nodes if n.output[0] in keep or
                  any(o in keep for o in n.output)]
    # 保留 initializer: 只留被用到的
    used_inits = set()
    for n in kept_nodes:
        for inp in n.input:
            if inp in init_names:
                used_inits.add(inp)
    inits = [i for i in graph.initializer if i.name in used_inits]

    new_graph = helper.make_graph(
        kept_nodes,
        "trunc_encoder",
        [helper.make_tensor_value_info("speech", onnx.TensorProto.FLOAT,
                                       [1, 18, 560])],
        [helper.make_tensor_value_info(target, onnx.TensorProto.FLOAT,
                                       [1, 18, 512])],
        inits,
    )
    new_m = helper.make_model(new_graph, opset_imports=[helper.make_opsetid("", 14)])
    new_m.ir_version = 9
    onnx.checker.check_model(new_m, full_check=False)
    onnx.save(new_m, out_path)
    print("截断模型: %d 节点, %d initializers -> %s" %
          (len(kept_nodes), len(inits), out_path))

    # ORT 一致性校验: 截断模型输出 == 全模型同张量输出。
    # ORT 只能请求图输出, 故把 target 临时加入全模型副本的输出 (不动原文件);
    # onnxruntime 1.16 不支持 BytesIO, 用临时文件
    import os
    import onnxruntime as ort
    m2 = onnx.load(SIM)
    m2.graph.output.append(
        helper.make_tensor_value_info(target, onnx.TensorProto.FLOAT,
                                      [1, 18, 512]))
    full_tmp = out_path + ".fullcheck.onnx"
    onnx.save(m2, full_tmp)
    sess_full = ort.InferenceSession(full_tmp,
                                     providers=["CPUExecutionProvider"])
    sess_trunc = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 18, 560)).astype(np.float32)
    out_full = sess_full.run([target], {"speech": x})[0]
    out_trunc = sess_trunc.run(None, {"speech": x})[0]
    diff = float(np.abs(out_full - out_trunc).max())
    print("ORT 一致性: max|diff|=%.3e" % diff)
    assert diff < 1e-5, "截断模型与全模型输出不一致!"
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
