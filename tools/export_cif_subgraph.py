#!/usr/bin/env python3
"""从 encoder.onnx 提取 CIF 分支子图 (alphas 生产者链), 导出独立 float32 ONNX.

背景: RKNN int8 模型内量化后的 CIF 分支保真度差 (板上实测 alphas 比参考
系统性偏高 1.3~5.3 倍, 导致 NPU 识别乱码), 而 RKNN int8 的 enc 输出质量可
接受。因此把 CIF 移出 NPU 模型, 用本脚本提取的 cif.onnx 在 CPU(ORT,
float32) 上计算 alphas。输入 [enc(1,18,512), speech_lengths(1)], 输出
[alphas(1,18)], 与编码器内置 alphas 在主机上验证逐位一致
(见 tools/validate_cif.py, max|C-A|=0.0)。

用法:
    python3 tools/export_cif_subgraph.py [encoder.onnx] [输出 cif.onnx]
"""
import sys

import onnx
from onnx import helper

DEFAULT_ENC = ("third_party/models/"
               "sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.onnx")
DEFAULT_OUT = "entry/src/main/resources/rawfile/models/cif.onnx"


def extract(model):
    # 停止点 = 原图输入 ∪ 原图输出(enc/enc_len): 防止 BFS 越过 enc 钻入编码器主干
    stop_tensors = ({i.name for i in model.graph.input} |
                    {o.name for o in model.graph.output})
    stop_tensors.discard("alphas")
    pred = {}
    for n in model.graph.node:
        for o in n.output:
            pred[o] = n
    init_map = {t.name: t for t in model.graph.initializer}
    needed, inits, used = set(), [], set()
    reached = set()
    queue = ["alphas"]
    while queue:
        t = queue.pop()
        if t in stop_tensors:
            reached.add(t)
            continue
        n = pred.get(t)
        if n is None:
            if t in init_map and t not in used:
                inits.append(init_map[t])
                used.add(t)
            continue
        if n.name in needed:
            continue
        needed.add(n.name)
        for i in n.input:
            if i:
                queue.append(i)
    sel = [n for n in model.graph.node if n.name in needed]
    shape_of = {"enc": [1, 18, 512], "speech_lengths": [1],
                "speech": [1, 18, 560]}
    dtype_of = {"speech_lengths": onnx.TensorProto.INT32}
    ins = [helper.make_tensor_value_info(
        nm, dtype_of.get(nm, onnx.TensorProto.FLOAT), shape_of[nm])
        for nm in sorted(reached)]
    g = helper.make_graph(
        sel, "cif_subgraph", ins,
        [helper.make_tensor_value_info("alphas", onnx.TensorProto.FLOAT,
                                       [1, 18])],
        inits)
    sm = helper.make_model(g, opset_imports=[helper.make_opsetid("", 14)])
    sm.ir_version = 9  # 与主机/板卡 ORT 1.16 兼容
    onnx.checker.check_model(sm)
    return sm, sorted(reached)


def main():
    enc_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ENC
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    model = onnx.load(enc_path)
    sm, ins = extract(model)
    onnx.save(sm, out_path)
    import os
    print("CIF 子图: %d 节点, %d 初始化器, 输入=%s -> %s (%.1f KB)" % (
        len(sm.graph.node), len(sm.graph.initializer), ins, out_path,
        os.path.getsize(out_path) / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
