#!/usr/bin/env python3
"""把 SCRFD 的 onnx 模型的 batch 维改成动态（支持 batch size > 1）。

固定 batch=1 的模型，逐帧推理（per-frame）开销大、吃单核 CPU。
改成动态 batch 后，配合批量推理代码可一次送 N 帧，大幅减少 Python/调用开销。

用法（在装了 onnx + onnxruntime 的环境，比如 deface 那个 conda 环境里跑）：
    python make_dynamic_batch.py 输入.onnx [输出.onnx]
例：
    python make_dynamic_batch.py scrfd_1g.onnx scrfd_1g_dyn.onnx

脚本会：
  1) 把输入/输出的第 0 维(batch)改成动态('batch')，H/W 等其它维保持不变；
  2) 另存为新模型；
  3) 用 batch=2 实际跑一遍验证——若成功并且每个输出第 0 维都是 2，说明可用；
     若失败，说明模型内部有写死 batch=1 的节点(Reshape 等)，需要进一步改图。
"""
import sys
import os


def shape_of(value_info):
    dims = []
    for d in value_info.type.tensor_type.shape.dim:
        if d.HasField("dim_value") and d.dim_value > 0:
            dims.append(d.dim_value)
        else:
            dims.append(d.dim_param or "?")
    return dims


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(inp)[0] + "_dyn.onnx"

    if not os.path.exists(inp):
        print(f"找不到输入模型: {inp}")
        return 1

    try:
        import onnx
        from onnx.tools import update_model_dims
    except ImportError:
        print("缺少 onnx 库，请先: pip install onnx")
        return 1

    model = onnx.load(inp)
    g = model.graph

    print("== 原始 input/output 形状 ==")
    in_dims = {}
    for i in g.input:
        s = shape_of(i)
        print(f"  input  {i.name}: {s}")
        s = list(s)
        s[0] = "batch"          # 只把 batch 维改成动态
        in_dims[i.name] = s
    out_dims = {}
    for o in g.output:
        s = shape_of(o)
        print(f"  output {o.name}: {s}")
        s = list(s)
        s[0] = "batch"
        out_dims[o.name] = s

    model = update_model_dims.update_inputs_outputs_dims(model, in_dims, out_dims)
    onnx.save(model, out)
    print(f"\n已保存动态 batch 模型: {out}")

    # ---- 用 batch=2 实测验证 ----
    print("\n== 用 batch=2 验证是否真的能跑 ==")
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("缺少 onnxruntime/numpy，无法验证。请装好后再跑一次，或直接在 deface 里试。")
        return 0

    try:
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        inp_meta = sess.get_inputs()[0]
        ishape = inp_meta.shape  # 形如 ['batch', 3, 640, 640]
        concrete = []
        for d in ishape:
            if isinstance(d, str) or d is None or (isinstance(d, int) and d <= 0):
                concrete.append(2)   # 动态维（batch）填 2
            else:
                concrete.append(d)
        x = np.zeros(concrete, dtype=np.float32)
        outs = sess.run(None, {inp_meta.name: x})
        print(f"✓ batch=2 推理成功! 输入 {concrete}")
        print("  输出形状（第 0 维应为 2）:")
        all_ok = True
        for o, arr in zip(sess.get_outputs(), outs):
            ok = arr.shape[0] == 2
            all_ok = all_ok and ok
            print(f"    {o.name}: {arr.shape}  {'OK' if ok else '<-- 第0维不是2!'}")
        if all_ok:
            print("\n>>> 转换成功，可以用作动态 batch 模型。把这个输出贴给我，我来改批量推理代码。")
        else:
            print("\n>>> 有输出的 batch 维不对（可能 Reshape 把多张图合并了），把上面形状贴我。")
    except Exception as e:
        print(f"✗ batch=2 验证失败: {repr(e)}")
        print(">>> 模型内部有写死 batch=1 的节点，需要进一步改图。把这段报错完整发我。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
