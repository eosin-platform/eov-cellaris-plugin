#!/usr/bin/env python3
"""Writes tests/data/markers-synthetic.onnx, a stand-in for cellaris's
markers.onnx with the same input, outputs and metadata: markers are a 1x1
convolution of the input, recon_error is the mean absolute input. It
predicts nothing; it only lets the ONNX Runtime path be tested.

    python3 tools/make_test_model.py tests/data/markers-synthetic.onnx

Needs `onnx` and `numpy`.
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

names = ["Hoechst", "AF1", "CD31", "CD45", "CD68", "Argo550", "CD4", "FOXP3", "CD8a",
         "CD45RO", "CD20", "PD-L1", "CD3e", "CD163", "E-cadherin", "PD-1", "Ki67",
         "Pan-CK", "SMA"]
c = len(names)
rng = np.random.default_rng(7)
w = rng.normal(0, 20, size=(c, 3, 1, 1)).astype(np.float32)
b = np.linspace(50, 400, c).astype(np.float32)

x = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, "height", "width"])
markers = helper.make_tensor_value_info("markers", TensorProto.FLOAT, ["batch", c, "height", "width"])
recon = helper.make_tensor_value_info("recon_error", TensorProto.FLOAT, ["batch", 1, "height", "width"])
nodes = [
    helper.make_node("Conv", ["input", "W", "B"], ["markers"]),
    helper.make_node("Abs", ["input"], ["abs"]),
    helper.make_node("ReduceMean", ["abs", "axes"], ["recon_error"], keepdims=1),
]
graph = helper.make_graph(nodes, "synthetic-markers", [x], [markers, recon],
                          initializer=[numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B"),
                                       numpy_helper.from_array(np.array([1], dtype=np.int64), "axes")])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], producer_name="cellaris-smoke")
model.ir_version = 9
for key, value in {"channel_names": ",".join(names), "raw_intensity": "true", "mpp": "0.325",
                   "name": "synthetic cellaris markers", "version": "smoke-1"}.items():
    entry = model.metadata_props.add()
    entry.key, entry.value = key, value
onnx.checker.check_model(model)
onnx.save(model, sys.argv[1])
print("wrote", sys.argv[1])
