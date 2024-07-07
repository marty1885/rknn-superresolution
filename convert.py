from rknn.api import RKNN

rknn = RKNN()
rknn.config(target_platform='rk3588')
rknn.load_onnx("super-resolution-10.onnx", inputs=['input'], input_size_list=[[1, 1, 224, 224]])
rknn.build(do_quantization=False)
rknn.export_rknn("super-resolution-10.rknn")
