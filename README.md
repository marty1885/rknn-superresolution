# RKNN Superresolution

Demo code to run image superresolution on Rockchip NPU

## How to use

1. Download `super-resolution-10.onnx` from the [ONNX model zoo](https://github.com/onnx/models/tree/bec48b6a70e5e9042c0badbaafefe4454e072d08/validated/vision/super_resolution/sub_pixel_cnn_2016/model)
    * [https://github.com/onnx/models/blob/main/validated/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx](https://github.com/onnx/models/blob/bec48b6a70e5e9042c0badbaafefe4454e072d08/validated/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx)
2. Run `convert.py` on a PC with RKNN installed
    * Modify the `target_platform` variable to sute your board
3. Move the generated `super-resolution-10.rknn` to your dev board
4. Prepare any image, name it `test.jpg` on your board. Run `infer.py`
5. `out.jpg` is your output
