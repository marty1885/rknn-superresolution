from rknnlite.api import RKNNLite
import numpy as np
from PIL import Image
import math

model_path = "super-resolution-10.rknn"
image_path = "test.jpg"

input_size = 224
output_size = 672

rknn = RKNNLite()
rknn.load_rknn(model_path)
rknn.init_runtime()

# load the image as rgb
img = Image.open(image_path)
img = img.convert('YCbCr')
img_y, img_cb, img_cr = img.split()
img_y = np.array(img_y, dtype=np.float32)/255.0

width = img.width
height = img.height
scale_factor = output_size / input_size
scaled_width = int(width * scale_factor)
scaled_height = int(height * scale_factor)

print(f"width: {width}, height: {height}")
print(f"scaled_width: {scaled_width}, scaled_height: {scaled_height}")

out_y = np.zeros((scaled_height, scaled_width), dtype=np.float32)
for y in range(0, scaled_height-input_size, input_size):
    yp = np.clip(y, 0, height - input_size)
    sy = np.clip(int(y*scale_factor), 0, scaled_height - output_size)
    for x in range(0, scaled_width-input_size, input_size):
        xp = np.clip(x, 0, width - input_size)
        sx = np.clip(int(x*scale_factor), 0, scaled_width - output_size)
        img_crop = img_y[yp:yp+input_size, xp:xp+input_size][np.newaxis, np.newaxis, :, :]
        out_y[sy:sy+output_size, sx:sx+output_size] = rknn.inference(inputs=[img_crop])[0][0]

final_image = Image.merge("YCbCr", [
    Image.fromarray(np.uint8(np.clip(out_y*255.0, 0, 255.0))),
    img_cb.resize((scaled_width, scaled_height), Image.BICUBIC),
    img_cr.resize((scaled_width, scaled_height), Image.BICUBIC)
]).convert("RGB")
final_image.save("out.jpg")

