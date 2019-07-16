# Copyright 2019 DayHR Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from PIL import Image
import numpy as np

L = np.asarray(Image.open(
    '../example/raw_img/demo1.png').convert('L')).astype('float')  # 取得图像灰度

depth = 10.                                     # (0-100)
grad = np.gradient(L)                           # 取图像灰度的梯度值
grad_x, grad_y = grad                           # 分别取横纵图像梯度值
grad_x = grad_x * depth / 100.
grad_y = grad_y * depth / 100.
A = np.sqrt(grad_x**2 + grad_y**2 + 1.)
uni_x = grad_x / A
uni_y = grad_y / A
uni_z = 1. / A

el = np.pi / 2.2                              # 光源的俯视角度，弧度值
az = np.pi / 4                               # 光源的方位角度，弧度值
dx = np.cos(el) * np.cos(az)              # 光源对x轴的影响
dy = np.cos(el) * np.sin(az)              # 光源对y轴的影响
dz = np.sin(el)                             # 光源对z轴的影响

gd = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)        # 光源归一化
gd = gd.clip(0, 255)  # 避免数据越界，将生成的灰度值裁剪至0-255之间

im = Image.fromarray(gd.astype('uint8'))         # 重构图像
im.save('../example/out_img/demo1.png')         # 保存图像
