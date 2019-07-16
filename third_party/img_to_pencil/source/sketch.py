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

import cv2
import os


def dodgeV2(image, mask):
    return cv2.divide(image, 255 - mask, scale=256)


def rgb_to_sketch(src_image_name, dst_image_name):
    img_rgb = cv2.imread(src_image_name)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    img_gray_inv = 255 - img_gray
    img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21),
                                sigmaX=0, sigmaY=0)
    img_blend = dodgeV2(img_gray, img_blur)

    cv2.imwrite(dst_image_name, img_blend)


if __name__ == '__main__':
    for image_path in os.listdir('../example/raw_img'):
        src_image_name = os.path.join('../example/raw_img', image_path)
        dst_image_name = '../example/out_img/' + 'sketch' + '_' + os.path.splitext(image_path)[0] + '.png'
        rgb_to_sketch(src_image_name, dst_image_name)
