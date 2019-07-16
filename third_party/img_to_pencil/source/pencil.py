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
import os


def img_to_pencil(raw_image, threshold=15):
    """ Convert color map to simple pencil style.

    Args:
        raw_image: Image path to be processed.
        threshold: Thresholds are defined between 0 and 100.

    Returns:
        array.
    """

    if threshold < 0:
        threshold = 0
    if threshold > 100:
        threshold = 100

    width, height = raw_image.size
    raw_image = raw_image.convert('L')  # convert to gray scale mode
    pixel = raw_image.load()  # get pixel matrix

    for w in range(width):
        for h in range(height):
            if w == width - 1 or h == height - 1:
                continue

            src = pixel[w, h]
            dst = pixel[w + 1, h + 1]

            diff = abs(src - dst)

            if diff >= threshold:
                pixel[w, h] = 0
            else:
                pixel[w, h] = 255

    return raw_image


if __name__ == "__main__":
    for image_path in os.listdir('../example/raw_img'):
        file_path = os.path.join('../example/raw_img', image_path)
        image = Image.open(file_path)
        image = img_to_pencil(image)
        image.save('../example/out_img/' + 'pencil' + '_' + os.path.splitext(image_path)[0] + '.png')
