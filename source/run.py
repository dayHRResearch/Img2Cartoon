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


import argparse
import os
import sys
sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], '../'))

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils

from torch.autograd import Variable

from source.network.Cartoon import Cartoon

parser = argparse.ArgumentParser('Image to Cartoon Img.')
# parser.add_argument('--input_dir', required=True, type=str,
#                    help='Image path to request processing.')
parser.add_argument('--img_size', required=False, type=int, default=512,
                    help='Input image size.'
                         'default: 512.')
parser.add_argument('--model', required=False, type=str, default='./model',
                    help='Model file address.'
                         'default: `./model`.')
parser.add_argument('--style', required=False, type=str, default='hayao',
                    help='Styles to be changed for pictures.'
                         'default: hayao.'
                         'option: [`hayao`, `hosoda`, `paprika`, `shinkai`].')
parser.add_argument('--output_dir', required=True, type=str,
                    help='Output image path after style conversion.')
parser.add_argument('--mode', required=False, type=str, default='gpu',
                    help='Which model of GPU to use, or use cpu.'
                         'default: `gpu`'
                         'option: [`gpu`, `cpu`].')

args = parser.parse_args()

img_suffix = ['.jpg', '.jpeg', '.png']

MODELFILE = os.path.join(args.model, args.style + '.pth')

# Create if the output save directory does not exist.
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)


def preprocess(file_path):
    """ Preprocess the image.

    Args:
        file_path: Directory of files to be processed.

    Returns:
        numpy.array()

    """
    raw_image = cv2.imread(file_path)

    # resize image, keep aspect ratio
    image_height = raw_image.shape[0]
    image_width = raw_image.shape[1]
    ratio = image_height * 1.0 / image_width

    if ratio > 1:
        image_height = args.img_size
        image_width = int(image_height * 1.0 / ratio)
    else:
        image_width = args.img_size
        image_height = int(image_width * ratio)

    cv2.resize(raw_image, (image_height, image_width), cv2.INTER_CUBIC)
    return raw_image


def main(inputs_dir):
    """ File main function access point.
    """
    # build model
    model = Cartoon()

    # load model weights
    model.load_state_dict(torch.load(MODELFILE))

    # set model mode is eval
    model.eval()

    # check mode status
    if args.mode == 'gpu':
        if torch.cuda.is_available():
            print('Use GPU mode!')
            model.cuda()
        else:
            raise (
                'Please check if your system is properly installed with CUDA'
                'and if PyTorch`s GPU version is installed.')
    else:
        print('Use CPU mode!')
        model.float()

    # Get all the files in the specified directory
    for img_path in os.listdir(inputs_dir):
        # Intercept file suffix
        suffix = os.path.splitext(img_path)[1]
        if suffix not in img_suffix:
            continue
        # load image
        file_path = os.path.join(inputs_dir, img_path)

        raw_image = preprocess(file_path)

        raw_image = np.asarray(raw_image)

        # RGB -> BGR
        raw_image = raw_image[:, :, [2, 1, 0]]
        raw_image = transforms.ToTensor()(raw_image).unsqueeze(0)

        # preprocess, (-1, 1)
        raw_image = -1 + 2 * raw_image

        with torch.no_grad():
            if args.mode == 'gpu':
                raw_image = Variable(raw_image).cuda()
            else:
                raw_image = Variable(raw_image).float()

        # forward
        cartoon_image = model(raw_image)
        cartoon_image = cartoon_image[0]
        # BGR -> RGB
        cartoon_image = cartoon_image[[2, 1, 0], :, :]

        # deprocess, (0, 1)
        cartoon_image = cartoon_image.data.cpu().float() * 0.5 + 0.5

        img_path = os.path.join(args.output_dir,
                                img_path[:-4] + '_' + args.style + '.png')
        vutils.save_image(cartoon_image, img_path)


if __name__ == '__main__':
    while True:
        input_dir = input("Process dir path (input `0` exit.):")
        if input_dir == '0':
            print("Seccessful exit!\nreturn code -1.")
            exit(0)
        elif input_dir != '0':
            main(input_dir)
            print("Img transfer source successful!\nreturn code 0")
        else:
            continue
