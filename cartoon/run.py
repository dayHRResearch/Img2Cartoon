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

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils

from PIL import Image
from torch.autograd import Variable

from cartoon.network.Transformer import Transformer

parser = argparse.ArgumentParser()
parser.add_argument('--input_img', required=True, type=str,
                    help='Image path to request processing.')
parser.add_argument('--load_size', default=500)
parser.add_argument('--model_path', default='./pretrained_model')
parser.add_argument('--style', default='Hayao')
parser.add_argument('--output_dir', default='test_output')
parser.add_argument('--gpu', type=int, default=0)

opt = parser.parse_args()

valid_ext = ['.jpg', '.png']

if not os.path.exists(opt.output_dir):
    os.mkdir(opt.output_dir)

# load pretrained network
model = Transformer()
model.load_state_dict(
    torch.load(
        os.path.join(
            opt.model_path,
            opt.style +
            '_net_G_float.pth')))
model.eval()

if opt.gpu > -1:
    print('GPU mode')
    model.cuda()
else:
    print('CPU mode')
    model.float()

for files in os.listdir(opt.input_dir):
    ext = os.path.splitext(files)[1]
    if ext not in valid_ext:
        continue
    # load image
    input_image = Image.open(os.path.join(opt.input_dir, files)).convert("RGB")
    # resize image, keep aspect ratio
    h = input_image.size[0]
    w = input_image.size[1]
    ratio = h * 1.0 / w
    if ratio > 1:
        h = opt.load_size
        w = int(h * 1.0 / ratio)
    else:
        w = opt.load_size
        h = int(w * ratio)
    input_image = input_image.resize((h, w), Image.BICUBIC)
    input_image = np.asarray(input_image)
    # RGB -> BGR
    input_image = input_image[:, :, [2, 1, 0]]
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    # preprocess, (-1, 1)
    input_image = -1 + 2 * input_image
    if opt.gpu > -1:
        input_image = Variable(input_image, volatile=True).cuda()
    else:
        input_image = Variable(input_image, volatile=True).float()
    # forward
    output_image = model(input_image)
    output_image = output_image[0]
    # BGR -> RGB
    output_image = output_image[[2, 1, 0], :, :]

    # deprocess, (0, 1)
    output_image = output_image.data.cpu().float() * 0.5 + 0.5
    # save
    vutils.save_image(output_image, os.path.join(
        opt.output_dir, files[:-4] + '_' + opt.style + '.jpg'))

print('Done!')
