# Img2Cartoon (CartoonGAN)

Pytorch and Torch testing code of [CartoonGAN](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2205.pdf) `[Chen et al., CVPR18]`. With the released pre train model file in  **[URL](http://cg.cs.tsinghua.edu.cn/people/~Yongjin/Yongjin.htm)** by Yongjin, thanks.

## Getting started

### Operational requirements

- Linux or Windows
- NVIDIA GPU
- [Python](https://www.python.org/downloads/release/python-374/) >= 3.7
- [PyTorch](https://pytorch.org/) >= 1.0
- [Torchvision](https://pytorch.org/) >= 0.3.0
- [Numpy](https://www.numpy.org/) >= 1.14.0
- [Pillow](https://python-pillow.org/) >= 0.5.0
- CUDA >= 10.0

### Quick installer

```text
git clone https://github.com/dayHRResearch/Img2Cartoon
cd Img2Cartoon
```

#### Linux

##### CPU

```text
pip3 install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
pip3 install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp37-cp37m-linux_x86_64.whl
pip3 install -r requirements.txt
```

##### GPU

```text
pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp37-cp37m-linux_x86_64.whl
pip3 install -r requirements.txt
```

#### Windows

##### CPU

```text
pip3 install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-win_amd64.whl
pip3 install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp37-cp37m-win_amd64.whl
pip3 install -r requirements.txt
```

##### GPU

```text
pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-win_amd64.whl
pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp37-cp37m-win_amd64.whl
pip3 install -r requirements.txt
```

## Download pre trained model file.

You can load the trained model `pth` file directly.

- Download the converted models:

```text
sh model/download_pth.sh
```

- For testing:

```python
python test.py --input_dir YourImgDir --style Hosoda --gpu 0
```

## Torch

Working with the original models in Torch is also fine. I just convert the weights (bias) in their models from CudaTensor to FloatTensor so that `cudnn` is not required for loading models.

- Download the converted models:

```
sh pretrained_model/download_t7.sh
```

- For testing:

```
th test.lua -input_dir YourImgDir -style Hosoda -gpu 0
```

## Examples (Left: input, Right: output)

<p>
    <img src='test_img/in2.png' width=300 />
    <img src='test_output/in2_Hayao.png' width=300 />
</p>

<p>
    <img src='test_img/in3.png' width=300 />
    <img src='test_output/in3_Hayao.png' width=300 />
</p>

<p>
    <img src='test_img/5--26.jpg' width=300 />
    <img src='test_output/5--26_Hosoda.jpg' width=300 />
</p>

<p>
    <img src='test_img/7--136.jpg' width=300 />
    <img src='test_output/7--136_Hayao.jpg' width=300 />
</p>

<p>
    <img src='test_img/15--324.jpg' width=300 />
    <img src='test_output/15--324_Hosoda.jpg' width=300 />
</p>


## Note

- The training code should be similar to the popular GAN-based image-translation frameworks and thus is not included here.

## Acknowledgement

- Many thanks to the authors for this cool work.

- Part of the codes are borrowed from [DCGAN](https://github.com/soumith/dcgan.torch), [TextureNet](https://github.com/DmitryUlyanov/texture_nets), [AdaIN](https://github.com/xunhuang1995/AdaIN-style) and [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

