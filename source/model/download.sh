#!/usr/bin/env bash

wget -c http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Hayao_net_G_float.pth
wget -c http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Hosoda_net_G_float.pth
wget -c http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Paprika_net_G_float.pth	
wget -c http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Shinkai_net_G_float.pth

mv Hayao_net_G_float.pth hayao.pth
mv Hosoda_net_G_float.pth hosoda.pth
mv Paprika_net_G_float.pth paprika.pth
mv Shinkai_net_G_float.pth shinkai.pth