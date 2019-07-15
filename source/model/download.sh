#!/usr/bin/env bash
wget -c http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Hayao_net_G_float.pth
wget -c http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Hosoda_net_G_float.pth
wget -c http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Paprika_net_G_float.pth	
wget -c http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Shinkai_net_G_float.pth

mv Hayao_net_G_float.pth model/hayao.pth
mv Hosoda_net_G_float.pth model/hosoda.pth
mv Paprika_net_G_float.pth model/paprika.pth
mv Shinkai_net_G_float.pth model/shinkai.pth