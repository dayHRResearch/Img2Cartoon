#!/usr/bin/env bash
cd model/download.sh

wget -c http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Hayao_net_G_float.pth
wget -c http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Hosoda_net_G_float.pth
wget -c http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Paprika_net_G_float.pth	
wget -c http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Shinkai_net_G_float.pth

cd ..

mv model/Hayao_net_G_float.pth model/hayao.pth
mv model/Hosoda_net_G_float.pth model/hosoda.pth
mv model/Paprika_net_G_float.pth model/paprika.pth
mv model/Shinkai_net_G_float.pth model/shinkai.pth