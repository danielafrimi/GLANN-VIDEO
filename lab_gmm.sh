#!/bin/bash

echo "Script is working - GMM"
source /cs/labs/yedid/daniel023/cifar_glo_clean/glan/bin/activate
module load tensorflow
python3 -W ignore -u /cs/labs/yedid/daniel023/cifar_glo_clean/train_gmm.py
echo "Script is running - GMM"


