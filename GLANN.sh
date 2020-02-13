#!/bin/bash

echo "Script is working - GLO"
source /cs/labs/yedid/daniel023/cifar_glo_clean/glan/bin/activate
module load tensorflow
echo "Script is running"
python3 -W ignore -u /cs/labs/yedid/daniel023/cifar_glo_clean/tester_video.py
echo "Script is DONE - GLO"


