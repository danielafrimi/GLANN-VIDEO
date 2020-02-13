#!/bin/bash

echo "Script is working - IMLE"
source /cs/labs/yedid/daniel023/cifar_glo_clean/glan/bin/activate
module load tensorflow
python3 -W ignore -u /cs/labs/yedid/daniel023/cifar_glo_clean/tester_icp.py
echo "Script is running - IMLE"


