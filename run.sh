#!/bin/bash
source /home/TUE/20201168/miniconda3/etc/profile.d/conda.sh
conda activate dropedge
./run_local_community_global.sh
conda deactivate
