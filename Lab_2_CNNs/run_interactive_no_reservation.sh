#!/bin/bash
module add libs/tensorflow/1.2
srun -p gpu --gres=gpu:1 -A comsm0018 -t 0-00:15 --mem=4G  sh files_to_run.sh