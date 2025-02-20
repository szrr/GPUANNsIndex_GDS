#!/usr/bin/env bash
set -euo pipefail

for i in {1}
do
    sudo ../../build/bin/nvm-iodepth-block-bench --threads=$((1)) --blk_size=1 --reqs=1 --pages=$((2)) --queue_depth=128  --page_size=$((512)) --num_blks=$((1024)) --gpu=0 --n_ctrls=1 --num_queues=8

    echo "******************** $i *********************"
done
