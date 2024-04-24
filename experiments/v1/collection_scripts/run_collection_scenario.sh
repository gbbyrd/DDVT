#!/bin/bash

# (bash c.sh)
# bash b.sh & PID=$! & ( sleep 15; )


# echo "starting subshell and sleeping for 10 seconds"
# bash b.sh & 
# PID=$!
# echo "what $PID" 
# sleep 2
# kill -2 $PID
# # & ( sleep 2s; kill -2 $PID)


# echo "done"

# kill -15 9742
# docker-compose stop

echo "Collecting data on $1"

# start up the docker compose and wait for it to fully start
BASE_DIR=/home/nianyli/Desktop/code/DiffViewTrans
(cd $BASE_DIR; docker-compose up) &
echo "waiting for docker-compose"
sleep 10

# run the data collection python script
python collect_data_rgb_only.py --num_frames 10000 \
    --dataset_path /home/nianyli/Desktop/code/DDVT/experiments/v1/fundamental_diffusion_model_training \
    --town $1 \
    --skip_frames 5

# clean the dataset to ensure synchronization
python collect_data_rgb_only.py --clean_dataset \
    --dataset_path /home/nianyli/Desktop/code/DDVT/experiments/v1/fundamental_diffusion_model_training

# verify the dataset to make sure there are not outstanding synchronization errors
python collect_data_rgb_only.py --verify_dataset \
    --dataset_path /home/nianyli/Desktop/code/DDVT/experiments/v1/fundamental_diffusion_model_training

docker-compose stop

# kill -15 $generate_traffic_pid

echo "done!"