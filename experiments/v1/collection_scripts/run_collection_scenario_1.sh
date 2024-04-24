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

# start up the docker compose and wait for it to fully start
BASE_DIR=/home/nianyli/Desktop/code/DiffViewTrans
(cd $BASE_DIR; docker-compose up) &
echo "waiting for docker-compose"
sleep 10

# run the data collection script and generate traffic
DATA_COLL_UTILS_DIR=/home/nianyli/Desktop/code/DiffViewTrans/carla_data_collection_utils
echo "changing the map"
python $DATA_COLL_UTILS_DIR/change_town.py --town Town01

# generate traffic and wait for all cars to spawn
python collect_data.py --num_frames 10000 \
    --dataset_path /home/nianyli/Desktop/code/DiffViewTrans/experiments/v1/dataset \
    --skip_frames 1 & (sleep 10; python $DATA_COLL_UTILS_DIR/generate_traffic.py & generate_traffic_pid=$!)

docker-compose stop

echo "waiting for generate traffic script to abort"
sleep 10

kill -15 $generate_traffic_pid

echo "done!"