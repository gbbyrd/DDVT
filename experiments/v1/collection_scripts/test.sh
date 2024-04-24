BASE_DIR=/home/nianyli/Desktop/code/DiffViewTrans

(cd $BASE_DIR; docker-compose up) &
echo "waiting for docker-compose"
sleep 10

DATA_COLL_UTILS_DIR=/home/nianyli/Desktop/code/DiffViewTrans/carla_data_collection_utils
echo "changing the map"
python $DATA_COLL_UTILS_DIR/change_town.py --town Town01

# generate traffic and wait for all cars to spawn
# (python $DATA_COLL_UTILS_DIR/generate_traffic.py & generate_traffic_pid=$!) &
# sleep 15

python collect_data.py --num_frames 100 \
    --dataset_path /home/nianyli/Desktop/code/DiffViewTrans/experiments/v1/dataset_varied_yaw \
    --skip_frames 1

docker-compose stop