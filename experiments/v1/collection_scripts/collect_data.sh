#!/bin/bash

# store an array of the towns to collect data on
# array[0]="Town01"
# array[1]="Town02"
# array[2]="Town03"
# array[3]="Town04"
# array[4]="Town05"

array[0]="Town02"
array[1]="Town03"
array[2]="Town04"
array[3]="Town05"
array[4]="Town06"
array[5]="Town07"
array[6]="Town10HD"

# size=${#array[@]}
# index=$(($RANDOM % $size))
# bash run_collection_scenario.sh ${array[$index]}

for i in {1..200};
do
    # bash run_collection_scenario.sh
    # bash run_collection_scenario_1.sh
    echo "data collection iteration :$i"
    size=${#array[@]}
    index=$(($RANDOM % $size))
    bash run_collection_scenario.sh ${array[$index]}
done