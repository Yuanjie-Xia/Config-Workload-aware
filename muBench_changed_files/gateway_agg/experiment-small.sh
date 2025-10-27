#! /bin/bash
# Set the working directory to the project root
cd "$(dirname "$0")/../.." || exit

# Add the current directory to PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

export PROM_USER=admin
export PROM_PASS=prom-operator

export K8S_DNS_RESOLVER=192.168.38.233 

export SERVER_API_ENDPOINT=http://129.97.165.151:31113
export PROMETHEUS_API_ENDPOINT=http://129.97.165.151:30000

export USE_MINIKUBE=false

echo "Shell working directory:"
pwd

echo "Python working directory:"
python3 -c "import os; print(os.getcwd())"

EXP_NAME=yjx
BASE_PATH=./gssi_experiment/gateway_aggregator/results/$EXP_NAME

LOGS_PATH=$BASE_PATH/logs.out
mkdir -p $BASE_PATH
echo > $BASE_PATH/logs.out


DELAY=60
BIG_NODE=node-3
SMALL_NODE_1=node-1
SMALL_NODE_2=node-2

STEPS=5

RERUNS=3
VARIABLE=1

counter=1

for RUN in {1..1}
do
    echo Starting run $RUN
    python3 ./gssi_experiment/gateway_aggregator/experiment_runner_wrapper.py \
        --wait-for-pods $DELAY \
        --node-selector fse \
        --steps $STEPS \
        --seed $counter \
        --cpu-limit 1000m \
        --replicas 1 \
        --name $EXP_NAME/experiment_$counter
    counter=$((counter + 1))
done

cd ./gssi_experiment/gateway_aggregator/gateway_aggregator_service/
make delete
cd ../../..
