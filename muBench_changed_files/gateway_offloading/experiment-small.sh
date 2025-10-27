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
BASE_PATH=./gssi_experiment/gateway_offloading/results/$EXP_NAME

LOGS_PATH=$BASE_PATH/logs.out
mkdir -p $BASE_PATH
echo > $LOGS_PATH

DELAY=60
BIG_NODE=node-3
SMALL_NODE_1=node-1
SMALL_NODE_2=node-2

STEPS=5

counter=1

for RUN in {1..1}
do
    echo Start run $RUN
    python3 ./gssi_experiment/gateway_offloading/experiment_runner_wrapper.py \
        --wait-for-pods $DELAY \
        --node-selector fse \
        --steps $STEPS \
        --seed $counter \
        --replicas 1 \
        --cpu-limit 1000m \
        --gateway-load "[0,10,5]" \
        --name $EXP_NAME/experiment_$counter
    counter=$((counter + 1))
done
