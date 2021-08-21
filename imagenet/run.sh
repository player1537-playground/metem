#!/bin/bash
set -eu

usage()
{
  echo "usage: IMAGE_ZIP IMAGE_DIR"
  echo "IMAGE_ZIP: the original tiny-imagenet zip file"
  echo "IMAGE_DIR: a working directory for this run,"
  echo "           which will be removed and re-created"
  echo "LOG_PATTERN: the Python %-formatting pattern to write logs to"
}

if (( ${#} != 5 ))
then
  usage
  exit 1
fi

IMAGE_ZIP=$1
IMAGE_DIR=$2
LOG_PATTERN=$3
ITERATION=$4
NGRADIENTS=$5

# IMAGE_ZIP=$HOME/work/ai/tiny-imagenet-200.zip

# IMAGE_DIR=/dev/shm/resnet-input

# rm -rf $IMAGE_DIR; mkdir -p $IMAGE_DIR
echo Unzipping images...
# time unzip $IMAGE_ZIP -d $IMAGE_DIR &> /dev/null
echo Running ResNet50...
# --exclude-range string is of the form 1.1.2.1, which means the dataset into 8 pieces, keep the 7th and delete everyting else
which python
if [ $ITERATION -eq 0 ]; then
    $HOME/src/triple-r/go.sh python-with-env keras_resnet50.py \
            --tar $IMAGE_ZIP  \
            --data-dir $IMAGE_DIR \
            --epochs 1 \
            --log-to $LOG_PATTERN \
            --reload /lus/theta-fs0/projects/VeloC/metem/logs/ai-apps/checkpoint-4e.h5 \
            --checkpoint /lus/theta-fs0/projects/VeloC/metem/logs/ai-apps-case1-reboot-baseline-4e-${NGRADIENTS}g/partial.h5 \
            --initial-epoch 4 \
            --ngradients $((53*NGRADIENTS))
elif [ $ITERATION -eq 1 ]; then
    $HOME/src/triple-r/go.sh python-with-env keras_resnet50.py \
            --tar $IMAGE_ZIP  \
            --data-dir $IMAGE_DIR \
            --epochs 1 \
            --log-to $LOG_PATTERN \
            --reload /lus/theta-fs0/projects/VeloC/metem/logs/ai-apps-case1-reboot-baseline-4e-${NGRADIENTS}g/partial.h5 \
            --initial-epoch 4 \
    && touch /lus/theta-fs0/projects/VeloC/metem/logs/ai-apps-case1-reboot-baseline-4e-${NGRADIENTS}g/flag.txt
elif [ $ITERATION -eq 2 ]; then
    $HOME/src/triple-r/go.sh python-with-env keras_resnet50.py \
            --tar $IMAGE_ZIP  \
            --data-dir $IMAGE_DIR \
            --epochs 1 \
            --log-to $LOG_PATTERN \
            --reload /lus/theta-fs0/projects/VeloC/metem/logs/ai-apps/checkpoint-10e.h5 \
            --checkpoint /lus/theta-fs0/projects/VeloC/metem/logs/ai-apps-case1-reboot-baseline-10e-${NGRADIENTS}g/partial.h5 \
            --initial-epoch 10 \
            --ngradients $((53*NGRADIENTS))
elif [ $ITERATION -eq 3 ]; then
    $HOME/src/triple-r/go.sh python-with-env keras_resnet50.py \
            --tar $IMAGE_ZIP  \
            --data-dir $IMAGE_DIR \
            --epochs 1 \
            --log-to $LOG_PATTERN \
            --reload /lus/theta-fs0/projects/VeloC/metem/logs/ai-apps-case1-reboot-baseline-10e-${NGRADIENTS}g/partial.h5 \
            --initial-epoch 10 \
    && touch /lus/theta-fs0/projects/VeloC/metem/logs/ai-apps-case1-reboot-baseline-10e-${NGRADIENTS}g/flag.txt
elif [ $ITERATION -eq 4 ]; then
    $HOME/src/triple-r/go.sh python-with-env keras_resnet50.py \
            --tar $IMAGE_ZIP  \
            --data-dir $IMAGE_DIR \
            --epochs 1 \
            --log-to $LOG_PATTERN \
            --reload /lus/theta-fs0/projects/VeloC/metem/logs/ai-apps/checkpoint-73e.h5 \
            --checkpoint /lus/theta-fs0/projects/VeloC/metem/logs/ai-apps-case1-reboot-baseline-73e-${NGRADIENTS}g/partial.h5 \
            --initial-epoch 73 \
            --ngradients $((53*NGRADIENTS))
elif [ $ITERATION -eq 5 ]; then
    $HOME/src/triple-r/go.sh python-with-env keras_resnet50.py \
            --tar $IMAGE_ZIP  \
            --data-dir $IMAGE_DIR \
            --epochs 1 \
            --log-to $LOG_PATTERN \
            --reload /lus/theta-fs0/projects/VeloC/metem/logs/ai-apps-case1-reboot-baseline-73e-${NGRADIENTS}g/partial.h5 \
            --initial-epoch 73 \
    && touch /lus/theta-fs0/projects/VeloC/metem/logs/ai-apps-case1-reboot-baseline-73e-${NGRADIENTS}g/flag.txt
elif [ $ITERATION -eq 6 ]; then
    $HOME/src/triple-r/go.sh python-with-env keras_resnet50.py \
            --tar $IMAGE_ZIP  \
            --data-dir $IMAGE_DIR \
            --epochs 1 \
            --log-to $LOG_PATTERN \
            --reload /lus/theta-fs0/projects/VeloC/metem/logs/ai-apps/checkpoint-73e.h5 \
            --initial-epoch 73 \
            --nbatches $((97*3)) \
            --checkpoint /lus/theta-fs0/projects/VeloC/metem/logs/ai-apps/checkpoint-73e.h5
fi
