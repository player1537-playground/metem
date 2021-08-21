#!/usr/bin/env bash

i2e=( 4 4 10 10 73 73 )

for ngradients in 1 2 3; do
for iteration in 0 2 4; do
root=/lus/theta-fs0/projects/VeloC/metem/logs/ai-apps-case1-reboot-baseline-${i2e[$iteration]}e-${ngradients}g
mkdir -p ${root:?}
if [ -e ${root:?}/1of8.log ] && [ -e ${root:?}/partial.h5 ] && [ -e ${root:?}/flag.txt ]; then continue; fi
if [ $iteration -eq 0 ] || [ $iteration -eq 2 ] || [ $iteration -eq 4 ]; then
    rm ${root:?}/{1..8}of8.log ${root:?}/{partial,recovery}.h5
fi
sleep 1 || break

mpirun -np 8 bash -x ./run.sh \
    /lus/theta-fs0/projects/VeloC/metem/data/tiny-imagenet-200.tar.gz \
    /dev/shm/metem \
    /lus/theta-fs0/projects/VeloC/metem/logs/ai-apps-case1-reboot-baseline-${i2e[$iteration]}e-${ngradients}g/'%(rank+1)dof%(size)d.log' \
    $iteration \
    $ngradients
done
done
