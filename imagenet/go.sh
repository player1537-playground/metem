#!/usr/bin/env bash
# vim: sta:et:sw=4:ts=4:sts=4:ai

die() { printf $'Error: %s\n' "$*" >&2; exit 1; }

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
name=  # the name of this run e.g. name=resnet50-baseline
tar=/lus/theta-fs0/projects/VeloC/metem/data/tiny-imagenet-200.tar.gz
shm=/dev/shm/metem
epochs=  # number of epochs to run e.g. epochs=1
logs=/lus/theta-fs0/projects/VeloC/metem/logs
logpattern='%(rank+1)dof%(size)d.log'
reload=  # optional, the checkpoint to reload before training e.g. reload=path/to=checkpoint.h5
checkpoint=${logs:?}/checkpoint.h5  # optional, the path to checkpoint the model to at the end of training, e.g. checkpoint=path/to/checkpoint.h5
initial_epoch=1  # the epoch to start counting from, in case of restarting trials e.g. initial_epoch=10
ngradients=  # the number of gradients to run before simulating a failure e.g. ngradients=53

[ -f "${root:?}/env.sh" ] && source "${root:?}/env.sh"
[ -f "${root:?}/${USER:?}.env.sh" ] && source "${root:?}/${USER:?}.env.sh"

go-runtrial() {
    python -u "${root:?}/keras_resnet50.py" \
        --tar "${tar:?}" \
        --data-dir "${shm:?}" \
        --epochs "${epochs:?}" \
        --log-to "${logs:?}/${logpattern:?}" \
        ${reload:+--reload "${reload:?}"} \
        --checkpoint "${checkpoint:?} \
        ${ngradients:+--ngradients "${ngradients:?}"}
}

go() {
    case "${1:-}" in
    (go) "$@";;
    (go-*) "$@";;
    (*) go-"$@";;
    esac
}

go "$@"
