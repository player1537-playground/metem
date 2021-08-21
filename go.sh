#!/usr/bin/env bash
# vim: sta:et:sw=4:ts=4:sts=4:ai
set -x

die() { printf $'Error: %s\n' "$*" >&2; exit 1; }

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
build=${root:?}/build
venv=${root:?}/venv
spack=${root:?}/spack
whatreallyhappened=${root:?}/whatreallyhappened
python=python3.8
stag=docker://nvcr.io/nvidia/tensorflow:20.12-tf2-py3
simg=${root:?}/tensorflow-20.12-tf2-py3.simg
scache=${root:?}/scache
stmp=${root:?}/stmp

[ -f ${root:?}/env.sh ] && . ${root:?}/env.sh

go-singularity() {
    go-singularity-"$@"
}

go-singularity-build() {
    SINGULARITY_CACHEDIR=${scache:?} \
    SINGULARITY_TMPDIR=${stmp:?} \
    exec singularity build \
        ${simg:?} \
        ${stag:?}
}

go-singularity-exec() {
    exec singularity exec \
        --nv \
        -B /soft,/gpfs/mira-home,/home,/lus,/scratch \
        ${simg:?} \
        ${root:?}/go.sh \
        "$@"
}

go-spack() {
    go-singularity-exec \
    go-spack-"$@"
}

go-spack-install() {
    if ! [ -x ${spack:?}/bin/spack ]; then
        git clone https://github.com/spack/spack.git ${spack:?} >&2 || die "Could not clone spack"
    fi

    exec ${spack:?}/bin/spack --env ${root:?} install
}

go-spack-exec() (
    eval $(${spack:?}/bin/spack env activate --sh --dir ${root:?})
    "$@"
)

go-virtualenv() {
    go-spack-exec \
    go-virtualenv-"$@"
}

go-virtualenv-setup() {
    if ! [ -x ${venv:?}/bin/python ]; then
        if ! ${python:?} -c 'import virtualenv' &>/dev/null; then
            if ! ${python:?} -c 'import pip' &>/dev/null; then
                if ! ${python:?} -c 'import ensurepip' &>/dev/null; then
                    die "Cannot import ensurepip"
                fi
                ${python:?} -m ensurepip || die "Cannot ensurepip"
            fi
            ${python:?} -m pip install --user virtualenv || die "Cannot install virtualenv"
        fi
        ${python:?} -m virtualenv -p ${python:?} ${venv:?} || die "Cannot setup virtualenv"
    fi
}

go-virtualenv-python() {
    exec ${venv:?}/bin/python -u "$@"
}

go-virtualenv-exec() (
    . ${venv:?}/bin/activate
    "$@"
)

go-wrh() {
    if ! [ -x ${whatreallyhappened:?}/go.sh ]; then
        git clone https://github.com/player1537-playground/whatreallyhappened.git ${whatreallyhappened:?}
    fi

    go-virtualenv-exec \
    go-wrh-"$@"
}

go-wrh-configure() {
    exec ${whatreallyhappened:?}/go.sh cmake
}

go-wrh-build() {
    exec ${whatreallyhappened:?}/go.sh make
}

go-wrh-install() {
    exec ${whatreallyhappened:?}/go.sh make install
}

go-wrh-exec() {
    exec ${whatreallyhappened:?}/go.sh exec ${root:?}/go.sh "$@"
}

go-exec() {
    go-wrh-exec "$@"
}

go-clean() {
    if [ $# -eq 0 ]; then
        set -- data spack venv
    fi
    for arg; do
        case "$arg" in
        (data) rm -rf ${data:?};;
        (spack) rm -rf ${spack:?} ${root:?}/.spack-env;;
        (venv) rm -rf {venv:?};;
        esac
    done
}

case "${1:-}" in
(go-*) "$@";;
(*) go-"$@";;
esac
