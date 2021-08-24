# Metem Scripts

This repository includes all of the scripts written for the Metem paper as part
of the Triple-R subgoal of the Triple-Convergence project.

The code is split into 4 categories:
- Overall environment setup (`/go.sh`)
- Metem-specific code (`/metem/`)
- ImageNet/ResNet50-specific code (`/imagenet/`)
- NT3-specific code (`/nt3/`)


## Setup

To setup the environment, run the following command:

```console
$ ./go.sh buildall
```

This command runs a lot of separate commands in order. Those separate commands
are:

```console
$ ./go.sh singularity build
$ ./go.sh spack install
$ ./go.sh virtualenv setup
$ ./go.sh wrh configure
$ ./go.sh wrh build
$ ./go.sh wrh install
```


## Existing Data

There is already some existing data, primarily checkpoints and log files.

For ResNet50, these are at:

```console
$ ls /lus/theta-fs0/projects/VeloC/metem/logs/ai-apps/checkpoint-*e.h5
$ ls /lus/theta-fs0/projects/VeloC/metem/logs/ai-apps/*of8.log
```

For example,
`/lus/theta-fs0/projects/VeloC/metem/logs/ai-apps/checkpoint-5e.h5` is the
checkpoint taken after 5 epochs have completed.

For NT3, these are at:

```console
$ ls /lus/theta-fs0/projects/VeloC/metem/logs/BL\,dataset\=NT3\,model\=default\,nworkers\=8\,seed\=1337\,div\=1\,nepochs\=200/checkpoint-*.h5
$ ls /lus/theta-fs0/projects/VeloC/metem/logs/BL\,dataset\=NT3\,model\=default\,nworkers\=8\,seed\=1337\,div\=1\,nepochs\=200/*of8.log
```
