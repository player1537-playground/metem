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

