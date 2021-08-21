#!/bin/bash
#COBALT -A VeloC
#COBALT -q debug-flat-quad
#COBALT -t 1:00:00
#COBALT --attrs ssds=required:ssd_size=120

module load datascience/tensorflow-2.3

do_aprun() {
    CKPT_TYPE=$1
    # Init
    rm -rf ckpt-logs; mkdir -p ckpt-logs

    # Launch application
    echo "Starting Cobalt job script $1 with $COBALT_JOBSIZE nodes, out: $CKPT_TYPE"
    rpn=1
    threads=$((128/$rpn))
    aprun -n $(($COBALT_JOBSIZE*rpn)) -N $rpn \
    -e HDF5_USE_FILE_LOCKING=FALSE \
    -e OMP_NUM_THREADS=$threads -e KMP_BLOCKTIME=0 -e NUM_INTER_THREADS=2 -e NUM_INTRA_THREADS=$threads \
    -cc depth -d $threads -j 2 \
    python3 ./nt3_baseline_keras2.py

    # Cleanup
    OUT=NT3-$CKPT_TYPE-$COBALT_JOBSIZE
    rm -rf $OUT; mkdir -p $OUT
    cp *.cobaltlog *.output *.error $OUT
    truncate -s0 *.cobaltlog *.output *.error
    mv ckpt-logs/*.log $OUT
} 


do_aprun "HDF5"
rm -rf *.cobaltlog *.output *.error
