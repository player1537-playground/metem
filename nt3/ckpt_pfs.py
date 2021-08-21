import timeit, socket, os, sys, functools, time
import numpy
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class BatchMonitor(Callback):
    def __init__(self, rank, size):
        self.THRESHOLD = 10
        self.counter = 0
        self.rank = rank
        self.size = size
        self.ckpt_no = 0
        os.makedirs("ckpt-logs", exist_ok = True)
        self.fn = open(os.path.join("ckpt-logs", "batch-" + socket.gethostname() + "-" + str(self.rank) + ".log"), 'a')

        self.ts = timeit.default_timer()
        self.batch_ts = self.ts
        self.log("begin batch monitoring, assuming role: %d" % (rank % 2))

    def on_batch_begin(self, batch, logs):
        self.counter += 1
        self.log("starting batch %d" % self.counter, self.batch_ts)
        self.batch_ts = timeit.default_timer()

    def on_batch_end(self, batch, logs):
        if self.counter == self.THRESHOLD:
            self.do_ckpt()
        self.fn.flush()

    def log(self, line, ref=0.0):
        now = timeit.default_timer()
        if ref == 0.0:
            print("[%.3f] Rank %d: %s" % (now - self.ts, self.rank, line), file = self.fn)
        else:
            print("[%.3f] [duration: %.3f] Rank %d: %s" % (now - self.ts, now - ref, self.rank, line), file = self.fn)
        self.fn.flush()

    def do_ckpt(self):
        t = timeit.default_timer()
        self.log("starting checkpoint at epoch %d" % self.counter)
        if self.rank == 0:
            self.model.save_weights(os.path.join("ckpt-logs", "model-%d.%d.h5" % (self.rank, self.counter)))
        self.log("finsihed checkpoint at epoch %d" % self.counter, t)
        self.ckpt_no += 1
        self.fn.flush()
