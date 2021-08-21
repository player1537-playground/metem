# This model is an example of a computation-intensive model that achieves good accuracy on an image
# classification task.  It brings together distributed training concepts such as learning rate
# schedule adjustments with a warmup, randomized data reading, and checkpointing on the first worker
# only.
#
# Note: This model uses Keras native ImageDataGenerator and not the sophisticated preprocessing
# pipeline that is typically used to train state-of-the-art ResNet-50 model.  This results in ~0.5%
# increase in the top-1 validation error compared to the single-crop top-1 validation error from
# https://github.com/KaimingHe/deep-residual-networks.
#

import mypatch

import tensorflow.keras
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os, time, timeit
from datetime import datetime
import numpy as np
import random

import preprocess

import whatreallyhappened as wrh
from mpi4py import MPI

def set_parallelism_threads():
    """ Set the number of parallel threads according to the number available on the hardware
    """

    if 'NUM_INTRA_THREADS' in os.environ and 'NUM_INTER_THREADS' in os.environ:
        print('Using Thread Parallelism: {} NUM_INTRA_THREADS, {} NUM_INTER_THREADS'.format(os.environ['NUM_INTRA_THREADS'], os.environ['NUM_INTER_THREADS']))
        tf.config.threading.set_inter_op_parallelism_threads(int(os.environ['NUM_INTER_THREADS']))
        tf.config.threading.set_intra_op_parallelism_threads(int(os.environ['NUM_INTRA_THREADS']))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Keras ImageNet Example',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--zip', default=None,
                        help='path to training zip (optional)')
    parser.add_argument('--tar', default=None,
                        help='path to training tar (optional)')
    parser.add_argument('--data-dir', default=None,
                        help='path to data')
    # Default settings from https://arxiv.org/abs/1706.02677.
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=32,
                        help='input batch size for validation')
    parser.add_argument('--epochs', type=int, default=90,
                        help='number of epochs to train')
    parser.add_argument('--base-lr', type=float, default=0.0125,
                        help='learning rate for a single GPU')
    parser.add_argument('--warmup-epochs', type=float, default=5,
                        help='number of warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--wd', type=float, default=0.00005,
                        help='weight decay')
    parser.add_argument('--exclude-range', default='1', help='exclusion range in 1.x.y.z format')
    parser.add_argument('--model-type', type=int, default=50, help='model type: 50 - ResNet50, 121 - DenseNet121, 152 - ResNet152')
    parser.add_argument('--log-to', type=str)
    parser.add_argument('--reload', default=None)
    parser.add_argument('--initial-epoch', type=int, default=0)
    parser.add_argument('--ngradients', type=int, default=None)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--nbatches', default=None, type=int)

    args = parser.parse_args()

    if args.data_dir == None:
        raise ValueError("You must provide --data-dir !")
    return args

def check_exists(label, fn):
    if not os.path.exists(fn):
        raise Exception("does not exist: (%s) '%s'" % (label, fn))


class StdoutLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f'{epoch}: ')

    def on_epoch_end(self, epoch, logs=None):
        print()
        for k, v in logs.items():
            print('  %r = %r\n' % (k, v))

    def on_train_batch_begin(self, batch, logs=None):
        print(f'{batch%10}', end='')


class WRHLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        wrh.push('epoch')
        wrh.log('epoch', '%d', epoch)

    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            wrh.log(k, '%r', v)
        wrh.pop('epoch')

    def on_train_batch_begin(self, batch, logs=None):
        wrh.push('batch')
        wrh.log('batch', '%d', batch)

    def on_train_batch_end(self, batch, logs=None):
        for k, v in logs.items():
            wrh.log(k, '%r', v)
        wrh.pop('batch')


class PreciseEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, nepochs, nbatches):
        self.nepochs = nepochs
        self.nbatches = nbatches
        self.epoch = None
        self.batch = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self._check_condition_and_stop()

    def on_train_batch_begin(self, batch, logs=None):
        self.batch = batch
        self._check_condition_and_stop()

    def _check_condition_and_stop(self):
        has_epoch_limit = self.nepochs is not None
        has_batch_limit = self.nbatches is not None

        on_epoch_limit = has_epoch_limit and self.epoch == self.nepochs - 1
        on_batch_limit = has_batch_limit and self.batch == self.nbatches - 1

        if has_epoch_limit:
            if has_batch_limit:
                if on_epoch_limit and on_batch_limit:
                    self.model.stop_training = True
                else:
                    pass  # wait to reach limits
            else:
                if on_epoch_limit:
                    self.model.stop_training = True
                else:
                    pass  # wait to reach limits
        else:
            if has_batch_limit:
                if on_batch_limit:
                    self.model.stop_training = True
                else:
                    pass  # wait to reach limits
            else:
                pass  # there are no limits


def run(args):
    import horovod.tensorflow.keras as hvd
    hvd.init()

    print('opening wrh log...')
    wrh.open(str(args.log_to) % {
        'rank': hvd.rank(),
        'size': hvd.size(),
        'rank+1': hvd.rank() + 1,
    }, 'a', always_flush=True)
    print('opened wrh log')

    if hvd.rank() == 0:
        wrh.push('master')
        for i in range(1, hvd.size()):
            wrh.push('worker')
            info = wrh.save()
            MPI.COMM_WORLD.send(info, dest=i, tag=i)
            wrh.pop('worker')
        wrh.push('worker')
    else:
        info = MPI.COMM_WORLD.recv(source=0, tag=hvd.rank())
        wrh.load(info)

    print("TF version: " + tf.__version__)

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    set_parallelism_threads()
    verbose = 0

    words_txt = args.data_dir + "/tiny-imagenet-200/words.txt"
    print(words_txt)
    if os.path.exists(words_txt):
        preprocess.restore_dir(args.data_dir + "/tiny-imagenet-200")
    elif args.zip != None:
        print("extracting zip: '%s' to '%s'" % (args.zip, args.data_dir))
        import zipfile
        start = time.time()
        with zipfile.ZipFile(args.zip, "r") as zf:
            zf.extractall(path=args.data_dir)
        stop = time.time()
        print("time unzip: %2.3f" % (stop-start))
    elif args.tar != None:
        import subprocess
        subprocess.run([
            'mkdir', '-p', str(args.data_dir),
        ])

        subprocess.run([
            'tar', 'xf', args.tar,
        ], cwd=str(args.data_dir))
    else:
        print("using existing data_dir: '%s'" % args.data_dir)

    # Training data iterator.
    train_dir = args.data_dir + "/tiny-imagenet-200/train"
    val_dir   = args.data_dir + "/tiny-imagenet-200/val"
    check_exists("train_dir", train_dir)
    check_exists("val_dir",   val_dir)

    #preprocess.delete_region(train_dir, args.exclude_range)

    train_gen = image.ImageDataGenerator(
        width_shift_range=0.33, height_shift_range=0.33, zoom_range=0.5, horizontal_flip=True,
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
    train_iter = train_gen.flow_from_directory(train_dir,
                                               batch_size=args.batch_size,
                                               target_size=(224, 224))
    for _ in range(10):
        next(train_iter)


    # Validation data iterator.
    test_gen = image.ImageDataGenerator(
        zoom_range=(0.875, 0.875), preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
    test_iter = test_gen.flow_from_directory(val_dir,
                                             batch_size=args.val_batch_size,
                                             target_size=(224, 224))

    for _ in range(10):
        next(test_iter)

    if args.reload is None:

        # Set up standard ResNet-50 model.
        if args.model_type == 50:
            print("Running with ResNet50")
            model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights=None, classes=200)
        elif args.model_type == 121:
            print("Running with DenseNet121")
            model = tf.keras.applications.DenseNet121(include_top=True, weights=None, classes=200)
        else:
            print("Running with ResNet152")
            model = tf.keras.applications.ResNet152(include_top=True, weights=None, classes=200)

        # ResNet-50 model that is included with Keras is optimized for inference.
        # Add L2 weight decay & adjust BN settings.
        model_config = model.get_config()
        for layer, layer_config in zip(model.layers, model_config['layers']):
            if hasattr(layer, 'kernel_regularizer'):
                regularizer = tf.keras.regularizers.l2(args.wd)
                layer_config['config']['kernel_regularizer'] = \
                    {'class_name': regularizer.__class__.__name__,
                     'config': regularizer.get_config()}
            if type(layer) == tf.keras.layers.BatchNormalization:
                layer_config['config']['momentum'] = 0.9
                layer_config['config']['epsilon'] = 1e-5

        model = tf.keras.models.Model.from_config(model_config)
        opt = tf.keras.optimizers.SGD(lr=args.base_lr, momentum=args.momentum)
        opt = hvd.DistributedOptimizer(opt)

        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=opt,
                      experimental_run_tf_function=True,
                      metrics=['accuracy'])
        # , 'top_k_categorical_accuracy'

        # ,              experimental_run_tf_function=False

    else:
        model = hvd.load_model(args.reload)

    base_callbacks = [
              hvd.callbacks.BroadcastGlobalVariablesCallback(0),
              StdoutLoggingCallback(),
              WRHLoggingCallback(),
    ]


    callbacks = base_callbacks[:]

    if hvd.rank() == 0 and False:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath='/lus/theta-fs0/projects/VeloC/metem/logs/ai-apps-case3/checkpoint-{epoch}e.h5',
        ))

    mypatch.set_params(
        rank=hvd.rank(),
        divisor=hvd.size(),
        act_after_gradient=100000,
        act_after_layer=None,
        action=None,
    )
    callbacks.append(PreciseEarlyStopping(
        nepochs=None,
        nbatches=5,
    ))

    tf.random.set_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    wrh.push('model.fit')
    start_ts = timeit.default_timer()
    print("train_iter: ", len(train_iter))
    model.fit(train_iter,
              steps_per_epoch=len(train_iter) // hvd.size(),
              epochs=args.initial_epoch + args.epochs,
              verbose=verbose if hvd.rank() == 0 else 0,
              initial_epoch=args.initial_epoch,
              callbacks=callbacks,
              validation_data=test_iter,
              validation_steps=len(test_iter)) # // hvd.size()
    wrh.pop('model.fit')


    callbacks = base_callbacks[:]

    if hvd.rank() == 0 and False:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath='/lus/theta-fs0/projects/VeloC/metem/logs/ai-apps-case3/checkpoint-{epoch}e.h5',
        ))

    mypatch.set_params(
        rank=hvd.rank(),
        divisor=hvd.size(),
        act_after_gradient=1000 if args.ngradients is None else args.ngradients,
        act_after_layer=None,
        action='stop-8',
    )
    callbacks.append(PreciseEarlyStopping(
        nepochs=None,
        nbatches=args.nbatches if args.nbatches is not None else 1,
    ))

    tf.random.set_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    wrh.push('model.fit')
    start_ts = timeit.default_timer()
    print("train_iter: ", len(train_iter))
    model.fit(train_iter,
              steps_per_epoch=len(train_iter) // hvd.size(),
              epochs=args.initial_epoch + args.epochs,
              verbose=verbose if hvd.rank() == 0 else 0,
              initial_epoch=args.initial_epoch,
              callbacks=callbacks,
              validation_data=test_iter,
              validation_steps=len(test_iter)) # // hvd.size()
    wrh.pop('model.fit')


    if args.checkpoint is not None:
        wrh.push('checkpoint')
        if hvd.rank() == 0:
            model.save(args.checkpoint)
        wrh.pop('checkpoint')

    if hvd.rank() == 0:
        wrh.pop('worker')
        wrh.pop('master')

if __name__ == "__main__":
    args = parse_args()
    run(args)
