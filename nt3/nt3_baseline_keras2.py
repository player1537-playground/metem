import mypatch

import numpy as np
import os, sys, gzip, random, timeit, pickle, time
import types
import random

# import MPI
from mpi4py import MPI

# import Tensorflow + Horovod
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.models import Sequential, Model, model_from_json, model_from_yaml
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
import horovod.tensorflow.keras as hvd
import tensorflow as tf
import whatreallyhappened as wrh

from dataclasses import dataclass
from pathlib import Path
import re

#TIMEOUT=3600 # in sec; set this to -1 for no timeout 

import nt3 as bmk
import candle
import ckpt_pfs as ckpt_util

class PreciseEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, nepochs, nbatches):
        self.nepochs = nepochs
        self.nbatches = nbatches
        self.epoch = None
        self.batch = None
        self.batch_start = None

    def on_epoch_begin(self, epoch, logs=None):
        print(f'{epoch}: ')
        wrh.push('epoch')
        wrh.log('epoch', '%d', epoch)
        self.epoch = epoch
        self._check_condition_and_stop()

    def on_epoch_end(self, epoch, logs=None):
        print()
        for k, v in logs.items():
            wrh.log(k, '%r', v)
        wrh.pop('epoch')

    def on_train_batch_begin(self, batch, logs=None):
        print(f'{batch%10}', end='')
        wrh.push('batch')
        wrh.log('batch', '%d', batch)
        self.batch = batch
        self._check_condition_and_stop()

    def on_train_batch_end(self, batch, logs=None):
        for k, v in logs.items():
            wrh.log(k, '%r', v)
        wrh.pop('batch')

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


def initialize_parameters():

    # Initialize CANDLE environment
    candle.set_seed(1234)
    candle.set_parallelism_threads()

    # Build benchmark object
    nt3Bmk = bmk.BenchmarkNT3(bmk.file_path, 'nt3_default_model.txt', 'keras',
    prog='nt3_baseline', desc='Multi-task (DNN) for data extraction from clinical reports - Pilot 3 Benchmark 1')

    # Initialize parameters
    gParameters = candle.initialize_parameters(nt3Bmk)
    #benchmark.logger.info('Params: {}'.format(gParameters))

    return gParameters

def load_data(pickled, rank, size):
    start = timeit.default_timer()
    f = open(pickled, 'rb')
    X_train = pickle.load(f)
    Y_train = pickle.load(f)
    X_test = pickle.load(f)
    Y_test = pickle.load(f)
    f.close()

    # Extract the training data shard of the current rank
    shard_size = 20 * gParameters['batch_size']
    dindex = rank * shard_size
    if ((dindex + shard_size) > X_train.shape[0]):
        dindex = 0
    X_train = X_train[dindex: dindex + shard_size, :]
    Y_train = Y_train[dindex: dindex + shard_size, :]
    print("Rank %d loaded data shard (%d: %d): %.3f s" % (rank, dindex, dindex + shard_size, timeit.default_timer() - start))

    # Extract the training data shard of the current rank
    #train_len = X_train.shape[0]
    #shard_size = train_len // size
    #X_train = X_train[rank * shard_size: (rank + 1) * shard_size, :]
    #Y_train = Y_train[rank * shard_size: (rank + 1) * shard_size, :]
    #print("Rank %d loaded data shard (%d: %d): %.3f s" % (rank, rank * shard_size, (rank + 1) * shard_size, timeit.default_timer() - start))

    return X_train, Y_train, X_test, Y_test

def run(gParameters):
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    size = world.Get_size()

    wrh.open(str(args.log_to) % {
        'rank': rank,
        'size': size,
        'rank+1': rank+1,
    }, 'a', always_flush=True)

    if rank == 0:
        wrh.push('master')
        for i in range(1, size):
            wrh.push('worker')
            info = wrh.save()
            world.send(info, dest=i, tag=i)
            wrh.pop('worker')
        wrh.push('worker')
    else:
        info = world.recv(source=0, tag=rank)
        wrh.load(info)

    wrh.push('triple-r.py')
    wrh.log('rank', '%d', rank)
    wrh.log('size', '%d', size)
    wrh.log('model', '%s', args.make_model_fn)
    wrh.log('dataset', '%s', args.dataset)
    wrh.log('events', '%s', args.events)
    wrh.log('div', '%d', args.div)
    wrh.log('data_dir', '%s', args.data_dir)

    wrh.push('loading dataset')
    print ('Params:', gParameters)

    #hvd.init()

    X_train, Y_train, X_test, Y_test = load_data('/lus/theta-fs0/projects/VeloC/bogdan/candle/input/csv_input.pickled', rank, size)

    #hvd.shutdown()

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)

    x_train_len = X_train.shape[1]

    # this reshaping is critical for the Conv1D to work

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    wrh.pop('loading dataset')

    wrh.push('creating model')

    model = Sequential()

    layer_list = list(range(0, len(gParameters['conv']), 3))
    for l, i in enumerate(layer_list):
        filters = gParameters['conv'][i]
        filter_len = gParameters['conv'][i+1]
        stride = gParameters['conv'][i+2]
        print(int(i/3), filters, filter_len, stride)
        if gParameters['pool']:
            pool_list=gParameters['pool']
            if type(pool_list) != list:
                pool_list=list(pool_list)

        if filters <= 0 or filter_len <= 0 or stride <= 0:
                break
        if 'locally_connected' in gParameters:
                model.add(LocallyConnected1D(filters, filter_len, strides=stride, padding='valid', input_shape=(x_train_len, 1)))
        else:
            #input layer
            if i == 0:
                model.add(Conv1D(filters=filters, kernel_size=filter_len, strides=stride, padding='valid', input_shape=(x_train_len, 1)))
            else:
                model.add(Conv1D(filters=filters, kernel_size=filter_len, strides=stride, padding='valid'))
        model.add(Activation(gParameters['activation']))
        if gParameters['pool']:
                model.add(MaxPooling1D(pool_size=pool_list[int(i/3)]))

    model.add(Flatten())

    for layer in gParameters['dense']:
        if layer:
            model.add(Dense(layer))
            model.add(Activation(gParameters['activation']))
            if gParameters['drop']:
                    model.add(Dropout(gParameters['drop']))
    model.add(Dense(gParameters['classes']))
    model.add(Activation(gParameters['out_act']))

#Reference case
#model.add(Conv1D(filters=128, kernel_size=20, strides=1, padding='valid', input_shape=(P, 1)))
#model.add(Activation('relu'))
#model.add(MaxPooling1D(pool_size=1))
#model.add(Conv1D(filters=128, kernel_size=10, strides=1, padding='valid'))
#model.add(Activation('relu'))
#model.add(MaxPooling1D(pool_size=10))
#model.add(Flatten())
#model.add(Dense(200))
#model.add(Activation('relu'))
#model.add(Dropout(0.1))
#model.add(Dense(20))
#model.add(Activation('relu'))
#model.add(Dropout(0.1))
#model.add(Dense(CLASSES))
#model.add(Activation('softmax'))

    wrh.pop('creating model')

    initial_epoch = args.initial_epoch
    for event in args.events:
        world.Barrier()

        wrh.push('event')
        wrh.log('event', '%r', event)

#        wrh.push('hvd.init')
#        color = 1
#        if rank >= event.nworkers:
#            color = MPI.UNDEFINED
#
#        new_world = world.Split(color, rank)
#
#        if new_world == MPI.COMM_NULL:
#            wrh.log('bailing', '%r', (rank, event.nworkers))
#            wrh.pop('hvd.init')
#            wrh.pop('event')
#            continue
#
#        hvd.init(new_world)
#        wrh.pop('hvd.init')

        np.random.seed(event.seed)
        tf.random.set_seed(event.seed)
        random.seed(event.seed)

        kerasDefaults = candle.keras_default_config()

        # Define CkptOptimzier: pass COMM_WORLD to it
        bm = ckpt_util.BatchMonitor(rank, size)
        optimizer = candle.build_optimizer(gParameters['optimizer'],
                                           gParameters['learning_rate'] * hvd.size(),
                                           kerasDefaults)

        # Convert into distributed Horovod optimizer
        optimizer = hvd.DistributedOptimizer(optimizer)
        bm.optimizer = optimizer

        mypatch.set_params(
            rank=rank,
            divisor=event.nworkers,
            act_after_layer=event.nlayers,
            act_after_gradient=event.ngradients,
            action=event.action,  # action='abort' or 'stop' or 'stop-4'
        )

        if event.reload is not None:
            wrh.push('reload')
            wrh.log('event.reload', '%r', args.checkpoint_dir / event.reload)
            print(f'Reloading weights')
            model = hvd.load_model(args.checkpoint_dir / event.reload)
            wrh.pop('reload')

        else:
            model.summary()
            model.compile(loss=gParameters['loss'],
                          optimizer=optimizer,
                          metrics=[gParameters['metrics']],
                          experimental_run_tf_function=False)

        output_dir = gParameters['save']

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # calculate trainable and non-trainable params
        gParameters.update(candle.compute_trainable_params(model))

        cbacks = []

        # set up a bunch of callbacks to do work during model training..
        model_name = gParameters['model_name']
        path = '{}/{}.autosave.model.h5'.format(output_dir, model_name)
        cbacks.append(CSVLogger('{}/training.log'.format(output_dir)))
        cbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0))
        cbacks.append(candle.CandleRemoteMonitor(params=gParameters))
        cbacks.append(candle.TerminateOnTimeOut(gParameters['timeout']))
        cbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        cbacks.append(bm)

        nbatches = event.nbatches
        if nbatches is not None and nbatches < 0:
            nbatches = X_train.shape[0] // event.batch // event.nworkers // args.div - abs(nbatches)
        cbacks.append(PreciseEarlyStopping(nepochs=None, nbatches=nbatches))

        if rank == 0 and False:
            cbacks.append(tf.keras.callbacks.ModelCheckpoint(
                args.checkpoint_dir / event.checkpoint,
            ))

        wrh.push('train')
        try:
            history = model.fit(X_train, Y_train,
                            shuffle=True,
                            batch_size=event.batch,
                            steps_per_epoch=X_train.shape[0] // event.batch // event.nworkers // args.div,
                            initial_epoch=initial_epoch,
                            epochs=initial_epoch + event.nepochs,
                            verbose=1,
                            validation_data=(X_test, Y_test),
                            callbacks = cbacks,
                            )
        except Exception as e:
            import traceback; traceback.print_exc()
            wrh.log('exception', '%r', e)
            wrh.pop('batch')
            wrh.pop('epoch')
            wrh.pop('train')
            wrh.pop('event')
            return
        wrh.pop('train')

        wrh.push('valid')
        score = model.evaluate(X_test, Y_test, verbose=0)
        if rank == 0:
            print(f'stats = {" ".join(f"{name}={value}" for name, value in zip(model.metrics_names, score))}')
        for name, value in zip(model.metrics_names, score):
            wrh.log(name, '%r', value)
        wrh.pop('valid')

        if False and event.checkpoint is not None and rank == 0:
            wrh.push('checkpoint')
            wrh.log('event.checkpoint', '%r', args.checkpoint_dir / event.checkpoint)
            model.save(args.checkpoint_dir / event.checkpoint)
            wrh.pop('checkpoint')

        world.Barrier()

        initial_epoch += event.nepochs

#        wrh.push('hvd.shutdown')
#        hvd.shutdown()
#        wrh.pop('hvd.shutdown')

        wrh.pop('event')

    wrh.pop('triple-r.py')

    if rank == 0:
        wrh.pop('worker')
        wrh.pop('master')

        


    if False:
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        # serialize model to JSON
        model_json = model.to_json()
        with open("{}/{}.model.json".format(output_dir, model_name), "w") as json_file:
            json_file.write(model_json)

        # serialize model to YAML
        model_yaml = model.to_yaml()
        with open("{}/{}.model.yaml".format(output_dir, model_name), "w") as yaml_file:
            yaml_file.write(model_yaml)

        # serialize weights to HDF5
        model.save_weights("{}/{}.weights.h5".format(output_dir, model_name))
        print("Saved model to disk")

        # load json and create model
        json_file = open('{}/{}.model.json'.format(output_dir, model_name), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model_json = model_from_json(loaded_model_json)

        # load yaml and create model
        yaml_file = open('{}/{}.model.yaml'.format(output_dir, model_name), 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model_yaml = model_from_yaml(loaded_model_yaml)


        # load weights into new model
        loaded_model_json.load_weights('{}/{}.weights.h5'.format(output_dir, model_name))
        print("Loaded json model from disk")

        # evaluate json loaded model on test data
        loaded_model_json.compile(loss=gParameters['loss'],
            optimizer=gParameters['optimizer'],
            metrics=[gParameters['metrics']])
        score_json = loaded_model_json.evaluate(X_test, Y_test, verbose=0)

        print('json Test score:', score_json[0])
        print('json Test accuracy:', score_json[1])

        print("json %s: %.2f%%" % (loaded_model_json.metrics_names[1], score_json[1]*100))

        # load weights into new model
        loaded_model_yaml.load_weights('{}/{}.weights.h5'.format(output_dir, model_name))
        print("Loaded yaml model from disk")

        # evaluate loaded model on test data
        loaded_model_yaml.compile(loss=gParameters['loss'],
            optimizer=gParameters['optimizer'],
            metrics=[gParameters['metrics']])
        score_yaml = loaded_model_yaml.evaluate(X_test, Y_test, verbose=0)

        print('yaml Test score:', score_yaml[0])
        print('yaml Test accuracy:', score_yaml[1])

        print("yaml %s: %.2f%%" % (loaded_model_yaml.metrics_names[1], score_yaml[1]*100))

    return None

@dataclass
class Event:
    nepochs: int
    nworkers: int
    batch: int
    reload: Path
    checkpoint: Path
    ngradients: int
    nlayers: int
    seed: int
    nbatches: int
    action: str

    @classmethod
    def parse(cls, s):
        # 0e/nworkers=12
        # 12e/nworkers=6
        match = re.match(r'^(?P<nepochs>[0-9]+)e/(?P<options>[a-z_]+=[a-z0-9A-Z._{}-]+(?:,[a-z_]+=[a-z0-9A-Z._{}-]+)*)$', s)
        nepochs = int(match.group('nepochs'))
        options = match.group('options')
        options = dict((k, v) for x in options.split(',') for k, v in (x.split('=', 1),))
        nworkers = options.get('nworkers', None)
        if nworkers is not None:
            nworkers = int(nworkers)
        batch = int(options.get('batch', 32))
        reload = options.get('reload', None)
        if reload is not None:
            reload = Path(reload)
        checkpoint = options.get('checkpoint', None)
        if checkpoint is not None:
            checkpoint = Path(checkpoint)
        ngradients = options.get('ngradients', None)
        if ngradients is not None:
            ngradients = int(ngradients)
        nlayers = options.get('nlayers', None)
        if nlayers is not None:
            nlayers = int(nlayers)
        seed = options.get('seed', None)
        if seed is not None:
            seed = int(seed)
        nbatches = options.get('nbatches', None)
        if nbatches is not None:
            nbatches = int(nbatches)
        action = options.get('action', None)

        return cls(nepochs, nworkers, batch, reload, checkpoint, ngradients, nlayers, seed, nbatches, action)


if __name__ == '__main__':
    def event(s):
        try:
            return Event.parse(s)
        except Exception as e:
            print(e)
            raise argparse.ArgumentError() from e

    def make_model_fn(s):
        if s.startswith('CNN-'):
            num_conv_layers = int(s[len('CNN-'):])
            fn = partial(make_simple_cnn_model, num_conv_layers=num_conv_layers)

        elif s == 'ResNet50':
            fn = make_resnet50_model

        fn.__name__ = s
        return fn

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('events', nargs='+', type=event)
    parser.add_argument('--go-model', dest='make_model_fn', required=True)
    parser.add_argument('--go-dataset', dest='dataset', required=True)
    parser.add_argument('--go-div', dest='div', required=True, type=int)
    parser.add_argument('--go-default-verbosity', dest='default_verbosity', required=True)
    parser.add_argument('--go-data-dir', dest='data_dir', type=Path, required=True)
    parser.add_argument('--go-log-to', dest='log_to', required=True, type=Path)
    parser.add_argument('--go-log-info', dest='log_info', required=False)
    parser.add_argument('--go-checkpoint-dir', dest='checkpoint_dir', required=True)
    parser.add_argument('--go-initial-epoch', dest='initial_epoch', required=True, type=int)
    args, left = parser.parse_known_args()

    import sys; sys.argv = sys.argv[:1] + left

    gParameters = initialize_parameters()

    run(gParameters)
    try:
        K.clear_session()
    except AttributeError:      # theano does not have this function
        pass
