"""

"""

from __future__ import annotations
from contextlib import contextmanager
import csv
from dataclasses import dataclass, field
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
import re
from statistics import mean, stdev, quantiles
import subprocess
import sys
from typing import *

from typing_extensions import get_type_hints, get_origin, get_args, Annotated
import numpy as np
import scipy.optimize


class SkipWithBlock(Exception):
    pass


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    from itertools import tee
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    from itertools import zip_longest
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


@contextmanager
def subprocess_as_stdin(args):
    process = subprocess.Popen(
        args=args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    old_stdin = sys.stdin
    sys.stdin = process.stdout

    try:
        yield
    finally:
        process.stdout.close()
        process.terminate()
        process.wait(timeout=1)
        process.kill()
        process.wait()

        sys.stdin = old_stdin


@dataclass
class Tree:
    attributes: Dict[str, Tuple[datetime, str]]
    logs: Dict[str, Tuple[datetime, str]]
    children: List[Tree] = field(repr=False)
    parent: Tree
    depth: int = field(init=False)

    def __post_init__(self):
        if self.parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1

    RE = re.compile(r'''
        ^
        (?:.*)
        (?P<task_id>[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-z0-9]{12})  # e.g. "6a3ad23f-b0c1-47dd-9913-cf537e950839"
        /(?P<task_level>[0-9]+(?:/[0-9]+)*)                                        # e.g. "/8/5/17/5/4"
        [\t]
        (?P<timestamp>[0-9]+\.[0-9]*)                                              # e.g. "1613403872.549264"
        [\t]
        (?P<variable>[^\t]+)                                                       # e.g. "@finished"
        [\t]
        (?P<value>.*)
        $
    ''', re.VERBOSE)

    @classmethod
    def readiter(cls, infiles: List[Path], *, presorted: bool=False) -> Iterator[Tree]:
        if not presorted:
            args = [
                'sort',
                '-t\t',      # tab delimited
                '-k1,1.36',  # sort by first column, first 36 characters (task id)
                '-k1.36V',   # sort by first column, characters after first 36, as a version (task levels)
                '-k2,2n',    # sort by second column, as a number (timestamp)
                *infiles,    # sort these files
            ]
        else:
            args = [
                'sort',
                '-t\t',      # tab delimited
                '-k1,1.36',  # sort by first column, first 36 characters (task id)
                '-k1.36V',   # sort by first column, characters after first 36, as a version (task levels)
                '-k2,2n',    # sort by second column, as a number (timestamp)
                '-m',        # merge already sorted files
                *infiles,    # sort these files
            ]
            
        with subprocess_as_stdin(args):
            last_task_id = None
            last_task_level = None
            last_timestamp = None
            last_variable = None
            last_value = None

            root = None
            tree = None

            for line in sys.stdin:
                line = line.rstrip()
                match = cls.RE.match(line)
                if not match:
                    print( ValueError(f'{line=} does not match {cls.RE=}') )
                    continue

                task_id = match.group('task_id')
                task_level = tuple(map(int, match.group('task_level').split('/')))
                timestamp = datetime.fromtimestamp(float(match.group('timestamp')))
                variable = match.group('variable')
                value = match.group('value')

                #print(f'{task_id=} {task_level=} {timestamp=} {variable=} {value=}')

                if last_task_id is not None and task_id != last_task_id:
                    yield root
                    root = None
                    tree = None

                if tree is None:
                    tree = cls({}, {}, [], None)
                    last_task_id = None
                    last_task_level = None
                    last_timestamp = None
                    last_variable = None
                    last_value = None
                    #print('new tree')

                if root is None:
                    root = tree

                if last_task_id is not None and len(task_level) > len(last_task_level):
                    tree = cls({}, {}, [], tree)
                    tree.parent.children.append(tree)

                if last_task_id is not None and len(task_level) < len(last_task_level):
                    tree = tree.parent

                if last_task_id is not None and len(task_level) == len(last_task_level):
                    if task_level[-1] < last_task_level[-1]:
                        tree = cls({}, {}, [], tree.parent)
                        tree.parent.children.append(tree)

                if variable.startswith('@'):
                    tree.attributes[variable] = (timestamp, value)
                else:
                    tree.logs[variable] = (timestamp, value)

                last_task_id = task_id
                last_task_level = task_level
                last_timestamp = timestamp
                last_variable = variable
                last_value = value

        if root is not None:
            yield root
    
    def pprint(self):
        def p(tree, depth=0):
            for attr, (timestamp, value) in tree.attributes.items():
                print(f'{" "*depth}{timestamp} {attr}={value!r}')
            for log, (timestamp, value) in tree.logs.items():
                print(f'{" "*depth}{timestamp} {log}={value!r}')
            for tree in tree.children:
                p(tree, depth+2)

        p(self)

    # matching

    def __call__(self, started=None, finished=None):
        if started is not None and finished is None:
            finished = started

        for child in self.children:
            if not (started is not None and (child.attributes.get('@started', None) is None or started == child.attributes['@started'][1])):
                continue

            if not (finished is not None and (child.attributes.get('@finished', None) is None or finished == child.attributes['@finished'][1])):
                continue

            yield child

    def __getitem__(self, key) -> str:
        try:
            key, convert = key
        except ValueError:
            convert = str

        if (value := self.attributes.get(f'@{key}', None)) is not None:
            pass
        elif (value := self.logs.get(key, None)) is not None:
            pass
        else:
            return None

        if convert is datetime:
            value = value[0]
        else:
            value = convert(value[1])

        return value

    @property
    def duration(self) -> timedelta:
        started = self.attributes['@started'][0]
        finished = self.attributes['@finished'][0]

        return finished - started


def main_batch_csv(infiles, presorted, outfile):
    writer = csv.writer(outfile)
    writer.writerow(['iteration', 'epoch', 'batch', 'duration', 'loss', 'accuracy'])

    i = 0

    for tree in Tree.readiter(infiles, presorted=presorted):
        #for tree in tree('master'):
            for tree in tree('worker'):
                for tree in tree('model.fit'):
                    for tree in tree('epoch'):
                        epoch = tree['epoch', int]

                        for tree in tree('batch'):
                            batch_duration = tree.duration.total_seconds()
                            batch = tree['batch', int]
                            loss = tree['loss', float]
                            accuracy = tree['accuracy', float]

                            writer.writerow([i, epoch, batch, batch_duration, loss, accuracy])
                            i += 1


def main_epoch_csv(infiles, presorted, outfile):
    writer = csv.writer(outfile)
    writer.writerow(['iteration', 'epoch', 'duration', 'val_loss', 'val_accuracy'])

    i = 0

    for tree in Tree.readiter(infiles, presorted=presorted):
        #for tree in tree('master'):
            for tree in tree('worker'):
                for tree in tree('model.fit'):
                    for tree in tree('epoch'):
                        epoch = tree['epoch', int]
                        epoch_duration = tree.duration.total_seconds()
                        val_loss = tree['val_loss', float]
                        val_accuracy = tree['val_accuracy', float]

                        writer.writerow([i, epoch, epoch_duration, val_loss, val_accuracy])
                        i += 1


def main_case1(infiles, outfile):
    writer = csv.writer(outfile)

    writer.writerow(['loss1', 'loss2', 'accuracy1', 'accuracy2', 'duration1', 'duration2'])

    losses = []
    accuracies = []
    durations  = []

    for tree in Tree.readiter(infiles, presorted=True):
        #for tree in tree('master'):
            for tree in tree('worker'):
                for tree in tree('model.fit'):
                    for tree in tree('epoch'):
                        epoch = tree['epoch', int]
                        accuracy = tree['val_accuracy', float]
                        loss = tree['val_loss', float]

                        duration = None
                        for tree in tree('batch'):
                            duration = tree.duration.total_seconds()

                        losses.append(loss)
                        accuracies.append(accuracy)
                        durations.append(duration)

    while len(losses) < 2:
        losses.append(None)

    while len(accuracies) < 2:
        accuracies.append(None)

    while len(durations) < 2:
        durations.append(None)

    writer.writerow([*losses, *accuracies, *durations])


def main_batch_model(infiles, presorted, outfile):
    writer = csv.writer(outfile)
    writer.writerow(['iteration', 'epoch', 'duration', 'loss_change', 'accuracy_change'])

    epochs = []
    durations = []
    losses = []
    accuracies = []

    for tree in Tree.readiter(infiles, presorted=presorted):
        #for tree in tree('master'):
            for tree in tree('worker'):
                for tree in tree('model.fit'):
                    for tree in tree('epoch'):
                        epoch = tree['epoch', int]

                        for tree in tree('batch'):
                            duration = tree.duration.total_seconds()
                            batch = tree['batch', int]
                            loss = tree['loss', float]
                            accuracy = tree['accuracy', float]

                            epochs.append(epoch)
                            durations.append(duration)
                            losses.append(loss)
                            accuracies.append(accuracy)

    indices = np.argsort(epochs)
    epochs = [epochs[i] for i in indices]
    durations = [durations[i] for i in indices]
    losses = [losses[i] for i in indices]
    accuracies = [accuracies[i] for i in indices]

    iterations = []
    epochs_ = []
    durations_ = []
    loss_changes = []
    accuracy_changes = []

    last_loss = None
    last_accuracy = None
    iteration = 0
    for epoch, duration, loss, accuracy in zip(epochs, durations, losses, accuracies):
        if last_loss is not None:
            iteration += 1
            iterations.append(iteration)
            epochs_.append(epoch)
            durations_.append(duration)
            loss_changes.append(loss - last_loss)
            accuracy_changes.append(accuracy - last_accuracy)

            writer.writerow([iteration, epoch, duration, loss - last_loss, accuracy - last_accuracy])

        last_loss = loss
        last_accuracy = accuracy

    epochs = epochs_
    durations = durations_

    print(f'{len(iterations) = }')
    print(f'{len(durations) = }')
    print(f'{len(loss_changes) = }')
    print(f'{len(accuracy_changes) = }')

    a, b, c = np.polyfit(iterations, durations, 2)
    print(f'def predict_batch_duration(batch: int) -> float:')
    print(f'    return {a} * batch ** 2 + {b} * batch + {c}')
    print(rf'$$\textrm{{dur}}(x) = \num{{{a:0.4e}}}x^2 + \num{{{b:0.4e}}}x + \num{{{c:0.4e}}}$$')

    a, b, c = np.polyfit(iterations, accuracy_changes, 2)
    print(f'def predict_batch_accuracy_change(batch: int) -> float:')
    print(f'    return {a} * batch ** 2 + {b} * batch + {c}')
    print(rf'$$\textrm{{\Delta acc}}(x) = \num{{{a:0.4e}}}x^2 + \num{{{b:0.4e}}}x + \num{{{c:0.4e}}}$$')

    a, b, c = np.polyfit(iterations, loss_changes, 2)
    print(f'def predict_batch_loss_change(batch: int) -> float:')
    print(f'    return {a} * batch ** 2 + {b} * batch + {c}')
    print(rf'$$\textrm{{\Delta loss}}(x) = \num{{{a:0.4e}}}x^2 + \num{{{b:0.4e}}}x + \num{{{c:0.4e}}}$$')


def cli():
    import argparse

    parser = argparse.ArgumentParser()

    parser.set_defaults(main=None)
    subparsers = parser.add_subparsers(required=True)

    batch_csv = subparsers.add_parser('batch_csv')
    batch_csv.set_defaults(main=main_batch_csv)
    batch_csv.add_argument('--outfile', '-o', default=sys.stdout, type=argparse.FileType('w'))
    batch_csv.add_argument('--presorted', action='store_true')
    batch_csv.add_argument('infiles', nargs='+', type=Path)

    epoch_csv = subparsers.add_parser('epoch_csv')
    epoch_csv.set_defaults(main=main_epoch_csv)
    epoch_csv.add_argument('--outfile', '-o', default=sys.stdout, type=argparse.FileType('w'))
    epoch_csv.add_argument('--presorted', action='store_true')
    epoch_csv.add_argument('infiles', nargs='+', type=Path)

    case1 = subparsers.add_parser('case1')
    case1.set_defaults(main=main_case1)
    case1.add_argument('--outfile', '-o', default=sys.stdout, type=argparse.FileType('w'))
    case1.add_argument('infiles', nargs='+', type=Path)

    batch_model = subparsers.add_parser('batch_model')
    batch_model.set_defaults(main=main_batch_model)
    batch_model.add_argument('--outfile', '-o', default=sys.stdout, type=argparse.FileType('w'))
    batch_model.add_argument('--presorted', action='store_true')
    batch_model.add_argument('infiles', nargs='+', type=Path)

    args = vars(parser.parse_args())
    main = args.pop('main')
    main(**args)


if __name__ == '__main__':
    cli()
