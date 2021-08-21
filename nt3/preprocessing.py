from __future__ import print_function
from keras.utils import np_utils
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
import pandas as pd
import os, sys, pickle

def load_data(train_path, test_path, classes):
    print('Loading data...')
    df_train = (pd.read_csv(train_path,header=None).values).astype('float32')
    df_test = (pd.read_csv(test_path,header=None).values).astype('float32')
    print('done')

    print('df_train shape:', df_train.shape)
    print('df_test shape:', df_test.shape)

    seqlen = df_train.shape[1]

    df_y_train = df_train[:,0].astype('int')
    df_y_test = df_test[:,0].astype('int')

    Y_train = np_utils.to_categorical(df_y_train, classes)
    Y_test = np_utils.to_categorical(df_y_test, classes)

    df_x_train = df_train[:, 1:seqlen].astype(np.float32)
    df_x_test = df_test[:, 1:seqlen].astype(np.float32)

#        X_train = df_x_train.as_matrix()
#        X_test = df_x_test.as_matrix()

    X_train = df_x_train
    X_test = df_x_test

    scaler = MaxAbsScaler()
    mat = np.concatenate((X_train, X_test), axis=0)
    mat = scaler.fit_transform(mat)

    X_train = mat[:X_train.shape[0], :]
    X_test = mat[X_train.shape[0]:, :]

    f = open(os.path.join(os.path.dirname(train_path), "csv_input.pickled"), 'wb')
    pickle.dump(X_train, f)
    pickle.dump(Y_train, f)
    pickle.dump(X_test, f)
    pickle.dump(Y_test, f)
    f.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: %s <train_csv> <test_csv> <classes>" % sys.argv[0])
        sys.exit(1)

    print("Processing input from %s" % os.path.dirname(sys.argv[1]))
    load_data(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    print("Done!")
