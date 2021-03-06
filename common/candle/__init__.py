from __future__ import absolute_import

#__version__ = '0.0.0'

#import from data_utils
from data_utils import load_csv_data
from data_utils import load_Xy_one_hot_data2
from data_utils import load_Xy_data_noheader

#import from file_utils
from file_utils import get_file

#import from default_utils
from default_utils import ArgumentStruct
from default_utils import Benchmark
from default_utils import str2bool
from default_utils import initialize_parameters
from default_utils import fetch_file
from default_utils import verify_path
from default_utils import keras_default_config
from default_utils import set_up_logger

from generic_utils import Progbar

#import from viz_utils
#from viz_utils import plot_history
#from viz_utils import plot_scatter

# import benchmark-dependent utils
import sys
if 'tensorflow.keras' in sys.modules:
    print ('Importing candle utils for keras')
    #import from keras_utils
    #from keras_utils import dense
    #from keras_utils import add_dense
    from keras_utils import build_initializer
    from keras_utils import build_optimizer
    from keras_utils import get_function
    from keras_utils import set_seed
    from keras_utils import set_parallelism_threads
    from keras_utils import PermanentDropout
    from keras_utils import register_permanent_dropout
    from keras_utils import LoggingCallback

    from solr_keras import CandleRemoteMonitor
    from solr_keras import compute_trainable_params
    from solr_keras import TerminateOnTimeOut

elif 'torch' in sys.modules:
    print ('Importing candle utils for pytorch')
    from pytorch_utils import set_seed
    from pytorch_utils import build_optimizer
    from pytorch_utils import build_activation
    from pytorch_utils import get_function
    from pytorch_utils import initialize
    from pytorch_utils import xent
    from pytorch_utils import mse
    from pytorch_utils import set_parallelism_threads # for compatibility

else:
    raise Exception('No backend has been specified.')


