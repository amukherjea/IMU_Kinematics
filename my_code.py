import numpy as np
import torch
import pickle
from math import pi
from torch.nn import *  # noqa
from torch.optim import *  # noqa
from torch.optim.lr_scheduler import *  # noqa
#from nn_models.utils import prep_utils, data_utils, dloader_utils  # noqa
#from nn_models.utils import log_utils, train_utils  # noqa
from nn_models.models.pure_conv import *  # noqa
from nn_models.models.pure_lstm import *  # noqa

model.load_state_dict(torch.load('9.t2'))