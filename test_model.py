import numpy as np
import torch
import pickle
from math import pi
from torch.nn import *  # noqa
from torch.optim import *  # noqa
from torch.optim.lr_scheduler import *  # noqa
from nn_models.utils import prep_utils, data_utils, dloader_utils  # noqa
from nn_models.utils import log_utils, train_utils  # noqa
from nn_models.models.pure_conv import *  # noqa
from nn_models.models.pure_lstm import *  # noqa


def main(general_spec, model_spec, show_bar=True):
    np.random.seed(42)
    torch.random.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Parse arguments of the training script
    scheduler_spec = prep_utils.parse_scheduler(general_spec)
    model_type = general_spec['model_type']
    stage_spec = prep_utils.parse_stages(general_spec)
    h5path = general_spec['data_path']

    # Create logger and specify what to log
    log_spec = [general_spec['result_path'], general_spec['name'], general_spec['log_metrics'],
                general_spec['log_freq'], general_spec['check_freq']]
    logger = log_utils.TrainLogger(*log_spec)

    # Get device and parse model, optimizer etc
    device = prep_utils.get_device(show_bar)
    model = globals()[model_type](**model_spec).to(device)
    optimizer = globals()[general_spec['optim']](model.parameters(), lr=general_spec['lr'])
    scheduler = globals()[scheduler_spec['type']](optimizer, **scheduler_spec['args'])
    criterion = globals()[general_spec['loss']]()
