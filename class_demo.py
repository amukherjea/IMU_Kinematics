# ------------------------------------
#
# Kinematics project demo code
# for Wearable Health Technology
#
# 2021 Spring Semester
# Professor Eni Halilaj
#
# ------------------------------------

from nn_models.models.pure_conv import CustomConv1D
from nn_models.models.pure_lstm import CustomLSTM
from nn_models.utils.train_cfg import get_model_spec
from nn_models.train_model import main as run_train

from _2_optimization._22_get_optimization_parameters import *
from _2_optimization._23_run_optimization import run_optimization

import torch
import numpy as np

from tqdm import tqdm
import os
import os.path as osp
from copy import copy




# ======================================
# Step1: Basic configuration declaration
# =====================================

# Folder structure
base_dir = 'Data'
data_fldr = '2_Processed'
result_fldr = 'my_result'            # Define your result folder name

data_path = osp.join(base_dir, data_fldr)
result_path = osp.join(base_dir, result_fldr)

# Estimation
activity = 'Walking'        # Target activity to estimate joint angle   choose 'Walking' or 'Running'
joint = 'Knee'              # Target joint to estimate angle        choose 'Knee', 'Hip', or 'Ankle'




# =====================================
# Step2: Neural network configuration declaration
# =====================================

## 1) Angle prediction model

"""Here is to define neural network model and training configuration.
You can manually change the values as far as they follow the regulation.
"""

params_angle = {'activity': activity, 'prediction': 'angle', 'meta_arch': 'conv'}
general_spec_angle, model_spec_angle = get_model_spec(params_angle)
my_general_spec, my_model_spec = {}, {}

# Neural network model configuration
my_general_spec['model_type'] = 'CustomConv1D'              # Your model type; choose CustomLSTM or CustomConv1D
my_general_spec['name'] = 'my_angle_model'                  # Your model name   ex) 'my_model'
my_model_spec['layers'] = [70]                              # Size of hidden layers     ex) [70, 70, 70]

# --------- Conv1D network configuration --------- #
my_model_spec['conv_dropout'] = [0, 0, 0]                   # Probability of dropout for Conv network       ex) [0, 0, 0]
my_model_spec['lin_dropout'] = []                           # Probability of dropout for linear layer       ex) []
my_model_spec['window'] = 41                                # Size of window (kernel) for Conv network      ex) 41
my_model_spec['conv_activation'] = ['ReLU']                 # Type of activation function for Conv network  ex) ['ReLU']
my_model_spec['lin_activation'] = ['ReLU']                  # Type of activation function for linear network    ex) ['ReLU']

# --------- LSTM network configuration --------- #
my_model_spec['dropout'] = [0, 0, 0]                        # Probability dropout       ex) [0, 0, 0]
my_model_spec['bidir'] = True                               # Bidirectional configuration of LSTM network, choose True or False


# Training configuration
my_general_spec['lr'] = 1e-3                                # Leraning rate     ex) 1e-2
my_general_spec['lr_decay'] = 0.9997                        # Learning rate decay   ex) 0.9997
my_general_spec['seq_length'] = [200, 400]                  # Sequence length of data fed into network      ex) [200, 400]
my_general_spec['num_iter'] = [5000, 5000]                  # Number of iteration       ex) [5000, 5000]
my_general_spec['batch_size'] = [16, 16]                    # Size of mini-batch        ex) [16, 16]


# Update configurations as given values
for my_key, my_value in my_general_spec.items():
    general_spec_angle.update({my_key: my_value})

for my_key, my_value in my_model_spec.items():
    model_spec_angle.update({my_key: my_value})


## 2) Orientation prediction model

"""Here is to define orientation prediction model.
Since we will conduct optimization loop to fine-tune the prediction,
neural network and training configuration are set as default values.
You may change the values, but the result will vary significantly.
"""

params_orient = {'activity': activity, 'prediction': 'orientation', 'meta_arch': 'lstm'}
general_spec_orient, model_spec_orient = get_model_spec(params_orient)

# Setup few things for orientation model
my_general_spec['model_type'] = 'CustomLSTM'
my_general_spec['name'] = 'my_orientation_model'
my_model_spec['layers'] = [70]
my_model_spec['dropout'] = [0, 0, 0]
my_model_spec['bidir'] = True


# Update configurations as given values
for my_key, my_value in my_general_spec.items():
    general_spec_orient.update({my_key: my_value})

for my_key, my_value in my_model_spec.items():
    model_spec_orient.update({my_key: my_value})




# =====================================
# Step3: Train network
# =====================================

"""Here is to run neural network prediction to estimate angle and orientation."""

if joint == 'Hip':
    general_spec_angle['inp'] = ['pelvis', 'thigh']; general_spec_angle['outp'] = ['hip']
    general_spec_orient['inp'] = ['pelvis', 'thigh']; general_spec_orient['outp'] = ['hip']
elif joint == 'Knee':
    general_spec_angle['inp'] = ['thigh', 'shank']; general_spec_angle['outp'] = ['knee']
    general_spec_orient['inp'] = ['thigh', 'shank']; general_spec_orient['outp'] = ['knee']
elif joint == 'Ankle':
    general_spec_angle['inp'] = ['shank, foot']; general_spec_angle['outp'] = ['ankle']
    general_spec_orient['inp'] = ['shank, foot']; general_spec_orient['outp'] = ['ankle']
else:
    AssertionError, "Wrong configuration for joint name {}".format(joint)

final_result_path = osp.join(result_path, activity, joint) + '/'
general_spec_angle['result_path'] = final_result_path
general_spec_orient['result_path'] = final_result_path


# Running angle model training
try:
    run_train(copy(general_spec_angle), copy(model_spec_angle))
except:
    print('Folder already exists! Please rename result path or remove existing folder.')
    # You can keep running new training by deleting existing folder as below:
    # os.system('rm -rf {}'.format(final_result_path))
    # run_train(copy(general_spec_angle), copy(model_spec_angle))

# Running orientation model training
try:
    run_train(copy(general_spec_orient), copy(model_spec_orient))
except:
    print('Folder already exists! Please rename result path or remove existing folder.')
    # You can keep running new training by deleting existing folder as below:
    # os.system('rm -rf {}'.format(final_result_path))
    # run_train(copy(general_spec_orient), copy(model_spec_orient))




# =====================================
# Step4: Optimize orientation
# =====================================


## 1) Run optimization code

"""Here is to run optimization loop to fine-tune orientation prediction.
The optimization algorithm requires input IMU gyroscope value.
"""

opt_fldr = 'my_optimization'            # Define your optimization folder name

pred_path = osp.join(final_result_path, general_spec_orient['name'], 'predictions')
pred_file_orient= osp.join(pred_path, 'y_pred_test.npy')
pred_file_gyro = osp.join(pred_path, 'x_test.npy')

pred_orient= np.load(pred_file_orient)
input_gyro = np.load(pred_file_gyro)[:, :, [4, 5, 6, 12, 13, 14]]

opt_output_path = osp.join(final_result_path, opt_fldr)
opt_output_file = 'optim_angle.npy'
os.makedirs(opt_output_path, exist_ok=True)

if osp.exists(osp.join(opt_output_path, opt_output_file)):
    print("Optimization result exists! Skip this stage.")
else:
    print("Starting optimization... Result will be saved in %s"%(osp.join(opt_output_path, opt_output_file)))
    run_optimization(pred_orient, input_gyro, opt_output_path, joint, result_file=opt_output_file)



## 2) Get best weight

"""Here is to get best weight and scale factor.
Assuming angle prediction from neural network as \alpha, angle calculated by segments orientation as \beta.
Final angle estimation value \theta will be defined as:
\theta = w*\alpha + (1-w)*SF*\beta

And we get the optimal weight w and scale factor SF from validation set.
"""

pred_angle_path = osp.join(final_result_path, general_spec_angle['name'], 'predictions')
gt_angle = np.load(osp.join(pred_angle_path, 'y_val.npy'))
pred_angle = np.load(osp.join(pred_angle_path, 'y_pred_val.npy'))

pred_orient_path = osp.join(final_result_path, general_spec_orient['name'], 'predictions')
gt_orient = np.load(join(pred_orient_path, 'y_val.npy'))

std_ratio = calculate_std_ratio(gt_angle, gt_orient, joint)
weight = calculate_weight(gt_angle, pred_angle, gt_orient, std_ratio, joint)




# =====================================
# Step4: Analyze results
# =====================================

"""Here is your code to analyze the result.

   Notation.
        alpha = angle directly predicted by neural network
        beta = angle calculated by optimized segmentations orientation
        theta = angle after weighted sum
        theta_gt = groundtruth angle from IK (marker)
"""

alpha_path = osp.join(final_result_path, general_spec_angle['name'], 'predictions')
alpha = np.load(osp.join(alpha_path, 'y_pred_test.npy'))

beta_path = osp.join(final_result_path, opt_fldr)
beta = np.load(osp.join(beta_path, opt_output_file))
scaled_beta = (beta - beta.mean(1)[:, None]) * std_ratio + alpha.mean(1)[:, None]

theta = weight * alpha + (1-weight) * scaled_beta

theta_gt = np.load(osp.join(alpha_path, 'y_test.npy'))

# Begin Here
