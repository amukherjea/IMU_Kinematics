from nn_models.models.pure_conv import CustomConv1D
from nn_models.models.pure_lstm import CustomLSTM, rot6_to_rotmat
from _2_optimization.utils.optimization_utils import *

import torch
import numpy as np
import pickle
import os.path as osp
from pdb import set_trace as st     # Debugging tips


# Define your custom data
left_seg1_acc = np.load('Data/Exp_data/Knee/Left_seg1_acc.npy')
left_seg1_gyr = np.load('Data/Exp_data/Knee/Left_seg1_gyr.npy')
left_seg2_acc = np.load('Data/Exp_data/Knee/Left_seg2_acc.npy')
left_seg2_gyr = np.load('Data/Exp_data/Knee/Left_seg2_gyr.npy')

right_seg1_acc = np.load('Data/Exp_data/Knee/Right_seg1_acc.npy')
right_seg1_gyr = np.load('Data/Exp_data/Knee/Right_seg1_gyr.npy')
right_seg2_acc = np.load('Data/Exp_data/Knee/Right_seg2_acc.npy')
right_seg2_gyr = np.load('Data/Exp_data/Knee/Right_seg2_gyr.npy')

# TODO: Match your custom data with the data you used for training your model.



#######################################################################################################
# I will add some code lines that only consider left leg (you can repeat for the right leg) - Soyong  #
#######################################################################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float

# Add new dimension for norm (magnitude)
left_seg1_acc = np.concatenate((left_seg1_acc, np.linalg.norm(left_seg1_acc, axis=-1)[:, :, None]), axis=-1)
left_seg1_gyr = np.concatenate((left_seg1_gyr, np.linalg.norm(left_seg1_gyr, axis=-1)[:, :, None]), axis=-1)
left_seg2_acc = np.concatenate((left_seg2_acc, np.linalg.norm(left_seg2_acc, axis=-1)[:, :, None]), axis=-1)
left_seg2_gyr = np.concatenate((left_seg2_gyr, np.linalg.norm(left_seg2_gyr, axis=-1)[:, :, None]), axis=-1)

right_seg1_acc = np.concatenate((right_seg1_acc, np.linalg.norm(right_seg1_acc, axis=-1)[:, :, None]), axis=-1)
right_seg1_gyr = np.concatenate((right_seg1_gyr, np.linalg.norm(right_seg1_gyr, axis=-1)[:, :, None]), axis=-1)
right_seg2_acc = np.concatenate((right_seg2_acc, np.linalg.norm(right_seg2_acc, axis=-1)[:, :, None]), axis=-1)
right_seg2_gyr = np.concatenate((right_seg2_gyr, np.linalg.norm(right_seg2_gyr, axis=-1)[:, :, None]), axis=-1)

input_data = np.concatenate((left_seg1_acc, left_seg1_gyr, left_seg2_acc, left_seg2_gyr), axis=-1)
input_data = torch.from_numpy(input_data).to(device=device, dtype=dtype)


# TODO: Match your custom data axis with the data you used for training your model.



#######################################################################################################
# Angle prediction                                                                                    #
#######################################################################################################

# Load angle prediction model
angle_model_fldr = 'Data/my_result/Walking/Knee/my_angle_model'
with open(osp.join(angle_model_fldr, 'model_kwargs.pkl'), 'rb') as fopen:
    angle_model_kwargs = pickle.load(fopen)
angle_model = globals()['CustomConv1D'](**angle_model_kwargs) if angle_model_kwargs['model_type'] == 'CustomConv1D' \
                                                              else globals()['CustomLSTM'](**angle_model_kwargs)
angle_state_dict = torch.load(osp.join(angle_model_fldr, 'model.pt'))
angle_model.load_state_dict(angle_state_dict)
angle_model.to(device=device)

# Load angle normalization dict
angle_norm_dict = torch.load(osp.join(angle_model_fldr, 'norm_dict.pt'))['params']
for key, value in angle_norm_dict.items():
    angle_norm_dict[key] = value.to(device=device, dtype=dtype)

# Running angle prediction
with torch.no_grad():
    input_data_angle = (input_data - angle_norm_dict['x_mean']) / angle_norm_dict['x_std']    
    angle_model.eval()
    alpha = angle_model(input_data_angle)
    alpha = alpha * angle_norm_dict['y_std'] + angle_norm_dict['y_mean']

alpha = alpha.detach().cpu().double().numpy()       # Angle prediction from neural network


#######################################################################################################
# Orientation prediction                                                                              #
#######################################################################################################

# Load orientation prediction model
ori_model_fldr = 'Data/my_result/Walking/Knee/my_orientation_model/'
with open(osp.join(ori_model_fldr, 'model_kwargs.pkl'), 'rb') as fopen:
    ori_model_kwargs = pickle.load(fopen)
ori_model = globals()['CustomConv1D'](**ori_model_kwargs) if ori_model_kwargs['model_type'] == 'CustomConv1D' \
                                                          else globals()['CustomLSTM'](**ori_model_kwargs)

ori_state_dict = torch.load(osp.join(ori_model_fldr, 'model.pt'))
ori_model.load_state_dict(ori_state_dict)
ori_model.to(device=device)

# Load orientation normalization dict
ori_norm_dict = torch.load(osp.join(ori_model_fldr, 'norm_dict.pt'))['params']
for key, value in ori_norm_dict.items():
    ori_norm_dict[key] = value.to(device=device, dtype=dtype)

# Running orientation prediction
with torch.no_grad():
    input_data_ori = (input_data - ori_norm_dict['x_mean']) / ori_norm_dict['x_std']
    ori_model.eval()
    ori_pred = ori_model(input_data_ori)

ori_pred = ori_pred.detach().cpu().double().numpy()     # Rotation matrix form of orientation prediction




#######################################################################################################
# Optimization                                                                                        #
#######################################################################################################
std_ratio = 1
weight = 0.5
start, end = 0, -1
## Run optimization
gyro_data = np.concatenate([left_seg1_gyr[:, :, :-1], left_seg2_gyr[:, :, :-1]], axis=-1)
beta = optimization_demo(ori_pred, gyro_data, joint='Knee', leg='Left')

# Get theta from alpha and beta
# TODO: Please save std_ratio and weight before running this
beta = (beta - beta.mean(axis=1)[:, None]) * std_ratio + alpha.mean(axis=1)[:, None]
print(beta.shape, alpha.shape)
theta = weight * alpha + (1 - weight) * beta
np.save('pred_angles.npy',theta)



# TODO: Analyze your result