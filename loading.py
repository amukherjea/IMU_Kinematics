import numpy as np

left_seg1_acc = np.load('Data/Exp_data/Knee/Left_seg1_acc.npy')
left_seg1_gyr = np.load('Data/Exp_data/Knee/Left_seg1_gyr.npy')
left_seg2_acc = np.load('Data/Exp_data/Knee/Left_seg2_acc.npy')
left_seg2_gyr = np.load('Data/Exp_data/Knee/Left_seg2_gyr.npy')

right_seg1_acc = np.load('Data/Exp_data/Knee/Right_seg1_acc.npy')
right_seg1_gyr = np.load('Data/Exp_data/Knee/Right_seg1_gyr.npy')
right_seg2_acc = np.load('Data/Exp_data/Knee/Right_seg2_acc.npy')
right_seg2_gyr = np.load('Data/Exp_data/Knee/Right_seg2_gyr.npy')

#for both acc and gyr shape is [SubjectxFramex3(x,y,z)]

print(left_seg1_acc.shape)
