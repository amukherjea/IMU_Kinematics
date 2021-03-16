import numpy as np
import os
import os.path as osp
from numpy import genfromtxt
import h5py

base_dir = 'Data'
data_fldr = 'Exp_data'
result_fldr = 'my_new_result'            # Define your result folder name

data_path = osp.join(base_dir, data_fldr)

## ACC 
path = osp.join(data_path,'Local_SensorData_LeftThigh.csv')
data = genfromtxt(path, delimiter=',')
data = data[1:,1:-1]
acc = data[:,:3]
gyro = data[:,3:]
acc = acc[np.newaxis,:] #Add new axis for gyro
gyro = gyro[np.newaxis,:]
acc_file = np.save('Left_seg1_acc.npy',acc)
gyr_file = np.save('Left_seg1_gyr.npy',gyro)
path = osp.join(data_path,'Local_SensorData_LeftLowerLeg.csv')
data = genfromtxt(path, delimiter=',')
data = data[1:,1:-1]
acc = data[:,:3]
gyro = data[:,3:]
acc = acc[np.newaxis,:]
gyro = gyro[np.newaxis,:]
acc_file = np.save('Left_seg2_acc.npy',acc)
gyr_file = np.save('Left_seg2_gyr.npy',gyro)
path = osp.join(data_path,'Local_SensorData_RightThigh.csv')
data = genfromtxt(path, delimiter=',')
data = data[1:,1:-1]
acc = data[:,:3]
gyro = data[:,3:]
acc = acc[np.newaxis,:]
gyro = gyro[np.newaxis,:]
acc_file = np.save('Right_seg1_acc.npy',acc)
gyr_file = np.save('Right_seg1_gyr.npy',gyro)
path = osp.join(data_path,'Local_SensorData_RightLowerLeg.csv')
data = genfromtxt(path, delimiter=',')
data = data[1:,1:-1]
acc = data[:,:3]
gyro = data[:,3:]
acc = acc[np.newaxis,:]
gyro = gyro[np.newaxis,:]
acc_file = np.save('Right_seg2_acc.npy',acc)
gyr_file = np.save('Right_seg2_gyr.npy',gyro)


acc_file = h5py.File('acc_file.h5','w')
acc_file.create_dataset('dataset_1', data=acc)
acc_file.close()
print(acc.shape,gyro.shape)

