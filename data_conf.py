import numpy as np
import os
import os.path as osp
from numpy import genfromtxt
import h5py

base_dir = 'Data/Vary_Length_Trials/Vary_Length_Trials'
name_dir = 'Ankita/ankita_varylength_1/'
result_fldr = 'my_new_result'            # Define your result folder name

pos = ['high','low','middle']
locations = ['high_BackLeftShank_Local','high_BackLeftThigh_Local','low_BackLeftShank_Local','low_BackLeftThigh_Local','middle_BackLeftShank_Local','middle_BackLeftThigh_Local']

data_path = osp.join(base_dir, name_dir)
result_path = osp.join('Data',result_fldr)
count = 0
i =0
for loc in range(0,len(locations),2):
    path = osp.join(data_path,(locations[loc]+'.csv'))
    data = genfromtxt(path, delimiter=',')
    data = data[1:,1:-1]
    acc = data[:,:3]
    gyro = data[:,3:]
    acc = acc[np.newaxis,:] #Add new axis for gyro
    gyro = gyro[np.newaxis,:]
    res_path = osp.join(result_path,('Left_seg1_acc_'+pos[i]+'.npy'))
    acc_file = np.save(res_path,acc)
    res_path = osp.join(result_path,('Left_seg1_gyr_'+pos[i]+'.npy'))
    gyr_file = np.save(res_path,gyro)

    path = osp.join(data_path,(locations[loc+1]+'.csv'))
    data = genfromtxt(path, delimiter=',')
    data = data[1:,1:-1]
    acc = data[:,:3]
    gyro = data[:,3:]
    acc = acc[np.newaxis,:] #Add new axis for gyro
    gyro = gyro[np.newaxis,:]
    res_path = osp.join(result_path,('Left_seg2_acc_'+pos[i]+'.npy'))
    acc_file = np.save(res_path,acc)
    res_path = osp.join(result_path,('Left_seg2_gyr_'+pos[i]+'.npy'))
    gyr_file = np.save(res_path,gyro)
    i+=1