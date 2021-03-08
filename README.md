# JointAnglePrediction_JOB
​
 # Installation:
 ## 1. Download source code via git clone
 ```bash
 # Clone the repository
  git clone https://github.com/CMU-MBL/JointAnglePrediction_JOB.git
  ```
​
  ## 2. Set up data folder structure
  ```bash
  cd JointAnglePrediction_JOB && mkdir Data
  mv <your extracted data folder> ./Data/
  ```
  or you can put your data at ```SomeDir``` and soft-link to the directory by:
  ```bash
  cd JointAnglePrediction_JOB
  ln -s <dir to SomeDir> ./Data
  ```
  ```
  # Desired data folder structure
  JointAnglePrediction_JOB | Data  | 1_Extracted           | walking/running_meta.h5
                                   | 2_Processed           | walking/running_data.h5
  ```
  The scripts will automatically generate result folders under ```Data```. But if you have your own preferrence in data structure, please go to the script and modify each path values.
  
  ## 3. Install requirement libraries
  You can install all required libraries by running:
  ```bash
  pip install -r requirements.txt
  ```
  However, we recommend you to create new virtual environment (such as conda) and install those requirements. But in any case your workstation has problem with running or installing the libraries, please report it on Issues tab.
  
  # How to run (Demo version):
  This demo version of the code allows you to run the framework using your own IMU data. The following steps are required.
  ## 1. Preparation
  IMU data (acceleration and angular velocity) from two segments and ground truth joint angle (optional), as well as the trained model of angle and orientation are needed. In the model folder, both ```model.pt``` (pretrained model), ```model_kwargs.pkl``` (model key arguments), and ```norm_dict.pt``` (normalization dictionary) should exist. Which means that your custom data structure should look like
```
Your root-path          | <Left or Right>_seg1_acc.npy
                        | <Left or Right>_seg2_acc.npy
                        | <Left or Right>_seg1_gyr.npy
                        | <Left or Right>_seg2_gyr.npy
                        | <Left or Right>_mocap_angle.npy (optional)

Your angle model-path   | model.pt
                        | model_kwargs.pkl
                        | norm_dict.pt
                        
Your orient model-path  | model.pt
                        | model_kwargs.pkl
                        | norm_dict.pt
```
  
  ## 2. Run the code
  ```bash
  python demo.py --joint <the type of joint ('Knee', 'Hip', 'Ankle') \
                 --activity <the type of activity ('Walking', 'Running') \
                 --root-path <path to the folder containing your data> \
                 --angle-model-fldr <folder path of angle prediction model> \
                 --ori-model-fldr <folder path of orientation prediction model> \
                 --result-fldr <folder to save the result files> \
                 --use-cuda <cuda configuration (True, False)>
  ```
  
  # How to run (Entire Framework):
  ## 1. Data preprocessing  
  ```bash
  # Given marker cluster data, this piece of code will create coordinate systems and generate simulated inertial data.
  python 0_preprocessing/01_preproc_dataset.py
  ```
  
  ```bash
  # Given processed dataset of inertial data, this script checks for unusual features in dataset and excludes those subjects. 
  # Refer to Calgary_issue_report.pdf for examples of checks
  python 0_preprocessing/02_check_dataset.py
  ```
  
  ## 2. Get best neural network model
  ```bash
  # Given processed and checked dataset, this script trains both CNNs and LSTMs utilizing hyperparameter optimization to predict joint kinematics.
  # Hyperopt sweeps over given sets of parameters, and each evaluation tries a different combination of those parameters.
  python _1_nn_hyperopt_training/11_optimize_hyperparams.py
  ```
  
  ```bash
  # This script compiles the model results from optimizing the hyperparameters and outputs an Excel file to compare the different performances.
  python _1_nn_hyperopt_training/12_summarize_results.py
  ```
  
  ```bash
  # This script compares the performances of the model results and saves the best performing model configuration in a separate directory for use in the framework.
  python _1_nn_hyperopt_training/13_get_best_results.py
  ```
  
  ## 3. Run optimization
  ```bash
  # This script predicts orientation of sensors with fixed hyperparameters of neural networks. The result will be saved in 5_Optimization/NN_Prediction folder
  python _2_optimization/21_predict_orientation.py
  ```
  
  ```bash
  # This scripts calculate priors for weighted sum (weight, scale factor) using validation data.
  python _2_optimization/22_get_optimization_parameters.py
  ```
  
  ```bash
  # This script runs top-down optimization which minimizes reconstruction error of angular velocity data
  python _2_optimization/23_run_optimization.py
  ```
  
  ```bash
  # This script calculates performance of optimization and shows it on the terminal screen
  python _2_optimization/24_get_final_results.py
  ```
# IMU_Kinematics
# IMU_Kinematics
# IMU_Kinematics
# IMU_Kinematics
