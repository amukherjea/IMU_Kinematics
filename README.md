# IMU_Kinematics
This project uses Intertial Measurement Unit (IMU) data to predict Joint angles using Deep Learning Models (CNN and LSTM). 
 # Installation
   ## Set up environment
   1. install conda
   2. conda create --name kinematics python=3.8
   3. conda activate kinematics
   4. ``` pip install -r requirements.txt ```
   5. Clone the repository
      ```
      git clone https://github.com/amukherjea/IMU_Kinematics.git
      ```
   ## Use pretrained model to run the data
   1. Download Data folder from https://drive.google.com/drive/folders/1zjZtYtb_GI7DjsFT1ksmGwf4FcZ7NbS2?usp=sharing
   2. Place this Data folder in the parent folder (IMU kinematics)
   3. cd IMU_Kinematics
   4. Run the gen_var.py to generate sensor data files from .csv file containing consolidated data
   5. Run main.py to use the pretrained model and optimization on the sensor data
   6. Find the predicted angles .npy files in Data/my_new_results/Results 
 
