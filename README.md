# IMU_Kinematics
This project uses Intertial Measurement Unit (IMU) data to predict Joint angles using Deep Learning Models (CNN and LSTM). 
 # Installation
   ## Use pretrained model to run the data
   1. ### Clone the repository
        ```
        git clone https://github.com/amukherjea/IMU_Kinematics.git
        ```
   2. Download Data folder from https://drive.google.com/drive/folders/1zjZtYtb_GI7DjsFT1ksmGwf4FcZ7NbS2?usp=sharing
   3. Place this Data folder in the parent folder (IMU kinematics)
   4. Run the gen_var.py to generate sensor data files from .csv file containing consolidated data
   5. Run main.py to use the pretrained model and optimization on the sensor data
   6. Find the predicted angles .npy files in Data/my_new_results/Results 
 

