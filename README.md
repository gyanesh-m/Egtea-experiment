# Egtea-experiment
EGTEA+ dataset include three different train/test splits for action recognition as train/test_split(1-3). This model takes a small subset of dataset from the EGTEA Gaze+ dataset. This dataset belongs to split 1 and includes cropped videos corresponding to the folder P17-R04-ContinentalBreakfast under the ContinentalBreakfast category.

Because of the constraint of resources, I have trained the model on a custom CNN architecture similar to VGGNet.
This model covers only one stream as of now which is **single frame**.

The dataset used is available [here](https://drive.google.com/file/d/1PgGLO81rXCMoPIqR9Pu-S0MLxxIm51Ob/view?usp=sharing).

The logs of the training session is available [here](https://www.floydhub.com/api/v1/resources/MpStApfiMXE26V3zqeZYrd?content=true) and model weights can be downloaded from [here](https://www.floydhub.com/api/v1/resources/MBCFXTUse5UjUESgZKKT37/weights.23-loss0.68-acc0.81-.hdf5?content=true&rename=weights23-loss068-acc081-hdf5). 
Also the plots from the training session are as follows:

![accuracy](https://github.com/gyanesh-m/Egtea-experiment/blob/master/images/acc.png)

![val-acc](https://github.com/gyanesh-m/Egtea-experiment/blob/master/images/val-acc.png)

![loss](https://github.com/gyanesh-m/Egtea-experiment/blob/master/images/loss.png)

![val-loss](https://github.com/gyanesh-m/Egtea-experiment/blob/master/images/val-loss.png)

