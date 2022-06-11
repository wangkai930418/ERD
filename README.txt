#########################################################

Here we attach the core codes which are used to reproduce 
the experimental results on CIFAR100 for our method ERD 
in Figure 4 of the main text.

********************* required packages *****************

All installed packages in our running environment are in 
requirements.txt, please check whether you have any conf-
licts if enconterring any problem.

****** 1st step: download dataset and create split ******

1,Pleas run get_cifar_data.py to download cifar dataset 
from torchvision.datasets automatically and it will be 
saved at "./data/cifar".

2,cd ./data/cifar
  Run "./data/cifar/create_cifar_split_auto.py"

        to create the 16-task cifar split.

****** 2nd step: run the experiments ********************
### cd back to the root directory 
cd ..
cd ..

### 
Run reproduce.sh to reproduce the CIFAR100 experiments of
Figure 4 in our main paper. 

### FT_cifar.py is the finetuning baseline.
### ERD_cifar.py is our method ERD implementation.


****** 3rd step: test the experiments *******************

Run test_cifar.sh to test the model performance.

### By default, we print out the test accuracy and save
### the results in "./results/" directory. 