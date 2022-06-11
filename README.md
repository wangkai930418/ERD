# ERD for IML
 code for our CVPRW 2022 paper [Incremental Meta-Learning via Episodic Replay Distillation for Few-Shot Image Recognition](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/html/Wang_Incremental_Meta-Learning_via_Episodic_Replay_Distillation_for_Few-Shot_Image_Recognition_CVPRW_2022_paper.html) by [Kai Wang](https://wangkai930418.github.io/), [Xialei Liu](https://xialeiliu.github.io/), [Andrew D. Bagdanov](https://scholar.google.com/citations?user=_Fk4YUcAAAAJ&hl=en), [Luis Herranz](http://www.lherranz.org/), Shangling Jui, and [Joost van de Weijer](http://www.cvc.uab.es/LAMP/joost/).

Our supplementary material is also attached here as [supp.pdf](https://github.com/wangkai930418/ERD/blob/c72da697da5378026eff51c359984d1cefe359ab/supp.pdf).

## required packages

All installed packages in our running environment are in *requirements.txt*, please check whether you have any conflicts if enconterring any problem.

### 1st step: download dataset and create split

1,Pleas run 

```
get_cifar_data.py
```

to download cifar dataset 
from torchvision.datasets automatically and it will be 
saved at *./data/cifar*.

2, Run
```
cd ./data/cifar
./data/cifar/create_cifar_split_auto.py
```
to create the 16-task cifar split.

### 2nd step: run the experiments 

cd back to the root directory. Then run

```
cd ../..
reproduce.sh
```
to reproduce the CIFAR100 experiments in our paper. 

*FT_cifar.py* is the finetuning baseline.
*ERD_cifar.py* is our method ERD implementation.


### 3rd step: test the experiments

Run 
```
test_cifar.sh
```
to test the model performance.

By default, we print out the test accuracy and save the results in *./results/* directory. 

### **REMIND ME** if you need further information and details on other datasets since I may forget to update the repositories later.

### Our bibtex is:
```
@InProceedings{Wang_2022_CVPR,
    author    = {Wang, Kai and Liu, Xialei and Bagdanov, Andrew D. and Herranz, Luis and Jui, Shangling and van de Weijer, Joost},
    title     = {Incremental Meta-Learning via Episodic Replay Distillation for Few-Shot Image Recognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {3729-3739}
}
```