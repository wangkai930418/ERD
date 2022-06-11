import os.path as osp

import pickle as pkl
import PIL
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import ImageEnhance


##########################  3rd step: select 100 for test for CL training classes ##################

TOTAL_TASK_NUM=100
TASK_NUM=16
BASE_CLS_NUM=5
META_TEST_CLS_NUM=20
CL_TEST_IMG_NUM_PER_TASK=100
IMG_PER_TASK_TOTAL=600

if TASK_NUM>1:
    CLS_NUM_PER_TASK=(TOTAL_TASK_NUM-BASE_CLS_NUM-META_TEST_CLS_NUM)//(TASK_NUM-1)
else:
    BASE_CLS_NUM=80

######## we save this file in the current directory
ROOT_PATH = './'

with open(osp.join(ROOT_PATH, "all.pkl"),'rb') as h:
    all_data=pkl.load(h)

data=all_data['data']
label=all_data['label']

CL_data_split={}

for session in range(TASK_NUM):
    train_data=[]
    test_data=[]
    train_label=[]
    test_label=[]

    curr_task_data={}
    if session==0:
        task_id_range=range(BASE_CLS_NUM)
    else:
        task_id_range=range(BASE_CLS_NUM+(session-1)*CLS_NUM_PER_TASK,
                            BASE_CLS_NUM+session*CLS_NUM_PER_TASK)
    test_count={}
    for cls_id in task_id_range:
        test_count[cls_id]=0

    for idx in range(len(label)):
        if label[idx] in task_id_range:
            if test_count[label[idx]]<CL_TEST_IMG_NUM_PER_TASK:
                if session ==0:
                    test_label.append(label[idx])
                else:
                    test_label.append(label[idx]-(session-1)*CLS_NUM_PER_TASK-BASE_CLS_NUM)
                test_data.append(data[idx])
                test_count[label[idx]]+=1
            else:
                if session ==0:
                    train_label.append(label[idx])
                else:
                    train_label.append(label[idx]-(session-1)*CLS_NUM_PER_TASK-BASE_CLS_NUM)
                train_data.append(data[idx])
        else:
            continue
    curr_task_data['test_label']=test_label
    curr_task_data['test_data']=test_data
    curr_task_data['train_label']=train_label
    curr_task_data['train_data']=train_data
    CL_data_split[session]=curr_task_data

################### saving for meta testing 20 extra classes ###################
train_data=[]
train_label=[]
curr_task_data={}

task_id_range=range(TOTAL_TASK_NUM-META_TEST_CLS_NUM, TOTAL_TASK_NUM)
test_count={}
for cls_id in task_id_range:
    test_count[cls_id]=0

for idx in range(len(label)):
    if label[idx] in task_id_range:
        train_label.append(label[idx]-TOTAL_TASK_NUM+META_TEST_CLS_NUM)
        train_data.append(data[idx])
    else:
        continue

curr_task_data['data']=train_data
curr_task_data['label']=train_label
CL_data_split['meta_test']=curr_task_data

file_name='_'.join(('base',str(BASE_CLS_NUM),'task',str(TASK_NUM),'meta_test',str(META_TEST_CLS_NUM)))+'.pkl'

with open(osp.join(ROOT_PATH, file_name),'wb') as h:
    pkl.dump(CL_data_split,h)

