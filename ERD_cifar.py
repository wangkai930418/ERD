import argparse
import os.path as osp
import os

import copy
import pickle as pkl
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from dataloader.samplers import Categories_Sampler_Exem_Distill_with_args as CategoriesSampler
from dataloader.samplers import Hard_Mine_Sampler
from models.protonet import ProtoNet
from torch.utils.data import DataLoader
from utils import (
    pprint,
    set_gpu,
    ensure_path,
    make_path,
    Averager,
    Timer,
    count_acc,
    compute_confidence_interval,
    save_model,
    pairwise_distances,
)




def train(epoch, model, model_old, train_loader, optimizer,args):
    model.train()
    model_old.eval()
    tl = Averager()
    ta = Averager()

    label = torch.arange(args.way).repeat(args.query)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)

    NUM_SHOT=args.shot+args.query
    NUM_WAY=2*args.way
    IMG_NUM=args.way*NUM_SHOT
    for i, batch in enumerate(train_loader, 1):
        if torch.cuda.is_available():
            data, ori_lbl = [_.to(device) for _ in batch]
        else:
            data, ori_lbl = batch[0],batch[1]

        encoded_data=model.encode(data)
        old_encoded_data=model_old.encode(data)

        prev_task_encoded_data=encoded_data.reshape(NUM_SHOT,NUM_WAY,-1)[:,:args.way,:].reshape(IMG_NUM,-1)
        curr_task_encoded_data=encoded_data.reshape(NUM_SHOT,NUM_WAY,-1)[:,args.way:,:].reshape(IMG_NUM,-1)

        prev_task_old_encoded_data=old_encoded_data.reshape(NUM_SHOT,NUM_WAY,-1)[:,:args.way,:].reshape(IMG_NUM,-1)
        curr_task_old_encoded_data=old_encoded_data.reshape(NUM_SHOT,NUM_WAY,-1)[:,args.way:,:].reshape(IMG_NUM,-1)

        ############################# current task prediction #######################################
        p = args.shot * args.way
        enc_data_shot, enc_data_query= curr_task_encoded_data[:p], curr_task_encoded_data[p:]
        enc_proto = enc_data_shot.reshape(args.shot, args.way, -1).mean(dim=0)
        logits=model.decode(enc_proto,enc_data_query)
        cls_loss = F.cross_entropy(logits, label)

        ################### previours task prediction ##############
        enc_data_shot, enc_data_query= prev_task_encoded_data[:p], prev_task_encoded_data[p:]
        enc_proto = enc_data_shot.reshape(args.shot, args.way, -1).mean(dim=0)
        prev_task_curr_logits=model.decode(enc_proto,enc_data_query)
        prev_task_curr_logits=F.softmax(prev_task_curr_logits)

        enc_data_shot, enc_data_query= prev_task_old_encoded_data[:p], prev_task_old_encoded_data[p:]
        enc_proto = enc_data_shot.reshape(args.shot, args.way, -1).mean(dim=0)

        prev_task_old_logits=model.decode(enc_proto,enc_data_query)
        prev_task_old_logits=F.softmax(prev_task_old_logits)
        
        kl_div_prev_task=(prev_task_old_logits.clamp(min=1e-4) * (prev_task_old_logits.clamp(min=1e-4)
                    / prev_task_curr_logits.clamp(min=1e-4)).log()).sum()/len(prev_task_old_logits)

        ############################## current task alignment ################################ 
        curr_task_new_logits=model.decode(enc_proto,curr_task_encoded_data)
        curr_task_new_logits=F.softmax(curr_task_new_logits)
        curr_task_old_logits=model.decode(enc_proto,curr_task_old_encoded_data)
        curr_task_old_logits=F.softmax(curr_task_old_logits)

        kl_div_curr_task=(curr_task_old_logits.clamp(min=1e-4) * (curr_task_old_logits.clamp(min=1e-4)
                    / curr_task_new_logits.clamp(min=1e-4)).log()).sum()/len(curr_task_old_logits)

        ############################################################## losses
        loss = cls_loss + kl_div_prev_task*args.lambda_distill+kl_div_curr_task*args.lambda_align
        ##############################################################
        acc = count_acc(logits, label)

        if i % 50 == 0:
            print(
                "epoch {}, train {}/{}, loss={:.4f}, cls_loss={:.4f}, kl_div_prev={:.4f},"
                "kl_div_curr={:.4f}, acc={:.4f}".format(
                    epoch, i, len(train_loader), loss.item(), \
                    cls_loss.item(), kl_div_prev_task.item(), kl_div_curr_task.item(),  acc
                )
            )

        tl.add(loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return tl, ta


def hard_mine(loader, model, args):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                data, ori_lbl, paths = batch[0].cuda(), batch[1].cuda(), batch[2]
            else:
                data, ori_lbl, paths = batch[0], batch[1], batch[2]
            encoded_data = model.encode(data)
            proto = encoded_data.mean(0)
            dist = torch.sum((encoded_data - proto) ** 2, dim=1)
            topk_indx = torch.topk(-dist, args.EXEM_NUM)[1]
            path_list = []
            label_list = []
            for i in range(args.EXEM_NUM):
                label_list.append(ori_lbl[0].item())
                path_list.append(paths[topk_indx[i]].numpy())
        return path_list, label_list


def main():
    THIS_PATH = osp.dirname(__file__)
    if args.dataset=='CIFAR100':
        ROOT_PATH = osp.join(THIS_PATH, "data/cifar")
        TOTAL_TASK_NUM = 100
        META_TEST_CLS_NUM = 20
        origin_path = 'logs/CIFAR100-ProtoNet_' + str(args.shot) + \
                      '_15_5_5_50_0.8_0.001_1_' + str(args.reverse) + \
                      '_1600_0.01_False_False_' + str(args.model) + '_' + str(args.TASK_NUM) + '_ft'

    TASK_NUM=args.TASK_NUM
    BASE_CLS_NUM=args.BASE_CLS_NUM

    if TASK_NUM > 1:
        CLS_NUM_PER_TASK = (TOTAL_TASK_NUM - BASE_CLS_NUM - META_TEST_CLS_NUM) // (TASK_NUM - 1)
    else:
        BASE_CLS_NUM = TOTAL_TASK_NUM-META_TEST_CLS_NUM

    if args.dataset=='CIFAR100':
        file_name = '_'.join(('base', str(BASE_CLS_NUM),
                              'task', str(TASK_NUM), 'meta_test',
                              str(META_TEST_CLS_NUM))) + '.pkl'

    with open(osp.join(ROOT_PATH, file_name), 'rb') as h:
        all_data = pkl.load(h)
    exem_all_data = copy.deepcopy(all_data)
    model = ProtoNet(args)
    ckpt_name = '_'.join(('session', str(0), 'base', str(BASE_CLS_NUM), 'task', str(TASK_NUM),
                          'meta_test', str(META_TEST_CLS_NUM), 'method', 'ft'))

    state_dict = torch.load(osp.join(origin_path, ckpt_name + ".pth"), map_location='cpu')
    model.load_state_dict(state_dict['params'])
    exem_dict = {}

    for session in range(1, TASK_NUM):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            model = model.to(device)

        if session >1:
            prev_exem_file_name = '_'.join(('session', str(session - 2), 'exem', str(args.EXEM_NUM))) + '.pkl'
            prev_exem_file_path =args.save_path
            with open(osp.join(prev_exem_file_path, prev_exem_file_name), 'rb') as h:
                exem_dict = pkl.load(h)
        ##################### hard mining for exemplars #######################

        prev_task_data = exem_all_data[session-1]
        prev_train_label = prev_task_data['train_label']
        prev_train_data = prev_task_data['train_data']

        prev_set = Hard_Mine_Dataset("test", args, prev_train_data, prev_train_label)

        curr_task_exem_paths = []
        curr_task_exem_label = []
        for class_id in range(CLS_NUM_PER_TASK):
            prev_sampler = Hard_Mine_Sampler(
                prev_set.label, class_id, 1, CLS_NUM_PER_TASK, 250, 250
            )
            prev_loader = DataLoader(
                dataset=prev_set, batch_sampler=prev_sampler, num_workers=8, pin_memory=True,
            )

            path_list, label_list = hard_mine(prev_loader, model, args)
            curr_task_exem_paths += path_list
            curr_task_exem_label += label_list

        exem_dict[session-1] = (curr_task_exem_paths, curr_task_exem_label)
        exem_file_name = '_'.join(('session', str(session-1), 'exem', str(args.EXEM_NUM))) + '.pkl'
        with open(osp.join(args.save_path, exem_file_name), 'wb') as h:
            pkl.dump(exem_dict, h)

        #####################################
        if session > 1:
            ckpt_name = '_'.join(('session', str(session - 1), 'base', str(BASE_CLS_NUM), 'task', str(TASK_NUM),
                                  'meta_test', str(META_TEST_CLS_NUM), 'method', str(args.method)))

            state_dict = torch.load(osp.join(args.save_path, ckpt_name + ".pth"), map_location='cpu')
            model.load_state_dict(state_dict['params'])

        #####################################

        curr_task_data = all_data[session]
        train_label = curr_task_data['train_label']
        train_data = curr_task_data['train_data']

        for j in range(len(train_label)):
            train_label[j] += (BASE_CLS_NUM + (session - 1) * CLS_NUM_PER_TASK)

        comb_data = train_data
        comb_label = train_label
        for exem_id in range(session):
            exem_data, exem_label = exem_dict[exem_id]
            if exem_id > 0 :
                for j in range(len(exem_label)):
                    exem_label[j] += (BASE_CLS_NUM + (exem_id - 1) * CLS_NUM_PER_TASK)

            comb_data = comb_data + exem_data
            comb_label = comb_label + exem_label

        batch_size = args.batch_size

        trainset = Dataset("train", args, comb_data, comb_label, location=args.location)
        train_sampler = CategoriesSampler(
            trainset.label, batch_size, args.way, args.shot, args.query, \
                args, cls_per_task=CLS_NUM_PER_TASK,
        )
        train_loader = DataLoader(
            dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        if args.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.step_size, gamma=args.gamma
            )

        ###########################################
        model_old = copy.deepcopy(model)

        for name,para in model_old.named_parameters():
            para.requires_grad=False
            
        timer = Timer()

        for epoch in range(1, args.max_epoch + 1):

            tl, ta = train(epoch, model, model_old, train_loader, optimizer,args)
            if epoch % 10 == 0:
                print(
                    "ETA:{}/{}".format(timer.measure(), timer.measure(epoch / args.max_epoch))
                )
            if args.lr_decay:
                lr_scheduler.step()
        ckpt_name = '_'.join(('session', str(session), 'base', str(BASE_CLS_NUM), 'task', str(TASK_NUM),
                              'meta_test', str(META_TEST_CLS_NUM), 'method', str(args.method)))
        save_model(model, ckpt_name, args)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument(
        "--model",
        type=str,
        default="convnet",
        choices=[
            "convnet",
            "resnet12",
            "resnet18",
            "resnet34",
            "densenet121",
            "wideres",
        ],
    )
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--query", type=int, default=15)
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--validation_way", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--step_size", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument(
        "--dataset", type=str, default="CIFAR100",
        choices=["MiniImageNet", "CUB", "CIFAR100"]
    )
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--hyperbolic", action="store_true", default=False)
    parser.add_argument("--c", type=float, default=0.01)
    parser.add_argument("--dim", type=int, default=1600)
    parser.add_argument("--location", type=int, default=105)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--batch_times", type=int, default=1)
    parser.add_argument("--init_weights", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr_decay", type=bool, default=True)
    parser.add_argument("--train_c", action="store_true", default=False)
    parser.add_argument("--train_x", action="store_true", default=False)
    parser.add_argument("--not_riemannian", action="store_true")
    parser.add_argument("--TASK_NUM", type=int, default=16)
    parser.add_argument("--EXEM_NUM", type=int, default=20)
    parser.add_argument("--BASE_CLS_NUM", type=int, default=5)
    parser.add_argument("--method", type=str, default='ERD')
    parser.add_argument("--hardmine_min", action="store_false", default=True)
    ################## for cosine linear ###################
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--sigma", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    ################## for lambdas ###################
    parser.add_argument("--lambda_distill", type=float, default=0.5)
    parser.add_argument("--lambda_align", type=float, default=0.5)
    parser.add_argument("--probability", type=float, default=0.2)

    args = parser.parse_args()
    pprint(vars(args))
    if torch.cuda.is_available():
        print("CUDA IS AVAILABLE")

    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    print(torch.cuda.current_device())

    if not os.path.exists('logs'):
        os.makedirs('logs')

    if args.save_path is None:
        save_path1 = "logs/" + "-".join([args.dataset, "ProtoNet"])
        save_path2 = "_".join(
            [
                str(args.shot),
                str(args.query),
                str(args.way),
                str(args.validation_way),
                str(args.step_size),
                str(args.gamma),
                str(args.lr),
                str(args.temperature),
                str(args.reverse),
                str(args.dim),
                str(args.probability), ##################
                str(args.lambda_distill), ##################
                str(args.lambda_align), ##################
                str(args.model),
                str(args.TASK_NUM),
                str(args.method),
                str(args.EXEM_NUM), ##################

            ]
        )
        args.save_path = save_path1 + "_" + save_path2
        make_path(args.save_path)
    else:
        make_path(args.save_path)

    if args.dataset=='CIFAR100':
        from dataloader.my_cifar import CIFAR100 as Dataset
        from dataloader.my_cifar import Hard_Mine_CIFAR as Hard_Mine_Dataset
    else:
        raise ValueError("Non-supported Dataset.")
    
    main()