import argparse
import os.path as osp
import os

import pickle as pkl
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from dataloader.samplers import CategoriesSampler
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
)


def meta_val(epoch, model,trlog,val_loader):
    model.eval()

    vl = Averager()
    va = Averager()

    label = torch.arange(args.validation_way).repeat(args.query)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)
    test_acc_record = np.zeros((len(val_loader),))

    with torch.no_grad():
        for i, batch in enumerate(val_loader, 1):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            p = args.shot * args.validation_way
            data_shot, data_query = data[:p], data[p:]
            logits = model(data_shot, data_query)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            vl.add(loss.item())
            va.add(acc)
            test_acc_record[i - 1] = acc

    m, pm = compute_confidence_interval(test_acc_record)
    vl = vl.item()
    va = va.item()
    return vl,va,m,pm


def main():
    THIS_PATH = osp.dirname(__file__)
    if args.dataset=='CIFAR100':
        ROOT_PATH = osp.join(THIS_PATH, "data/cifar")
        TOTAL_TASK_NUM = 100
        META_TEST_CLS_NUM = 20

    TASK_NUM=args.TASK_NUM
    BASE_CLS_NUM=args.BASE_CLS_NUM

    if TASK_NUM > 1:
        CLS_NUM_PER_TASK = (TOTAL_TASK_NUM - BASE_CLS_NUM - META_TEST_CLS_NUM) // (TASK_NUM - 1)
    else:
        BASE_CLS_NUM = TOTAL_TASK_NUM-META_TEST_CLS_NUM

    file_name = '_'.join(('base', str(BASE_CLS_NUM),
                          'task', str(TASK_NUM), 'meta_test',
                          str(META_TEST_CLS_NUM)))+'.pkl'

    with open(osp.join(ROOT_PATH, file_name),'rb') as h:
        all_data=pkl.load(h)

    curr_task_data=all_data['meta_test']
    meta_label=curr_task_data['label']
    meta_data=curr_task_data['data']
    model = ProtoNet(args)

    meta_valset = Dataset("val", args, meta_data, meta_label,location=args.location)
    meta_val_sampler = CategoriesSampler(
        meta_valset.label, args.batch_size, args.validation_way, args.shot + args.query
    )
    meta_val_loader = DataLoader(
        dataset=meta_valset, batch_sampler=meta_val_sampler, num_workers=8, pin_memory=True
    )
    text_file = './results/' + args.save_path[4:] + '.txt'
    f = open(text_file, 'a')
    for session in range(0, TASK_NUM):
        ckpt_name = '_'.join(('session',str(session),'base', str(BASE_CLS_NUM), 'task', str(TASK_NUM),
                            'meta_test', str(META_TEST_CLS_NUM), 'method', str(args.method)))

        state_dict = torch.load(osp.join(args.save_path, ckpt_name + ".pth"), map_location='cpu')
        model.load_state_dict(state_dict['params'])

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            model = model.cuda()

        _, meta_acc, m, pm=meta_val(0, model, None, meta_val_loader)
        print('training on task id {}, meta test:\n {:.4f} , {:.4f}\n'.format(session, m, pm))
        print('training on task id {}, meta test:\n {:.4f} , {:.4f}\n'.format(session, m, pm),file=f)

        batch_size=args.batch_size
        
        ############# compute mean acc on seen tasks
        sum_acc=0.0

        for test_id in range(session+1):
            curr_task_data=all_data[test_id]
            test_label=curr_task_data['test_label']
            test_data=curr_task_data['test_data']
        
            meta_testset = Dataset("test", args, test_data,test_label,location=args.location)
            meta_test_sampler = CategoriesSampler(
                meta_testset.label, args.batch_size, args.validation_way, args.shot + args.query
            )
            meta_test_loader = DataLoader(
                dataset=meta_testset, batch_sampler=meta_test_sampler, num_workers=8, pin_memory=True,
            )
            _, meta_acc_seen,m_seen,pm_seen=meta_val(0, model, None, meta_test_loader)
            print('test on task id {}, {:.4f} , {:.4f}'.format(test_id,m_seen, pm_seen))
            print('test on task id {}, {:.4f} , {:.4f}'.format(test_id,m_seen, pm_seen),file=f)

            sum_acc+=m_seen

        sum_acc/=(float(session+1))
        print('mean acc on seen, {:.4f}'.format(sum_acc))
        print('mean acc on seen, {:.4f}'.format(sum_acc),file=f)

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument(
        "--model",
        type=str,
        default="resnet12",
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
    ####################### location
    parser.add_argument("--location", type=int, default=105)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--batch_times", type=int, default=1)

    ##############################################
    parser.add_argument("--init_weights", type=str, default=None)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--lr_decay", type=bool, default=True)
    parser.add_argument("--train_c", action="store_true", default=False)
    parser.add_argument("--train_x", action="store_true", default=False)
    parser.add_argument("--not_riemannian", action="store_true")

    # TASK_NUM=3
    # BASE_CLS_NUM=60
    parser.add_argument("--TASK_NUM", type=int, default=16)
    parser.add_argument("--BASE_CLS_NUM", type=int, default=5)
    parser.add_argument("--method", type=str, default='ft')
    ################## for cosine linear ###################
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--sigma", action="store_true")
    parser.add_argument("--reverse", action="store_true")

    args = parser.parse_args()
    pprint(vars(args))
    args.riemannian = not args.not_riemannian

    if torch.cuda.is_available():
        print("CUDA IS AVAILABLE")

    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('results'):
        os.makedirs('results')

    if args.save_path is None:
        save_path1 = "logs/"+"-".join([args.dataset, "ProtoNet"])
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
                str(args.c)[:5],
                str(args.train_c),
                str(args.train_x),
                str(args.model),
                str(args.TASK_NUM),
                str(args.method),
            ]
        )
        args.save_path = save_path1 + "_" + save_path2
        make_path(args.save_path)
    else:
        make_path(args.save_path)

    if args.dataset=='CIFAR100':
        from dataloader.my_cifar import CIFAR100 as Dataset
    else:
        raise ValueError("Non-supported Dataset.")

    main()