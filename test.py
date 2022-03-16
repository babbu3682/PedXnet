import os
import sys
sys.path.append(os.path.abspath('/workspace/sunggu'))
sys.path.append(os.path.abspath('/workspace/sunggu/3.Child'))
sys.path.append(os.path.abspath('/workspace/sunggu/3.Child/modules'))
sys.path.append(os.path.abspath('/workspace/sunggu/3.Child/utils'))

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import random

from create_model import create_model
from dataset.prepare_datasets import build_test_dataset
from engine import *
from losses import Uptask_Loss, Downtask_Loss
import utils
import functools
from pathlib import Path


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('Sunggu Deeplearning Train and Evaluation script', add_help=False)

    # Model parameters
    parser.add_argument('--input-size',           default=224,                     type=int,      help='images input size')
    parser.add_argument('--patch-size',           default=224,                     type=int,      help='images patch size')
    parser.add_argument('--backbone-name',        default='resnet50',              type=str,      help='backbone-name')
    parser.add_argument('--model-name',           default='Uptask_Sup_Classifier', type=str,      help='model or method name')    

    ## Select Training-Mode [Upstream, Downstream]
    parser.add_argument('--training-stream', default='Upstream', choices=['Upstream', 'Downstream'],     type=str, help='training stream')  
    parser.add_argument('--training-mode',   default='Supervised', choices=['Supervised', 'Unsupervised'], type=str, help='training mode')  

    # Dataset parameters
    parser.add_argument('--data-set',  default='CIFAR10', type=str, help='dataset name')    
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem',     default=False, action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # DataParrel or Single GPU train
    parser.add_argument('--device',         default='cuda', help='device to use for training / testing')

    # Continue Training from checkpoint
    parser.add_argument('--resume',     default='', help='resume from checkpoint')  # '' = None
    
    # Prediction and Save setting
    parser.add_argument('--print-freq',     default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--test-name', default='Downstream_3d_seg_model1', type=str, help='test name')    
    parser.add_argument('--save_dir',   default='', help='path where to prediction PNG save')
    
    return parser



# fix random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

def default_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def collate_fn(batches):
    if isinstance(batches[0], (list, tuple)):
        X          = [ batch[0]['image'] for batch in batches ]
        Y          = [ batch[0]['label'] for batch in batches ]
    else : 
        X          = [ batch['image'] for batch in batches ]
        Y          = [ batch['label'] for batch in batches ]
        
    except_img = [i for i, value in enumerate(X) if torch.isnan(value.sum())]
        
    for index in except_img:
        del X[index] 
        del Y[index]     
    
    batch = dict()
    batch['image'] = torch.stack(X, dim=0)
    batch['label'] = Y
    return batch


def main(args):
       
    device = torch.device(args.device)

    print("Loading dataset ....")
    dataset_test     = build_test_dataset(args=args)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=args.pin_mem, drop_last=False)

    # Select Loss
    if args.training_stream == 'Upstream':
        criterion = Uptask_Loss(mode=args.training_mode, model_name=args.model_name)
    elif args.training_stream == 'Downstream':
        criterion = Downtask_Loss(mode=args.training_mode, model_name=args.model_name)
    else: 
        raise Exception('Error...! args.training_stream')

    #### Select Model
    print(f"Creating model  : {args.model_name}")
    model = create_model(stream=args.training_stream, name=args.model_name, pretrained=args.pretrained_weight)  # linear protocol이랑 pretrained_weight 넣기
    print(model)
    
    # Resume
    print('Resume From: ', args.resume)
    if args.resume:
        print("Loading... Resume")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'best_metric' in checkpoint:
            print("Epoch: ", checkpoint['epoch'], " Best Metric ==> ", checkpoint['best_metric'])

    model.to(device)        
    start_time = time.time()

    # TEST
    if args.training_stream == 'Upstream':
        if args.model_name == 'Uptask_Sup_Classifier':
            test_stats = test_Uptask_Sup(model, criterion, data_loader_test, device, args.num_class, args.print_freq)  
        elif args.model_name == 'Uptask_Unsup_AutoEncoder':
            test_stats = test_Uptask_Unsup_AE(model, criterion, data_loader_test, device, args.print_freq, args.save_dir)  
        elif args.model_name == 'Uptask_Unsup_ModelGenesis':
            test_stats = test_Uptask_Unsup(model, criterion, data_loader_test, device, args.print_freq)
        else : 
            raise Exception('Error...! args.training_mode')    

    elif args.training_stream == 'Downstream':
        if args.training_mode == '1.General_Fracture':
            test_stats = test_Downtask_General_Fracture(model, criterion, data_loader_test, device)
        elif args.training_mode == '2.RSNA_BoneAge':
            test_stats = test_Downtask_RSNA_BoneAge(model, criterion, data_loader_test, device)
        elif args.training_mode == '3.Nodule_Detection':
            test_stats = test_Downtask_Nodule_Detection(model, criterion, data_loader_test, device)
        else : 
            raise Exception('Error...! args.training_mode')    

    else :
        raise Exception('Error...! args.training_stream')    


    # Finish
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sunggu Test script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    main(args)

