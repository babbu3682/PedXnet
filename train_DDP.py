# ref : https://github.com/lukemelas/EfficientNet-PyTorch/blob/7e8b0d312162f335785fb5dcfa1df29a75a1783a/examples/imagenet/main.py

import os
import sys
sys.path.append(os.path.abspath('/workspace/sunggu'))
sys.path.append(os.path.abspath('/workspace/sunggu/3.Child'))
sys.path.append(os.path.abspath('/workspace/sunggu/3.Child/modules'))
sys.path.append(os.path.abspath('/workspace/sunggu/3.Child/utils'))

import warnings
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import random
import json
from create_model import create_model
from dataset.prepare_datasets import build_dataset
from engine import *
from losses import Uptask_Loss, Downtask_Loss
import utils as sunggu_utils
import functools
from pathlib import Path
from lr_scheduler import create_scheduler

import torch.multiprocessing as mp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler

def lambda_rule(epoch, warm_up_epoch, start_decay_epoch, total_epoch, min_lr):
    # Linear WarmUP
    if (epoch < warm_up_epoch):
        return max(0, epoch / warm_up_epoch)
    else :
        lr = 1.0 - max(0, epoch - start_decay_epoch) /(float(total_epoch) - start_decay_epoch)

        if lr <= min_lr:
            lr = min_lr

    return lr

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('Sunggu Deeplearning Train and Evaluation script', add_help=False)

    # Dataset parameters
    parser.add_argument('--data-set', default='CIFAR10', type=str, help='dataset name')    

    # DataLoader setting
    parser.add_argument('--batch-size',  default=72, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem',    action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)

    ## Select Training-Mode [Upstream, Downstream]
    parser.add_argument('--training-stream', default='Upstream', choices=['Upstream', 'Downstream'],     type=str, help='training stream')  
    parser.add_argument('--training-mode',   default='Supervised', choices=['Supervised', 'Unsupervised'], type=str, help='training mode')  

    # Model parameters
    parser.add_argument('--model-name',      default='Sequence_SkipHidden_Unet_ALL',  type=str, help='model name')    
    # parser.add_argument('--criterion',       default='Sequence_SkipHidden_Unet_loss', type=str, help='criterion name')    
    # parser.add_argument('--criterion_mode',  default='none', type=str,  help='criterion mode')
    # parser.add_argument('--patch_training',  default="FALSE",   type=str2bool, help='patch_training')    

    # Optimizer parameters
    parser.add_argument('--optimizer', default='AdamW', type=str, metavar='OPTIMIZER', help='Optimizer (default: "AdamW"')
    
    # Learning rate and schedule and Epoch parameters
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--lr_scheduler', default='cosine_annealing_warm_restart', type=str, metavar='lr_scheduler', help='lr_scheduler (default: "cosine_annealing_warm_restart"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='learning rate (default: 5e-4)')

    # DataParrel or Single GPU train
    parser.add_argument('--multi-gpu-mode', default='DataParallel', choices=['DataParallel', 'Single', 'DDP'], type=str, help='multi-gpu-mode')          
    parser.add_argument('--device',         default='cuda', help='device to use for training / testing')

    # Continue Training
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')    
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrained_weight', default='', help='pretrained_weight')
    # parser.add_argument('--linear_protocol', default='', help='freeze the backbone')
    parser.add_argument('--from_pretrained', default="FALSE",   type=str2bool, help='just start from the checkpoint')    
    
    # Validation setting
    parser.add_argument('--print-freq',     default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--validate-every', default=2,  type=int, help='validate and save the checkpoints every n epochs')  

    # Prediction and Save setting
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--save_dir',   default='', help='path where to prediction PNG save')

    # parser.add_argument('--end2end',    default="FALSE", type=str2bool, help='Downtask option end2end')
    # parser.add_argument('--progressive-transfer', default="FALSE",                 type=str2bool, help='progressive_transfer_learning')
    # parser.add_argument('--freeze-backbone',      default="FALSE",                 type=str2bool, help='freeze_backbone_learning')
    # parser.add_argument('--test-name', default='Downstream_3d_seg_model1', type=str, help='test name')    

    # Multi-Processing DDP 
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')  # we selected the rank each node ex) node1 => rank 0, node2 => rank 1, node3 => rank2
    parser.add_argument('--dist-url', default='tcp://192.168.45.53:3683', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--ngpus-per-node', default=2, type=int, help='the number of gpus each node')
    parser.add_argument('--cuda-device-order', default='PCI_BUS_ID', type=str, help='cuda_device_order')
    parser.add_argument('--cuda-visible-devices', default='0', type=str, help='cuda_visible_devices')
    # parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--master-addr', default='192.168.45.53', type=str, help='MASTER_ADDR')
    parser.add_argument('--master-port', default='3684', type=str, help='MASTER_PORT')
    parser.add_argument('--nccl-socket-ifname', default='eno1', type=str, help='NCCL_SOCKET_IFNAME')

    return parser


# Fix random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

best_acc = 0

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
    # Multi processing DDP setting
    # if args.multiprocessing_distributed:
    #     mp.spawn(spawn_main, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    # else:
    #     main_worker(args.gpu, ngpus_per_node, args)

    mp.spawn(spawn_main, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))

def spawn_main(gpu, ngpus_per_node, args):
    # Multi processing DDP setting --> mp.spawn are changing the 'gpu' varient value
    global best_acc
    print("Use GPU: {} for training".format(gpu))
    args.rank = args.rank * ngpus_per_node + gpu   # arrange the rank number in total gpus ex) nodeA 1,2 gpu -> rank 0,1 / nodeB 1,2 gpu -> rank 2,3 / end so on...
    print("Arrange Rank: {} for training".format(args.rank))
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank) 
    torch.cuda.set_device(gpu)

    # Real Main
    sunggu_utils.print_args(args)
    device = torch.device(args.device)

    print("Loading dataset ....")
    dataset_train = build_dataset(is_train=True,  args=args)   
    dataset_valid = build_dataset(is_train=False, args=args)
    
    # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    valid_sampler = torch.utils.data.SequentialSampler(dataset_valid)

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=args.pin_mem, drop_last=True, sampler=train_sampler)#,  collate_fn=default_collate_fn)
    data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1,               num_workers=args.num_workers, shuffle=False, pin_memory=args.pin_mem, drop_last=False, sampler=valid_sampler)#, collate_fn=default_collate_fn)

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
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    print(model)
    

    # Optimizer & lr schedule
    optimizer    = torch.optim.AdamW(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)
    lr_scheduler = create_scheduler(name=args.lr_scheduler, optimizer=optimizer, args=args)
    loss_scaler  = GradScaler()
    dist.barrier()

    # Resume
    if args.resume:
        print("Loading... Resume")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])        
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])        

        if not args.from_pretrained: # if you want finetuning, Commenting below lines
            args.start_epoch = checkpoint['epoch'] + 1  
            if 'best_metric' in checkpoint:
                print("Epoch: ", checkpoint['epoch'], " Best Metric ==> ", checkpoint['best_metric'])

    # Multi GPU
    if args.multi_gpu_mode == 'DataParallel':
        model = torch.nn.DataParallel(model).to(device)
    elif args.multi_gpu_mode == 'Single':
        model.to(device)
    elif args.multi_gpu_mode == 'DDP':
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model)
    else :
        raise Exception('Error...! args.multi_gpu_mode')    

    dist.barrier()

    #### Etc training setting
    output_dir = Path(args.output_dir)
    print(f"Start training for {args.epochs} epochs")
    start_time  = time.time()
    best_epoch  = 0
    best_metric = best_metric1 = best_metric2 = 0.0
    best_mse    = 9999999

    #### Whole LOOP Train & Valid #####
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        # Train & Valid
        if args.training_stream == 'Upstream':
            if args.model_name == 'Uptask_Sup_Classifier':
                train_stats = train_Uptask_Sup(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq)
                valid_stats = valid_Uptask_Sup(model, criterion, data_loader_valid, device, args.num_class, args.print_freq)  

            elif args.model_name == 'Uptask_Unsup_AutoEncoder':
                train_stats = train_Uptask_Unsup_AE(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq, loss_scaler)
                valid_stats = valid_Uptask_Unsup_AE(model, criterion, data_loader_valid, device, epoch, args.print_freq, args.save_dir)  

            elif args.model_name == 'Uptask_Unsup_ModelGenesis':
                train_stats = train_Uptask_Unsup(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq)

            else : 
                raise Exception('Error...! args.training_mode')    

        elif args.training_stream == 'Downstream':
            if args.training_mode == '1.General_Fracture':
                train_stats = train_Downtask_General_Fracture(model, criterion, data_loader_train, optimizer, device, epoch)
                valid_stats = valid_Downtask_General_Fracture(model, criterion, data_loader_valid, device)
            elif args.training_mode == '2.RSNA_BoneAge':
                train_stats = train_Downtask_RSNA_BoneAge(model, criterion, data_loader_train, optimizer, device, epoch, args.progressive_transfer)
                valid_stats = valid_Downtask_RSNA_BoneAge(model, criterion, data_loader_valid, device)
            elif args.training_mode == '3.Nodule_Detection':
                train_stats = train_Downtask_Nodule_Detection(model, criterion, data_loader_train, optimizer, device, epoch, args.progressive_transfer)                
                valid_stats = valid_Downtask_Nodule_Detection(model, criterion, data_loader_valid, device)
            else : 
                raise Exception('Error...! args.training_mode')    

        else :
            raise Exception('Error...! args.training_stream')    


        ##### Summary #####
        if args.training_stream == 'Upstream':
            if args.model_name == 'Uptask_Sup_Classifier':
                print(f"AUC of the network on the {len(dataset_valid)} valid images: {valid_stats['AUC']:.3f}")
                if valid_stats["AUC"] > best_metric1 :    
                    best_metric1 = valid_stats["AUC"]
                    best_metric = best_metric1
                    best_epoch = epoch
                print(f'Max AUC: {best_metric:.3f}')
                print(f'Best Epoch: {best_epoch:.3f}')                  

            elif args.model_name == 'Uptask_Unsup_AutoEncoder':
                print(f"MAE of the network on the {len(dataset_valid)} valid images: {valid_stats['MAE']:.3f}")
                if valid_stats["MAE"] < best_metric1 :    
                    best_metric1 = valid_stats["MAE"]
                    print(f'Min MAE: {best_metric1:.3f}')
                    best_metric = best_metric1
                    best_epoch = epoch                
                print(f'Min Dice: {best_metric:.3f}')
                print(f'Best Epoch: {best_epoch:.3f}')    

            elif args.model_name == 'Uptask_Unsup_ModelGenesis':
                print(f"DICE of the network on the {len(dataset_valid)} valid images: {valid_stats['Dice']:.3f}")
                if valid_stats["Dice"] > best_metric1 :    
                    best_metric1 = valid_stats["AUC"]
                    print(f'Max Dice: {best_metric1:.3f}')
                    best_metric = best_metric1
                    best_epoch = epoch                
                print(f'Max Dice: {best_metric:.3f}')
                print(f'Best Epoch: {best_epoch:.3f}')    

            else :
                raise Exception('Error...! args.training_mode')  

        elif args.training_stream == 'Downstream':
            if args.model_name == '1.General_Fracture':
                print(f"AUC of the network on the {len(dataset_valid)} valid images: {valid_stats['AUC']:.3f}%")                
                if valid_stats["AUC"] > best_metric1 :    
                    best_metric1 = valid_stats["AUC"]
                    best_metric = best_metric1
                    best_epoch = epoch  
                print(f'Max AUC: {best_metric:.3f}')
                print(f'Best Epoch: {best_epoch:.3f}')               

            elif args.model_name == '2.RSNA_BoneAge':
                print(f"DICE of the network on the {len(dataset_valid)} valid images: {valid_stats['Dice']:.3f}%")                
                if valid_stats["Dice"] > best_metric1 :    
                    best_metric1 = valid_stats["Dice"]
                    best_metric = best_metric1
                    best_epoch = epoch         
                print(f'Max Dice: {best_metric:.3f}')    
                print(f'Best Epoch: {best_epoch:.3f}')     

            elif args.model_name == '3.Nodule_Detection':
                print(f"DICE of the network on the {len(dataset_valid)} valid images: {valid_stats['Dice']:.3f}%")                
                if valid_stats["Dice"] > best_metric1 :    
                    best_metric1 = valid_stats["Dice"]
                    best_metric = best_metric1
                    best_epoch = epoch         
                print(f'Max Dice: {best_metric:.3f}')    
                print(f'Best Epoch: {best_epoch:.3f}')                       

            else :
                raise Exception('Error...! args.training_mode')  

        else :
            raise Exception('Error...! args.training_stream')    
            
        # Save & Prediction png
        if dist.get_rank() == 0:
            save_name = 'epoch_' + str(epoch) + '_checkpoint.pth'
            checkpoint_path = args.output_dir + str(save_name)
            torch.save({
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),  # Save only Single Gpu mode
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'loss_scaler': loss_scaler.state_dict(),
                'epoch': epoch,
                'best_epoch': best_epoch,
                'best_metric': best_metric,
                'args': args,
            }, checkpoint_path)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'valid_{k}': v for k, v in valid_stats.items()},
                        'epoch': epoch}

            if args.output_dir:
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        lr_scheduler.step(epoch)

    # Finish
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sunggu training and evaluation script', parents=[get_args_parser()])
    args   = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, mode=0o777, exist_ok=True)
        
    os.environ["CUDA_DEVICE_ORDER"]     =  args.cuda_device_order
    os.environ["CUDA_VISIBLE_DEVICES"]  =  args.cuda_visible_devices
    os.environ['MASTER_ADDR']           =  args.master_addr
    os.environ['MASTER_PORT']           =  args.master_port
    os.environ['RANK']                  =  str(args.rank)
    os.environ['NCCL_SOCKET_IFNAME']    =  args.nccl_socket_ifname
    
    main(args)
