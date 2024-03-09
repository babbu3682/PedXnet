import os
import argparse
import datetime
import time
import json
import random
import torch
import numpy as np
import utils
from dataloaders_train import get_train_dataloader
from models import get_model
from schedulers import get_scheduler
from optimizers import get_optimizer
from losses import get_loss
from engine import *


def get_args_parser():
    parser = argparse.ArgumentParser('PedXNet Deep-Learning Train script', add_help=False)

    # Dataset parameters
    parser.add_argument('--dataset',               default="amc", type=str, help='dataset name')
    parser.add_argument('--train-batch-size',      default=72, type=int)
    parser.add_argument('--valid-batch-size',      default=72, type=int)
    parser.add_argument('--train-num-workers',     default=10, type=int)
    parser.add_argument('--valid-num-workers',     default=10, type=int)

    # Model parameters
    parser.add_argument('--model',                 default='Sequence_SkipHidden_Unet_ALL',  type=str, help='model name')    
    parser.add_argument('--loss',                  default='Sequence_SkipHidden_Unet_loss', type=str, help='loss name')    
    parser.add_argument('--method',                default='', help='multi-task weighting name')

    # Optimizer parameters
    parser.add_argument('--optimizer',             default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "AdamW"')
    
    # Learning rate and schedule and Epoch parameters
    parser.add_argument('--scheduler',             default='poly_lr', type=str, metavar='scheduler', help='scheduler (default: "poly_learning_rate"')
    parser.add_argument('--epochs',                default=1000, type=int, help='Upstream 1000 epochs, Downstream 500 epochs')  
    parser.add_argument('--warmup-epochs',         default=10, type=int, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--lr',                    default=5e-4, type=float, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--min-lr',                default=1e-5, type=float, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    # DataParrel or Single GPU train
    parser.add_argument('--multi-gpu-mode',        default='DataParallel', choices=['Single', 'DataParallel'], type=str, help='multi-gpu-mode')          
    parser.add_argument('--device',                default='cuda', help='device to use for training / testing')
    
    # Validation setting
    parser.add_argument('--print-freq',            default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--save-checkpoint-every', default=1,  type=int, help='save the checkpoints every n epochs')  

    # Prediction and Save setting
    parser.add_argument('--checkpoint-dir',        default='', help='path where to save checkpoint or output')
    parser.add_argument('--save-dir',              default='', help='path where to prediction PNG save')

    # Continue Training
    parser.add_argument('--from-pretrained',       default='',  help='pre-trained from checkpoint')
    parser.add_argument('--resume',                default='',  help='resume from checkpoint')  # '' = None

    # Memo
    parser.add_argument('--memo',                  default='', help='memo for script')

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



def main(args):
    start_epoch = 0
    utils.print_args(args)
    device = torch.device(args.device)
    print("cpu == ", os.cpu_count())
    
    # Dataloader
    train_loader, valid_loader = get_train_dataloader(name=args.dataset, args=args)

    # Model
    model = get_model(name=args.model)

    # Pretrained
    if args.from_pretrained:
        print("Loading... Pretrained")
        checkpoint = torch.load(args.from_pretrained)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Multi-GPU & CUDA
    if args.multi_gpu_mode == 'DataParallel':
        model = torch.nn.DataParallel(model)
        model = model.to(device)
    else :
        model = model.to(device)

    # Optimizer & LR Schedule & Loss
    optimizer = get_optimizer(name=args.optimizer, model=model, lr=args.lr)
    scheduler = get_scheduler(name=args.scheduler, optimizer=optimizer, warm_up_epoch=args.warmup_epochs, start_decay_epoch=args.epochs/10, total_epoch=args.epochs, min_lr=1e-6)
    criterion = get_loss(name=args.loss)

    # Resume
    if args.resume:
        print("Loading... Resume")
        checkpoint = torch.load(args.resume, map_location='cpu')
        checkpoint['model_state_dict'] = {k.replace('.module', ''):v for k,v in checkpoint['model_state_dict'].items()} # fix loading multi-gpu 
        model.load_state_dict(checkpoint['model_state_dict'])   
        start_epoch = checkpoint['epoch'] + 1 
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        utils.fix_optimizer(optimizer) 

    # Etc traing setting
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    # Whole Loop Train & Valid 
    for epoch in range(start_epoch, args.epochs):

        # Upstream
        if args.model == 'Uptask_Sup_Classifier':
            train_stats = train_Uptask_Sup(train_loader, model, criterion, optimizer, device, epoch)
            print("Averaged train_stats: ", train_stats)
            valid_stats = valid_Uptask_Sup(valid_loader, model, device, epoch)
            print("Averaged valid_stats: ", valid_stats)

        # Downstream
        elif args.model == 'Downtask_General_Fracture' or args.model == 'Downtask_General_Fracture_ImageNet' or args.model == 'Downtask_General_Fracture_PedXNet_7Class' or args.model == 'Downtask_General_Fracture_PedXNet_30Class' or args.model == 'Downtask_General_Fracture_PedXNet_68Class':
            train_stats = train_Downtask_General_Fracture(train_loader, model, criterion, optimizer, device, epoch)
            print("Averaged train_stats: ", train_stats)
            valid_stats = valid_Downtask_General_Fracture(valid_loader, model, device, epoch)
            print("Averaged valid_stats: ", valid_stats)
        
        elif args.model == 'Downtask_RSNA_Boneage':
            train_stats = train_Downtask_RSNA_BAA(train_loader, model, criterion, optimizer, device, epoch)
            print("Averaged train_stats: ", train_stats)
            valid_stats = valid_Downtask_RSNA_BAA(valid_loader, model, device, epoch)
            print("Averaged valid_stats: ", valid_stats)

        else : 
            raise Exception('Error...! args.model')    

        # LR scheduler update
        scheduler.step(epoch)         

        # Save checkpoint & Prediction png
        if epoch % args.save_checkpoint_every == 0:
            checkpoint_path = args.checkpoint_dir + '/epoch_' + str(epoch) + '_checkpoint.pth'
            torch.save({
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),  # Save only Single Gpu mode
                'optimizer': optimizer.state_dict(), 
                'scheduler': scheduler.state_dict(), 
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)               

        # Log & Save
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'valid_{k}': v for k, v in valid_stats.items()},
                     'epoch': epoch}
        
        with open(args.checkpoint_dir + "/log.txt", "a") as f:
            f.write(json.dumps(log_stats) + "\n")

    # Finish
    total_time_str = str(datetime.timedelta(seconds=int(time.time()-start_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PedXNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    # Make folder if not exist
    os.makedirs(args.checkpoint_dir + "/args", mode=0o777, exist_ok=True)
    os.makedirs(args.save_dir, mode=0o777, exist_ok=True)
    
    # Save args to json
    if not os.path.isfile(args.checkpoint_dir + "/args/args_" + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".json"):
        with open(args.checkpoint_dir + "/args/args_" + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".json", "w") as f:
            json.dump(args.__dict__, f, indent=2)

    main(args)