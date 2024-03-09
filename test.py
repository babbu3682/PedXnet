import os
import argparse
import datetime
import time
import json
import random
import torch
import numpy as np
import utils
from dataloaders_test import get_test_dataloader
from models import get_model
from losses import get_loss
from engine import *




def get_args_parser():
    parser = argparse.ArgumentParser('PedXNet Deeplearning Evaluation script', add_help=False)

    # Dataset parameters
    parser.add_argument('--dataset',               default="amc", type=str, help='dataset name')
    parser.add_argument('--test-batch-size',       default=72, type=int)
    parser.add_argument('--test-num-workers',      default=10, type=int)

    # Model parameters
    parser.add_argument('--model',                 default='Sequence_SkipHidden_Unet_ALL',  type=str, help='model name')    
    parser.add_argument('--loss',                  default='Sequence_SkipHidden_Unet_loss', type=str, help='loss name')    
    parser.add_argument('--method',                default='', help='multi-task weighting name')

    # DataParrel or Single GPU train
    parser.add_argument('--multi-gpu-mode',        default='DataParallel', choices=['Single', 'DataParallel'], type=str, help='multi-gpu-mode')          
    parser.add_argument('--device',                default='cuda', help='device to use for training / testing')
    
    # Resume
    parser.add_argument('--resume',                default='',  help='resume from checkpoint')  # '' = None

    # Validation setting
    parser.add_argument('--print-freq',            default=10, type=int, metavar='N', help='print frequency (default: 10)')

    # Prediction and Save setting
    parser.add_argument('--checkpoint-dir',        default='', help='path where to save checkpoint or output')
    parser.add_argument('--save-dir',              default='', help='path where to prediction PNG save')
    parser.add_argument('--epoch',                 default=10, type=int)

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
    print('Available CPUs: ', os.cpu_count())
    utils.print_args_test(args)
    device = torch.device(args.device)

    # Dataloader
    test_loader = get_test_dataloader(name=args.dataset, args=args)   

    # Model
    model = get_model(name=args.model)

    # Multi-GPU & CUDA
    if args.multi_gpu_mode == 'DataParallel':
        model = torch.nn.DataParallel(model)
        model = model.to(device)
    else :
        model = model.to(device)

    # Loss
    loss = get_loss(name=args.loss)

    # Resume
    if args.resume:
        print("Loading... Resume")
        print("Before Weight ==> ", model.state_dict()['feat_extractor.0.conv.weight'].mean().item())
        checkpoint = torch.load(args.resume, map_location='cpu')
        checkpoint['model_state_dict'] = {k.replace('.module', ''):v for k,v in checkpoint['model_state_dict'].items()} # fix loading multi-gpu 
        model.load_state_dict(checkpoint['model_state_dict'])
        print("After Weight ==> ", model.state_dict()['feat_extractor.0.conv.weight'].mean().item())

    # Etc testing setting
    print(f"Start testing for {args.epoch} epoch")
    start_time = time.time()

    # Upstream
    if args.model == 'Uptask_Sup_Classifier':
        test_stats = test_Uptask_Sup(test_loader, model, device, args.epoch, args.save_dir)
        print("Averaged test_stats: ", test_stats)

    # Downstream
    elif args.model == 'Downtask_General_Fracture' or args.model == 'Downtask_General_Fracture_ImageNet' or args.model == 'Downtask_General_Fracture_PedXNet_7Class' or args.model == 'Downtask_General_Fracture_PedXNet_30Class' or args.model == 'Downtask_General_Fracture_PedXNet_68Class':
        test_stats = test_Downtask_General_Fracture(test_loader, model, device, args.epoch, args.save_dir)
        print("Averaged test_stats: ", test_stats)
    
    elif args.model == 'Downtask_RSNA_Boneage' or args.model == 'Downtask_RSNA_Boneage_ImageNet' or args.model == 'Downtask_RSNA_Boneage_PedXNet_7Class' or args.model == 'Downtask_RSNA_Boneage_PedXNet_30Class' or args.model == 'Downtask_RSNA_Boneage_PedXNet_68Class':
        test_stats = test_Downtask_RSNA_BAA(test_loader, model, device, args.epoch, args.save_dir)
        print("Averaged test_stats: ", test_stats)

    else : 
        raise Exception('Error...! args.model')

    # Log & Save
    log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': args.epoch}
    
    with open(args.checkpoint_dir + "/test_log.txt", "a") as f:
        f.write(json.dumps(log_stats) + "\n")

    # Finish
    total_time_str = str(datetime.timedelta(seconds=int(time.time()-start_time)))
    print('Testing time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PedXnet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()        
    
    # Make folder if not exist
    os.makedirs(args.checkpoint_dir + "/args", exist_ok =True)
    os.makedirs(args.save_dir, mode=0o777, exist_ok=True)

    # Save args to json
    if not os.path.isfile(args.checkpoint_dir + "/args/test_args_" + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".json"):
        with open(args.checkpoint_dir + "/args/test_args_" + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".json", "w") as f:
            json.dump(args.__dict__, f, indent=2)

    main(args)





