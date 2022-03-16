import math
from pathlib import Path
from typing import Iterable, Optional
import utils
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt

import monai
# sys.path.append(os.path.abspath('/workspace/sunggu/MONAI'))
# from monai.metrics.utils import MetricReduction, do_metric_reduction
from monai.metrics import compute_roc_auc, ConfusionMatrixMetric   
from monai.transforms import AsDiscrete, Activations
from torch.cuda.amp import autocast

fn_tonumpy = lambda x: x.cpu().detach().numpy().transpose(0, 2, 3, 1)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

features   = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

def Activation_Map(x):
    # print(x.shape) torch.Size([32, 2048, 16, 16])
    mean = torch.mean(x, dim=1)
    mean = torch.sigmoid(mean).squeeze().cpu().detach().numpy()
    # print("mean shape1 == ", mean.shape) # (32, 16, 16)
    mean = np.stack([ cv2.resize(i, (512, 512), interpolation=cv2.INTER_CUBIC) for i in mean ], axis=0)
    # print("mean shape2 == ", mean.shape) # (32, 512, 512)
    return mean

# Metric
confuse_metric = ConfusionMatrixMetric(metric_name=['f1 score', 'accuracy', 'sensitivity', 'specificity'])  # input, target must be one-hot format

# Post-processing
Pred_To_Prob   = Activations(sigmoid=False, softmax=True, other=None)

######################################################                    Uptask Task                         ########################################################
############################ Supervised ####################################
def train_Uptask_Sup(model, criterion, data_loader, optimizer, device, epoch, print_freq):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)   # (B, 1, H, W) ---> (B, 1, H, W)
        cls_gt  = batch_data["label"].to(device)   # (B, 1)
        
        cls_pred = model(inputs)
        
        if isinstance(cls_pred, (list, tuple)):
            loss, loss_detail = criterion(cls_pred=cls_pred[0], cls_aux=cls_pred[1], cls_gt=cls_gt)

        else: 
            loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# Evaluation code 
@torch.no_grad()
def valid_Uptask_Sup(model, criterion, data_loader, device, num_class, print_freq):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid:'

    # Post-processing define
    Pred_To_Onehot  = AsDiscrete(argmax=True,  to_onehot=True, num_classes=num_class, threshold_values=False, logit_thresh=0.5, rounding=None, n_classes=None)
    Label_To_Onehot = AsDiscrete(argmax=False, to_onehot=True, num_classes=num_class, threshold_values=False, logit_thresh=0.5, rounding=None, n_classes=None)

    # switch to evaluation mode
    model.eval()
    
    total_cls_pred  = torch.tensor([], dtype=torch.float32, device='cpu')
    total_cls_gt    = torch.tensor([], dtype=torch.float32, device='cpu')

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)   # (B, 1, H, W) ---> (B, 1, H, W)
        cls_gt  = batch_data["label"].to(device)   # (B, 1)

        with torch.no_grad():
            cls_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        total_cls_pred  = torch.cat([total_cls_pred,  cls_pred],   dim=0)
        total_cls_gt    = torch.cat([total_cls_gt,    cls_gt],     dim=0)

    # Metric CLS        
    AUC                 = compute_roc_auc(total_cls_pred, total_cls_gt, to_onehot_y=True, softmax=True)

    total_cls_pred      = Pred_To_Onehot(total_cls_pred)
    total_cls_gt        = Label_To_Onehot(total_cls_gt)
    
    confuse_metric(y_pred=total_cls_pred, y=total_cls_gt) # execute
    metric_result       = confuse_metric.aggregate()

    F1  = metric_result[0].item()
    Acc = metric_result[1].item()
    Sen = metric_result[2].item()
    Spe = metric_result[3].item()
    
    print('* Loss:{losses.global_avg:.3f} | AUC:{AUC:.3f} F1:{F1:.3f} Acc:{Acc:.3f} Sen:{Sen:.3f} Spe:{Spe:.3f} '.format(losses=metric_logger.loss, AUC=AUC, F1=F1, Acc=Acc, Sen=Sen, Spe=Spe))
    return {'loss': metric_logger.loss.global_avg, 'AUC':AUC, 'F1':F1, 'Acc': Acc, 'Sen': Sen, 'Spe': Spe}

# test code 
@torch.no_grad()
def test_Uptask_Sup(model, criterion, data_loader, device, num_class, print_freq):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'TEST:'
    
    # Post-processing define
    Pred_To_Onehot  = AsDiscrete(argmax=True,  to_onehot=True, num_classes=num_class, threshold_values=False, logit_thresh=0.5, rounding=None, n_classes=None)
    Label_To_Onehot = AsDiscrete(argmax=False, to_onehot=True, num_classes=num_class, threshold_values=False, logit_thresh=0.5, rounding=None, n_classes=None)

    # switch to evaluation mode
    model.eval()
    print_freq = 10
    
    total_cls_pred  = torch.tensor([], dtype=torch.float32, device='cpu')
    total_cls_gt    = torch.tensor([], dtype=torch.float32, device='cpu')

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)   # (B, 1, H, W) ---> (B, 1, H, W)
        cls_gt  = batch_data["label"].to(device)   # (B, 1)

        with torch.no_grad():
            cls_pred = model(inputs)
            

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        total_cls_pred  = torch.cat([total_cls_pred,  cls_pred],   dim=0)
        total_cls_gt    = torch.cat([total_cls_gt,    cls_gt],     dim=0)

    # Metric CLS        
    AUC                 = compute_roc_auc(total_cls_pred, total_cls_gt, to_onehot_y=True, softmax=True)

    total_cls_pred      = Pred_To_Onehot(total_cls_pred)
    total_cls_gt        = Label_To_Onehot(total_cls_gt)
    
    confuse_metric(y_pred=total_cls_pred, y=total_cls_gt) # execute
    metric_result       = confuse_metric.aggregate()

    F1  = metric_result[0].item()
    Acc = metric_result[1].item()
    Sen = metric_result[2].item()
    Spe = metric_result[3].item()
    
    print('* Loss:{losses.global_avg:.3f} | AUC:{AUC:.3f} F1:{F1:.3f} Acc:{Acc:.3f} Sen:{Sen:.3f} Spe:{Spe:.3f} '.format(losses=metric_logger.loss, AUC=AUC, F1=F1, Acc=Acc, Sen=Sen, Spe=Spe))
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return {'loss': metric_logger.loss.global_avg, 'AUC':AUC, 'F1':F1, 'Acc': Acc, 'Sen': Sen, 'Spe': Spe}


# to do... 
############################ Unsupervised ####################################
def train_Uptask_Unsup_AE(model, criterion, data_loader, optimizer, device, epoch, print_freq, loss_scaler):
    model.train(True)
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)   # (B, 1, H, W) ---> (B, 1, H, W)

        # print("check == ", inputs.shape)
        # print("C3 == ", batch_data['path'][0])
        with autocast(enabled=True):
            pred = model(inputs)
            loss, loss_detail = criterion(pred=pred, gt=inputs)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print("C1 == ", inputs.shape, inputs.max(), inputs.min())
            print("C2 == ", pred.shape, pred.max(), pred.min())
            print("C3 == ", batch_data['path'])

        optimizer.zero_grad()
        loss_scaler.scale(loss).backward()
        loss_scaler.step(optimizer)
        loss_scaler.update()

        # loss.backward()
        # optimizer.step()
        
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        
        
    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# Evaluation code 
@torch.no_grad()
def valid_Uptask_Unsup_AE(model, criterion, data_loader, device, epoch, print_freq, save_dir):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid:'

    os.makedirs(save_dir, mode=0o777, exist_ok=True)

    # switch to evaluation mode
    model.eval()
    
    # total_pred      = torch.tensor([], dtype=torch.float32, device='cuda')
    # total_inputs    = torch.tensor([], dtype=torch.float32, device='cuda')

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)   # (B, 1, H, W) ---> (B, 1, H, W)

        with torch.no_grad():
            pred = model(inputs)
            loss, loss_detail = criterion(pred=pred, gt=inputs)
            loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # Metric REC
        MAE = F.l1_loss(input=pred, target=inputs).item()

        # LOSS
        metric_logger.update(loss=loss_value)
        metric_logger.update(metric=MAE)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # total_pred    = torch.cat([total_pred,   pred],   dim=0)
        # total_inputs  = torch.cat([total_inputs, inputs], dim=0)

    # Metric REC        
    # MAE               = F.l1_loss(input=total_pred, target=total_inputs).item()
    
    # PNG Save Last sample
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_input.png', fn_tonumpy(inputs).squeeze(), cmap="gray")
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_pred.png',  fn_tonumpy(pred).squeeze(),   cmap="gray")

    imagesToPrint = torch.cat([inputs[0:15].cpu(), inputs[15:30].cpu(), inputs[30:45].cpu(),
                                pred[0:15].cpu(), pred[15:30].cpu(), pred[30:45].cpu()], dim=0)

    torchvision.utils.save_image(imagesToPrint, save_dir+'epoch_'+str(epoch)+'_total.jpg', nrow=15, normalize=True)

    # print('* Loss:{losses.global_avg:.3f} | MAE:{MAE:.3f} '.format(losses=metric_logger.loss, MAE=MAE))
    print('* Loss:{losses.global_avg:.3f} | Metric:{metric.global_avg:.3f} '.format(losses=metric_logger.loss, metric=metric_logger.metric))

    # return {'loss': metric_logger.loss.global_avg, 'MAE':MAE}
    return {'loss': metric_logger.loss.global_avg, 'MAE':metric_logger.metric.global_avg}
    

# test code 
@torch.no_grad()
def test_Uptask_Unsup_AE(model, criterion, data_loader, device, print_freq, save_dir):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'TEST:'

    os.makedirs(save_dir, mode=0o777, exist_ok=True)

    feat_list = []
    save_dict = dict()
    # switch to evaluation mode
    model.eval()
    MAE_score = 0
    cnt = 0
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)   # (B, 1, H, W) ---> (B, 1, H, W)

        with torch.no_grad():
            pred = model(inputs)
            feat = model.feature_extract(inputs)
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(start_dim=1)

            loss, loss_detail = criterion(pred=pred, gt=inputs)
            loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # Metric REC
        MAE = F.l1_loss(input=pred, target=inputs).item()

        # LOSS
        metric_logger.update(loss=loss_value)
        metric_logger.update(metric=MAE)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # 여기수정...
        # PNG Save
        plt.imsave(save_dir+str(cnt)+'_input.png', fn_tonumpy(inputs).squeeze(), cmap="gray")
        plt.imsave(save_dir+str(cnt)+'_pred.png',  fn_tonumpy(pred).squeeze(),   cmap="gray")

        # npy Save
        np.save(save_dir+'features_'+str(cnt)+'.npy', feat.detach().cpu().numpy())
        
        cnt+=1
        # too much GPU resource
        # total_pred    = torch.cat([total_pred,   pred],   dim=0)
        # total_inputs  = torch.cat([total_inputs, inputs], dim=0)

    # print('* Loss:{losses.global_avg:.3f} | MAE:{MAE:.3f} '.format(losses=metric_logger.loss, MAE=MAE))
    print('* Loss:{losses.global_avg:.3f} | Metric:{metric.global_avg:.3f} '.format(losses=metric_logger.loss, metric=metric_logger.metric))

    # return {'loss': metric_logger.loss.global_avg, 'MAE':MAE}
    return {'loss': metric_logger.loss.global_avg, 'MAE':metric_logger.metric.global_avg}
    

############################ Previous Works ####################################
# 1. Model Genesis
def train_Uptask_ModelGenesis(model, criterion, data_loader, optimizer, device, epoch):
    # 2d slice-wise based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):

        inputs  = batch_data["distort"].squeeze(4).to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        rec_gt  = batch_data["image"].squeeze(4).to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)

        rec_pred = model(inputs)

        loss, loss_detail = criterion(rec_pred=rec_pred, rec_gt=rec_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
            # metric_logger.update(RotationLoss=loss1.data.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
# Evaluation code 
@torch.no_grad()
def valid_Uptask_ModelGenesis(model, criterion, data_loader, device):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid:'
    
    # switch to evaluation mode
    model.eval()
    print_freq = 10
    metric_count = metric_sum = 0
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["distort"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        rec_gt  = batch_data["image"]                # (B, C, H, W, 1) ---> (B, C, H, W)

        with torch.no_grad():
            rec_pred = torch.stack([ model(inputs[..., i]).detach().cpu() for i in range(inputs.shape[-1]) ], dim = -1)    #    ---> torch.Size([1, 1, 256, 256, 36])

        # Metrics
        loss, loss_detail = criterion(rec_pred=rec_pred, rec_gt=rec_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)


    print('* Loss:{losses.global_avg:.3f} '.format(losses=metric_logger.loss))
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return {'loss': metric_logger.loss.global_avg}
# test code 
@torch.no_grad()
def test_Uptask_ModelGenesis(model, criterion, data_loader, device, epoch, output_dir):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid:'

    save_check = os.path.join(output_dir, 'check_samples')
    Path(save_check).mkdir(parents=True, exist_ok=True)
    
    # switch to evaluation mode
    model.eval()
    print_freq = 10

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)

        with torch.no_grad():
            seg_pred = torch.stack([ model(inputs[..., i]).detach().cpu() for i in range(inputs.shape[-1]) ], dim = 0)    #    ---> (B, 1)

        loss, loss_detail = criterion(seg_pred=seg_pred, seg_gt=seg_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # Metrics
        value, not_nans = dice_metric(y_pred=torch.sigmoid(seg_pred), y=seg_gt)
        metric_count   += not_nans
        metric_sum     += value * not_nans

            
    # Metric SEG
    Dice = (metric_sum / metric_count).item()

    print('* Loss:{losses.global_avg:.3f} | Dice:{dice:.3f} '.format(losses=metric_logger.loss, dice=Dice))
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return {'loss': metric_logger.loss.global_avg, 'Dice': Dice}






######################################################                    Down Task                         ##########################################################
############################ 1. General Fracture ####################################
def train_Downtask_General_Frac(model, criterion, data_loader, optimizer, device, epoch):
    # 2d slice-wise based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = batch_data["label"].flatten(1).bool().any(dim=1).float().unsqueeze(1).to(device) #    ---> (B, 1)
        x_lens  = batch_data["z_shape"]            #    ---> (B, 1) 최근에 Bug 생김. cpu로 넣어줘야 함.
        # x_lens  = batch_data["z_shape"].to(device) #    ---> (B, 1)

        cls_pred = model(inputs, x_lens)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# Evaluation code 
@torch.no_grad()
def valid_Downtask_General_Frac(model, criterion, data_loader, device):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid:'
    
    # switch to evaluation mode
    model.eval()
    print_freq = 10

    total_cls_pred  = torch.tensor([], dtype=torch.float32, device='cpu')
    total_cls_gt    = torch.tensor([], dtype=torch.float32, device='cpu')

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = batch_data["label"].flatten(1).bool().any(dim=1).float().unsqueeze(1).to(device) #    ---> (B, 1)
        x_lens  = batch_data["z_shape"]            #    ---> (B, 1) 최근에 Bug 생김. cpu로 넣어줘야 함.
        # x_lens  = batch_data["z_shape"].to(device) #    ---> (B, 1)

        with torch.no_grad():
            cls_pred = model(inputs, x_lens)

        total_cls_pred  = torch.cat([total_cls_pred,  torch.sigmoid(cls_pred).detach().cpu()],   dim=0)
        total_cls_gt    = torch.cat([total_cls_gt,    cls_gt.detach().cpu()],     dim=0)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

            
    # Metric CLS
    AUC = roc_auc_score(total_cls_gt, total_cls_pred)
    f1  = f1_score(y_true=total_cls_gt, y_pred=total_cls_pred.round())
    print("F1 score == ",  f1)    
    confuse_result      = confuse_metric(y_pred=total_cls_pred.round(), y=total_cls_gt) 
    confusion_matrix, _ = do_metric_reduction(confuse_result, MetricReduction.SUM)

    TP = confusion_matrix[0].item()
    FP = confusion_matrix[1].item()
    TN = confusion_matrix[2].item()
    FN = confusion_matrix[3].item()
    
    Acc = (TP+TN) / (TP+FP+TN+FN)
    Sen = TP / (TP+FN)
    Spe = TN / (TN+FP)


    print('* Loss:{losses.global_avg:.3f} | AUC:{AUC:.3f} Acc:{acc:.3f} Sen:{sen:.3f} Spe:{spe:.3f} '.format(losses=metric_logger.loss, AUC=AUC, acc=Acc, sen=Sen, spe=Spe))
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return {'loss': metric_logger.loss.global_avg, 'AUC':AUC, 'Acc': Acc, 'Sen': Sen, 'Spe': Spe}


# test code 
@torch.no_grad()
def test_Downtask_General_Frac(model, criterion, data_loader, device, test_name, save_path):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'TEST:'
    
    # switch to evaluation mode
    model.eval()
    print_freq = 10

    # Save npz path 
    save_dict = dict()
    
    total_cls_pred  = torch.tensor([], dtype=torch.float32, device='cpu')
    total_cls_gt    = torch.tensor([], dtype=torch.float32, device='cpu')

    img_path_list  = []
    mask_path_list = []
    img_list       = []
    mask_list      = []
    label_list     = []
    cls_prob_list  = []
    feature_list   = []
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        img_path   = batch_data["img_path"][0]        # batch 1. so indexing [0]
        mask_path  = batch_data["mask_path"][0]       # batch 1. so indexing [0]
        inputs     = batch_data["image"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt     = batch_data["label"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt     = batch_data["label"].flatten(1).bool().any(dim=1).float().unsqueeze(1).to(device) #    ---> (B, 1)
        x_lens     = batch_data["z_shape"]            #    ---> (B, 1) 최근에 Bug 생김. cpu로 넣어줘야 함.
        # x_lens     = batch_data["z_shape"].to(device) #    ---> (B, 1)

        with torch.no_grad():
            model.linear1.register_forward_hook(get_features('feat')) # for Representation
            cls_pred = model(inputs, x_lens)
            # print("체크 확인용", features['feat'].shape)  #torch.Size([1, 512]) 

            # Save
            img_path_list.append(img_path)
            mask_list.append(seg_gt.detach().cpu().numpy())
            mask_path_list.append(mask_path)
            img_list.append(inputs.detach().cpu().numpy())
            label_list.append(cls_gt.detach().cpu().numpy())            
            cls_prob_list.append(torch.sigmoid(cls_pred).detach().cpu().numpy())
            feature_list.append(features['feat'].detach().cpu().numpy())
            
        total_cls_pred  = torch.cat([total_cls_pred,  torch.sigmoid(cls_pred).detach().cpu()],   dim=0)
        total_cls_gt    = torch.cat([total_cls_gt,    cls_gt.detach().cpu()],     dim=0)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
    
    # Metric CLS
    AUC = roc_auc_score(total_cls_gt, total_cls_pred)
    confuse_result      = confuse_metric(y_pred=total_cls_pred.round(), y=total_cls_gt) 
    confusion_matrix, _ = do_metric_reduction(confuse_result, MetricReduction.SUM)
    f1  = f1_score(y_true=total_cls_gt, y_pred=total_cls_pred.round())
    print("F1 score == ",  f1)
    TP = confusion_matrix[0].item()
    FP = confusion_matrix[1].item()
    TN = confusion_matrix[2].item()
    FN = confusion_matrix[3].item()
    
    Acc = (TP+TN) / (TP+FP+TN+FN)
    Sen = TP / (TP+FN)
    Spe = TN / (TN+FP)

    # Save Prediction by using npz
    save_dict['gt_img_path']  = img_path_list
    save_dict['gt_mask_path'] = mask_path_list
    save_dict['gt_img']       = img_list
    save_dict['gt_mask']      = mask_list
    save_dict['gt_label']     = label_list
    save_dict['pred_label']   = cls_prob_list
    save_dict['feature']      = feature_list

    print("Saved npz...! => ", save_path + test_name + '_cls_3d[AUC_' + str(round(AUC, 3)) + '].npz')
    np.savez(save_path + test_name + '_cls_3d[AUC_' + str(round(AUC, 3)) + '].npz', cls_3d=save_dict) 

    print('* Loss:{losses.global_avg:.3f} | AUC:{AUC:.3f} Acc:{acc:.3f} Sen:{sen:.3f} Spe:{spe:.3f} '.format(losses=metric_logger.loss, AUC=AUC, acc=Acc, sen=Sen, spe=Spe))
    return {'loss': metric_logger.loss.global_avg, 'AUC':AUC, 'Acc': Acc, 'Sen': Sen, 'Spe': Spe}


############################ 2. RSNA_BoneAge ####################################
def train_Downtask_RSNA_BoneAge(model, criterion, data_loader, optimizer, device, epoch):
    # 2d slice-wise based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = batch_data["label"].flatten(1).bool().any(dim=1).float().unsqueeze(1).to(device) #    ---> (B, 1)
        x_lens  = batch_data["z_shape"]            #    ---> (B, 1) 최근에 Bug 생김. cpu로 넣어줘야 함.
        # x_lens  = batch_data["z_shape"].to(device) #    ---> (B, 1)

        cls_pred = model(inputs, x_lens)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# Evaluation code 
@torch.no_grad()
def valid_Downtask_RSNA_BoneAge(model, criterion, data_loader, device):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid:'
    
    # switch to evaluation mode
    model.eval()
    print_freq = 10

    total_cls_pred  = torch.tensor([], dtype=torch.float32, device='cpu')
    total_cls_gt    = torch.tensor([], dtype=torch.float32, device='cpu')

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = batch_data["label"].flatten(1).bool().any(dim=1).float().unsqueeze(1).to(device) #    ---> (B, 1)
        x_lens  = batch_data["z_shape"]            #    ---> (B, 1) 최근에 Bug 생김. cpu로 넣어줘야 함.
        # x_lens  = batch_data["z_shape"].to(device) #    ---> (B, 1)

        with torch.no_grad():
            cls_pred = model(inputs, x_lens)

        total_cls_pred  = torch.cat([total_cls_pred,  torch.sigmoid(cls_pred).detach().cpu()],   dim=0)
        total_cls_gt    = torch.cat([total_cls_gt,    cls_gt.detach().cpu()],     dim=0)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

            
    # Metric CLS
    AUC = roc_auc_score(total_cls_gt, total_cls_pred)
    f1  = f1_score(y_true=total_cls_gt, y_pred=total_cls_pred.round())
    print("F1 score == ",  f1)    
    confuse_result      = confuse_metric(y_pred=total_cls_pred.round(), y=total_cls_gt) 
    confusion_matrix, _ = do_metric_reduction(confuse_result, MetricReduction.SUM)

    TP = confusion_matrix[0].item()
    FP = confusion_matrix[1].item()
    TN = confusion_matrix[2].item()
    FN = confusion_matrix[3].item()
    
    Acc = (TP+TN) / (TP+FP+TN+FN)
    Sen = TP / (TP+FN)
    Spe = TN / (TN+FP)


    print('* Loss:{losses.global_avg:.3f} | AUC:{AUC:.3f} Acc:{acc:.3f} Sen:{sen:.3f} Spe:{spe:.3f} '.format(losses=metric_logger.loss, AUC=AUC, acc=Acc, sen=Sen, spe=Spe))
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return {'loss': metric_logger.loss.global_avg, 'AUC':AUC, 'Acc': Acc, 'Sen': Sen, 'Spe': Spe}


# test code 
@torch.no_grad()
def test_Downtask_RSNA_BoneAge(model, criterion, data_loader, device, test_name, save_path):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'TEST:'
    
    # switch to evaluation mode
    model.eval()
    print_freq = 10

    # Save npz path 
    save_dict = dict()
    
    total_cls_pred  = torch.tensor([], dtype=torch.float32, device='cpu')
    total_cls_gt    = torch.tensor([], dtype=torch.float32, device='cpu')

    img_path_list  = []
    mask_path_list = []
    img_list       = []
    mask_list      = []
    label_list     = []
    cls_prob_list  = []
    feature_list   = []
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        img_path   = batch_data["img_path"][0]        # batch 1. so indexing [0]
        mask_path  = batch_data["mask_path"][0]       # batch 1. so indexing [0]
        inputs     = batch_data["image"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt     = batch_data["label"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt     = batch_data["label"].flatten(1).bool().any(dim=1).float().unsqueeze(1).to(device) #    ---> (B, 1)
        x_lens     = batch_data["z_shape"]            #    ---> (B, 1) 최근에 Bug 생김. cpu로 넣어줘야 함.
        # x_lens     = batch_data["z_shape"].to(device) #    ---> (B, 1)

        with torch.no_grad():
            model.linear1.register_forward_hook(get_features('feat')) # for Representation
            cls_pred = model(inputs, x_lens)
            # print("체크 확인용", features['feat'].shape)  #torch.Size([1, 512]) 

            # Save
            img_path_list.append(img_path)
            mask_list.append(seg_gt.detach().cpu().numpy())
            mask_path_list.append(mask_path)
            img_list.append(inputs.detach().cpu().numpy())
            label_list.append(cls_gt.detach().cpu().numpy())            
            cls_prob_list.append(torch.sigmoid(cls_pred).detach().cpu().numpy())
            feature_list.append(features['feat'].detach().cpu().numpy())
            
        total_cls_pred  = torch.cat([total_cls_pred,  torch.sigmoid(cls_pred).detach().cpu()],   dim=0)
        total_cls_gt    = torch.cat([total_cls_gt,    cls_gt.detach().cpu()],     dim=0)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
    
    # Metric CLS
    AUC = roc_auc_score(total_cls_gt, total_cls_pred)
    confuse_result      = confuse_metric(y_pred=total_cls_pred.round(), y=total_cls_gt) 
    confusion_matrix, _ = do_metric_reduction(confuse_result, MetricReduction.SUM)
    f1  = f1_score(y_true=total_cls_gt, y_pred=total_cls_pred.round())
    print("F1 score == ",  f1)
    TP = confusion_matrix[0].item()
    FP = confusion_matrix[1].item()
    TN = confusion_matrix[2].item()
    FN = confusion_matrix[3].item()
    
    Acc = (TP+TN) / (TP+FP+TN+FN)
    Sen = TP / (TP+FN)
    Spe = TN / (TN+FP)

    # Save Prediction by using npz
    save_dict['gt_img_path']  = img_path_list
    save_dict['gt_mask_path'] = mask_path_list
    save_dict['gt_img']       = img_list
    save_dict['gt_mask']      = mask_list
    save_dict['gt_label']     = label_list
    save_dict['pred_label']   = cls_prob_list
    save_dict['feature']      = feature_list

    print("Saved npz...! => ", save_path + test_name + '_cls_3d[AUC_' + str(round(AUC, 3)) + '].npz')
    np.savez(save_path + test_name + '_cls_3d[AUC_' + str(round(AUC, 3)) + '].npz', cls_3d=save_dict) 

    print('* Loss:{losses.global_avg:.3f} | AUC:{AUC:.3f} Acc:{acc:.3f} Sen:{sen:.3f} Spe:{spe:.3f} '.format(losses=metric_logger.loss, AUC=AUC, acc=Acc, sen=Sen, spe=Spe))
    return {'loss': metric_logger.loss.global_avg, 'AUC':AUC, 'Acc': Acc, 'Sen': Sen, 'Spe': Spe}


