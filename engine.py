import os
import math
import utils
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from metrics import *



def freeze_params(model: torch.nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False

def unfreeze_params(model: torch.nn.Module):
    """Set requires_grad=True for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = True

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


# Uptask Task
    # Supervised 
def train_Uptask_Sup(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)   # (B, 1, H, W) ---> (B, 1, H, W)
        cls_gt  = batch_data["label"].to(device)   # (B, 1)
        
        cls_pred = model(inputs)
        
        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print("image == ", batch_data['image_meta_dict']['filename_or_obj'])
            print("label == ", batch_data['label_meta_dict']['filename_or_obj'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_Uptask_Sup(model, criterion, data_loader, device, print_freq, batch_size):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)   # (B, 1, H, W) ---> (B, 1, H, W)
        cls_gt  = batch_data["label"].to(device)   # (B, 1)

        cls_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # Metric CLS
        auc            = auc_metric(y_pred=Softmax_To_Prob(cls_pred), y=Label_To_16_Onehot(cls_gt))
        confuse_matrix = confuse_metric(y_pred=Pred_To_16_Onehot(cls_pred), y=Label_To_16_Onehot(cls_gt)) 
        

    # Aggregatation
    auc                = auc_metric.aggregate()
    f1, acc, sen, spe  = confuse_metric.aggregate()
    
    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)          
    
    auc_metric.reset()
    confuse_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_Uptask_Sup(model, criterion, data_loader, device, print_freq, batch_size):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)   # (B, 1, H, W) ---> (B, 1, H, W)
        cls_gt  = batch_data["label"].to(device)   # (B, 1)

        cls_pred = model(inputs)
            
        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # Metric CLS
        auc            = auc_metric(y_pred=Softmax_To_Prob(cls_pred), y=Label_To_16_Onehot(cls_gt))
        confuse_matrix = confuse_metric(y_pred=Pred_To_16_Onehot(cls_pred), y=Label_To_16_Onehot(cls_gt)) 
        

    # Aggregatation
    auc                = auc_metric.aggregate()
    f1, acc, sen, spe  = confuse_metric.aggregate()
    
    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)          
    
    auc_metric.reset()
    confuse_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


    # Unsupervised
def train_Uptask_Unsup_AE(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)   # (B, 1, H, W) ---> (B, 1, H, W)

        pred = model(inputs)
        loss, loss_detail = criterion(pred=pred, gt=inputs)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print("C1 == ", inputs.shape, inputs.max(), inputs.min())
            print("C2 == ", pred.shape, pred.max(), pred.min())
            print("C3 == ", batch_data['path'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        
        
    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_Uptask_Unsup_AE(model, criterion, data_loader, device, epoch, print_freq, save_dir, batch_size):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    os.makedirs(save_dir, mode=0o777, exist_ok=True)

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

        # print("check list == ", inputs.shape) torch.Size([720, 1, 256, 256])

    # Metric REC        
    # MAE               = F.l1_loss(input=total_pred, target=total_inputs).item()
    
    # PNG Save Last sample
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_input.png', fn_tonumpy(inputs).squeeze()[0], cmap="gray")
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_pred.png',  fn_tonumpy(pred).squeeze()[0],   cmap="gray")
    
    # print("check list == ", inputs.shape) torch.Size([89, 1, 256, 256])
    
    imagesToPrint = torch.cat([inputs[0:15].cpu(), inputs[15:30].cpu(), inputs[30:45].cpu(),
                                pred[0:15].cpu(), pred[15:30].cpu(), pred[30:45].cpu()], dim=0)

    torchvision.utils.save_image(imagesToPrint, save_dir+'epoch_'+str(epoch)+'_total.jpg', nrow=15, normalize=True)

    # print('* Loss:{losses.global_avg:.3f} | MAE:{MAE:.3f} '.format(losses=metric_logger.loss, MAE=MAE))
    print('* Loss:{losses.global_avg:.3f} | Metric:{metric.global_avg:.3f} '.format(losses=metric_logger.loss, metric=metric_logger.metric))

    # return {'loss': metric_logger.loss.global_avg, 'MAE':MAE}
    return {'loss': metric_logger.loss.global_avg, 'MAE':metric_logger.metric.global_avg}
    

@torch.no_grad()
def test_Uptask_Unsup_AE(model, criterion, data_loader, device, print_freq, save_dir, batch_size):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'

    os.makedirs(save_dir, mode=0o777, exist_ok=True)

    feat_list = []
    save_dict = dict()
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
    

    # Previous Works - 1. Model Genesis
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


# Down Task
    # General Fracture - Need for Customizing ...!
def train_Downtask_General_Fracture(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size, gradual_unfreeze):
    # 2d slice-wise based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    if gradual_unfreeze:
        # Gradual Unfreezing
        # 10 epoch 씩 one stage block 풀기, 100 epoch까지는 아예 고정
        if epoch >= 0 and epoch <= 100:
            freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)
            print("Freeze encoder ...!!")
        elif epoch >= 101 and epoch < 111:
            print("Unfreeze encoder.layer4 ...!")
            unfreeze_params(model.module.encoder.layer4) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer4)
        elif epoch >= 111 and epoch < 121:
            print("Unfreeze encoder.layer3 ...!")
            unfreeze_params(model.module.encoder.layer3) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer3)
        elif epoch >= 121 and epoch < 131:
            print("Unfreeze encoder.layer2 ...!")
            unfreeze_params(model.module.encoder.layer2) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer2)
        elif epoch >= 131 and epoch < 141:
            print("Unfreeze encoder.layer1 ...!")
            unfreeze_params(model.module.encoder.layer1) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer1)
        else :
            print("Unfreeze encoder.stem ...!")
            unfreeze_params(model.module.encoder) if hasattr(model, 'module') else unfreeze_params(model.encoder)
    else :
        print("Freeze encoder ...!")
        freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)     # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = batch_data["label"].to(device).unsqueeze(-1)     #                 ---> (B, 1)
        assert len(cls_gt.shape) == 2

        cls_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred.to(torch.float32), cls_gt=cls_gt.to(torch.float32))
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
        
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_Downtask_General_Fracture(model, criterion, data_loader, device, print_freq, batch_size):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = batch_data["label"].to(device).unsqueeze(-1)   #                 ---> (B, 1)

        cls_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred.to(torch.float32), cls_gt=cls_gt.to(torch.float32))
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value) 
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # # Post-processing
        cls_pred = torch.sigmoid(cls_pred)

        # Metric CLS
        auc                 = auc_metric(y_pred=cls_pred, y=cls_gt)
        confuse_matrix      = confuse_metric(y_pred=cls_pred.round(), y=cls_gt)


    # Aggregatation
    auc                = auc_metric.aggregate()
    f1, acc, sen, spe  = confuse_metric.aggregate()
    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)          
    
    auc_metric.reset()
    confuse_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_Downtask_General_Fracture(model, criterion, data_loader, device, print_freq, batch_size):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)     # (B, C, H, W, D)
        cls_gt  = batch_data["label"].to(device).unsqueeze(-1)     #  ---> (B, 1)

        cls_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred.to(torch.float32), cls_gt=cls_gt.to(torch.float32))
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value) 
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # # Post-processing
        cls_pred = torch.sigmoid(cls_pred)
        
        # Metric CLS
        auc                 = auc_metric(y_pred=cls_pred, y=cls_gt)
        confuse_matrix      = confuse_metric(y_pred=cls_pred.round(), y=cls_gt)   # pred_cls must be round() !!


    # Aggregatation
    auc                = auc_metric.aggregate()
    f1, acc, sen, spe  = confuse_metric.aggregate() 
    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)          
    
    auc_metric.reset()
    confuse_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


def train_Downtask_Pneumonia(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size, gradual_unfreeze):
    # 2d slice-wise based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    if gradual_unfreeze is not None:
        # Gradual Unfreezing
        # 10 epoch 씩 one stage block 풀기, 100 epoch까지는 아예 고정
        if epoch >= 0 and epoch <= 100:
            freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)
            print("Freeze encoder ...!")
        elif epoch >= 101 and epoch < 111:
            print("Unfreeze encoder.layer4 ...!")
            unfreeze_params(model.module.encoder.layer4) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer4)
        elif epoch >= 111 and epoch < 121:
            print("Unfreeze encoder.layer3 ...!")
            unfreeze_params(model.module.encoder.layer3) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer3)
        elif epoch >= 121 and epoch < 131:
            print("Unfreeze encoder.layer2 ...!")
            unfreeze_params(model.module.encoder.layer2) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer2)
        elif epoch >= 131 and epoch < 141:
            print("Unfreeze encoder.layer1 ...!")
            unfreeze_params(model.module.encoder.layer1) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer1)
        else :
            print("Unfreeze encoder.stem ...!")
            unfreeze_params(model.module.encoder) if hasattr(model, 'module') else unfreeze_params(model.encoder)
    else :
        print("Freeze encoder ...!")
        freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)     # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = batch_data["label"].to(device).unsqueeze(-1)     #                 ---> (B, 1)
        # print(cls_gt.shape)

        cls_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred.to(torch.float32), cls_gt=cls_gt.to(torch.float32))
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
        
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_Downtask_Pneumonia(model, criterion, data_loader, device, print_freq, batch_size):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = batch_data["label"].to(device).unsqueeze(-1)   #                 ---> (B, 1)

        cls_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred.to(torch.float32), cls_gt=cls_gt.to(torch.float32))
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value) 
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # # Post-processing
        cls_pred = torch.sigmoid(cls_pred)

        # Metric CLS
        auc                 = auc_metric(y_pred=cls_pred, y=cls_gt)
        confuse_matrix      = confuse_metric(y_pred=cls_pred.round(), y=cls_gt)


    # Aggregatation
    auc                = auc_metric.aggregate()
    f1, acc, sen, spe  = confuse_metric.aggregate()
    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)          
    
    auc_metric.reset()
    confuse_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}
@torch.no_grad()
def test_Downtask_Pneumonia(model, criterion, data_loader, device, print_freq, batch_size):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)     # (B, C, H, W, D)
        cls_gt  = batch_data["label"].to(device).unsqueeze(-1)     #  ---> (B, 1)

        cls_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred.to(torch.float32), cls_gt=cls_gt.to(torch.float32))
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value) 
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # # Post-processing
        cls_pred = torch.sigmoid(cls_pred)

        # Metric CLS
        auc                 = auc_metric(y_pred=cls_pred, y=cls_gt)
        confuse_matrix      = confuse_metric(y_pred=cls_pred.round(), y=cls_gt)


    # Aggregatation
    auc                = auc_metric.aggregate()
    f1, acc, sen, spe  = confuse_metric.aggregate()
    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)          
    
    auc_metric.reset()
    confuse_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


def train_Downtask_RSNA_BAA(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size, gradual_unfreeze):
    # 2d slice-wise based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    # if gradual_unfreeze is not None:
    #     # Gradual Unfreezing
    #     # 10 epoch 씩 one stage block 풀기, 100 epoch까지는 아예 고정
    #     if epoch >= 0 and epoch <= 100:
    #         freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)
    #         print("Freeze encoder ...!")
    #     elif epoch >= 101 and epoch < 111:
    #         print("Unfreeze encoder.layer4 ...!")
    #         unfreeze_params(model.module.encoder.layer4) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer4)
    #     elif epoch >= 111 and epoch < 121:
    #         print("Unfreeze encoder.layer3 ...!")
    #         unfreeze_params(model.module.encoder.layer3) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer3)
    #     elif epoch >= 121 and epoch < 131:
    #         print("Unfreeze encoder.layer2 ...!")
    #         unfreeze_params(model.module.encoder.layer2) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer2)
    #     elif epoch >= 131 and epoch < 141:
    #         print("Unfreeze encoder.layer1 ...!")
    #         unfreeze_params(model.module.encoder.layer1) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer1)
    #     else :
    #         print("Unfreeze encoder.stem ...!")
    #         unfreeze_params(model.module.encoder) if hasattr(model, 'module') else unfreeze_params(model.encoder)
    # else :
    #     print("Freeze encoder ...!")
    #     freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].float().to(device)     # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = batch_data["label"].float().to(device)     #                 ---> (B, 1)
        gender  = batch_data['gender'].to(device)
        
        cls_pred = model(inputs, gender)

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
        
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_Downtask_RSNA_BAA(model, criterion, data_loader, device, print_freq, batch_size):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].float().to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = batch_data["label"].float().to(device)   #                 ---> (B, 1)
        gender  = batch_data['gender'].to(device) 

        cls_pred = model(inputs, gender)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
        
        MAE = F.l1_loss(input=cls_pred, target=cls_gt).item()

        # LOSS
        # Metric
        metric_logger.update(loss=loss_value)
        metric_logger.update(metric=MAE)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

    print('* Loss:{losses.global_avg:.3f} | Metric:{metric.global_avg:.3f} '.format(losses=metric_logger.loss, metric=metric_logger.metric))

    return {'loss': metric_logger.loss.global_avg, 'MAE':metric_logger.metric.global_avg}

@torch.no_grad()
def test_Downtask_RSNA_BAA(model, criterion, data_loader, device, print_freq, batch_size):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].float().to(device)     # (B, C, H, W, D)
        cls_gt  = batch_data["label"].float().to(device)     #  ---> (B, 1)
        gender  = batch_data['gender'].to(device) 

        cls_pred = model(inputs, gender)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
        
        MAE = F.l1_loss(input=cls_pred, target=cls_gt).item()

        # LOSS
        # Metric
        metric_logger.update(loss=loss_value)
        metric_logger.update(metric=MAE)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

    print('* Loss:{losses.global_avg:.3f} | Metric:{metric.global_avg:.3f} '.format(losses=metric_logger.loss, metric=metric_logger.metric))

    return {'loss': metric_logger.loss.global_avg, 'MAE':metric_logger.metric.global_avg}
