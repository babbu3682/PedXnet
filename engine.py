import os
import math
import utils
import numpy as np
import torch
import cv2
import skimage
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from metrics import f1_metric, accuracy, sensitivity, specificity, calculate_ppv_npv
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
from monai.visualize import GradCAM


fn_tonumpy = lambda x: x.detach().cpu().numpy()

def freeze_params(model: torch.nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False

def unfreeze_params(model: torch.nn.Module):
    """Set requires_grad=True for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = True


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


# Uptask Task - Supervised 
def train_Uptask_Sup(train_loader, model, criterion, optimizer, device, epoch):
    model.train()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(train_loader))
    
    for step, batch_data in enumerate(epoch_iterator):

        image, label = batch_data
        image = image.to(device)
        label = label.to(device)

        pred = model(image)
        
        if isinstance(pred, tuple):
            loss_main, loss_dict_main = criterion(pred=pred[0], gt=label)
            loss_aux,  loss_dict_aux  = criterion(pred=pred[1], gt=label) # auxilary loss
            loss = loss_main + 0.3*loss_aux
            loss_dict = {k: loss_dict_main[k] + loss_dict_aux[k] for k in loss_dict_main}
        else:
            loss, loss_dict = criterion(pred=pred, gt=label)

        loss_value = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print("image == ", batch_data['image_meta_dict']['filename_or_obj'])
            print("label == ", batch_data['label_meta_dict']['filename_or_obj'])

        metric_logger.update(key='train_loss', value=loss_value, n=image.shape[0])
        for key in loss_dict:
            if key.startswith('cls_'):
                metric_logger.update(key='train_'+key, value=loss_dict[key].item(), n=image.shape[0])

        epoch_iterator.set_description("Training: Epochs %d (%d / %d Steps), (train_loss=%2.5f)" % (epoch, step, len(train_loader), loss_value))

    return {k: round(v, 7) for k, v in metric_logger.average().items()}

@torch.no_grad()
def valid_Uptask_Sup(valid_loader, model, device, epoch):
    model.eval()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(valid_loader, desc="Validating (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(valid_loader))

    pred_prob_list = []
    gt_binary_list = []
    
    for step, batch_data in enumerate(epoch_iterator):
        
        image, label = batch_data
        image = image.to(device)
        label = label.to(device)

        pred = model(image)

        epoch_iterator.set_description("Validating: Epochs %d (%d / %d Steps)" % (epoch, step, len(valid_loader)))
        
        pred_prob_list.append(fn_tonumpy(pred))
        gt_binary_list.append(fn_tonumpy(label))

    pred_prob_list  = np.concatenate(pred_prob_list, axis=0).squeeze() # (B,)
    gt_binary_list  = np.concatenate(gt_binary_list, axis=0).squeeze() # (B,)

    # Metric Multi-Class CLS
    f1  = f1_metric(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    acc = accuracy(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    sen = sensitivity(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    spe = specificity(y_true=gt_binary_list, y_pred=pred_prob_list.round())

    metric_logger.update(key='f1',  value=f1,  n=1)
    metric_logger.update(key='acc', value=acc, n=1)
    metric_logger.update(key='sen', value=sen, n=1)
    metric_logger.update(key='spe', value=spe, n=1)

    return {k: round(v, 7) for k, v in metric_logger.average().items()}

@torch.no_grad()
def test_Uptask_Sup(test_loader, model, device, epoch, save_dir):
    model.eval()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(test_loader, desc="TEST (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(test_loader))

    pred_prob_list = []
    gt_binary_list = []
    
    for step, batch_data in enumerate(epoch_iterator):
        
        image, label = batch_data
        image = image.to(device)
        label = label.to(device)

        pred = model(image)

        epoch_iterator.set_description("TEST: (%d / %d Steps)" % (step, len(test_loader)))
        
        pred_prob_list.append(fn_tonumpy(pred))
        gt_binary_list.append(fn_tonumpy(label))

    pred_prob_list  = np.concatenate(pred_prob_list, axis=0).squeeze() # (B,)
    gt_binary_list  = np.concatenate(gt_binary_list, axis=0).squeeze() # (B,)

    # Metric Multi-Class CLS
    f1  = f1_metric(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    acc = accuracy(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    sen = sensitivity(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    spe = specificity(y_true=gt_binary_list, y_pred=pred_prob_list.round())

    metric_logger.update(key='f1',  value=f1,  n=1)
    metric_logger.update(key='acc', value=acc, n=1)
    metric_logger.update(key='sen', value=sen, n=1)
    metric_logger.update(key='spe', value=spe, n=1)

    return {k: round(v, 7) for k, v in metric_logger.average().items()}



# Uptask Task - Unsupervised 미완성....
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
    

# Down Task - General Fracture
def train_Downtask_General_Fracture(train_loader, model, criterion, optimizer, device, epoch):
    model.train()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(train_loader))
    
    for step, batch_data in enumerate(epoch_iterator):

        image, label = batch_data
        image = image.to(device)
        label = label.to(device)

        pred = model(image).sigmoid()

        loss, loss_dict = criterion(pred, label)
        loss_value = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        metric_logger.update(key='train_loss', value=loss_value, n=image.shape[0])
        for key in loss_dict:
            if key.startswith('cls_'):
                metric_logger.update(key='train_'+key, value=loss_dict[key].item(), n=image.shape[0])

        epoch_iterator.set_description("Training: Epochs %d (%d / %d Steps), (train_loss=%2.5f)" % (epoch, step, len(train_loader), loss_value))

    return {k: round(v, 7) for k, v in metric_logger.average().items()}

@torch.no_grad()
def valid_Downtask_General_Fracture(valid_loader, model, device, epoch):
    model.eval()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(valid_loader, desc="Validating (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(valid_loader))

    pred_prob_list = []
    gt_binary_list = []
    
    for step, batch_data in enumerate(epoch_iterator):
        
        image, label = batch_data
        image = image.to(device)
        label = label.to(device)

        pred = model(image).sigmoid()

        epoch_iterator.set_description("Validating: Epochs %d (%d / %d Steps)" % (epoch, step, len(valid_loader)))
        
        pred_prob_list.append(fn_tonumpy(pred))
        gt_binary_list.append(fn_tonumpy(label))

    pred_prob_list  = np.concatenate(pred_prob_list, axis=0).squeeze() # (B,)
    gt_binary_list  = np.concatenate(gt_binary_list, axis=0).squeeze() # (B,)

    # Metric Binary-Class CLS
    auc = roc_auc_score(y_true=gt_binary_list, y_score=pred_prob_list)
    f1  = f1_metric(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    acc = accuracy(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    sen = sensitivity(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    spe = specificity(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    ppv, npv = calculate_ppv_npv(y_true=gt_binary_list, y_pred=pred_prob_list.round())

    metric_logger.update(key='auc', value=auc, n=1)
    metric_logger.update(key='f1',  value=f1,  n=1)
    metric_logger.update(key='acc', value=acc, n=1)
    metric_logger.update(key='sen', value=sen, n=1)
    metric_logger.update(key='spe', value=spe, n=1)
    metric_logger.update(key='ppv', value=ppv, n=1)
    metric_logger.update(key='npv', value=npv, n=1)

    return {k: round(v, 7) for k, v in metric_logger.average().items()}

# @torch.no_grad()
def test_Downtask_General_Fracture(test_loader, model, device, epoch, save_dir):
    model.eval()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(test_loader, desc="TEST (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(test_loader))
    grad_cam = GradCAM(nn_module=model, target_layers='feat_extractor') # Grad-CAM

    pred_prob_list  = []
    gt_binary_list  = []
    patient_id_list = []

    cnt = 0
    for step, batch_data in enumerate(epoch_iterator):
        
        image, label, patient_id = batch_data
        image = image.to(device)
        label = label.to(device)

        pred = model(image).sigmoid()

        epoch_iterator.set_description("TEST: (%d / %d Steps)" % (step, len(test_loader)))
        
        patient_id_list.append(patient_id)
        pred_prob_list.append(fn_tonumpy(pred))
        gt_binary_list.append(fn_tonumpy(label))

        # Grad-CAM
        result = grad_cam(x=image, layer_idx=-1)
        uint8_mask = skimage.img_as_ubyte(result.detach().cpu().numpy().squeeze())
        heatmap = cv2.applyColorMap( uint8_mask, cv2.COLORMAP_JET )
        heatmap = skimage.img_as_float32(heatmap)
        cam_img = heatmap + image[0].cpu().detach().numpy().transpose(1,2,0)
        # Min Max Normalize
        cam_img = cam_img - cam_img.min() 
        cam_img /= cam_img.max()            

        # PNG Save
        plt.imsave(save_dir+'/idx_'+str(cnt)+'_input_'+str(label.item())+'.png', image.detach().cpu().numpy().squeeze(), cmap="gray")
        plt.imsave(save_dir+'/idx_'+str(cnt)+'_pred_cam_'+str(pred.round().item())+'.png', cam_img)
        cnt += 1


    patient_id_list = np.concatenate(patient_id_list, axis=0).squeeze() # (B,)
    pred_prob_list  = np.concatenate(pred_prob_list, axis=0).squeeze() # (B,)
    gt_binary_list  = np.concatenate(gt_binary_list, axis=0).squeeze() # (B,)

    # Metric Binary-Class CLS
    auc = roc_auc_score(y_true=gt_binary_list, y_score=pred_prob_list)
    f1  = f1_metric(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    acc = accuracy(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    sen = sensitivity(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    spe = specificity(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    ppv, npv = calculate_ppv_npv(y_true=gt_binary_list, y_pred=pred_prob_list.round())

    metric_logger.update(key='auc', value=auc, n=1)
    metric_logger.update(key='f1',  value=f1,  n=1)
    metric_logger.update(key='acc', value=acc, n=1)
    metric_logger.update(key='sen', value=sen, n=1)
    metric_logger.update(key='spe', value=spe, n=1)
    metric_logger.update(key='ppv', value=ppv, n=1)
    metric_logger.update(key='npv', value=npv, n=1)

    # DataFrame
    df = pd.DataFrame()
    df['Patient_id'] = patient_id_list
    df['Prob']       = pred_prob_list
    df['Label']      = gt_binary_list
    df['Decision']   = pred_prob_list.round()
    df.to_csv(save_dir+'/pred_results.csv')

    return {k: round(v, 7) for k, v in metric_logger.average().items()}


# Down Task - BAA
def train_Downtask_RSNA_BAA(train_loader, model, criterion, optimizer, device, epoch):
    model.train()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(train_loader))
    
    for step, batch_data in enumerate(epoch_iterator):

        image, age, gender = batch_data
        image  = image.to(device)
        age    = age.to(device)
        gender = gender.to(device)

        pred = model(image, gender)

        loss, loss_dict = criterion(pred=pred, gt=age)
        loss_value = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        metric_logger.update(key='train_loss', value=loss_value, n=image.shape[0])
        for key in loss_dict:
            if key.startswith('cls_'):
                metric_logger.update(key='train_'+key, value=loss_dict[key].item(), n=image.shape[0])

        epoch_iterator.set_description("Training: Epochs %d (%d / %d Steps), (train_loss=%2.5f)" % (epoch, step, len(train_loader), loss_value))

    return {k: round(v, 7) for k, v in metric_logger.average().items()}

@torch.no_grad()
def valid_Downtask_RSNA_BAA(valid_loader, model, device, epoch):
    model.eval()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(valid_loader, desc="Validating (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(valid_loader))

    pred_age_list = []
    gt_age_list   = []
    
    for step, batch_data in enumerate(epoch_iterator):
        
        image, age, gender = batch_data
        image  = image.to(device)
        age    = age.to(device)
        gender = gender.to(device)

        pred = model(image, gender)

        epoch_iterator.set_description("Validating: Epochs %d (%d / %d Steps)" % (epoch, step, len(valid_loader)))
        
        pred_age_list.append(fn_tonumpy(pred))
        gt_age_list.append(fn_tonumpy(age))

    pred_age_list = np.concatenate(pred_age_list, axis=0).squeeze() # (B,)
    gt_age_list   = np.concatenate(gt_age_list, axis=0).squeeze() # (B,)

    # regresion
    mae = mean_absolute_error(y_true=gt_age_list, y_pred=pred_age_list)
    mse = mean_squared_error(y_true=gt_age_list, y_pred=pred_age_list)
    r2  = r2_score(y_true=gt_age_list, y_pred=pred_age_list)

    metric_logger.update(key='mae', value=mae, n=1)
    metric_logger.update(key='mse', value=mse,  n=1)
    metric_logger.update(key='r2', value=r2, n=1)

    return {k: round(v, 7) for k, v in metric_logger.average().items()}

@torch.no_grad()
def test_Downtask_RSNA_BAA(test_loader, model, device, epoch, save_dir):
    model.eval()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(test_loader, desc="TEST (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(test_loader))

    pred_age_list   = []
    gt_age_list     = []
    patient_id_list = []
    for step, batch_data in enumerate(epoch_iterator):
        
        image, age, gender, patient_id = batch_data
        image  = image.to(device)
        age    = age.to(device)
        gender = gender.to(device)

        pred = model(image, gender)

        epoch_iterator.set_description("TEST: (%d / %d Steps)" % (step, len(test_loader)))
        
        patient_id_list.append(patient_id)
        pred_age_list.append(fn_tonumpy(pred))
        gt_age_list.append(fn_tonumpy(age))
    
    patient_id_list = np.concatenate(patient_id_list, axis=0).squeeze() # (B,)
    pred_age_list   = np.concatenate(pred_age_list, axis=0).squeeze() # (B,)
    gt_age_list     = np.concatenate(gt_age_list, axis=0).squeeze() # (B,)

    # regresion
    mae = mean_absolute_error(y_true=gt_age_list, y_pred=pred_age_list)
    mse = mean_squared_error(y_true=gt_age_list, y_pred=pred_age_list)
    r2  = r2_score(y_true=gt_age_list, y_pred=pred_age_list)

    metric_logger.update(key='mae', value=mae, n=1)
    metric_logger.update(key='mse', value=mse, n=1)
    metric_logger.update(key='r2', value=r2, n=1)

    # DataFrame
    df = pd.DataFrame()
    df['Patient_id'] = patient_id_list
    df['Age']        = pred_age_list
    df['Target']     = gt_age_list
    df.to_csv(save_dir+'/pred_results.csv')

    return {k: round(v, 7) for k, v in metric_logger.average().items()}