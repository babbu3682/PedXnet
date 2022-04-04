import torch

# class Dice_BCE_Loss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.loss_function_1 = DiceLoss(mode='binary', from_logits=True)
#         self.loss_function_2 = torch.nn.BCEWithLogitsLoss()
#         self.dice_weight     = 1.0   
#         self.bce_weight      = 1.0   

#     def forward(self, y_pred, y_true):
#         dice_loss  = self.loss_function_1(y_pred, y_true)
#         bce_loss   = self.loss_function_2(y_pred, y_true)

#         return self.dice_weight*dice_loss + self.bce_weight*bce_loss


######################################################                    Uptask Loss                         ########################################################

class Uptask_Loss(torch.nn.Module):
    def __init__(self, mode='Supervised', model_name='Uptask_Sup_Classifier'):
        super().__init__()
        self.mode       = mode
        self.model_name = model_name
        self.CE_loss    = torch.nn.CrossEntropyLoss()
        self.L1_loss    = torch.nn.L1Loss()
        self.L2_loss    = torch.nn.MSELoss()
        self.weight_Loss1   = 1.0

    def forward(self, pred=None, gt=None):
        if self.mode == 'Supervised':
            if self.model_name == 'Uptask_Sup_Classifier':
                Loss_1 = self.CE_loss(pred, gt)
                return self.weight_Loss1*Loss_1, {'CE_Loss':(self.weight_Loss1*Loss_1).item()}
        
        elif self.mode == 'Unsupervised':
            if self.model_name == 'Uptask_Unsup_AutoEncoder':
                Loss_1 = self.L1_loss(pred, gt)
                return self.weight_Loss1*Loss_1, {'L1_loss':(self.weight_Loss1*Loss_1).item()}

            elif self.model_name == 'Uptask_Unsup_ModelGenesis':
                Loss_1 = self.CE_loss(pred, gt)
                return self.weight_Loss1*Loss_1, {'CE_Loss':(self.weight_Loss1*Loss_1).item()}

        else: 
            raise Exception('Error...! self.mode')


######################################################                    Downtask Loss                       ########################################################


class Downtask_Loss(torch.nn.Module):
    def __init__(self, mode, task_name):
        super().__init__()
        self.mode      = mode
        self.task_name = task_name
        self.BCE_loss  = torch.nn.BCEWithLogitsLoss() # Sigmoid
        self.Loss_1_W  = 1.0

    def forward(self, cls_pred=None, cls_gt=None):
        if self.mode == 'Supervised':
            
            if self.task_name=='1.General_Fracture':
                Loss_1 = self.BCE_loss(cls_pred, cls_gt)
                return self.Loss_1_W*Loss_1, {'CE_Loss':(self.Loss_1_W*Loss_1).item()}
            
            elif self.task_name=='2.RSNA_BoneAge':
                Loss_1 = self.BCE_loss(cls_pred, cls_gt)
                return self.Loss_1_W*Loss_1, {'CE_Loss':(self.Loss_1_W*Loss_1).item()}
            
            elif self.task_name=='3.Ped_Pneumo':
                Loss_1 = self.BCE_loss(cls_pred, cls_gt)
                return self.Loss_1_W*Loss_1, {'CE_Loss':(self.Loss_1_W*Loss_1).item()}
        
        elif self.mode == 'Unsupervised':
            Loss_1 = self.BCE_loss(cls_pred, cls_gt)
            return self.Loss_1_W*Loss_1, {'CE_Loss':(self.Loss_1_W*Loss_1).item()}

        else: 
            raise Exception('Error...! self.mode')      