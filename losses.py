import torch


# Uptask Loss 
class Uptask_Loss(torch.nn.Module):
    def __init__(self, model_name='Uptask_Sup_Classifier'):
        super().__init__()
        self.model_name = model_name

        self.CE_loss    = torch.nn.CrossEntropyLoss()  #  you should not use softmax before.
        self.L1_loss    = torch.nn.L1Loss()
        self.L2_loss    = torch.nn.MSELoss()

        self.loss1_weight = 1.0
        self.loss2_weight = 1.0
        self.loss3_weight = 1.0

    def forward(self, pred=None, gt=None):
        if self.model_name == 'Uptask_Sup_Classifier':
            Loss_1 = self.CE_loss(pred, gt)
            total  = self.loss1_weight*Loss_1
            return total, {'CE_Loss':(self.loss1_weight*Loss_1).item()}
    
        elif self.model_name == 'Uptask_Unsup_AutoEncoder':
            Loss_1 = self.L1_loss(pred, gt)
            total  = self.loss1_weight*Loss_1
            return total, {'L1_loss':(self.loss1_weight*Loss_1).item()}

        elif self.model_name == 'Uptask_Unsup_ModelGenesis':
            Loss_1 = self.CE_loss(pred, gt)
            total  = self.loss1_weight*Loss_1
            return total, {'CE_Loss':(self.loss1_weight*Loss_1).item()}

        else: 
            raise Exception('Error...! self.model_name in Loss')      

# Downtask Loss
class Downtask_Loss(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        
        self.BCE_loss  = torch.nn.BCEWithLogitsLoss()
        self.L1_loss   = torch.nn.L1Loss()
        self.L2_loss   = torch.nn.MSELoss()

        self.loss1_weight = 1.0
        self.loss2_weight = 1.0
        self.loss3_weight = 1.0

    def forward(self, cls_pred=None, cls_gt=None):
        if self.model_name == 'Downtask_General_Fracture' or self.model_name == 'Downtask_Pneumonia':
            Loss_1 = self.BCE_loss(cls_pred, cls_gt)
            total  = self.loss1_weight*Loss_1
            return total, {'BCE_Loss':(self.loss1_weight*Loss_1).item()}
        
        elif self.model_name == 'Downtask_RSNA_Boneage':
            Loss_1 = self.L2_loss(cls_pred, cls_gt)
            total  = self.loss1_weight*Loss_1
            return total, {'L2_Loss':(self.loss1_weight*Loss_1).item()}

        else: 
            raise Exception('Error...! self.model_name in Loss')      