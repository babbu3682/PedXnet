import torch

# Downtask_GRAZPEDWRI_DX_Loss
class Downtask_GRAZPEDWRI_DX_Loss(torch.nn.Module):
    def __init__(self):
        super(Downtask_GRAZPEDWRI_DX_Loss, self).__init__()
        self.bce_loss = torch.nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce_loss(pred, target)

        loss_details = {'cls_bce_loss': bce_loss}

        return bce_loss, loss_details


def get_loss(name):
    # Upstream    
    if name == 'Uptask_PedXNet_Supervised_Loss':
        return torch.nn.CrossEntropyLoss()

    # Downstream
    elif name == 'Downtask_BoneAge_Loss':
        return torch.nn.MSELoss()

    elif name == 'Downtask_GeneralFracture_Loss':
        return torch.nn.BCELoss()

    elif name == 'Downtask_GRAZPEDWRI_DX_Loss':
        return Downtask_GRAZPEDWRI_DX_Loss()       

    else:
        raise NotImplementedError