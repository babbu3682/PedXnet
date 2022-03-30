from datasets.PedXnet import *

from datasets.General_Fracture import *
from datasets.RSNA_BoneAge import *
from datasets.Ped_Pneumo import *



# For Train & Valid
def build_dataset(is_train, args):

    mode='train' if is_train else 'valid'

    # Upstream
        # Supervised
    if args.data_set == 'Old_PedXnet_Sup_7class':
        dataset = Old_PedXNet_Dataset(mode=mode, num_class=7)
    elif args.data_set == 'Old_PedXnet_Sup_30class':
        dataset = Old_PedXNet_Dataset(mode=mode, num_class=30)
    elif args.data_set == 'Old_PedXnet_Sup_68class':
        dataset = Old_PedXNet_Dataset(mode=mode, num_class=68)

        # Unsupervised
    elif args.data_set == 'Old_PedXnet_Unsup':
        dataset = Old_PedXNet_Dataset(mode=mode, num_class=0)
    elif args.data_set == 'New_PedXnet_Unsup':
        dataset = New_PedXNet_Dataset(mode=mode)

    # Downstream
    elif args.data_set == '1.General_Fracture':
        dataset = General_Fracture_Dataset(mode=mode)  
    elif args.data_set == '2.RSNA_BoneAge':
        mode='train' if is_train else 'val'
        dataset = RSNA_BoneAge_Dataset(mode=mode)
    elif args.data_set == '3.Ped_Pneumo':
        print('fuck')
        mode='train' if is_train else 'validation'
        print(mode)
        dataset = Ped_Pneumo_Dataset(mode=mode)

    return dataset



# For Test
def build_test_dataset(args):

    # Upstream
    if args.data_set == 'Old_PedXnet_Unsup':
        dataset = PedXNet_Dataset(mode='test', num_class=0)
        
    elif args.data_set == 'New_PedXNet_Dataset':
        dataset = Unsup_PedXNet_Dataset(mode=mode)

    # Downstream
    elif args.data_set == '1.General_Fracture':
        dataset = General_Fracture_Dataset(mode='test')  

    elif args.data_set == '2.RSNA_BoneAge':
        mode='train' if is_train else 'val'
        dataset = RSNA_BoneAge_Dataset(mode='test')

    elif args.data_set == '3.Ped_Pneumo':
        mode='train' if is_train else 'validation'
        dataset = Ped_Pneumo_Dataset(mode='test')

    return dataset


