# Uptask
from inspect import modulesbyfile
from create_datasets.PedXnet import *

# Downtask
from create_datasets.General_Fracture import *
from create_datasets.RSNA_BAA import *
from create_datasets.PedXnet import *
from create_datasets.Pneumonia import *


# For Train & Valid
def build_dataset(is_train, args):
    mode='train' if is_train else 'valid'

    # Upstream
        # Supervised
    if args.data_set == 'PedXnet_Sup_16class':
        dataset, collate_fn = Supervised_16Class(mode = mode, data_folder_dir=args.data_folder_dir)
    
    # Downstream
    elif args.data_set == 'General_Fracture':
        dataset, collate_fn = General_Fracture_Dataset(mode=mode, data_folder_dir=args.data_folder_dir)
    
    elif args.data_set == 'RSNA_BAA':
        dataset, collate_fn = RSNA_BAA_Dataset(mode=mode, data_folder_dir=args.data_folder_dir)
    
    elif args.data_set == 'Pneumonia':
        dataset, collate_fn = Pneumonia_Dataset(mode=mode, data_folder_dir=args.data_folder_dir)

    return dataset, collate_fn


# For Test
def build_test_dataset(args):

    # Upstream
    if args.data_set == 'PedXnet_Sup_16class':
        dataset, collate_fn = Supervised_16Class()

    # Downstream
    elif args.data_set == 'General_Fracture':
        dataset, collate_fn = General_Fracture_Dataset_TEST(data_folder_dir=args.data_folder_dir)

    elif args.data_set == 'RSNA_BAA':
        dataset, collate_fn = RSNA_BAA_Dataset_TEST(data_folder_dir=args.data_folder_dir)
    
    elif args.data_set =='Pneumonia':
        dataset, collate_fn = Pneumonia_Dataset_TEST(data_folder_dir=args.data_folder_dir)

    return dataset, collate_fn


