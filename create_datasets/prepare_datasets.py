# Uptask
from create_datasets.PedXnet import *

# Downtask
from create_datasets.General_Fracture import *
from create_datasets.RSNA_BAA import *



# For Train & Valid
def build_dataset(is_train, args):
    mode='train' if is_train else 'valid'

    # Upstream
        # Supervised
    if args.data_set == 'Old_PedXnet_Sup_7class':
        dataset, collate_fn = Old_PedXNet_Dataset(mode=mode, num_class=7)
    elif args.data_set == 'Old_PedXnet_Sup_30class':
        dataset, collate_fn = Old_PedXNet_Dataset(mode=mode, num_class=30)
    elif args.data_set == 'Old_PedXnet_Sup_68class':
        dataset, collate_fn = Old_PedXNet_Dataset(mode=mode, num_class=68)

        # Unsupervised
    elif args.data_set == 'Old_PedXnet_Unsup':
        dataset, collate_fn = Old_PedXNet_Dataset(mode=mode, num_class=0)
    elif args.data_set == 'New_PedXnet_Unsup':
        dataset, collate_fn = New_PedXNet_Dataset(mode=mode, data_folder_dir=args.data_folder_dir)

    # Downstream
    elif args.data_set == 'General_Fracture':
        dataset = General_Fracture_Dataset(mode=mode, data_folder_dir=args.data_folder_dir)

    elif args.data_set == 'RSNA_BAA':
        dataset = RSNA_BAA_Dataset(mode=mode, data_folder_dir=args.data_folder_dir)

    return dataset



# For Test
def build_test_dataset(args):

    # Upstream
    if args.data_set == 'Old_PedXnet_Unsup':
        dataset, collate_fn = PedXNet_Dataset(mode='test', num_class=0)
        
    elif args.data_set == 'New_PedXNet_Dataset':
        dataset, collate_fn = Unsup_PedXNet_Dataset(mode='test', data_folder_dir=args.data_folder_dir)

    # Downstream
    elif args.data_set == 'General_Fracture':
        dataset, collate_fn = General_Fracture_Dataset(mode='test', data_folder_dir=args.data_folder_dir)

    elif args.data_set == 'RSNA_BAA':
        dataset, collate_fn = RSNA_BoneAge_Dataset(mode='test', data_folder_dir=args.data_folder_dir)


    return dataset


