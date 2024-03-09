from torch.utils.data import DataLoader

# Uptask
from create_datasets.PedXnet import *

# Downtask
from create_datasets.General_Fracture import *
from create_datasets.RSNA_BAA import *
from create_datasets.GRAZPEDWRI_DX import *



def default_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_test_dataloader(name, args):
    # Upstream
    if name == 'PedXNet_7Class_Dataset':
        test_dataset = PedXNet_7Class_Dataset(mode='test')
        test_loader  = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.test_num_workers, shuffle=False, drop_last=False, collate_fn=default_collate_fn)

    # Downstream
    elif name == 'General_Fracture':
        test_dataset = General_Fracture_Dataset(mode='test')
        test_loader  = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.test_num_workers, shuffle=False, drop_last=False, collate_fn=default_collate_fn)

    elif name == 'RSNA_BAA':
        test_dataset = RSNA_BAA_Dataset(mode='test')
        test_loader  = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.test_num_workers, shuffle=False, drop_last=False, collate_fn=default_collate_fn)
    
    elif name == 'GRAZPEDWRI_DX':
        test_dataset = GRAZPEDWRI_DX_Dataset(mode='test')
        test_loader  = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.test_num_workers, shuffle=False, drop_last=False, collate_fn=default_collate_fn)

    else: 
        raise Exception('Error...! args.data_folder_dir')

    print("test [Total] number = ", len(test_dataset))

    return test_loader


