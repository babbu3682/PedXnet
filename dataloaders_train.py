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


def get_train_dataloader(name, args):
    # Upstream
    if name == 'PedXNet_7Class_Dataset':
        train_dataset = PedXNet_7Class_Dataset(mode='train')
        valid_dataset = PedXNet_7Class_Dataset(mode='valid')
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.train_num_workers, shuffle=True,  drop_last=True,  collate_fn=default_collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, num_workers=args.valid_num_workers, shuffle=False, drop_last=False, collate_fn=default_collate_fn)

    # Downstream
    elif name == 'General_Fracture':
        train_dataset = General_Fracture_Dataset(mode='train')
        valid_dataset = General_Fracture_Dataset(mode='valid')
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.train_num_workers, shuffle=True,  drop_last=True,  collate_fn=default_collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, num_workers=args.valid_num_workers, shuffle=False, drop_last=False, collate_fn=default_collate_fn)

    elif name == 'RSNA_BAA':
        train_dataset = RSNA_BAA_Dataset(mode='train')
        valid_dataset = RSNA_BAA_Dataset(mode='valid')
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.train_num_workers, shuffle=True,  drop_last=True,  collate_fn=default_collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, num_workers=args.valid_num_workers, shuffle=False, drop_last=False, collate_fn=default_collate_fn)
    
    elif name == 'GRAZPEDWRI_DX':
        train_dataset = GRAZPEDWRI_DX_Dataset(mode='train')
        valid_dataset = GRAZPEDWRI_DX_Dataset(mode='valid')
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.train_num_workers, shuffle=True,  drop_last=True,  collate_fn=default_collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, num_workers=args.valid_num_workers, shuffle=False, drop_last=False, collate_fn=default_collate_fn)

    else: 
        raise Exception('Error...! args.data_folder_dir')

    print("Train [Total] number = ", len(train_dataset))
    print("Valid [Total] number = ", len(valid_dataset))

    return train_loader, valid_loader


