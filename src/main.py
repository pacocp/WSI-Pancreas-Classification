import os
import argparse
import datetime
import pickle

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from wsi_model import *
from read_data import *
from resnet import resnet50
from utils import *

def collate_fn(batch):
    """Remove bad entries from the dataloader
    Args:
        batch (torch.Tensor): batch of tensors from the dataaset
    Returns:
        collate: Default collage for the dataloader
    """
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


parser = argparse.ArgumentParser(description='Generalization classification')
parser.add_argument('--path_csv', type=str, help='Path to the csv file')
parser.add_argument('--path_csv2', type=str, help='Path to the csv file for adding to training', default=None)
parser.add_argument('--checkpoint', type=str, default=None,
        help='File with the checkpoint to start with')
parser.add_argument('--save_dir', type=str, default=None,
        help='Where to save the checkpoints')
parser.add_argument('--flag', type=str, default=None,
        help='Flag to use for saving the checkpoints')
parser.add_argument('--seed', type=int, default=99,
        help='Seed for random generation')
parser.add_argument('--log', type=int, default=0,
        help='Use tensorboard for experiment logging')
parser.add_argument('--parallel', type=int, default=0,
        help='Use DataParallel training')
parser.add_argument('--fp16', type=int, default=0,
        help='Use mixed-precision training')
parser.add_argument('--bag_size', type=int, default=50,
                    help='Bag size to use')
parser.add_argument('--max_patch_per_wsi', type=int, default=100,
                    help='Maximum number of paches per wsi')
parser.add_argument('--k', type=int, default=10,
                    help='Number of folds to use')
parser.add_argument('--img_size', type=int, default=256,
                    help='Shape of the input images')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size to use')
parser.add_argument('--log_interval', type=int, default=30,
                    help='Interval for saving results to Tensorboard')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='Number of epochs to use')
parser.add_argument('--lr', type=float, default=3e-3,
                    help='Learning rate to use')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='Weight decay to use')
parser.add_argument("--quick", help="if a quick test is performed",
                    action="store_true")
parser.add_argument("--train", help="if the model is going to be trained",
                    action="store_true")
parser.add_argument("--fulltrain", help="if the model is going to be trained on the full dataset",
                    action="store_true")
parser.add_argument("--evaluate", help="if we just want to evaluate on a dataset a pretrained model",
                    action="store_true")
parser.add_argument("--country", help="ablation analysis on a given variable",
                    action="store_true")
parser.add_argument("--png", help="if the images are saved in PNG format",
                    action="store_true")
parser.add_argument('--normalizer', type=str, default='reinhard',
                    help='Stain normalizer to use')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

print(10*'-')
print('Args for this experiment \n')
print(args)
print(10*'-')

if not args.flag:
    args.flag = 'train_{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now())

if not os.path.exists(args.save_dir):
    if not os.path.exists(args.save_dir.split('/')[0]):
        os.mkdir(args.save_dir.split('/')[0])
    os.mkdir(args.save_dir)

path_csv = args.path_csv
img_size = args.img_size
max_patch_per_wsi = args.max_patch_per_wsi
quick = args.quick
bag_size = args.bag_size
batch_size = args.batch_size

transforms_ = torch.nn.Sequential(
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

transforms_val = torch.nn.Sequential(
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
print('Loading dataset...')

df = pd.read_csv(path_csv)
df = shuffle(df, random_state=args.seed)
le = preprocessing.LabelEncoder()
le.fit(np.array(['Tumor', 'Control']).ravel())
print(list(le.classes_))

if args.train:
    if args.path_csv2:
        df2 = pd.read_csv(args.path_csv2)
        df2 = shuffle(df2, random_state=args.seed)

    train_idxs, val_idxs, test_idxs = patient_kfold(df, n_splits=args.k)
    k_fold_idxs = {
        'train_idxs': train_idxs,
        'val_idxs': val_idxs,
        'test_idxs': test_idxs
    }
    with open(os.path.join(args.save_dir,'k-fold_splits.pkl'), 'wb') as file:
        pickle.dump(k_fold_idxs, file)

    test_results_splits = {}
    i = 0
    for train_idx, val_idx, test_idx in zip(train_idxs, val_idxs, test_idxs):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        test_df = df.iloc[test_idx]
        if args.path_csv2:
            tr_idx, v_idx = patient_split_full(df2, random_state=args.seed)
            tr = df2.iloc[tr_idx]
            val = df2.iloc[v_idx]
            train_df = pd.concat([train_df, tr], axis=0)
            val_df = pd.concat([val_df, val], axis=0)
            train_df = shuffle(train_df, random_state=args.seed)
            val_df = shuffle(val_df, random_state=args.seed)
        train_dataset = PatchBagDataset(train_df,
                                max_patches_total=max_patch_per_wsi,
                                bag_size=bag_size,
                                transforms=transforms_, quick=quick,
                                label_encoder=le,
                                normalize=False,
                                return_ids=True)
        val_dataset = PatchBagDataset(val_df,
                                max_patches_total=bag_size,
                                bag_size=bag_size,
                                transforms=transforms_val, quick=quick,
                                label_encoder=le,
                                normalize=False,
                                normalizer_type=args.normalizer,
                                return_ids=True)

        test_dataset = PatchBagDataset(test_df,
                                max_patches_total=bag_size,
                                bag_size=bag_size,
                                transforms=transforms_val, quick=quick,
                                label_encoder=le,
                                normalize=False,
                                normalizer_type=args.normalizer,
                                return_ids=True)

        if torch.cuda.is_available():
            print('There is a GPU!')
            num_workers = torch.cuda.device_count() * 4

        train_dataloader = DataLoader(train_dataset, 
                                      num_workers=num_workers, pin_memory=True, 
                                      shuffle=True, batch_size=batch_size,
                                      collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, num_workers=num_workers,
                                    shuffle=False, batch_size=batch_size,
                                    collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset,  
                                     num_workers=num_workers, 
                                     shuffle=False, batch_size=batch_size,
                                    collate_fn=collate_fn)

        dataloaders = {
                'train': train_dataloader,
                'val': val_dataloader}

        dataset_sizes = {
                'train': len(train_dataset),
                'val': len(val_dataset)
                }

        transforms = {
                'train': transforms_,
                'val': transforms_val
                }

        print('Finished loading dataset and creating dataloader')

        print('Initializing models')

        resnet50_ = resnet50(pretrained=True)

        layers_to_train = [resnet50_.layer4]

        for param in resnet50_.parameters():
            param.requires_grad = False
        for layer in layers_to_train:
            for n, param in layer.named_parameters():
                param.requires_grad = True

        resnet50_ = resnet50_.to('cuda:0')

        model = AggregationModel(resnet50_, num_outputs=2, resnet_dim=2048)
        use_attention = False
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        model.fc.apply(init_weights)
        if args.checkpoint is not None:
            print('Restoring from checkpoint')
            print(args.checkpoint)
            model.load_state_dict(torch.load(args.checkpoint))
            print('Loaded model from checkpoint')

        if args.parallel and torch.cuda.device_count() > 2:
            model = nn.DataParallel(model)
        if torch.cuda.is_available():
            model = model.to('cuda:0')
        
        # add optimizer
        lr = args.lr

        optimizer = AdamW(model.parameters(), weight_decay = args.weight_decay, lr=lr)
        criterion = nn.CrossEntropyLoss()

        # train model

        if args.log:
            day_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") 
            path_summary = os.path.join('summary',
                        str(day_time) + "_{0}".format(args.flag))
            #os.mkdir(path_summary)
            summary_writer = SummaryWriter(path_summary)

        else:
            summary_writer = None

        # train model
        model, results = train(model, criterion, optimizer, dataloaders, transforms, 
                    save_dir=args.save_dir,
                    device='cuda:0', log_interval=args.log_interval,
                    summary_writer=summary_writer,
                    num_epochs=args.num_epochs)

        #with open(args.save_dir+'train_val_results.pkl', 'wb') as file:
        #    pickle.dump(results, file)

        # test on test set

        test_results = evaluate(model, test_dataloader, len(test_dataset),
                                    transforms_val, criterion=criterion, device='cuda:0')

        test_results_splits[f'split_{i}'] = test_results
        i+= 1

    with open(os.path.join(args.save_dir,'test_results.pkl'), 'wb') as file:
        pickle.dump(test_results_splits, file)

elif args.fulltrain:
        train_idx, val_idx = patient_split_full(df)
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_dataset = PatchBagDataset(train_df,
                                max_patches_total=max_patch_per_wsi,
                                bag_size=bag_size,
                                transforms=transforms_, quick=quick,
                                label_encoder=le)
        val_dataset = PatchBagDataset(val_df,
                                max_patches_total=bag_size,
                                bag_size=bag_size,
                                transforms=transforms_val, quick=quick,
                                label_encoder=le)

        if torch.cuda.is_available():
            print('There is a GPU!')
            num_workers = torch.cuda.device_count() * 4

        train_dataloader = DataLoader(train_dataset, 
                                      num_workers=num_workers, pin_memory=True, 
                                      shuffle=True, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, num_workers=num_workers,
                                    shuffle=False, batch_size=batch_size)

        dataloaders = {
                'train': train_dataloader,
                'val': val_dataloader}

        dataset_sizes = {
                'train': len(train_dataset),
                'val': len(val_dataset)
                }

        transforms = {
                'train': transforms_,
                'val': transforms_val
                }

        print('Finished loading dataset and creating dataloader')

        print('Initializing models')

        resnet50_ = resnet50(pretrained=True)

        layers_to_train = [resnet50_.layer4]
        
        for param in resnet50_.parameters():
            param.requires_grad = False
        for layer in layers_to_train:
            for n, param in layer.named_parameters():
                param.requires_grad = True

        resnet50_ = resnet50_.to('cuda:0')

        model = AggregationModel(resnet50_, num_outputs=2, resnet_dim=2048)
        use_attention = False
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        model.fc.apply(init_weights)

        if args.checkpoint is not None:
            print('Restoring from checkpoint')
            print(args.checkpoint)
            model.load_state_dict(torch.load(args.checkpoint))
            print('Loaded model from checkpoint')

        if args.parallel and torch.cuda.device_count() > 2:
            model = nn.DataParallel(model)
        if torch.cuda.is_available():
            model = model.to('cuda:0')
        
        # add optimizer
        lr = args.lr

        optimizer = AdamW(model.parameters(), weight_decay = args.weight_decay, lr=lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = None
        # train model

        if args.log:
            day_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") 
            path_summary = os.path.join('summary',
                        str(day_time) + "_{0}".format(args.flag))
            #os.mkdir(path_summary)
            summary_writer = SummaryWriter(path_summary)

        else:
            summary_writer = None

        # train model
        model, results = train(model, criterion, optimizer, dataloaders, transforms, 
                    save_dir=args.save_dir,
                    device='cuda:0', log_interval=args.log_interval,
                    summary_writer=summary_writer,
                    num_epochs=args.num_epochs,
                    scheduler=None)
elif args.evaluate:
    
    if args.png:
        test_dataset = PatchBagMHMCDataset(df,
                                max_patches_total=max_patch_per_wsi,
                                bag_size=bag_size,
                                transforms=transforms_val, quick=quick,
                                label_encoder=le)
    else:
        test_dataset = PatchBagDataset(df,
                                max_patches_total=max_patch_per_wsi,
                                bag_size=bag_size,
                                transforms=transforms_val, quick=quick,
                                label_encoder=le,
                                return_ids=True)

    if torch.cuda.is_available():
        print('There is a GPU!')
        num_workers = torch.cuda.device_count() * 4
        
    test_dataloader = DataLoader(test_dataset,  
                                    num_workers=num_workers, 
                                    shuffle=False, batch_size=batch_size)

    print('Finished loading dataset and creating dataloader')

    print('Initializing models')

    resnet50_ = resnet50(pretrained=True)

    layers_to_train = [resnet50_.layer4]
    for param in resnet50_.parameters():
        param.requires_grad = False
    for layer in layers_to_train:
        for n, param in layer.named_parameters():
            param.requires_grad = True

    resnet50_ = resnet50_.to('cuda:0')

    model = AggregationModel(resnet50_, num_outputs=2, resnet_dim=2048)
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    model.fc.apply(init_weights)
   
    print('Restoring from checkpoint')
    print(args.checkpoint)
    model.load_state_dict(torch.load(args.checkpoint))
    print('Loaded model from checkpoint')

    if torch.cuda.is_available():
        model = model.to('cuda:0')
    
    # add optimizer
    lr = args.lr

    optimizer = AdamW(model.parameters(), weight_decay = args.weight_decay, lr=lr)
    criterion = nn.CrossEntropyLoss()
    
   
    # test on test set

    test_results = evaluate(model, test_dataloader, len(test_dataset),
                                transforms_val, criterion=criterion, device='cuda:0')

    with open(os.path.join(args.save_dir,'test_results_evaluation.pkl'), 'wb') as file:
        pickle.dump(test_results, file)
elif args.country:
    train_idxs, val_idxs, test_idxs = patient_kfold_variable(df, variable='country')
    test_results_splits = {}
    i = 0
    for train_idx, val_idx, test_idx in zip(train_idxs, val_idxs, test_idxs):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        test_df = df.iloc[test_idx]
        
        train_dataset = PatchBagDataset(train_df,
                                max_patches_total=max_patch_per_wsi,
                                bag_size=bag_size,
                                transforms=transforms_, quick=quick,
                                label_encoder=le,
                                normalize=True,
                                normalizer_type=args.normalizer,
                                return_ids=True)
        val_dataset = PatchBagDataset(val_df,
                                max_patches_total=bag_size,
                                bag_size=bag_size,
                                transforms=transforms_val, quick=quick,
                                label_encoder=le,
                                normalize=True,
                                normalizer_type=args.normalizer,
                                return_ids=True)

        test_dataset = PatchBagDataset(test_df,
                                max_patches_total=bag_size,
                                bag_size=bag_size,
                                transforms=transforms_val, quick=quick,
                                label_encoder=le,
                                normalize=True,
                                normalizer_type=args.normalizer,
                                return_ids=True)

        if torch.cuda.is_available():
            print('There is a GPU!')
            num_workers = torch.cuda.device_count() * 4

        train_dataloader = DataLoader(train_dataset, 
                                      num_workers=num_workers, pin_memory=True, 
                                      shuffle=True, batch_size=batch_size,
                                      collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, num_workers=num_workers,
                                    shuffle=False, batch_size=batch_size,
                                    collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset,  
                                     num_workers=num_workers, 
                                     shuffle=False, batch_size=batch_size,
                                    collate_fn=collate_fn)

        dataloaders = {
                'train': train_dataloader,
                'val': val_dataloader}

        dataset_sizes = {
                'train': len(train_dataset),
                'val': len(val_dataset)
                }

        transforms = {
                'train': transforms_,
                'val': transforms_val
                }

        print('Finished loading dataset and creating dataloader')

        print('Initializing models')

        resnet50_ = resnet50(pretrained=True)

        layers_to_train = [resnet50_.layer4]

        for param in resnet50_.parameters():
            param.requires_grad = False
        for layer in layers_to_train:
            for n, param in layer.named_parameters():
                param.requires_grad = True

        resnet50_ = resnet50_.to('cuda:0')

        model = AggregationModel(resnet50_, num_outputs=2, resnet_dim=2048)
        use_attention = False
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        model.fc.apply(init_weights)
        if args.checkpoint is not None:
            print('Restoring from checkpoint')
            print(args.checkpoint)
            model.load_state_dict(torch.load(args.checkpoint))
            print('Loaded model from checkpoint')

        if args.parallel and torch.cuda.device_count() > 2:
            model = nn.DataParallel(model)
        if torch.cuda.is_available():
            model = model.to('cuda:0')
        
        # add optimizer
        lr = args.lr

        optimizer = AdamW(model.parameters(), weight_decay = args.weight_decay, lr=lr)
        criterion = nn.CrossEntropyLoss()

        # train model

        if args.log:
            day_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") 
            path_summary = os.path.join('summary',
                        str(day_time) + "_{0}".format(args.flag))
            #os.mkdir(path_summary)
            summary_writer = SummaryWriter(path_summary)

        else:
            summary_writer = None

        # train model
        model, results = train(model, criterion, optimizer, dataloaders, transforms, 
                    save_dir=args.save_dir,
                    device='cuda:0', log_interval=args.log_interval,
                    summary_writer=summary_writer,
                    num_epochs=args.num_epochs)

        #with open(args.save_dir+'train_val_results.pkl', 'wb') as file:
        #    pickle.dump(results, file)

        # test on test set

        test_results = evaluate(model, test_dataloader, len(test_dataset),
                                    transforms_val, criterion=criterion, device='cuda:0')

        test_results_splits[f'split_{i}'] = test_results
        i+= 1

    with open(os.path.join(args.save_dir,'test_results.pkl'), 'wb') as file:
        pickle.dump(test_results_splits, file)
