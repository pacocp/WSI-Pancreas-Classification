import os
import random
import slideflow as sf

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch
from tqdm import tqdm
import h5py
from PIL import Image
import glob

class PatchBagDataset(Dataset):
    def __init__(self, csv_path, transforms=None, bag_size=40,
            max_patches_total=300, quick=False, label_encoder=None,
            img_size = 256, normalize=True, return_ids = False,
            normalizer_type = 'reinhard'):
        self.csv_path = csv_path
        self.transforms = transforms
        self.bag_size = bag_size
        self.max_patches_total = max_patches_total
        self.quick = quick
        self.le = label_encoder
        self.img_size = img_size
        self.index = []
        self.data = {}
        if normalizer_type == 'reinhard':
            self.normalizer = sf.norm.StainNormalizer('reinhard_fast')
        elif normalizer_type == 'macenko':
            self.normalizer = sf.norm.StainNormalizer('macenko')
        self.normalize = normalize
        self.return_ids = return_ids
        self._preprocess()

    def _preprocess(self):
        if type(self.csv_path) == str:
            csv_file = pd.read_csv(self.csv_path)
        else:
            csv_file = self.csv_path
        
        if self.quick:
            csv_file = csv_file.sample(15)
        
        for i, row in tqdm(csv_file.iterrows()):
            row = row.to_dict()
            path = row['Path']
            WSI = row['WSI_name']
            label = np.asarray(row['Label'])
            patient_id = row['Patient_ID']
            if self.le is not None:
                label = self.le.transform(label.reshape(-1,1))
           
            if not os.path.exists(path):
                print(f'Not exist {path}')
                continue
            n_patches = row['n_patches']
            n_selected = min(n_patches, self.max_patches_total)
            n_patches= list(range(n_selected))
            images = random.sample(n_patches, n_selected)
            self.data[WSI] = {w.lower(): row[w] for w in row.keys()}
            self.data[WSI].update({'WSI': WSI, 'images': images, 'n_images': len(images), 
                                   'wsi_path': path, 'patient_id': patient_id})
            for k in range(len(images) // self.bag_size):
                self.index.append((WSI, path, self.bag_size * k, label))
            
    def shuffle(self):
        for k in self.data.keys():
            wsi_row = self.data[k]
            np.random.shuffle(wsi_row['images'])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        (WSI, wsi_path, i, label) = self.index[idx]
        imgs = []
        row = self.data[WSI]
        with h5py.File(wsi_path, 'r') as h5_file:
            if self.normalize:
                imgs_aux = [self.normalizer.transform(h5_file[str(patch)][:])for patch in row['images'][i:i + self.bag_size]]
                imgs = [self.transforms(torch.from_numpy(img).permute(2,0,1)) for img in imgs_aux]
                
            else:
                imgs = [self.transforms(torch.from_numpy(h5_file[str(patch)][:]).permute(2,0,1)) for patch in row['images'][i:i + self.bag_size]]

        img = torch.stack(imgs, dim=0)
        if self.return_ids:
            return img, label, row['patient_id']+f'_{label}'
        return img, label

class PatchBagMHMCDataset(Dataset):
    def __init__(self, csv_path, transforms=None, bag_size=40,
            max_patches_total=300, quick=False, label_encoder=None,
            img_size = 256, normalize=True):
        self.csv_path = csv_path
        self.transforms = transforms
        self.bag_size = bag_size
        self.max_patches_total = max_patches_total
        self.quick = quick
        self.le = label_encoder
        self.img_size = img_size
        self.index = []
        self.data = {}
        self.normalizer = sf.norm.StainNormalizer('reinhard_fast')
        self.normalize = normalize
        self._preprocess()

    def _preprocess(self):
        if type(self.csv_path) == str:
            csv_file = pd.read_csv(self.csv_path)
        else:
            csv_file = self.csv_path
        
        if self.quick:
            csv_file = csv_file.sample(15)
        
        for i, row in tqdm(csv_file.iterrows()):
            row = row.to_dict()
            path = row['Path']
            WSI = row['WSI_name']
            label = np.asarray(row['Label'])
            if self.le is not None:
                label = self.le.transform(label.reshape(-1,1))
           
            if not os.path.exists(path):
                print(f'Not exist {path}')
                continue
            
            images = glob.glob(path+'/*.png')
            n_patches = len(images)
            n_selected = min(n_patches, self.max_patches_total)
            n_patches= list(range(n_selected))
            images_idx = random.sample(n_patches, n_selected)
            images = np.array(images)[images_idx]
            self.data[WSI] = {w.lower(): row[w] for w in row.keys()}
            self.data[WSI].update({'WSI': WSI, 'images': images, 'n_images': len(images), 
                                   'wsi_path': path})
            for k in range(len(images) // self.bag_size):
                self.index.append((WSI, path, self.bag_size * k, label))
            
    def shuffle(self):
        for k in self.data.keys():
            wsi_row = self.data[k]
            np.random.shuffle(wsi_row['images'])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        (WSI, wsi_path, i, label) = self.index[idx]
        imgs = []
        row = self.data[WSI]
        try:
            imgs_aux = [self.normalizer.transform(np.array(Image.open(patch))) for patch in row['images'][i: i + self.bag_size]]
            imgs = imgs = [self.transforms(torch.from_numpy(img).permute(2,0,1)) for img in imgs_aux]
        
            img = torch.stack(imgs, dim=0)
        except: 
            img = None
        return img, label, wsi_path
    
if __name__ == '__main__':
    file_name = '../data/tcia_ref.csv'
    dataset = PatchBagDataset(csv_path = file_name, quick=True)
    import pdb; pdb.set_trace()
    img, label = dataset.__getitem__(0)

    import pdb; pdb.set_trace()