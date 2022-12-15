from tqdm import tqdm
import h5py
import pandas as pd

data = pd.read_csv('../data/tcia_ref.csv')
n_patches = []
for i, row in tqdm(data.iterrows()):
    row = row.to_dict()
    path = row['Path']
    #else:
        #   label = label.astype(np.float32)
    with h5py.File(path, 'r') as h5_file: 
        n_patches.append(len(h5_file.keys()))

data['n_patches'] = n_patches
data.to_csv('../data/tcia_ref.csv', index=False, sep=',')
