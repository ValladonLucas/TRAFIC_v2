from torch.utils.data import Dataset
import torch
import pytorch_lightning as pl
from random import randint
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
"""
This Dataloader is used for training, validation and prediction.
"""

class Dataset_Fibers(Dataset):
    def __init__(self, bundle_file, buffer_size, number_of_samples, label, brain_bounding_box, subjectID):
        # Load the bundle
        self.bundle = np.load(bundle_file, allow_pickle=True)
        self.buffer_size = buffer_size
        self.number_of_samples = number_of_samples

        # Tensorize the label, brain bounding box and subjectID
        self.label = torch.tensor([label])
        self.brain_bounding_box = torch.tensor(brain_bounding_box)
        self.subjectID = torch.tensor([subjectID])

        self.fibers_buffer = []  # buffer to store the fibers

    def __len__(self):
        return self.number_of_samples  # number of fibers extracted in the bundle

    def fill_buffers(self):

        for _ in range(self.buffer_size):
            n = randint(0, len(self.bundle)- 1)
            verts = self.bundle[n]
            verts = torch.tensor(verts, dtype=torch.float32)
            self.fibers_buffer.append(verts)

    def __getitem__(self, idx):
        if not self.fibers_buffer:
            self.fill_buffers()
        return self.fibers_buffer.pop(), self.label, self.brain_bounding_box, self.subjectID

class Concat_Dataset(Dataset): # Concatenates multiple datasets
    def __init__(self, ds_num_items, datasets):

        self.datasets = datasets # array of datasets to concatenate
        self.ds_num_items = ds_num_items # number of items in each dataset
        self.count = sum(len(d) for d in self.datasets) # total number of items in the concatenated dataset

    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        ds_idx = idx//self.ds_num_items
        item_idx = idx%self.ds_num_items
        return self.datasets[ds_idx][item_idx]

class DataModule(pl.LightningDataModule):
    def __init__(self, train_path, val_path, test_path, number_of_samples, batch_size, buffer_size, num_workers):
        super().__init__()

        self.number_of_samples = number_of_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.buffer_size = buffer_size

        self.df_train = pd.read_csv(train_path)
        self.df_val = pd.read_csv(val_path)
        self.df_test = pd.read_csv(test_path)

    def setup(self, stage=None):
    
        self.trainDataset = self.create_dataset(self.df_train)
        self.valDataset = self.create_dataset(self.df_val)
        self.testDataset = self.create_dataset(self.df_test)

    def create_dataset(self, df):
        data_array = []
        for _, sample_row in df.iterrows():
            bundle_file, subjectID, label = sample_row['surf'], sample_row['id'], sample_row['label']
            brain_bounding_box = [sample_row['x_min'], sample_row['x_max'], sample_row['y_min'], sample_row['y_max'], sample_row['z_min'], sample_row['z_max']] # Bounds of the brain
            bundleData = Dataset_Fibers(bundle_file=bundle_file,
                                                 buffer_size=self.buffer_size,
                                                 number_of_samples=self.number_of_samples,
                                                 label=label,
                                                 brain_bounding_box=brain_bounding_box,
                                                 subjectID=subjectID)
            data_array.append(bundleData)
        return Concat_Dataset(self.number_of_samples, data_array)
    
    def train_dataloader(self):
        return DataLoader(self.trainDataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True, collate_fn=self.pad_verts_faces)

    def val_dataloader(self):
        return DataLoader(self.valDataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, collate_fn=self.pad_verts_faces)
    
    def test_dataloader(self):
        return DataLoader(self.testDataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, collate_fn=self.pad_verts_faces)

    def pad_verts_faces(self, batch):

        V = [v for v, _, _, _ in batch]
        L = [l for _, l, _, _ in batch]
        BBB = [bbb for _, _, bbb, _ in batch]
        SID = [sid for _, _, _, sid in batch]

        V = pad_sequence(V, batch_first=True, padding_value=0.0)
        L = torch.cat(L)
        BBB = pad_sequence(BBB, batch_first=True, padding_value=0.0)
        SID = torch.cat(SID)
 
        return V, L, BBB, SID

