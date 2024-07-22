from torch.utils.data import Dataset
import torch
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils.FuncUtils.utils import read_vtk_file, ExtractFiber

"""
This Dataloader is used for classification.
"""

class Dataset_Fibers_Classification(Dataset):
    def __init__(self, brain_file, number_of_samples, brain_bounding_box, subjectID):
        self.number_of_samples = number_of_samples
        self.brain_file = read_vtk_file(brain_file)
        self.brain_bounding_box = torch.tensor(brain_bounding_box, dtype=torch.float32)
        self.subjectID = torch.tensor([subjectID])

    def __len__(self):
        return self.number_of_samples

    def __getitem__(self, idx):
        fiber = ExtractFiber(self.brain_file, idx)
        verts = torch.tensor(fiber, dtype=torch.float32)
        return verts, self.brain_bounding_box, self.subjectID, torch.tensor([idx])

class Concat_Dataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_sizes = [0] + [len(d) for d in self.datasets] # Initialize to zero in case of empty datasets
        for i in range(1, len(self.cumulative_sizes)):
            self.cumulative_sizes[i] += self.cumulative_sizes[i - 1]

        self.count = sum(len(d) for d in self.datasets)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        ds_idx = 0
        while idx >= self.cumulative_sizes[ds_idx + 1]:
            ds_idx += 1
        item_idx = idx - self.cumulative_sizes[ds_idx]
        return self.datasets[ds_idx][item_idx]

class DataModule(pl.LightningDataModule):
    def __init__(self, class_path, batch_size, num_workers):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.df_class = pd.read_csv(class_path)

    def setup(self, stage=None):
    
        self.classificationDataset = self.create_dataset_classification(self.df_class)
    
    def create_dataset_classification(self,df): # Test dataset is whole brain not bundles
        data_array = []
        for _, sample_row in df.iterrows():
            number_of_samples = sample_row['num_cells']
            brain_file, subjectID = sample_row['surf'], sample_row['id']
            brain_bounding_box = [sample_row['x_min'], sample_row['x_max'], sample_row['y_min'], sample_row['y_max'], sample_row['z_min'], sample_row['z_max']]
            brainData = Dataset_Fibers_Classification(brain_file=brain_file,
                                                 number_of_samples=number_of_samples,
                                                 brain_bounding_box=brain_bounding_box,
                                                 subjectID=subjectID)
            data_array.append(brainData)
        return Concat_Dataset(data_array)


    def classification_dataloader(self):
        return DataLoader(self.classificationDataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.pad_verts_faces)

    def pad_verts_faces(self, batch):

        V = [v for v, _, _, _ in batch]
        BBB = [bbb for _, bbb, _, _ in batch]
        SID = [sid for _, _, sid, _ in batch]
        IDX = [idx for _, _, _, idx in batch]

        V = pad_sequence(V, batch_first=True, padding_value=0.0)
        BBB = pad_sequence(BBB, batch_first=True, padding_value=0.0)
        SID = torch.cat(SID)
        IDX = torch.cat(IDX)
 
        return V, BBB, SID, IDX