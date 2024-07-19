import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout
from torch.nn import Linear as Lin
from torch.nn import Sequential as Seq
from torch_geometric.nn import global_max_pool, DynamicEdgeConv
import torchmetrics
import pytorch_lightning as pl
from utils.FuncUtils.utils import *
torch.set_float32_matmul_precision('high')

class MLP(nn.Module):
    def __init__(self, channels, batch_norm=True):
        super(MLP, self).__init__()
        layers = []
        for i in range(1, len(channels)):
            layers.append(nn.Linear(channels[i - 1], channels[i]))
            layers.append(nn.ReLU())
            if batch_norm:
                layers.append(nn.BatchNorm1d(channels[i]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DEC(pl.LightningModule):
    def __init__(self, **kwargs):
        super(DEC, self).__init__()

        self.save_hyperparameters()

        # Hyperparameters
        input_size = kwargs['input_size']
        embedding_size = kwargs['embedding_size']
        n_classes = kwargs['n_classes']
        k = kwargs['k']
        aggr = kwargs['aggr']
        dropout = kwargs['dropout']
        noise, noise_range = kwargs['noise']
        shear, shear_range = kwargs['shear']
        rotation, rotation_range = kwargs['rotation']
        probability = kwargs['probability']

        # Variables
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.n_classes = n_classes
        self.k = k
        self.noise, self.noise_range = noise, noise_range
        self.shear, self.shear_range = shear, shear_range
        self.rotation, self.rotation_range = rotation, rotation_range
        self.probability = probability
        self.beta = 0.5
        self.lambda_val = 0.1
        self.eps = 1e-12
        self.label_list = []
        self.confidence_list = []

        self.conv1 = DynamicEdgeConv(MLP([2 * input_size, 64, 64, 64], batch_norm=True),k, aggr=aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128], batch_norm=True), k, aggr)
        self.lin1 = MLP([128 + 64, 1024])
        self.n_classes = n_classes
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes)

        self.pool = global_max_pool

        layers = [
            MLP([1024, 512], batch_norm=True),
            Dropout(0.5),
            MLP([512, self.embedding_size], batch_norm=True),
            Dropout(0.5)
        ] if dropout else [
            MLP([1024, 512]),
            MLP([512, self.embedding_size])
        ]

        self.mlp = Seq(*layers)
        self.c = Lin(self.embedding_size, self.n_classes)

    def forward(self, data):
        x = data.view(-1, data.shape[2]).to(data.device) # Reshape to [batch_size * num_points, num_features]
        batch = torch.arange(data.shape[0]).repeat_interleave(data.shape[1]).to(data.device) # Create batch index 
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = self.pool(out, batch=batch)
        out = self.mlp(out)
        out = self.c(out)
        return out
    
    def training_step(self, train_batch, batch_idx):
            batch_size = len(train_batch[0])
            V, L, BBB, _ = train_batch       
            
            # Curvature
            C = calculate_batch_curvature(V)
            V, C = normalize_batch(V, C, BBB, 0.1)

            # Adding features
            if self.input_size == 3:
                data = V
            elif self.input_size == 4:
                # Adding features
                data = torch.cat([V, C.unsqueeze(-1)], dim=2)

            # Forward pass
            Y = self(data)

            # Loss and accuracy calculation
            accuracy = self.accuracy(Y, L)
            loss = F.cross_entropy(Y, L)

            self.log('train_loss', loss, batch_size=batch_size, sync_dist=True, on_step=True, on_epoch=False)
            self.log('train_accuracy', accuracy, batch_size=batch_size, sync_dist=True, on_step=True, on_epoch=False)

            return {'loss' : loss}

    def validation_step(self, val_batch, batch_idx):
        batch_size = len(val_batch[0])
        V, L, BBB, _ = val_batch

        C = calculate_batch_curvature(V)
        V, C = normalize_batch(V, C, BBB, 0.1)
        if self.input_size == 3:
            data = V
        elif self.input_size == 4:
            # Adding features
            data = torch.cat([V, C.unsqueeze(-1)], dim=2)

        Y = self(data)

        loss = F.cross_entropy(Y, L)
        accuracy = self.accuracy(Y, L)

        self.log('val_loss', loss, batch_size=batch_size, sync_dist=True)
        self.log('val_accuracy', accuracy, batch_size=batch_size, sync_dist=True)

        return {'loss': loss, 'val_accuracy': accuracy} 
    
    def test_step(self, test_batch, batch_idx):
        batch_size = len(test_batch[0])
        V, L, BBB, _ = test_batch

        C = calculate_batch_curvature(V)
        V, C = normalize_batch(V, C, BBB, 0.1)
        if self.input_size == 3:
            data = V
        elif self.input_size == 4:
            # Adding features
            data = torch.cat([V, C.unsqueeze(-1)], dim=2)

        Y = self(data)

        loss = F.cross_entropy(Y, L)
        accuracy = self.accuracy(Y, L)

        self.log('test_loss', loss, batch_size=batch_size, sync_dist=True)
        self.log('test_accuracy', accuracy, batch_size=batch_size, sync_dist=True)

        return {'loss': loss, 'test_accuracy': accuracy} 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def update_hparams(self, **kwargs):
        # Update the model with the new hyperparameters
        self.hparams.update(kwargs)

        # Variables 
        self.n_classes = self.hparams['n_classes']
        self.noise, self.noise_range = self.hparams['noise']
        self.shear, self.shear_range = self.hparams['shear']
        self.rotation, self.rotation_range = self.hparams['rotation']

        # Augmentations
        # Augmentations
        self.aug_dict = {
            'noise': (self.noise, self.noise_range),
            'shear': (self.shear, self.shear_range),
            'rotation': (self.rotation, self.rotation_range),
            'probability': 0.2
        }

        # Functions
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.n_classes)

        # Layers
        self.c = Lin(self.hparams['embedding_size'], self.hparams['n_classes'])

        return self.hparams