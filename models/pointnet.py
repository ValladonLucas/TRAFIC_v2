import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import Linear as Lin
from torch.nn import ReLU
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool
import torchmetrics
from utils.utils import calculate_batch_curvature, calculate_batch_torsion, normalize_batch
torch.set_float32_matmul_precision('high')

class MLP(nn.Module):
    def __init__(self, channels, batch_norm=True):
        super(MLP, self).__init__()
        layers = []
        for i in range(1, len(channels)):
            layers.append(Lin(channels[i - 1], channels[i]))
            layers.append(ReLU())
            if batch_norm:
                layers.append(nn.BatchNorm1d(channels[i]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class PN(pl.LightningModule):

    def __init__(self, **kwargs):
        super(PN, self).__init__()

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

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes)

        self.fc_enc = MLP([input_size, 64, 64, 64, 128, 1024], batch_norm=False)
        self.pool = global_max_pool
        self.fc = MLP([1024, 512, 256, 128, embedding_size], batch_norm=True)
        self.c = Lin(embedding_size, n_classes)
        self.embedding = None

    def forward(self, data):
        bs = data.shape[0]
        x = data
        x = self.fc_enc(x)
        x = self.pool(x, batch=None)
        x = self.fc(x)
        x = self.c(x)
        return x

    def training_step(self, train_batch, batch_idx):
        batch_size = len(train_batch[0])
        V, L, BBB, _ = train_batch     
        
        # Curvature
        C = calculate_batch_curvature(V)
        # Torsion
        T = calculate_batch_torsion(V)
        V, C, T = normalize_batch(V, BBB, C=C, T=T, tolerance=0.1)

        if self.input_size == 3:
            data = V
        elif self.input_size == 4:
            data = torch.cat([V, C.unsqueeze(-1)], dim=2)
        elif self.input_size == 5:
            data = torch.cat([V, C.unsqueeze(-1), T.unsqueeze(-1)], dim=2)

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

        # Curvature
        C = calculate_batch_curvature(V)
        # Torsion
        T = calculate_batch_torsion(V)
        V, C, T = normalize_batch(V, BBB, C=C, T=T, tolerance=0.1)

        if self.input_size == 3:
            data = V
        elif self.input_size == 4:
            data = torch.cat([V, C.unsqueeze(-1)], dim=2)
        elif self.input_size == 5:
            data = torch.cat([V, C.unsqueeze(-1), T.unsqueeze(-1)], dim=2)

        Y = self(data)

        loss = F.cross_entropy(Y, L)
        accuracy = self.accuracy(Y, L)

        self.log('val_loss', loss, batch_size=batch_size, sync_dist=True)
        self.log('val_accuracy', accuracy, batch_size=batch_size, sync_dist=True)

        return {'loss': loss, 'val_accuracy': accuracy} 
    
    def test_step(self, test_batch, batch_idx):
        batch_size = len(test_batch[0])
        V, L, BBB, _ = test_batch

        # Curvature
        C = calculate_batch_curvature(V)
        # Torsion
        T = calculate_batch_torsion(V)
        V, C, T = normalize_batch(V, BBB, C=C, T=T, tolerance=0.1)

        if self.input_size == 3:
            data = V
        elif self.input_size == 4:
            data = torch.cat([V, C.unsqueeze(-1)], dim=2)
        elif self.input_size == 5:
            data = torch.cat([V, C.unsqueeze(-1), T.unsqueeze(-1)], dim=2)

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

