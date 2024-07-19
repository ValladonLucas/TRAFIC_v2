import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import global_max_pool
import torchmetrics
from utils.FuncUtils.utils import *
torch.set_float32_matmul_precision('high')
from random import randrange

from models.augmentations import RandomAugmentation, AddSmallFiber    
# MLP module
class MLP(nn.Module):
    def __init__(self, channels, batch_norm=True):
        super(MLP, self).__init__()
        layers = []
        for i in range(1, len(channels)):
            layers.append(nn.Linear(channels[i - 1], channels[i]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# PointNet with confidence
class PNConf(pl.LightningModule):
    def __init__(self, **kwargs):
        super(PNConf, self).__init__()

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

        # Augmentations
        self.aug_dict = {
            'noise': (self.noise, self.noise_range),
            'shear': (self.shear, self.shear_range),
            'rotation': (self.rotation, self.rotation_range),
            'probability': self.probability
        }
        self.random_augmentation = RandomAugmentation(**self.aug_dict)
        self.add_small_fiber = AddSmallFiber(num_points=128, length_percent=randrange(5, 50)*0.01).cuda()

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes)

        self.fc_enc = MLP([input_size, 64, 64, 64, 128, 1024], batch_norm=False)
        self.pool = global_max_pool
        self.fc = MLP([1024, 512, 256, 128, embedding_size], batch_norm=True)
        self.fc_cls = nn.Linear(embedding_size, n_classes)
        self.confidence_fc = nn.Sequential(
        nn.Linear(embedding_size, n_classes*2),
        nn.ReLU(),
        nn.Linear(n_classes*2, n_classes)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        x = batch
        x = self.fc_enc(x)
        x = self.pool(x, batch=None)
        x_emb = self.fc(x)
        x_cls = self.fc_cls(x_emb)
        confidence = self.sigmoid(self.confidence_fc(x_emb))
        return x_cls, confidence, x_emb

    def training_step(self, train_batch, batch_idx):
        batch_size = len(train_batch[0])
        V, L, BBB, _ = [x.cuda() for x in train_batch]

        # Data augmentation
        V = self.random_augmentation(V)
        V_rejection = self.add_small_fiber(V)
        L_rejection = (torch.ones(V_rejection.shape[0], dtype=torch.long) * (self.n_classes-1)).to(L.device)
        V = torch.cat([V, V_rejection], dim=0)
        L = torch.cat([L, L_rejection], dim=0)
        BBB = torch.cat([BBB, BBB], dim=0)
        
        # Curvature
        C = calculate_batch_curvature(V)
        V, C = normalize_batch(V, C, BBB, 0.1)

        if self.input_size == 3:
            data = V
        elif self.input_size == 4:
            # Adding features
            data = torch.cat([V, C.unsqueeze(-1)], dim=2)

        # Forward pass
        Y, confidence, _ = self(data)


        # Loss and accuracy calculation
        accuracy = self.accuracy(Y, L)
        loss, self.lambda_val = confidence_loss(pred = Y,
                                                confidence = confidence,
                                                labels = L,
                                                num_classes=self.n_classes,
                                                lambda_val = self.lambda_val,
                                                beta = self.beta,
                                                eps = self.eps)
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

        Y, _, _ = self(data)

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

        Y, confidence, _ = self(data)

        loss = F.cross_entropy(Y, L)
        accuracy = self.accuracy(Y, L)

        self.label_list.append(L.cpu().detach().numpy())
        self.confidence_list.append(confidence.cpu().detach().numpy())

        self.log('test_loss', loss, batch_size=batch_size, sync_dist=True)
        self.log('test_accuracy', accuracy, batch_size=batch_size, sync_dist=True)

        return {'loss': loss, 'test_accuracy': accuracy} 

    def on_test_end(self):
        labels = np.concatenate(self.label_list)
        confidence = np.concatenate(self.confidence_list)

        #Calculate the mean confidence for each class
        mean_confidence = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            mean_confidence[i] = np.mean(confidence[labels == i])
        labels = [str(i) for i in range(self.n_classes)]

        df = pd.DataFrame({'labels': labels, 'mean_confidence': mean_confidence})
        df.to_csv(f'mean_confidence.csv', index=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def update_hparams(self, **kwargs):
        # Update the model with the new hyperparameters
        self.hparams.update(kwargs)
        self.save_hyperparameters()

        # Variables 
        self.n_classes = self.hparams['n_classes']
        self.noise, self.noise_range = self.hparams['noise']
        self.shear, self.shear_range = self.hparams['shear']
        self.rotation, self.rotation_range = self.hparams['rotation']
        self.probability = self.hparams['probability']

        # Augmentations
        # Augmentations
        self.aug_dict = {
            'noise': (self.noise, self.noise_range),
            'shear': (self.shear, self.shear_range),
            'rotation': (self.rotation, self.rotation_range),
            'probability': self.probability
        }
        self.random_augmentation = RandomAugmentation(**self.aug_dict)

        # Functions
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.n_classes)

        # Layers
        self.fc_cls = nn.Linear(self.hparams['embedding_size'], self.hparams['n_classes'])

        return self.hparams
