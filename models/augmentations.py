import torch
import torch.nn as nn
from torchvision.transforms import RandomApply, Compose
from utils.utils import add_noise_to_fibers, shear_fibers, rotate_fibers, generate_small_fiber
torch.set_float32_matmul_precision('high')

class Noise(nn.Module):
    def __init__(self, noise_range):
        super(Noise, self).__init__()
        self.noise_range = noise_range
        self.add_noise = add_noise_to_fibers
    def forward(self, x):
        return self.add_noise(x, self.noise_range)
    
class Shear(nn.Module):
    def __init__(self, shear_range):
        super(Shear, self).__init__()
        self.shear_range = shear_range
        self.shear = shear_fibers
    def forward(self, x):
        return self.shear(x, self.shear_range)
    
class Rotate(nn.Module):
    def __init__(self, roation_range):
        super(Rotate, self).__init__()
        self.roation_range = roation_range
        self.rotate = rotate_fibers
    def forward(self, x):
        return self.rotate(x, self.roation_range)

class RandomAugmentation(nn.Module):
    def __init__(self, **kwargs):
        super(RandomAugmentation, self).__init__()

        noise, noise_range = kwargs['noise']
        shear, shear_range = kwargs['shear']
        rotation, rotation_range = kwargs['rotation']
        probability = kwargs['probability']

        self.probability = probability
        self.aug_list = []

        if shear:
            self.shear = Shear(shear_range)
            self.aug_list.append(self.shear)
        if noise:
            self.noise = Noise(noise_range)
            self.aug_list.append(self.noise)
        if rotation:
            self.rotate = Rotate(rotation_range)
            self.aug_list.append(self.rotate)
            self.aug_list.append(self.translate)
            
        self.compose_list = []
        for aug in self.aug_list:
            self.compose_list.append(RandomApply(nn.ModuleList([aug]), p=self.probability))
        self.compose = Compose(self.compose_list)

    def forward(self, x):
        x = self.compose(x)
        return x

class AddSmallFiber(nn.Module):
    def __init__(self, num_points=128, length_percent=0.1):
        super(AddSmallFiber, self).__init__()
        self.num_points = num_points
        self.length_percent = length_percent
        self.generate_small_fiber = generate_small_fiber

    def forward(self, x):
        return self.generate_small_fiber(x, self.num_points, self.length_percent)