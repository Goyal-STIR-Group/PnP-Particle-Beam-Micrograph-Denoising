"""One-stop DnCNN"""

import numpy as np
import torch
from torch import nn

class DnCNN(nn.Module):
    def __init__(self, numLayers, biasFirst = True):
        super().__init__()

        # Fixed parameters
        numChannels = 1
        kernelSize = 3
        padding = 1
        numFeatures = 64

        # Prepare layers
        layers = list()
        layers.append(nn.Conv2d(numChannels, numFeatures, kernelSize, padding=padding, bias=biasFirst))
        layers.append(nn.ReLU())
        for _ in range(numLayers - 2):
            layers.append(nn.Conv2d(numFeatures, numFeatures, kernelSize, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(numFeatures))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(numFeatures, numChannels, kernelSize, padding=padding, bias=False))

        # Pack layers
        self.layers = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        noise = self.layers(x)
        denoisedImage = x - noise
        return denoisedImage

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

def loadCheckpoint(cpdir, model, device=torch.device('cpu')):
    """Load model parameters from checkpoint
    Note:
        The checkpoint is expected to be a dict containing theh following keys,
            'model_state_dict': state dict of the model,
            'optimizer_state_dict': state dict of the optimizer,
            'epoch': the epoch count.
            'global_step': the global step count.
    Args:
        cpdir (str): path to the checkpoint.
        model: the model to load the parameters to.
    """
    checkpoint = torch.load(cpdir, map_location=device)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except KeyError:
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except KeyError:
            model.load_state_dict(checkpoint['model'][0])

class torchDenoiser:
    """Wrapper of torch denoiser for easier usage."""
    def __init__(self, model, device = torch.device('cpu')) -> None:
        model.eval()
        self.model = model
        self.device = device

    @torch.no_grad()
    def __call__(self, image):
        assert len(image.shape) == 2 # expect (H, W)-shape image
        image = torch.Tensor(image).view(1, 1, *image.shape).to(device=self.device)
        out = self.model(image).to(torch.device('cpu'))
        return np.array(out[0, 0])