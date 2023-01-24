
import typing

import torch
import torch.nn as nn
import torchvision.transforms as T


class PovertyMapTransform(nn.Module):
    _BAND_ORDER: list = [
        'BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS'
    ]
    _MEANS_2009_17: dict = {
        'BLUE':  0.059183,
        'GREEN': 0.088619,
        'RED':   0.104145,
        'SWIR1': 0.246874,
        'SWIR2': 0.168728,
        'TEMP1': 299.078023,
        'NIR':   0.253074,
        'NIGHTLIGHTS': 5.101585,
        'DMSP':  4.005496,  # does not exist in current dataset
        'VIIRS': 1.096089,  # does not exist in current dataset
    }
    _STD_DEVS_2009_17: dict = {
        'BLUE':  0.022926,
        'GREEN': 0.031880,
        'RED':   0.051458,
        'SWIR1': 0.088857,
        'SWIR2': 0.083240,
        'TEMP1': 4.300303,
        'NIR':   0.058973,
        'NIGHTLIGHTS': 23.342916,
        'DMSP':  23.038301,  # does not exist in current dataset
        'VIIRS': 4.786354,   # does not exist in current dataset
    }
    size: tuple = (224, 224)

    def __init__(self, augmentation: bool = False, randaugment: bool = True, **kwargs):
        """All images have already been mean-subtracted and normalized."""
        super(PovertyMapTransform, self).__init__()
        
        self.mean: tuple = (self._MEANS_2009_17[k] for k in self._BAND_ORDER)    # XXX; unnces?
        self.std: tuple = (self._STD_DEVS_2009_17[k] for k in self._BAND_ORDER)  # XXX; unnecs?
        self.augmentation: bool = augmentation
        self.randaugment: bool = randaugment
        
        if self.augmentation:
            self.base_transform_on_multispectral = nn.Sequential(
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.5),
                T.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=0.1, scale=(0.9, 1.1)),
            )
            if self.randaugment:
                self.color_transform_on_rgb = nn.Sequential(
                    T.ConvertImageDtype(torch.uint8),
                    T.RandAugment(num_ops=2, magnitude=9, ),
                    T.ConvertImageDtype(torch.float),
                    T.ColorJitter(brightness=.8, contrast=.8, saturation=.8, hue=.1)
                )
            else:
                self.color_transform_on_rgb = T.ColorJitter(
                    brightness=.8, contrast=.8, saturation=.8, hue=.1
                )
        else:
            self.base_transform_on_multispectral = None
            self.color_transform_on_rgb = None

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        
        # without augmentation
        if not self.augmentation:
            return x
        
        # with augmentation
        x = self.base_transform_on_multispectral(x)               # (N, 8, H, W)
        bgr_unnormalized = self._unnormalize_bgr(x[:, :3, :, :])  # (N, 3, H, W); in BGR order
        rgb_unnormalized = bgr_unnormalized[:, [2, 1, 0], :, :]   # (N, 3, H, W); in RGB order
        rgb_color_transformed = \
            self.color_transform_on_rgb(rgb_unnormalized)         # (N, 3, H, W); in RGB order
        bgr_color_transformed = \
            rgb_color_transformed[:, [2, 1, 0], :, :]             # (N, 3, H, W); in BGR order
        bgr_normalized = \
            self._normalize_bgr(bgr_color_transformed)            # (N, 3, H, W); in BGR order
        x[:, :3, :, :] = bgr_normalized
        x = self.cutout(x)
        
        return x

    def _unnormalize_bgr(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Unnormalize the first 3 channels in multispectral 8-channel image."""
        assert (x.ndim == 4) and (x.shape[1] == 3)
        return (x * self.bgr_stds.to(x.device)) + self.bgr_means.to(x.device)
    
    def _normalize_bgr(self, x: torch.FloatTensor) -> torch.FloatTensor:
        assert (x.ndim == 4) and (x.shape[1] == 3)
        return (x - self.bgr_means.to(x.device)) / self.bgr_stds.to(x.device)
        
    @property
    def bgr_means(self):
        return torch.tensor([self._MEANS_2009_17[c] for c in ['BLUE', 'GREEN', 'RED']]).view(-1, 1, 1)
    
    @property
    def bgr_stds(self):
        return torch.tensor([self._STD_DEVS_2009_17[c] for c in ['BLUE', 'GREEN', 'RED']]).view(-1, 1, 1)

    @staticmethod
    def cutout(img: torch.FloatTensor):
        
        assert (img.ndim == 4) and (img.shape[2] == img.shape[3])
        
        def _sample_uniform(a: float, b: float) -> float:
            return torch.empty(1).uniform_(a, b).item()
        
        _, _, _, width = img.shape
        cutout_width = _sample_uniform(0., width / 2.)
        cutout_center_x = _sample_uniform(0., width)
        cutout_center_y = _sample_uniform(0., width)
        
        x0 = int(max(0, cutout_center_x - cutout_width / 2))
        y0 = int(max(0, cutout_center_y - cutout_width / 2))
        x1 = int(min(width, cutout_center_x + cutout_width / 2))
        y1 = int(min(width, cutout_center_y + cutout_width / 2))

        # fill with zeros (expects zero-mean-normalized input for x)
        img[:, :, x0:x1, y0:y1] = 0.
        return img