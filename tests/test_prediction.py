"""PyTorch sliding-window prediction path (utils.pytorch_sliding_window_inference) with a stand-in model"""
import numpy as np
import SimpleITK as sitk
import torch
from torch import nn

import utils_sitk
from utils import pytorch_sliding_window_inference


class ThresholdModel(nn.Module):
    """3-class 'network': class 1 where the normalized intensity is bright, else class 0; class 2 never wins"""
    def forward(self, x):
        return torch.cat([0.5 - x, x - 0.5, torch.full_like(x, -10)], dim=1)


PARAMS = {'spacing': [1.0, 1.0, 1.0], 'clip_range': [-1024, 1024], 'roi_size': [16, 16, 16], 'normalize': True}


def _predict(image, **extra):
    return pytorch_sliding_window_inference(ThresholdModel(), image, params={**PARAMS, **extra},
                                            sw_batch_size=4, overlap=0.5, target_class=1)


def _dice(a, b):
    return 2 * np.logical_and(a, b).sum() / (a.sum() + b.sum())


def test_mask_is_on_original_grid(cube_image):
    """Prediction resamples / reorients internally, the mask must come back on the input grid"""
    mask = _predict(cube_image)
    assert utils_sitk.same_geometry(mask, cube_image)
    assert mask.GetPixelID() == sitk.sitkUInt8


def test_mask_matches_object(cube_image):
    mask  = sitk.GetArrayFromImage(_predict(cube_image)) > 0
    truth = sitk.GetArrayFromImage(cube_image) > 0
    assert _dice(mask, truth) > 0.9


def test_binary_vs_multilabel_output(cube_image):
    binary = sitk.GetArrayFromImage(_predict(cube_image))
    assert set(np.unique(binary)) <= {0, 1}

    labels = {1: {'name': 'Nerve', 'color': [255, 255, 0]}, 2: {'name': 'Fat', 'color': [255, 0, 255]}}
    multi  = sitk.GetArrayFromImage(_predict(cube_image, labels=labels))
    assert set(np.unique(multi)) <= {0, 1, 2}
    assert (multi == 1).sum() == (binary == 1).sum()
