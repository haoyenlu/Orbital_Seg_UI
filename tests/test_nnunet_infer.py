"""nnU-Net ensemble inference with a tiny fake results folder (same file layout as a real nnU-Net v2 model)"""
import json

import numpy as np
import pytest
import SimpleITK as sitk
import torch

import utils_sitk
from nnunet_infer import NNUNetEnsemble, PlainConvUNet

ARCH_KWARGS = {
    'n_stages': 3,
    'features_per_stage': [4, 8, 16],
    'kernel_sizes': [[3, 3, 3]] * 3,
    'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
    'n_conv_per_stage': [1, 1, 1],
    'n_conv_per_stage_decoder': [1, 1],
    'conv_bias': True,
    'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
    'nonlin_kwargs': {'inplace': True},
}
LABELS = {'background': 0, 'Nerve': 1, 'Fat': 2}


def write_results_folder(folder, folds=(0, 1), normalization='CTNormalization'):
    plans = {'configurations': {'3d_fullres': {
        'spacing': [1.0, 1.0, 1.0],
        'patch_size': [16, 16, 16],
        'normalization_schemes': [normalization],
        'architecture': {
            'network_class_name': 'dynamic_network_architectures.architectures.unet.PlainConvUNet',
            'arch_kwargs': ARCH_KWARGS,
        },
    }}}
    fingerprint = {'foreground_intensity_properties_per_channel': {'0': {
        'percentile_00_5': -1000.0, 'percentile_99_5': 1000.0, 'mean': 0.0, 'std': 100.0,
    }}}
    (folder / 'plans.json').write_text(json.dumps(plans))
    (folder / 'dataset.json').write_text(json.dumps({'labels': LABELS}))
    (folder / 'dataset_fingerprint.json').write_text(json.dumps(fingerprint))

    torch.manual_seed(0)
    for fold in folds:
        (folder / f'fold_{fold}').mkdir()
        net = PlainConvUNet(1, len(LABELS), ARCH_KWARGS)
        torch.save({'network_weights': net.state_dict(), 'inference_allowed_mirroring_axes': (0, 1, 2)},
                   folder / f'fold_{fold}' / 'checkpoint_final.pth')
    return folder


def test_network_output_shape():
    net = PlainConvUNet(1, 3, ARCH_KWARGS).eval()
    with torch.no_grad():
        assert net(torch.zeros(1, 1, 16, 16, 16)).shape == (1, 3, 16, 16, 16)


def test_state_dict_uses_nnunet_key_names():
    """Real nnU-Net checkpoints load with strict=True only if the module names match dynamic_network_architectures"""
    keys = PlainConvUNet(1, 3, ARCH_KWARGS).state_dict().keys()
    assert 'encoder.stages.0.0.convs.0.conv.weight' in keys
    assert 'decoder.encoder.stages.0.0.convs.0.conv.weight' in keys
    assert 'decoder.transpconvs.0.weight' in keys
    assert 'decoder.seg_layers.1.weight' in keys


def test_predict_returns_label_map_on_input_grid(tmp_path, cube_image):
    model = NNUNetEnsemble(str(write_results_folder(tmp_path)), folds=[0, 1], device=torch.device('cpu'))
    assert len(model.networks) == 2
    assert len(model.flips) == 8                                       # mirroring over 3 axes

    mask = model.predict(cube_image)
    assert utils_sitk.same_geometry(mask, cube_image)
    assert mask.GetPixelID() == sitk.sitkUInt8
    assert set(np.unique(sitk.GetArrayFromImage(mask))) <= set(LABELS.values())


def test_without_mirroring(tmp_path):
    model = NNUNetEnsemble(str(write_results_folder(tmp_path)), folds=[0], use_mirroring=False,
                           device=torch.device('cpu'))
    assert model.flips == [()]


def test_missing_fold_checkpoint(tmp_path):
    folder = write_results_folder(tmp_path, folds=(0,))
    with pytest.raises(FileNotFoundError):
        NNUNetEnsemble(str(folder), folds=[0, 1], device=torch.device('cpu'))


def test_unsupported_normalization(tmp_path):
    folder = write_results_folder(tmp_path, normalization='ZScoreNormalization')
    with pytest.raises(ValueError, match='normalization'):
        NNUNetEnsemble(str(folder), folds=[0], device=torch.device('cpu'))
