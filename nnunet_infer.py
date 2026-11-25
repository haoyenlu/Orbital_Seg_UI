"""
Stand-alone inference for nnU-Net v2 '3d_fullres' PlainConvUNet results folders (no nnunetv2 install needed).

Follows nnU-Net's default prediction:
  reorient to LPS (training data orientation) -> crop to nonzero -> CT normalization -> resample to plan spacing
  -> sliding window (patch size from plans, 50% overlap, gaussian weighting, mirroring TTA)
  -> average logits over folds -> softmax -> resample probabilities back to the input grid -> argmax
"""
import itertools
import json
import os

import numpy as np
import SimpleITK as sitk
import torch
from torch import nn
from monai.inferers import sliding_window_inference

import utils_sitk


# ---------------------------------------------------------------------------------------------------------------
# PlainConvUNet (same module names as dynamic_network_architectures, so nnU-Net checkpoints load with strict=True)
# ---------------------------------------------------------------------------------------------------------------
class ConvDropoutNormReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, conv_bias, norm_kwargs, nonlin_kwargs):
        super().__init__()
        padding     = [(k - 1) // 2 for k in kernel_size]
        self.conv   = nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding, bias=conv_bias)
        self.norm   = nn.InstanceNorm3d(out_ch, **norm_kwargs)
        self.nonlin = nn.LeakyReLU(**nonlin_kwargs)
        self.all_modules = nn.Sequential(self.conv, self.norm, self.nonlin)

    def forward(self, x):
        return self.all_modules(x)


class StackedConvBlocks(nn.Module):
    def __init__(self, n_convs, in_ch, out_ch, kernel_size, first_stride, **kw):
        super().__init__()
        self.convs = nn.Sequential(
            ConvDropoutNormReLU(in_ch, out_ch, kernel_size, first_stride, **kw),
            *[ConvDropoutNormReLU(out_ch, out_ch, kernel_size, 1, **kw) for _ in range(n_convs - 1)]
        )

    def forward(self, x):
        return self.convs(x)


class PlainConvEncoder(nn.Module):
    def __init__(self, in_ch, features, kernel_sizes, strides, n_conv_per_stage, **kw):
        super().__init__()
        stages = []
        for s in range(len(features)):
            stages.append(nn.Sequential(
                StackedConvBlocks(n_conv_per_stage[s], in_ch, features[s], kernel_sizes[s], strides[s], **kw)
            ))
            in_ch = features[s]
        self.stages          = nn.Sequential(*stages)
        self.output_channels = features
        self.strides         = strides
        self.kernel_sizes    = kernel_sizes

    def forward(self, x):
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


class UNetDecoder(nn.Module):
    def __init__(self, encoder: PlainConvEncoder, num_classes, n_conv_per_stage, conv_bias, **kw):
        super().__init__()
        self.encoder = encoder # registered like in nnU-Net (checkpoint has 'decoder.encoder.*' keys)
        n = len(encoder.output_channels)
        transpconvs, stages, seg_layers = [], [], []
        for s in range(1, n):
            below  = encoder.output_channels[-s]
            skip   = encoder.output_channels[-(s + 1)]
            stride = encoder.strides[-s]
            transpconvs.append(nn.ConvTranspose3d(below, skip, stride, stride, bias=conv_bias))
            stages.append(StackedConvBlocks(n_conv_per_stage[s - 1], 2 * skip, skip, encoder.kernel_sizes[-(s + 1)], 1,
                                            conv_bias=conv_bias, **kw))
            seg_layers.append(nn.Conv3d(skip, num_classes, 1, 1, 0, bias=True))
        self.transpconvs = nn.ModuleList(transpconvs)
        self.stages      = nn.ModuleList(stages)
        self.seg_layers  = nn.ModuleList(seg_layers) # all kept for loading; only the last one is used (no deep supervision)

    def forward(self, skips):
        x = skips[-1]
        for s in range(len(self.stages)):
            x = self.transpconvs[s](x)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
        return self.seg_layers[-1](x)


class PlainConvUNet(nn.Module):
    def __init__(self, in_ch, num_classes, arch_kwargs: dict):
        super().__init__()
        a  = arch_kwargs
        kw = dict(conv_bias=a['conv_bias'], norm_kwargs=a['norm_op_kwargs'], nonlin_kwargs=a['nonlin_kwargs'] or {})
        self.encoder = PlainConvEncoder(in_ch, a['features_per_stage'], a['kernel_sizes'], a['strides'],
                                        a['n_conv_per_stage'], **kw)
        kw.pop('conv_bias')
        self.decoder = UNetDecoder(self.encoder, num_classes, a['n_conv_per_stage_decoder'], a['conv_bias'], **kw)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------------------------------------------
# Fold ensemble predictor
# ---------------------------------------------------------------------------------------------------------------
class NNUNetEnsemble:
    def __init__(self, model_folder: str, folds=(0, 1, 2, 3, 4), checkpoint_name='checkpoint_final.pth',
                 configuration='3d_fullres', use_mirroring=True, step_size=0.5, device=None):
        with open(os.path.join(model_folder, 'plans.json')) as f:
            plans = json.load(f)
        with open(os.path.join(model_folder, 'dataset.json')) as f:
            dataset = json.load(f)
        with open(os.path.join(model_folder, 'dataset_fingerprint.json')) as f:
            fingerprint = json.load(f)

        cfg = plans['configurations'][configuration]
        arch = cfg['architecture']
        if not arch['network_class_name'].endswith('PlainConvUNet'):
            raise ValueError(f"Unsupported nnU-Net architecture: {arch['network_class_name']}")
        if cfg['normalization_schemes'] != ['CTNormalization']:
            raise ValueError(f"Unsupported normalization: {cfg['normalization_schemes']}")
        if plans.get('transpose_forward', [0, 1, 2]) != [0, 1, 2]:
            raise ValueError("Plans with transpose_forward != [0,1,2] are not supported")

        self.spacing    = cfg['spacing']    # (z, y, x) like nnU-Net arrays
        self.patch_size = cfg['patch_size'] # (z, y, x)
        self.step_size  = step_size
        self.num_classes = len(dataset['labels'])
        props = fingerprint['foreground_intensity_properties_per_channel']['0']
        self.clip_range = (props['percentile_00_5'], props['percentile_99_5'])
        self.mean, self.std = props['mean'], props['std']

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.networks, mirror_axes = [], None
        for fold in folds:
            ckpt = torch.load(os.path.join(model_folder, f'fold_{fold}', checkpoint_name),
                              map_location='cpu', weights_only=False) # own trusted checkpoint (holds non-tensor objects)
            net = PlainConvUNet(1, self.num_classes, arch['arch_kwargs'])
            net.load_state_dict(ckpt['network_weights'], strict=True)
            self.networks.append(net.to(self.device).eval())
            mirror_axes = ckpt.get('inference_allowed_mirroring_axes', (0, 1, 2))

        # every combination of flipped spatial axes (tensor dims 2,3,4); () = no flip
        axes = [a + 2 for a in (mirror_axes or ())] if use_mirroring else []
        self.flips = [c for r in range(len(axes) + 1) for c in itertools.combinations(axes, r)]

    @torch.no_grad()
    def _predict_patch(self, x):
        """Average logits over all folds and mirrorings"""
        total = None
        for net in self.networks:
            for flip in self.flips:
                y = net(torch.flip(x, flip) if flip else x)
                y = torch.flip(y, flip) if flip else y
                total = y.float() if total is None else total + y.float()
        return total / (len(self.networks) * len(self.flips))

    @torch.no_grad()
    def predict(self, simg: sitk.Image) -> sitk.Image:
        """Return a label map (uint8, dataset.json label values) on the grid of simg"""
        original = simg

        # training data was LPS at plan spacing
        simg = utils_sitk.reorient_to_identity(sitk.Cast(simg, sitk.sitkFloat32))
        if not np.allclose(simg.GetSpacing(), self.spacing[::-1]):
            simg = utils_sitk.resample(simg, self.spacing[::-1], sitk.sitkBSpline)
        arr = sitk.GetArrayFromImage(simg).astype(np.float32) # (z, y, x)

        # crop to nonzero region like nnU-Net
        nz = np.argwhere(arr != 0)
        lo, hi = (nz.min(0), nz.max(0) + 1) if len(nz) else (np.zeros(3, int), np.array(arr.shape))
        crop = arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

        # CTNormalization
        crop = (np.clip(crop, *self.clip_range) - self.mean) / max(self.std, 1e-8)

        x = torch.from_numpy(crop)[None, None].to(self.device)
        with torch.autocast(self.device.type, enabled=self.device.type == 'cuda'):
            logits = sliding_window_inference(
                inputs=x, roi_size=self.patch_size, sw_batch_size=1, predictor=self._predict_patch,
                overlap=1 - self.step_size, mode='gaussian', sigma_scale=0.125, padding_mode='constant', cval=0,
                progress=True,
            )
        probs_crop = torch.softmax(logits[0].float(), 0).cpu().numpy() # (C, z, y, x)

        # paste back into the full (resampled) volume; outside the crop is background
        probs = np.zeros((self.num_classes,) + arr.shape, np.float32)
        probs[0] = 1
        probs[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = probs_crop

        # resample probabilities (linear) onto the original grid, then argmax
        out = []
        for c in range(self.num_classes):
            p = sitk.GetImageFromArray(probs[c])
            p.CopyInformation(simg)
            p = sitk.Resample(p, original, sitk.Transform(), sitk.sitkLinear, 1.0 if c == 0 else 0.0, sitk.sitkFloat32)
            out.append(sitk.GetArrayFromImage(p))
        labels = np.argmax(np.stack(out), 0).astype(np.uint8)

        mask = sitk.GetImageFromArray(labels)
        mask.CopyInformation(original)
        return mask
