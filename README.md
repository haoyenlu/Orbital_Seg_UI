# SegUI

SegUI is a desktop application for segmenting orbital CT scans with deep learning models. Load a CT (DICOM, NIfTI or MetaImage), choose a model, run the prediction and inspect or export the mask.

Developed at the Beyond Vision Laboratory, University of Pittsburgh, for the Pediatric Orbital Project (NIH R01 EY033973).

> **Research use only.** SegUI is not a medical device and must not be used for clinical diagnosis or treatment decisions.

## Features

- **Open images:** DICOM series folders, NIfTI (`.nii`, `.nii.gz`) and MetaImage (`.mhd`) volumes, shown in axial, coronal and sagittal views with window/level presets.
- **Segmentation models** (configured in `config.yaml`):

  | Model | Output |
  |---|---|
  | Abscess | abscess mask |
  | Orbit Conal | orbit cone mask |
  | Fat, Nerve, Fat & Nerve | R2-UNet orbital fat and optic nerve masks |
  | nnUNet Fat & Nerve | 5-fold nnU-Net ensemble, fat and optic nerve |

- **Batch prediction:** select several images and predict them all; the first one is displayed.
- **Import Mask (File → Import Mask...):** attach an existing mask to the selected image. Its size must match the image.
- **Export (File → Export Result):** save a mask as NIfTI with the original image's spacing, origin and orientation.
- **Display:** multi-label masks shown in per-label colors; overlay or contour mode; adjustable opacity.
- **Info panel:** path, size, spacing, origin and orientation of the selected image or mask.

## Installation (from source)

Requires Python 3.10 and Windows or Linux. An NVIDIA GPU is recommended; the models also run on the CPU, but slowly.

```bash
git clone https://github.com/haoyenlu/Orbital_Seg_UI.git
cd Orbital_Seg_UI
python -m venv venv
venv\Scripts\activate            # Linux: source venv/bin/activate

# PyTorch with CUDA 12.6 (see https://pytorch.org for other versions)
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Model weights

The trained weights are **not** stored in this repository. They are distributed separately as assets on this repository's [Releases](../../releases) page. Download them and place them under `checkpoints/` so that the layout matches the `checkpoint` entries in `config.yaml`:

```
SegUI/
├── checkpoints/
│   ├── 033--0.6830-V1-0.5.h5                  # Abscess (Keras)
│   ├── Orbitconal-1.7414.h5                   # Orbit Conal (Keras)
│   ├── fat_nerve_r2_new_model.pth             # Fat / Nerve / Fat & Nerve (PyTorch)
│   └── nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/   # nnUNet Fat & Nerve
│       ├── plans.json
│       ├── dataset.json
│       ├── dataset_fingerprint.json
│       └── fold_0 … fold_4/checkpoint_final.pth
├── config.yaml
└── ui.py
```

You only need the weights for the models you use; a missing checkpoint only causes an error when that model is run.

Check that the weights load and predict correctly. This doesn't open a window:

```bash
python ui.py --selftest
```

### Adding your own model

Add an entry under `models:` in `config.yaml`. To reuse an existing architecture with a new checkpoint, use **Model → Add Model** in the app. For an nnU-Net v2 `3d_fullres` model, copy the trained results folder (with `plans.json`, `dataset.json`, `dataset_fingerprint.json` and `fold_*/checkpoint_final.pth`) into `checkpoints/` and add an entry of `type: 'nnunet'`; nnU-Net itself does not need to be installed.

## Usage

```bash
python ui.py
```

1. Open an image from the **File** menu (**Open Dicom Folder**, **Open Nifti File** or **Import MHD File**).
2. Choose a model from the **Model** menu.
3. Select one or more images in the left panel and click **Run Prediction**.
4. The predicted mask appears under its image. Select it and use **File → Export Result** to save it.

## Building the Windows executable

```bash
python -m PyInstaller SegUI.spec --noconfirm
```

The build takes about 10 minutes. It creates `dist/SegUI/` containing `SegUI.exe`, an `_internal/` folder, `config.yaml` and the checkpoints referenced by `config.yaml`. Distribute the whole folder. To skip copying the checkpoints, set the environment variable `SEGUI_COPY_CHECKPOINTS=0`.

- Check the build: `dist\SegUI\SegUI.exe --selftest`.
- The executable has no console window. Messages and errors are written to `SegUI.log` next to `SegUI.exe`.

## Data privacy

Never commit patient data to this repository. `.gitignore` excludes image and mask formats (`*.nii*`, `*.dcm`, `*.mhd`, …), model weights and notebooks. Check `git status` before every commit.

## License

The source code is licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attributions, including nnU-Net, whose network architecture and inference procedure `nnunet_infer.py` re-implements.

SegUI uses PyQt5, which is licensed under the GPL v3. The executable bundles PyQt5, so `SegUI.exe` is distributed under the GPL v3.

Model weights are distributed separately and may have their own terms of use.

## Citation

If you use SegUI in your research, please cite this repository. If you use the nnU-Net model, please also cite:

> Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18(2), 203–211.
