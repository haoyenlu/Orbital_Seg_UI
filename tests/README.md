# Tests

Unit tests for SegUI, written with [pytest](https://docs.pytest.org) and [pytest-qt](https://pytest-qt.readthedocs.io). They use only synthetic images created inside the tests, so they need no patient data. Only the tests marked `slow` need the model weights.

## Running

Run all commands from the repository root, with the development requirements installed (`pip install -r requirements.txt`).

```bash
pytest                                   # fast tests (about 10 s)
pytest -m slow                           # real-weights tests only (about 1 min, needs checkpoints/)
pytest -m "slow or not slow"             # everything
pytest tests/test_utils_sitk.py          # one file
pytest -k resample                       # tests whose name contains "resample"
pytest -x -v                             # stop at the first failure, list every test
```

Settings are in [`pytest.ini`](../pytest.ini). It skips `slow` tests by default and hides deprecation and user warnings from the libraries.

## What is tested

| File | Covers | Tests |
|---|---|---|
| [`test_utils_sitk.py`](test_utils_sitk.py) | `utils_sitk.py`: DICOM series reading (slice sorting, spacing, rescale, tags), display names, geometry comparison, resampling, reorientation, mask-import checks | 26 |
| [`test_prediction.py`](test_prediction.py) | `utils.pytorch_sliding_window_inference` with a stand-in threshold model: output on the input grid, object recovered, binary vs multi-label | 3 |
| [`test_nnunet_infer.py`](test_nnunet_infer.py) | `nnunet_infer.py` with a tiny fake nnU-Net results folder: nnU-Net weight names, fold ensemble with mirroring, output geometry, missing fold, unsupported plans | 6 |
| [`test_ui_workers.py`](test_ui_workers.py) | `ImageLoaderWorker`: slicing for each view, windowing, label colours, full image + mask overlay run | 7 |
| [`test_left_panel.py`](test_left_panel.py) | `LeftPanel`: adding images and masks, duplicate names, multi-selection for prediction, `itemChange(None)` regression | 5 |
| [`test_models_real.py`](test_models_real.py) | Every model in `config.yaml` with its real checkpoint (`slow`); a model whose checkpoint is missing is skipped | 6 |

**Regression tests.** Two tests guard against bugs that were fixed, so don't delete them:
- `test_resample_does_not_shift_objects`: resampling used to shift images by half a voxel, so masks came back 0.5 mm off.
- `test_reset_emits_none`: clearing the list used to crash the app.

**Not covered yet:**
- `MainWindow` dialog flows: opening files, Import Mask, Export.
- Menus.
- `utils.vPrediction` (Keras) with a stand-in model; the Keras models are only exercised by the `slow` tests.

## Shared helpers ([`conftest.py`](conftest.py))

| Name | What it is |
|---|---|
| `make_image(arr, spacing, origin, direction)` | Builds a SimpleITK image from a `(z, y, x)` NumPy array with the given geometry |
| `LPS`, `RAS` | Direction matrices. Use `RAS` to test the reorientation code paths. |
| `cube_image` (fixture) | Synthetic CT: air (-1000 HU) with a 40 HU cube. It is RAS at 0.5 × 0.5 × 1.5 mm, like a clinical scan, so tests exercise resampling and reorientation. |
| `qtbot` (pytest-qt fixture) | Creates widgets and waits for signals, e.g. `with qtbot.waitSignal(panel.itemChange) as blocker:` |

`conftest.py` also adds the repository root to `sys.path` so tests can `import utils_sitk` etc. It sets `QT_QPA_PLATFORM=offscreen`, so Qt widgets are created without opening windows and tests run on machines without a display (CI).

## Writing new tests

- **Patient data.** Never add patient scans, masks or real DICOM files here. Build the input in the test with `make_image`, or write synthetic DICOM as in `_write_ct_series`, and put temporary files in pytest's `tmp_path`.
- **Placement.** Test the logic, not the dialogs. If something is only reachable through a `QFileDialog` or `QMessageBox`, move the logic into a plain function, as `utils_sitk.validate_mask` was moved out of `MainWindow.import_mask`, and test that. To test a dialog flow itself, replace the dialog with `monkeypatch`, e.g. `monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (path, ''))`.
- **Geometry.** Check geometry, not only values. Every prediction path must return a mask on the input grid, so assert `utils_sitk.same_geometry(mask, image)`.
- **Speed.** Keep fast tests fast: small volumes, tiny networks, `device=torch.device('cpu')` where the result must be deterministic. Mark anything that needs real weights or takes more than a few seconds with `@pytest.mark.slow`.
- **Qt objects.** Register widgets with `qtbot.addWidget(widget)` so they are cleaned up. Don't call `show()`; it isn't needed to test behaviour.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: pytestqt` | `pip install pytest-qt` |
| `qt.qpa.plugin: Could not load the Qt platform plugin "offscreen"` on Linux | Install Qt's system libraries, e.g. `sudo apt-get install libegl1 libxkbcommon0` |
| `slow` tests are all skipped | The checkpoints in `config.yaml` aren't in `checkpoints/`; see the main [README](../README.md#model-weights) |
| Prediction tests differ slightly between GPU and CPU | Expected; the PyTorch tests use the GPU when available, so assert overlap (Dice) rather than exact voxels |
