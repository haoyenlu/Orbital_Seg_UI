"""Load each model in config.yaml with its real weights and predict a synthetic volume.

Skipped by default (needs checkpoints/ and is slow). Run with:  pytest -m slow
"""
import os

import numpy as np
import pytest
import SimpleITK as sitk
import yaml

import utils_sitk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(ROOT, 'config.yaml')) as f:
    MODELS = yaml.safe_load(f)['models']


@pytest.mark.slow
@pytest.mark.parametrize('name', list(MODELS))
def test_model_predicts(name, cube_image, monkeypatch):
    from ui_workers import PredictionWorker, load_model

    detail = MODELS[name]
    monkeypatch.chdir(ROOT)                                             # checkpoint paths are relative to the repo
    if not os.path.exists(detail['checkpoint']):
        pytest.skip(f"checkpoint not found: {detail['checkpoint']}")

    model = load_model(detail)
    mask  = PredictionWorker([], name, detail['function'], detail)._predict(model, cube_image)

    assert utils_sitk.same_geometry(mask, cube_image)
    values = set(np.unique(sitk.GetArrayFromImage(mask)).tolist())
    assert values <= ({0} | set(detail['labels']) if 'labels' in detail else {0, 1})
