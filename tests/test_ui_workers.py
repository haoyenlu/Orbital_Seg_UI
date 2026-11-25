"""Display pipeline of ImageLoaderWorker: slicing, windowing, label colors, and a full run"""
import numpy as np
import pytest
import SimpleITK as sitk

from conftest import make_image
from ui_data import DisplaySetting
from ui_workers import ImageLoaderWorker


def _worker(axis=0, **kw):
    return ImageLoaderWorker(display_setting=DisplaySetting(axis=axis, mask_color=[0, 255, 255]), **kw)


def test_get_slices_axial_coronal_sagittal():
    arr = np.arange(2 * 3 * 4).reshape(2, 3, 4)                         # (z, y, x), LPS
    axial    = _worker(0).get_slices(arr)
    coronal  = _worker(1).get_slices(arr)
    sagittal = _worker(2).get_slices(arr)

    assert len(axial) == 2    and axial[0].shape == (3, 4)
    assert len(coronal) == 3  and coronal[0].shape == (2, 4)
    assert len(sagittal) == 4 and sagittal[0].shape == (2, 3)

    # coronal / sagittal: superior (highest z) at the top of the screen
    assert np.array_equal(coronal[0][0], arr[-1, 0, :])
    assert np.array_equal(sagittal[0][0], arr[-1, :, 0])


def test_get_slices_invalid_axis():
    with pytest.raises(ValueError):
        _worker(3).get_slices(np.zeros((2, 2, 2)))


def test_window_and_normalize():
    w = _worker()
    windowed = w._apply_window(np.array([-1000.0, 0.0, 40.0, 1000.0]), width=80, level=40)
    assert np.array_equal(windowed, [0, 0, 40, 80])

    out = w._normalize(np.array([[0.0, 50.0, 100.0]]))
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 254  # 1e-6 guard against flat images
    assert w._normalize(np.zeros((2, 2))).max() == 0                      # flat image does not divide by 0


def test_color_regions_binary_and_multilabel():
    msk = np.array([0, 1, 2, 2])

    (region, color), = _worker()._color_regions(msk)
    assert region.tolist() == [False, True, True, True] and color == [0, 255, 255]

    labels  = [{'value': 1, 'color': [255, 255, 0]}, {'value': 2, 'color': [255, 0, 255]}]
    regions = _worker(labels=labels)._color_regions(msk)
    assert [r.tolist() for r, _ in regions] == [[False, True, False, False], [False, False, True, True]]
    assert [c for _, c in regions] == [[255, 255, 0], [255, 0, 255]]


@pytest.mark.parametrize('axis, n_frames', [(0, 36), (1, 24), (2, 24)])
def test_run_emits_one_frame_per_slice(qtbot, cube_image, axis, n_frames):
    """Full worker run: image + mask overlay, resampled to 1 mm for display (24 x 24 x 36 voxels)"""
    mask   = make_image((sitk.GetArrayFromImage(cube_image) > 0).astype(np.uint8),
                        cube_image.GetSpacing(), cube_image.GetOrigin(), cube_image.GetDirection())
    worker = _worker(axis, image=cube_image, mask=mask)
    errors = []
    worker.signal.error.connect(errors.append)

    with qtbot.waitSignal(worker.signal.result, timeout=10000) as blocker:
        worker.run()
    assert not errors
    assert len(blocker.args[0]) == n_frames
