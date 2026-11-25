import os
import sys

# Qt widgets are created off-screen: no window opens, and tests also run on machines without a display (CI)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

# the app modules live in the repository root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import SimpleITK as sitk


# direction matrices (row-major 3x3)
LPS = (1, 0, 0, 0, 1, 0, 0, 0, 1)
RAS = (-1, 0, 0, 0, -1, 0, 0, 0, 1)


def make_image(arr, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=LPS) -> sitk.Image:
    """sitk image from a (z, y, x) array with the given geometry"""
    img = sitk.GetImageFromArray(np.asarray(arr))
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(direction)
    return img


@pytest.fixture
def cube_image():
    """Synthetic CT: air (-1000 HU) with a soft-tissue cube (40 HU), RAS and anisotropic like a clinical scan"""
    arr = np.full((24, 48, 48), -1000, np.float32)
    arr[8:16, 16:32, 16:32] = 40
    return make_image(arr, spacing=(0.5, 0.5, 1.5), origin=(10.0, -20.0, 30.0), direction=RAS)
