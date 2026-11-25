import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

import utils_sitk
from conftest import LPS, RAS, make_image


# ---------------------------------------------------------------- display names
@pytest.mark.parametrize('name, expected', [
    ('scan.nii.gz', 'scan'),
    ('scan.NII.GZ', 'scan'),
    ('scan.nii',    'scan'),
    ('scan.mhd',    'scan'),
    ('scan.v2.nrrd', 'scan.v2'),
    ('folder',      'folder'),
])
def test_strip_image_ext(name, expected):
    assert utils_sitk.strip_image_ext(name) == expected


@pytest.mark.parametrize('path, expected', [
    ('E:/study/case_017.nii.gz',            'case_017'),            # informative file name is enough
    ('E:/study/case_017/CT/image.nii.gz',   'case_017/CT/image'),   # generic names pull in parents (max 3 parts)
    ('E:/study/case_017/SE0003',            'case_017/SE0003'),     # DICOM series folder
    ('E:/a/b/c/d/1.2.840.113619.2.55.3.6', 'c/d/1.2.840.113619.2.55.3.6'),  # UID folder
])
def test_name_from_path(path, expected):
    assert utils_sitk.name_from_path(path) == expected


def _tagged(pid='', desc='', num=''):
    img = sitk.Image(2, 2, 2, sitk.sitkInt16)
    for key, val in (('0010|0020', pid), ('0008|103e', desc), ('0020|0011', num)):
        if val:
            img.SetMetaData(key, val)
    return img


def test_dicom_display_name_uses_tags():
    assert utils_sitk.dicom_display_name('E:/x/SE0003', _tagged('P001', 'Orbit CT', '3')) == 'P001 - Orbit CT #3'
    assert utils_sitk.dicom_display_name('E:/x/SE0003', _tagged('P001', '', '3')) == 'P001 - Series 3'
    assert utils_sitk.dicom_display_name('E:/x/SE0003', _tagged('P001')) == 'P001'


def test_dicom_display_name_anonymous_falls_back_to_path():
    assert utils_sitk.dicom_display_name('E:/x/case_017/SE0003', _tagged('ANONYMOUS', 'Orbit CT', '3')) \
        == 'case_017/SE0003 - Orbit CT #3'


# ---------------------------------------------------------------- geometry
def test_same_geometry():
    a = make_image(np.zeros((4, 5, 6)), spacing=(0.5, 0.5, 1), origin=(1, 2, 3), direction=RAS)
    b = make_image(np.ones((4, 5, 6)),  spacing=(0.5, 0.5, 1), origin=(1, 2, 3.0001), direction=RAS)
    assert utils_sitk.same_geometry(a, b)                                       # within tolerance

    assert not utils_sitk.same_geometry(a, make_image(np.zeros((4, 5, 7)), (0.5, 0.5, 1), (1, 2, 3), RAS))
    assert not utils_sitk.same_geometry(a, make_image(np.zeros((4, 5, 6)), (0.5, 0.5, 2), (1, 2, 3), RAS))
    assert not utils_sitk.same_geometry(a, make_image(np.zeros((4, 5, 6)), (0.5, 0.5, 1), (1, 2, 4), RAS))
    assert not utils_sitk.same_geometry(a, make_image(np.zeros((4, 5, 6)), (0.5, 0.5, 1), (1, 2, 3), LPS))


def test_resample_keeps_physical_extent():
    img = make_image(np.zeros((50, 100, 100), np.float32), spacing=(0.5, 0.5, 1.0), origin=(1, 2, 3), direction=RAS)
    out = utils_sitk.resample(img, (1.0, 1.0, 1.0))
    assert out.GetSize() == (50, 50, 50)
    assert out.GetSpacing() == (1.0, 1.0, 1.0)
    assert out.GetDirection() == img.GetDirection()

    # both grids start and end at the same physical edges (voxel corners), so nothing shifts
    for corner in [(-0.5, -0.5, -0.5), (99.5, 99.5, 49.5)]:
        out_corner = [(c + 0.5) * s_in / s_out - 0.5 for c, s_in, s_out in zip(corner, img.GetSpacing(), out.GetSpacing())]
        assert np.allclose(img.TransformContinuousIndexToPhysicalPoint(corner),
                           out.TransformContinuousIndexToPhysicalPoint(out_corner))


def test_resample_does_not_shift_objects():
    arr = np.zeros((20, 40, 40), np.float32)
    arr[5:15, 10:30, 10:30] = 1
    img = make_image(arr, spacing=(0.5, 0.5, 1.0), origin=(3, 4, 5), direction=RAS)
    out = utils_sitk.resample(img, (1.0, 1.0, 1.0), sitk.sitkNearestNeighbor)

    def center(im):
        idx = np.argwhere(sitk.GetArrayFromImage(im) > 0).mean(0)[::-1]   # (x, y, z)
        return np.array(im.TransformContinuousIndexToPhysicalPoint(idx.tolist()))
    assert np.allclose(center(img), center(out), atol=1e-6)


def test_reorient_to_identity_keeps_anatomy(cube_image):
    out = utils_sitk.reorient_to_identity(cube_image)
    assert np.allclose(out.GetDirection(), LPS)

    # the same physical point has the same intensity before and after
    for idx in [(0, 0, 0), (20, 20, 10), (47, 10, 23)]:
        point = cube_image.TransformIndexToPhysicalPoint(idx)
        assert out[out.TransformPhysicalPointToIndex(point)] == cube_image[idx]


def test_reorient_to_identity_is_noop_for_lps():
    img = make_image(np.zeros((3, 3, 3)))
    assert utils_sitk.reorient_to_identity(img) is img


# ---------------------------------------------------------------- DICOM reading
def _write_ct_series(folder, n=5, dz=2.0):
    """Synthetic CT series; files are named in reverse slice order to check sorting by position"""
    series_uid = generate_uid()
    for k in range(n):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID    = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID          = ExplicitVRLittleEndian

        ds = Dataset()
        ds.file_meta         = meta
        ds.SOPClassUID       = CTImageStorage
        ds.SOPInstanceUID    = meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = series_uid
        ds.Modality          = 'CT'
        ds.PatientID         = 'TEST001'
        ds.SeriesDescription = 'Synthetic'
        ds.SeriesNumber      = 3
        ds.InstanceNumber    = k + 1
        ds.ImagePositionPatient    = [-10.0, -20.0, 5.0 + k * dz]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing            = [0.5, 0.75]   # [row (y), column (x)]
        ds.SliceThickness          = dz
        ds.Rows, ds.Columns        = 8, 6
        ds.SamplesPerPixel           = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
        ds.PixelData = np.full((8, 6), 1000 + k, np.uint16).tobytes()   # HU = k - 24
        ds.save_as(folder / f'IM{n - k:04d}', enforce_file_format=True)


def test_read_dicom_by_slice(tmp_path):
    _write_ct_series(tmp_path)
    (tmp_path / 'notes.txt').write_text('not a DICOM file')            # must be skipped

    img = utils_sitk.readDICOMbySlice(str(tmp_path))
    arr = sitk.GetArrayFromImage(img)

    assert img.GetSize() == (6, 8, 5)
    assert np.allclose(img.GetSpacing(), (0.75, 0.5, 2.0))
    assert np.allclose(img.GetOrigin(), (-10, -20, 5))                  # lowest slice
    assert np.allclose(img.GetDirection(), LPS)
    assert [arr[k, 0, 0] for k in range(5)] == [k - 24 for k in range(5)]  # sorted by position, rescaled
    assert img.GetMetaData('0010|0020') == 'TEST001'
    assert img.GetMetaData('0008|103e') == 'Synthetic'
    assert img.GetMetaData('0020|0011') == '3'


def test_read_dicom_by_slice_empty_folder(tmp_path):
    with pytest.raises(ValueError, match='No DICOM files'):
        utils_sitk.readDICOMbySlice(str(tmp_path))


# ---------------------------------------------------------------- mask import
def test_validate_mask_accepts_matching_label_map():
    image = make_image(np.zeros((4, 5, 6), np.float32))
    mask  = make_image(np.array([0, 1, 2] * 40).reshape(4, 5, 6).astype(np.float32))  # float but integer-valued
    assert list(utils_sitk.validate_mask(image, mask)) == [0, 1, 2]


@pytest.mark.parametrize('mask, message', [
    (make_image(np.zeros((4, 5, 7), np.uint8)),          'does not match image size'),
    (make_image(np.full((4, 5, 6), -1, np.int16)),       'between 0 and 255'),
    (make_image(np.full((4, 5, 6), 256, np.int16)),      'between 0 and 255'),
    (make_image(np.full((4, 5, 6), 0.5, np.float32)),    'between 0 and 255'),
    (sitk.Compose([make_image(np.zeros((4, 5, 6), np.uint8))] * 3), 'single-channel'),
])
def test_validate_mask_rejects(mask, message):
    image = make_image(np.zeros((4, 5, 6), np.float32))
    with pytest.raises(ValueError, match=message):
        utils_sitk.validate_mask(image, mask)


def test_mask_labels():
    palette = [[255, 0, 0], [0, 255, 0]]
    assert utils_sitk.mask_labels(np.array([0, 1]), palette) is None    # binary: single mask color
    assert utils_sitk.mask_labels(np.array([0]), palette) is None

    labels = utils_sitk.mask_labels(np.array([0, 1, 2, 5]), palette)
    assert [l['value'] for l in labels] == [1, 2, 5]
    assert [l['color'] for l in labels] == [[255, 0, 0], [0, 255, 0], [255, 0, 0]]  # palette wraps around
    assert labels[2]['name'] == 'Label 5'
