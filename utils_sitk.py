import SimpleITK as sitk
from matplotlib import image
import numpy as np
from skimage.transform import resize as img_resize
import os
import pydicom
from pydicom.errors import InvalidDicomError
import glob
import re
from pathlib import Path

def readDICOM(dicom_dir):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    dicom_files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    reader.SetFileNames(dicom_files)

    # load metadata too
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    img = reader.Execute()
    return img

def readDICOMbySlice(dicom_dir):
    print(f"Reading DICOM series from folder: {dicom_dir}")
    files     = []
    # DICOM files often have no extension, so try every file in the folder
    for fpath in sorted(glob.glob(os.path.join(dicom_dir,'*'))):
        if not os.path.isfile(fpath): continue
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
        except InvalidDicomError:
            continue
        # skip non-image objects (DICOMDIR, reports, ...)
        if 'ImagePositionPatient' not in ds or 'ImageOrientationPatient' not in ds: continue
        image_type = ds.get('ImageType', [])
        if 'AVERAGE' not in image_type and 'REFORMATTED' in image_type: continue
        files.append(fpath)



    if not files:
        raise ValueError("No DICOM files found in the specified directory.")
    
    # get orientation from first slice
    first_ds = pydicom.dcmread(files[0], stop_before_pixels=True)
    iop      = np.array(first_ds.ImageOrientationPatient, dtype=np.float32)
    row_cosine = iop[0:3] # x-axis
    col_cosine = iop[3:6] # y-axis


    row_cosine /= np.linalg.norm(row_cosine)
    col_cosine /= np.linalg.norm(col_cosine)
    normal      = np.cross(row_cosine, col_cosine) # z-axis
    normal     /= np.linalg.norm(normal)

    # spacing
    spacing_y = float(first_ds.PixelSpacing[0])
    spacing_x = float(first_ds.PixelSpacing[1])

    slice_info = []
    for f in files:
        ds = pydicom.dcmread(f,stop_before_pixels=True)
        ipp = np.array(ds.ImagePositionPatient, dtype=np.float32)
        pos = np.dot(ipp, normal)
        itn = ds.InstanceNumber if 'InstanceNumber' in ds else 0
        slice_info.append((pos, ipp, f, itn))
    
    slice_info.sort(key=lambda x: x[0]) # sort by position along normal direction
    
    positions = np.array([info[0] for info in slice_info]) 
    diffz = np.abs(np.diff(positions))
    if len(diffz) > 0:
        spacing_z = np.median(diffz)
    else: # single slice: fall back to slice thickness
        spacing_z = float(first_ds.get('SliceThickness', 1.0) or 1.0)

    # origin
    origin = slice_info[0][1] 

    # orientation code
    axis_labels = [('L', 'R'), ('P', 'A'), ('S', 'I')]
    def dominant_axis(cosine):
        idx = np.argmax(np.abs(cosine))
        return axis_labels[idx][0] if cosine[idx] >= 0 else axis_labels[idx][1]
    
    orientation_code = (
        dominant_axis(row_cosine) +
        dominant_axis(col_cosine) +
        dominant_axis(normal)
    )

    # direction
    direction = np.stack([row_cosine, col_cosine, normal], axis=1)
    direction = direction.flatten().tolist()

    # pixel data
    slices    = []
    for _, _, f, _ in slice_info:
        ds = pydicom.dcmread(f)
        slope = ds.RescaleSlope if 'RescaleSlope' in ds else 1.0
        inter = ds.RescaleIntercept if 'RescaleIntercept' in ds else 0.0
        pixel_array = ds.pixel_array.astype(np.float32)
        slices.append(pixel_array * slope + inter)
    volume = np.stack(slices, axis=0) # (Z,Y,X)

    # create SimpleITK image
    img = sitk.GetImageFromArray(volume)
    img.SetSpacing((float(spacing_x), float(spacing_y), float(spacing_z)))
    img.SetOrigin(tuple(origin.tolist()))
    img.SetDirection(tuple(direction))

    # keep identifying tags so the UI can give the image a readable name
    for key, attr in (('0010|0020', 'PatientID'), ('0008|103e', 'SeriesDescription'), ('0020|0011', 'SeriesNumber')):
        val = first_ds.get(attr, None)
        if val is not None and str(val).strip():
            img.SetMetaData(key, str(val).strip())



    return img


def flip_sitk(img: sitk.Image) -> sitk.Image:
    flip_filter = sitk.FlipImageFilter()
    flip_filter.SetFlipAxes([True, True, True])
    flipped = flip_filter.Execute(img)
    return flipped

def readNifti(fp: str) -> sitk.Image:
    img = sitk.ReadImage(fp)
    return img


def readMHD(fp: str) -> sitk.Image:
    img = sitk.ReadImage(fp)
    return img


def getFolderSeriesID(fp: str) -> str:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(fp)
    return series_ids[0]


def simg2arr(img: sitk.Image) -> np.array:
    arr = sitk.GetArrayFromImage(img)       # (Z,Y,X)
    # arr = np.transpose(arr, axes=(2, 1, 0)) # (X,Y,Z)
    return arr

def arr2simg(arr: np.array) -> sitk.Image:
    # arr = np.transpose(arr, (2, 1, 0)) # (D, H, W)
    simg = sitk.GetImageFromArray(arr) # (Z,Y,X)
    return simg

def resample(img: sitk.Image, target_spacing:  list|tuple = (0.5,0.5,0.5), interpolator: int = sitk.sitkLinear) -> sitk.Image:
    original_spacing = np.array(img.GetSpacing())
    original_size    = np.array(img.GetSize())
    new_spacing      = np.array(target_spacing)

    # calculate new size to preserve physical extent
    new_size = np.round(original_size * original_spacing / new_spacing).astype(int).tolist()

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing.tolist())
    resample.SetSize(new_size)
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetInterpolator(interpolator)

    return resample.Execute(img)




IMAGE_EXTS = ('.nii.gz', '.nii', '.mhd', '.mha', '.nrrd', '.dcm')

# path components that say nothing about which scan it is
GENERIC_NAMES = {
    'image', 'images', 'img', 'imgs', 'ct', 'cta', 'mr', 'mri', 'scan', 'scans', 'data', 'volume', 'vol',
    'series', 'dicom', 'dcm', 'study', 'studies', 'input', 'output', 'raw', 'nifti', 'nii', 'original',
    'export', 'files', 'sorted', 'unnamed', 'untitled', 'new folder', 'mask', 'masks', 'label', 'labels',
    'seg', 'segmentation', 'imagestr', 'imagests', 'labelstr', 'labelsts',
}

def strip_image_ext(name: str) -> str:
    """'scan.nii.gz' -> 'scan' (Path.stem would only drop '.gz')"""
    low = name.lower()
    for ext in IMAGE_EXTS:
        if low.endswith(ext):
            return name[:-len(ext)]
    return name

def _is_generic(part: str) -> bool:
    p = part.strip().lower()
    return (p in GENERIC_NAMES
            or len(p) <= 2
            or (p.isdigit() and len(p) <= 4)                                         # '3', '0001'
            or re.fullmatch(r'(se|s|sr|st|series|study|scan|img|image|ct|mr)[ _-]?\d+', p) is not None  # 'SE0003'
            or re.fullmatch(r'[\d.]{16,}', p) is not None)                           # DICOM UID

def name_from_path(path: str, max_parts: int = 3) -> str:
    """Last path component without extension; prepend parent folders while it is generic"""
    p     = Path(path)
    parts = [strip_image_ext(p.name)]
    for parent in p.parents:
        if not _is_generic(parts[0]) or len(parts) >= max_parts or not parent.name:
            break
        parts.insert(0, parent.name)
    return "/".join(parts)

def dicom_display_name(folder: str, img: sitk.Image) -> str:
    """'PatientID - SeriesDescription #SeriesNumber', falling back to the folder path"""
    def tag(key):
        return img.GetMetaData(key).strip() if img.HasMetaDataKey(key) else ''

    pid, desc, num = tag('0010|0020'), tag('0008|103e'), tag('0020|0011')
    anonymous = pid.lower() in ('', 'anonymous', 'anon', 'unknown', 'none')
    prefix    = name_from_path(folder) if anonymous else pid

    if desc:  series = f"{desc} #{num}" if num else desc
    elif num: series = f"Series {num}"
    else:     series = ''
    return f"{prefix} - {series}" if series else prefix


def same_geometry(a: sitk.Image, b: sitk.Image, tol: float = 1e-3) -> bool:
    """True if both images share size, spacing, origin and direction (i.e. voxels line up)"""
    return (a.GetSize() == b.GetSize() and
            np.allclose(a.GetSpacing(),   b.GetSpacing(),   atol=tol) and
            np.allclose(a.GetOrigin(),    b.GetOrigin(),    atol=tol) and
            np.allclose(a.GetDirection(), b.GetDirection(), atol=tol))


def check_metadata(img: sitk.Image):
    print(f"\nSimpleITK image metadata:")
    print(f"  Size      : {img.GetSize()}")      # (W, H, D)
    print(f"  Spacing   : {img.GetSpacing()}")   # (x, y, z)
    print(f"  Origin    : {img.GetOrigin()}")    # (x, y, z)
    print(f"  Direction : {np.array(img.GetDirection()).reshape(3,3)}")

    img_arr = sitk.GetArrayFromImage(img)
    print(f"  Distribution : {np.unique(img_arr)}")
    print(f"  Shape : {img_arr.shape}")

def reorient_to_identity(image):
    """
    Reorient image so direction matrix = identity.
    Useful for DL pipelines that assume axis-aligned volumes.
    """
    # identity direction = LPS

    current_dir = np.array(image.GetDirection()).reshape(3, 3)
    is_identity = np.allclose(current_dir, np.eye(3), atol=1e-4)

    if is_identity:
        print("Already Identity Direction")
        return image

    # reorient to LPS = identity direction
    reoriented = sitk.DICOMOrient(image, "LPS")
    return reoriented

