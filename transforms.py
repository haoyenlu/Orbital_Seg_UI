from monai import transforms as T
from monai.data import ITKReader


def get_transform(params):
    tfm = [
        T.Orientationd(keys=['img'], axcodes="SPL"),
        T.Spacingd(keys=['img'], pixdim=params['pixdim'], mode='bilinear'),
        T.ScaleIntensityRanged(keys=['img'], a_min=params['intensity_a_min'], a_max=params['intensity_a_max'], b_min = params['intensity_b_min'], b_max = params['intensity_b_max'], clip=True),
        T.CropForegroundd(keys=["img"], source_key="img", margin=5),
        T.SpatialPadd(keys=["img"], spatial_size=params['roi_size'], mode="constant", value=0),
    ]

    return T.Compose(tfm)
