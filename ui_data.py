from dataclasses import dataclass, fields

import SimpleITK as sitk



@dataclass
class ModelInfo:
    model_name: str
    model_function: str
    model_detail: dict


@dataclass
class ListItem:
    name: str        = ""
    data: sitk.Image = None
    type: str        = ""
    path: str        = "" # source file / folder, shown as tooltip

@dataclass
class DisplayItem:
    image_folder: str    = ""
    image_folder_id: str = ""
    image_type: str      = ""
    mask: sitk.Image     = None
    image: sitk.Image    = None
    model_name           = ""
    model_function       = ""

    def reset(self):
        self.image_folder = ""
        self.image_folder_id = ""
        self.image_type = ""
        self.mask = None
        self.image = None
        self.model_name = ""
        self.model_function = ""



@dataclass
class PredictItem:
    image_folder_id: str
    image: sitk.Image = None



@dataclass
class MaskResultItem:
    result: sitk.Image   = None
    model_name: str      = ""
    model_function: str  = ""
    labels: list         = None # [{'value', 'name', 'color'}] for multi-label masks, None for binary



@dataclass
class DisplaySetting:
    overlay: bool          = True
    contour: bool          = False
    axis: int              = 0
    window: list           = None
    mask_color: list       = None
    contour_width: float   = 1.0
    mask_alpha: int        = 100