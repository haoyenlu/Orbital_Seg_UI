import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QFileDialog, QMainWindow, QComboBox
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5 import QtWidgets as qwidget
from PyQt5.QtCore import pyqtSignal, Qt, QSize, QThread, QObject, pyqtSlot, QPointF, QRunnable
import traceback
from skimage import measure
from pathlib import Path
import utils_sitk
import SimpleITK as sitk
import numpy as np
import json

from tensorflow.keras.models import model_from_config
import h5py
import importlib
import torch
from utils import vPrediction, pytorch_sliding_window_inference
from ui_data import *


import vSegnets
# import onnx
# from onnx2torch import convert

def load_model(model_detail: dict):
    compiled_model = None
    if model_detail['type'] == 'keras':
        if "class" in model_detail:
            ModelClass     = get_class(model_detail['class'])
            model          = ModelClass(**model_detail['params'])
            compiled_model = model.compile_model(checkpoint=model_detail['checkpoint']) 

        else:
            compiled_model = load_keras_model_from_checkpoint(model_detail['checkpoint'])

    # if model_detail['type'] == 'onnx':
    #     onnx_model     = onnx.load(model_detail['checkpoint'])
    #     compiled_model = convert(onnx_model)
        
            
    elif model_detail['type'] == 'pytorch':
        compiled_model = get_pytorch_model(model_detail)

    elif model_detail['type'] == 'nnunet':
        from nnunet_infer import NNUNetEnsemble
        compiled_model = NNUNetEnsemble(
            model_folder    = model_detail['checkpoint'],
            folds           = model_detail.get('folds', [0, 1, 2, 3, 4]),
            checkpoint_name = model_detail.get('checkpoint_name', 'checkpoint_final.pth'),
            use_mirroring   = model_detail.get('use_mirroring', True),
            step_size       = model_detail.get('step_size', 0.5),
        )

    else:
        raise ValueError(f"{model_detail['type']} not supported!")
    return compiled_model


def load_keras_model_from_checkpoint(checkpoint_path):
    with h5py.File(checkpoint_path, "r") as f:
        if "model_config" in f.attrs:
            model_config = json.loads(f.attrs['model_config'])
            model = model_from_config(model_config)
            model.load_weights(checkpoint_path)
            return model
        else:
            raise ValueError("Model Config is not in the checkpoint")

# def get_onnx_model(checkpoint_path: str):
#     onnx_model  = onnx.load(checkpoint_path)
#     torch_model = convert(onnx_model)
#     return torch_model



def get_pytorch_model(model_detail: dict):
    from models.r2unet3d import R2UNet3D
    if model_detail['name'] == 'R2':
        model = R2UNet3D(
            in_ch=1, 
            out_ch=model_detail['n_classes'], 
            channels=model_detail['params']['channels'], 
            t=model_detail['params']['t'], 
            norm="INSTANCE"
        )
        ckpt = torch.load(model_detail['checkpoint'], map_location='cpu') # works without GPU; moved to device later
        model.load_state_dict(ckpt['state_dict'])
    else:
        raise ValueError(f"Unsupported model: {model_detail['name']}")
    return model


def get_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)



class PredictionSignal(QObject):
    result      = pyqtSignal(int, object) # (image index, MaskResultItem) for each predicted image
    image_error = pyqtSignal(int, str)    # (image index, traceback) when one image fails
    finished    = pyqtSignal()            # all images processed
    error       = pyqtSignal(str)         # fatal error (e.g. model failed to load)
    progress    = pyqtSignal(int)         # number of images processed
    started     = pyqtSignal()

class PredictionWorker(QRunnable):

    def __init__(self,
                 pred_images: list,
                 model_name: str,
                 model_function: str,
                 model_detail: dict):
        super().__init__()
        self.model_name     = model_name
        self.model_function = model_function
        self.model_detail   = model_detail
        self.pred_images    = list(pred_images)
        self.signal = PredictionSignal()


    def _predict(self, model, simg: sitk.Image) -> sitk.Image:
        if self.model_detail['type']  == 'keras':
            return vPrediction(model,  simg, params=self.model_detail)

        elif self.model_detail['type'] == 'nnunet':
            return model.predict(simg) # fold ensemble, returns label map on the input grid

        elif self.model_detail['type'] == 'pytorch' or self.model_detail['type'] == 'onnx':
            return pytorch_sliding_window_inference(model,
                                                    simg,
                                                    params = self.model_detail,
                                                    sw_batch_size=16,
                                                    overlap=0.75,
                                                    target_class = self.model_detail.get('target_class', 1))
        else:
            raise ValueError(f"Unsupported model type: {self.model_detail['type']}")


    @pyqtSlot()
    def run(self):
        self.signal.started.emit()
        try:
            assert self.pred_images, "There is no image for prediction!!"
            model  = load_model(self.model_detail) # load once for all images

            labels = self.model_detail.get('labels')
            if labels:
                labels = [{'value': int(v), 'name': l['name'], 'color': list(l['color'])} for v, l in labels.items()]
        except:
            self.signal.error.emit(traceback.format_exc())
            return

        for i, simg in enumerate(self.pred_images):
            try:
                result = MaskResultItem(
                    result         = self._predict(model, simg),
                    model_name     = self.model_name,
                    model_function = self.model_function,
                    labels         = labels,
                )
                self.signal.result.emit(i, result)
            except:
                self.signal.image_error.emit(i, traceback.format_exc())
            self.signal.progress.emit(i + 1)

        self.signal.finished.emit()







class ImageLoaderSignal(QObject):
    finished = pyqtSignal()
    error    = pyqtSignal(str)
    started  = pyqtSignal()
    result   = pyqtSignal(object)


class ImageLoaderWorker(QRunnable):

    def __init__(self, 
                 image: sitk.Image = None,
                 mask : sitk.Image = None,
                 display_setting: DisplaySetting = None,
                 labels: list = None,
                 *args, **kwargs):


        super().__init__()
        self.signal  = ImageLoaderSignal()
        self.image   = image
        self.mask    = mask
        self.setting = display_setting
        self.labels  = labels # per-label colors for multi-label masks

        self.args   = args
        self.kwargs = kwargs
    

    @pyqtSlot()
    def run(self):
        self.signal.started.emit()
        try:       
            # check if image is present
            image = self._standardize_image_for_display(self.image, sitk.sitkLinear)          if self.image is not None else None
            mask  = self._standardize_image_for_display(self.mask,  sitk.sitkNearestNeighbor) if self.mask is not None else None

            if mask is None:
                 # -> display image only
                frames = self.get_image_frames(image)
            elif mask is not None and not self.setting.overlay:
                 # -> display mask only
                frames = self.get_mask_frames(mask)
            else: 
                # -> overlay mask on image
                img_frames = self.get_image_frames(image)
                msk_frames = self.get_mask_frames(mask)

                frames = []
                for img_f, msk_f in zip(img_frames, msk_frames):
                    frames.append(self.overlay_qpixmap(img_f, msk_f))

            
            self.signal.result.emit(frames)
            self.signal.finished.emit()
        
        except:
            self.signal.error.emit(traceback.format_exc())


    def _standardize_image_for_display(self, simg: sitk.Image, interpolator: int = sitk.sitkLinear):
        simg  = utils_sitk.resample(simg, target_spacing=(1.0,1.0,1.0), interpolator=interpolator)
        simg  = utils_sitk.reorient_to_identity(simg)
        return simg


    def _apply_window(self, arr, width ,level):
        low = level - width / 2
        high = level + width / 2
        arr = np.clip(arr, low, high)
        return arr
    
    def _normalize(self, arr):
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        arr = (arr * 255).astype(np.uint8)
        return arr

    def get_qimage(self, arr):
        h, w       = arr.shape
        img_bytes  = arr.tobytes()
        qimage     = QImage(img_bytes, w, h, w, QImage.Format.Format_Grayscale8)
        return qimage, img_bytes


    def get_pixmap(self, arr):
        """Get QPixmap for a specific plane and slice index."""
        qimage, img_bytes = self.get_qimage(arr)
        self._img_bytes   = img_bytes       # keep reference
        return QPixmap.fromImage(qimage)
    
    def get_slices(self, arr):
        # arr is LPS (identity direction): X -> patient Left, Y -> Posterior, Z -> Superior
        # radiological convention: patient right on screen left, anterior / superior on top
        Z,Y,X = arr.shape
        slices = None
        if self.setting.axis==0:
            slices = [arr[z,:,:] for z in range(Z)]
        elif self.setting.axis==1:
            slices = [arr[::-1,y,:] for y in range(Y)]
        elif self.setting.axis==2:
            slices = [arr[::-1,:,x] for x in range(X)]
        else:
            raise ValueError(f"Invalid axis: {self.setting.axis}")
    
        return slices

    def get_qpixmap_frames_from_image(self, img_arr: np.array):
        slices = self.get_slices(img_arr)
        frames = []
        for img in slices:
            if self.setting.window is not None:
                level, width = self.setting.window
                img = self._apply_window(img, width, level)

            img    = self._normalize(img)
            pixmap = self.get_pixmap(img)
            frames.append(pixmap)
        return frames
    

    def _color_regions(self, msk: np.array):
        """(binary region, color) pairs: one per label for multi-label masks, else foreground in mask_color"""
        if self.labels:
            return [(msk == l['value'], l['color']) for l in self.labels]
        return [(msk > 0, self.setting.mask_color)]

    def get_qpixmap_frames_from_mask_contour(self, msk_arr: np.array):
        slices        = self.get_slices(msk_arr)
        contour_width = self.setting.contour_width
        frames = []
        for msk in slices:
            H,W = msk.shape
            qpixmap = QPixmap(W, H)
            qpixmap.fill(Qt.transparent)
            painter = QPainter(qpixmap)

            for region, color in self._color_regions(msk):
                if not region.any(): continue
                painter.setPen(QPen(QColor(*color), contour_width))
                for c in self.get_contour(region.astype(np.float32)):
                    if c.size == 0: continue
                    points = [QPointF(p[1], p[0]) for p in c]
                    if len(points) > 1:
                        for i in range(len(points)-1):
                            painter.drawLine(points[i], points[i+1])

            painter.end()
            frames.append(qpixmap)
        return frames


    def get_qpixmap_frames_from_mask_solid(self, msk_arr):
        slices     = self.get_slices(msk_arr)
        mask_alpha = self.setting.mask_alpha
        frames = []
        for msk in slices:
            h, w  = msk.shape
            rgba  = np.zeros((h, w, 4), dtype=np.uint8)
            for fg, color in self._color_regions(msk):
                rgba[fg, 0] = color[0]
                rgba[fg, 1] = color[1]
                rgba[fg, 2] = color[2]
                rgba[fg, 3] = mask_alpha
            img_bytes = rgba.tobytes()
            qimage = QImage(img_bytes, w, h, w*4, QImage.Format.Format_RGBA8888)
            qpixmap = QPixmap.fromImage(qimage)
            frames.append(qpixmap)
        return frames
    
    def get_contour(self, mask2D: np.array) -> np.array:
        """
        Get Contour from binary mask slices
        """
        contours = measure.find_contours(mask2D, 0.5)
        return contours

    def overlay_qpixmap(self, base, overlay):
        result = QPixmap(base.size())
        result.fill(Qt.transparent)

        painter = QPainter(result)
        painter.drawPixmap(0,0,base)
        painter.drawPixmap(0,0,overlay)
        painter.end()
        return result
    

    def get_image_frames(self, image: sitk.Image) ->  list[QPixmap]:
        image = sitk.GetArrayFromImage(image) # (Z,Y,X)
        frames = self.get_qpixmap_frames_from_image(image)
        return frames

    def get_mask_frames(self, mask: sitk.Image) -> list[QPixmap]:
        msk_arr    = sitk.GetArrayFromImage(mask) # (Z,Y,X)
        if self.setting.contour:
            frames = self.get_qpixmap_frames_from_mask_contour(msk_arr)
        else:
            frames = self.get_qpixmap_frames_from_mask_solid(msk_arr)
        return frames
