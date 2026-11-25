import os
import sys


def _setup_frozen():
    """Packaged exe: config.yaml / checkpoints/ live next to the exe, and there is no console.
    Runs before the heavy imports so loggers created at import time (tensorflow) see the log file."""
    app_dir = os.path.dirname(sys.executable)
    os.chdir(app_dir) # relative paths in config.yaml resolve against the exe folder

    # windowed exe has sys.stdout/stderr = None -> tqdm / progress bars would crash; send them to a log
    if sys.stdout is None or sys.stderr is None:
        import tempfile
        log = open(os.devnull, 'w')
        for folder in (app_dir, tempfile.gettempdir()):
            try:
                log = open(os.path.join(folder, 'SegUI.log'), 'w', buffering=1, encoding='utf-8')
                break
            except OSError:
                continue
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log

    # an unhandled exception in a slot would abort the app silently -> log it and show it instead
    import traceback
    def excepthook(exc_type, exc, tb):
        from PyQt5.QtWidgets import QApplication, QMessageBox
        msg = ''.join(traceback.format_exception(exc_type, exc, tb))
        print(msg, file=sys.stderr)
        if QApplication.instance() is not None:
            QMessageBox.critical(None, "Error", msg)
    sys.excepthook = excepthook

if getattr(sys, 'frozen', False):
    _setup_frozen()


import yaml
from pathlib import Path

from monai.inferers import sliding_window_inference

import sys
from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QMessageBox, QMenuBar,
    QPushButton, QFileDialog, QMainWindow, QComboBox, QListWidget, QSplitter, QAction
)
from PyQt5 import QtWidgets as qwidget
from PyQt5.QtCore import pyqtSignal, Qt, QSize, QThread, QObject, pyqtSlot, QPointF, QThreadPool

import SimpleITK as sitk


import numpy as np
import os

import utils_sitk


from ui_component import *
from ui_panel import LeftPanel, RightPanel, MenuBar
from ui_data import *
from ui_storage import ModelStorage, DisplayItemStorage, DisplaySettingStorage

class MainWindow(QMainWindow):
    def __init__(self, yaml_config:str):
        super().__init__( )
        self.config = self.load_yaml(yaml_config)
        self.setWindowTitle(self.config['UI']['title'])
        self.ModelStorage   = ModelStorage(self)
        self.DisplayStorage = DisplayItemStorage(self)
        self.DisplaySettingStorage = DisplaySettingStorage(self)


        display_setting = self.config['display']
        display_setting_item = DisplaySetting(**display_setting)
        self.DisplaySettingStorage.setSetting(display_setting_item)


        self.MenuBar = MenuBar(
            self, 
            cfg=self.config, 
            model_storage=self.ModelStorage, 
            display_setting_storage=self.DisplaySettingStorage
        )
        self.setMenuBar(self.MenuBar)
        self.MenuBar.signal.OpenDicomFolder.connect(self.open_dicom_folder)
        self.MenuBar.signal.OpenNiftiFolder.connect(self.open_nifti_file)
        self.MenuBar.signal.OpenMHDFile.connect(self.open_mhd_file)
        self.MenuBar.signal.ImportMask.connect(self.import_mask)
        self.MenuBar.signal.ExportResult.connect(self._export_result)
        self.MenuBar.signal.Reset.connect(self._reset)
        self.MenuBar.signal.DisplayChange.connect(self._on_display_change)
        self.MenuBar.signal.ModelChange.connect(self._on_model_change)
        self.MenuBar.signal.ShowAbout.connect(self._on_show_about_page)

        # Intialize
        self._on_display_change()
        self._on_model_change()

        self.LeftPanel  = LeftPanel(self, display_storage= self.DisplayStorage)
        self.RightPanel = RightPanel(parent=self, 
                                     model_config=self.config['models'], 
                                     model_storage=self.ModelStorage, 
                                     display_storage=self.DisplayStorage, 
                                     display_setting_storage=self.DisplaySettingStorage)
        self.RightPanel.PredictResult.connect(self._on_receive_prediction)


        
        self.DisplayStorage.DisplayItemChanged.connect(self.LeftPanel.update_info_panel)
        self.LeftPanel.itemChange.connect(self.RightPanel.display_image)
        self.LeftPanel.predTargetsChanged.connect(self.RightPanel.set_pred_items)
        self.MenuBar.signal.DisplayChange.connect(self.LeftPanel.on_item_select_change)

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.LeftPanel)
        splitter.addWidget(self.RightPanel)
        splitter.setStretchFactor(0,0)
        splitter.setStretchFactor(1,1)
        splitter.setHandleWidth(2)

        splitter.setStyleSheet(
            """
            QSplitter::handle{
                background: #C0C0C0;
            }
            Qsplitter::handle:pressed{
                background: #909090;
            }
            """
        )

        self.setCentralWidget(splitter)
        self.setMinimumSize(1000,800)
    

    def load_yaml(self, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config



    def open_dicom_folder(self):
        try:
            folder = qwidget.QFileDialog.getExistingDirectory(self,"Select Dicom Folder")
            if not folder: 
                return

            simg = self._handle_image_read(folder, image_type='DICOM')
            name = utils_sitk.dicom_display_name(folder, simg)

            item = ListItem(
                name = name,
                data = simg,
                type = 'image',
                path = folder
            )

            self.LeftPanel.add_item(item, parent=None)
        

        except Exception as e:
            QMessageBox.warning(self, "Warning", str(e))

    
    

    
    def open_nifti_file(self):
        try:
            file_path, _ = qwidget.QFileDialog.getOpenFileName(self, "Open Nifti File", "", "Nifti Files (*.nii *.nii.gz)")
            if not file_path:
                return
            
            
            simg = self._handle_image_read(file_path, image_type='Nifti')
            name = utils_sitk.name_from_path(file_path)

            item = ListItem(
                name = name,
                data = simg,
                type = 'image',
                path = file_path
            )

            self.LeftPanel.add_item(item, parent=None)
            

        except Exception as e:
            QMessageBox.warning(self, "Warning", str(e))

    def open_mhd_file(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open MHD File", "", "MetaImage Files (*.mhd)")
            if not file_path:
                return

            simg = utils_sitk.readMHD(file_path)
            self._check_3d(simg)
            item = ListItem(
                name = utils_sitk.name_from_path(file_path),
                data = simg,
                type = 'image',
                path = file_path
            )
            self.LeftPanel.add_item(item, parent=None)

        except Exception as e:
            QMessageBox.warning(self, "Warning", str(e))


    def import_mask(self):
        try:
            selected   = self.LeftPanel.selected_items()
            image_item = selected[0] if len(selected) == 1 else None
            if image_item is None or image_item.text(1) != 'image':
                QMessageBox.warning(self, "Warning", "Please select a single image first, then import its mask.")
                return

            file_path, _ = qwidget.QFileDialog.getOpenFileName(
                self, "Import Mask", "", "Mask Files (*.nii *.nii.gz *.mhd *.nrrd);;All (*)"
            )
            if not file_path:
                return

            image = image_item.data(0, Qt.UserRole)
            mask  = sitk.ReadImage(file_path)

            if mask.GetSize() != image.GetSize():
                QMessageBox.warning(
                    self, "Warning",
                    f"Mask size {mask.GetSize()} does not match image size {image.GetSize()}.\n"
                    f"The mask was not imported."
                )
                return

            if mask.GetNumberOfComponentsPerPixel() != 1:
                QMessageBox.warning(self, "Warning", "The mask must be a single-channel image.")
                return

            # same size but different spacing / origin / direction -> overlay would be misaligned
            if not utils_sitk.same_geometry(mask, image):
                ans = QMessageBox.question(
                    self, "Geometry Mismatch",
                    "The mask has the same size as the image, but its spacing, origin or direction differ.\n"
                    "Import it anyway and align it to the image geometry?"
                )
                if ans != QMessageBox.Yes:
                    return
                mask.CopyInformation(image)

            # masks hold integer labels 0-255
            values = np.unique(sitk.GetArrayViewFromImage(mask))
            if np.any(values < 0) or np.any(values > 255) or np.any(values != np.round(values)):
                QMessageBox.warning(self, "Warning", "The mask must contain integer label values between 0 and 255.")
                return
            mask = sitk.Cast(mask, sitk.sitkUInt8)

            # several labels -> give each its own color
            fg_values = [int(v) for v in values if v != 0]
            labels = None
            if len(fg_values) > 1:
                palette = self.config.get('label_palette', [[255,255,0],[255,0,255],[0,255,0],[255,128,0],[0,128,255],[255,0,0]])
                labels  = [
                    {'value': v, 'name': f"Label {v}", 'color': list(palette[i % len(palette)])}
                    for i, v in enumerate(fg_values)
                ]

            name = utils_sitk.strip_image_ext(Path(file_path).name) # already shown under its image
            self.LeftPanel.add_mask_child(image_item, name, mask, labels)

        except Exception as e:
            QMessageBox.warning(self, "Warning", str(e))


    def _handle_image_read(self, folder_path: str, image_type='DICOM'):
        simg = None
        if image_type == 'DICOM':
            simg = utils_sitk.readDICOMbySlice(folder_path)
                
        elif image_type == 'Nifti':
            simg   = utils_sitk.readNifti(folder_path)

        else:
            raise ValueError(f"{image_type} Not Supported")

        self._check_3d(simg)
        utils_sitk.check_metadata(simg)
        return simg

    def _check_3d(self, simg: sitk.Image):
        # viewer, info panel and models all assume 3D volumes
        if simg.GetDimension() != 3:
            raise ValueError(f"Only 3D images are supported (this image is {simg.GetDimension()}D).")

    def _check_mhd_file(self, folder_path: str):
        folder     = Path(folder_path)
        mhd_files  = list(folder.rglob("*.mhd"))
        return len(mhd_files) > 0

    def _on_display_change(self):
        setting = self.MenuBar.getDisplaySetting()
        self.DisplaySettingStorage.setSetting(setting)


    def _on_model_change(self):
        model_info = self.MenuBar.getCurrentModel()
        self.ModelStorage.setModel(model_info)


    def _on_show_about_page(self):
        dlg = AboutDialog(**self.config['about'], parent=self)
        dlg.exec_()


    def _on_receive_prediction(self, pred_result: dict = None):
        self.LeftPanel.add_mask_to_item(pred_result)

    def _reset(self):
        self.RightPanel.Viewer.reset()
        self.LeftPanel.reset()

    def _export_result(self): 
        path, _ = qwidget.QFileDialog.getSaveFileName(
            self,
            "Save File",                    # dialog title
            ".",                             # default directory
            "NIfTI (*.nii.gz);;All (*)"    # file filters
        )

        if not path: return # dialog cancelled
        self.LeftPanel.export_mask(path)

def _selftest(config_path: str):
    """Load every model in the config and predict a small synthetic volume, without opening a window
    (checks a packaged build)"""
    import traceback
    import torch
    from ui_workers import load_model, PredictionWorker
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"CUDA available: {torch.cuda.is_available()}")

    simg = sitk.GetImageFromArray(np.random.default_rng(0).normal(0, 300, (64, 96, 96)).astype(np.float32))
    simg.SetSpacing((0.5, 0.5, 1.0))

    failed = 0
    for name, detail in config['models'].items():
        try:
            model = load_model(detail)
            mask  = PredictionWorker([], name, detail['function'], detail)._predict(model, simg)
            assert mask.GetSize() == simg.GetSize(), f"mask size {mask.GetSize()} != image size {simg.GetSize()}"
            print(f"[OK]   {name}")
        except Exception:
            failed += 1
            print(f"[FAIL] {name}\n{traceback.format_exc()}")
    print(f"selftest: {len(config['models']) - failed}/{len(config['models'])} models loaded and predicted")
    return 1 if failed else 0


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    if '--selftest' in sys.argv:
        sys.exit(_selftest('config.yaml'))

    app = QApplication(sys.argv)
    window = MainWindow(yaml_config='config.yaml')
    window.show()
    sys.exit(app.exec_())