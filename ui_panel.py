from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QProgressBar, QHBoxLayout, QSlider,
    QPushButton, QFileDialog, QMainWindow, QComboBox, QListWidget, QAction, QDialog,QMenuBar,QTreeWidget, QTreeWidgetItem,
    QSplitter, QListWidgetItem, QMessageBox, QSizePolicy, QStackedLayout, QMenu, QActionGroup, QGroupBox, QFormLayout,
    QAbstractItemView, QTreeWidgetItemIterator
)
from PyQt5.QtCore import (
    pyqtSignal, Qt, QPoint, pyqtSlot, QPointF, QThreadPool
)
from PyQt5.QtGui import QImage, QPixmap

import copy
import numpy as np
import utils_sitk
import SimpleITK as sitk

from ui_workers import PredictionWorker, ImageLoaderWorker

from ui_data import *
from ui_storage import *
from ui_component import *


class InfoPanel(QWidget):

    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── group box ──────────────────────────────
        group  = QGroupBox("Image Information")
        form   = QFormLayout(group)

        # ── info rows ─────────────────────────────
        self.lbl_path    = QLabel("N/A")
        self.lbl_path.setFixedWidth(150)
        self.lbl_path.setWordWrap(True)

        self.lbl_size    = QLabel("N/A")
        self.lbl_spacing = QLabel("N/A")
        self.lbl_origin  = QLabel("N/A")
        self.lbl_orient  = QLabel("N/A")
        self.lbl_dtype   = QLabel("N/A")




        form.addRow("Size    :", self.lbl_size)
        form.addRow("Spacing :", self.lbl_spacing)
        form.addRow("Origin  :", self.lbl_origin)
        form.addRow("Orient  :", self.lbl_orient)

        layout.addWidget(group)

    def update_info(self, data: sitk.Image = None):
        """Update panel with new info dict."""
        if data is None: return

        orient = sitk.DICOMOrientImageFilter()
        code   = orient.GetOrientationFromDirectionCosines(data.GetDirection())

        size    = data.GetSize()
        spacing = data.GetSpacing()
        origin  = data.GetOrigin()

        size    = [round(x,4) for x in size]
        spacing = [round(x,4) for x in spacing]
        origin  = [round(x,4) for x in origin]

        info = {
            'size'   : size,
            'spacing': spacing,
            'origin' : origin,
            'orient' : code
        }


        self.lbl_size.setText(str(info.get("size",    "N/A")))
        self.lbl_spacing.setText(str(info.get("spacing", "N/A")))
        self.lbl_origin.setText( str(info.get("origin",  "N/A")))
        self.lbl_orient.setText( str(info.get("orient",  "N/A")))

    def reset(self):  
        self.lbl_size.setText("N/A")
        self.lbl_spacing.setText("N/A")
        self.lbl_origin.setText("N/A")
        self.lbl_orient.setText("N/A")



class LeftPanel(QWidget):
    itemChange = pyqtSignal(object) # QTreeWidgetItem, or None when nothing is selected
    ItemRemove = pyqtSignal()
    predTargetsChanged = pyqtSignal(list) # selected images to run prediction on
    def __init__(self,
                 parent = None,
                 display_storage: DisplayItemStorage = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.DisplayStorage = display_storage
        self._displayed_item = None

        self.ItemList = QTreeWidget()
        # self.ItemList.setHeaderHidden(True)
        self.ItemList.setHeaderLabels(["Name", "Type"])
        # Ctrl / Shift + click selects several images; only the first one is displayed
        self.ItemList.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.ItemList.itemSelectionChanged.connect(self._on_selection_changed)
        self.ItemList.setMinimumHeight(200)          # minimum height
        self.ItemList.setMaximumHeight(500)          # maximum height
        self.ItemList.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ItemList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ItemList.customContextMenuRequested.connect(self.item_context_menu)

        layout.addWidget(self.ItemList, alignment = Qt.AlignTop, stretch= 1)

        self.info_panel = InfoPanel()
        layout.addWidget(self.info_panel, alignment= Qt.AlignBottom, stretch=1)
        self.itemChange.connect(self.update_info_panel)
    
    def update_info_panel(self, item:QTreeWidgetItem = None):
        if item is None:
            self.info_panel.reset()
            return
        data = item.data(0, Qt.UserRole)
        self.info_panel.update_info(data)

    def add_item(self, item: ListItem, parent=None):
        """
        Add Display Item to List WIdget Item at UserRole
        Display Item on List with Folder ID
        """
        if parent is None:
            tree_item = QTreeWidgetItem(self.ItemList)
        else:
            tree_item = QTreeWidgetItem(parent)

        tree_item.setText(0, self._unique_name(item.name))
        tree_item.setText(1, item.type)
        if item.path:
            tree_item.setToolTip(0, item.path)
        tree_item.setData(0, Qt.UserRole, item.data)
        tree_item.setFlags(tree_item.flags() | Qt.ItemIsSelectable | Qt.ItemIsEnabled)

        self.ItemList.setCurrentItem(tree_item) # selection change displays it





    def _unique_name(self, name: str) -> str:
        """Append ' (2)', ' (3)', ... if a top-level item already has this name"""
        existing = {self.ItemList.topLevelItem(i).text(0) for i in range(self.ItemList.topLevelItemCount())}
        unique, n = name, 2
        while unique in existing:
            unique = f"{name} ({n})"
            n += 1
        return unique

    def reset(self):
        self.ItemList.clear()
        self._update_display()



    def add_mask_to_item(self, pred_result: dict):
        pred_item   = pred_result['pred_item']
        mask_result = pred_result['result']

        items   = self.ItemList.findItems("*",  Qt.MatchWildcard | Qt.MatchRecursive, 0)
        for item in items:
            if item == pred_item:
                self.add_mask_child(item, mask_result.model_function, mask_result.result, mask_result.labels,
                                    select=pred_result.get('select', True))

    def add_mask_child(self, parent_item: QTreeWidgetItem, name: str, mask: sitk.Image, labels: list = None,
                       select: bool = True):
        """Add a mask as a sub-item under an image item; select (and display) it if requested"""
        child_item = QTreeWidgetItem(parent_item)

        child_item.setText(0, name)
        child_item.setText(1, 'mask')
        child_item.setData(0, Qt.UserRole, mask)
        child_item.setData(0, Qt.UserRole + 1, labels) # per-label colors (multi-label masks)
        child_item.setFlags(parent_item.flags() | Qt.ItemIsSelectable | Qt.ItemIsEnabled)

        parent_item.setExpanded(True)
        if select:
            self.ItemList.setCurrentItem(child_item) # selection change displays it
            self.ItemList.scrollToItem(child_item)

    def selected_items(self) -> list:
        """Selected items in list order (top to bottom)"""
        it = QTreeWidgetItemIterator(self.ItemList, QTreeWidgetItemIterator.Selected)
        items = []
        while it.value():
            items.append(it.value())
            it += 1
        return items

    def selected_item(self) -> QTreeWidgetItem:
        """First selected item: the one shown in the viewer"""
        items = self.selected_items()
        return items[0] if items else None

    def selected_images(self) -> list:
        """Images to predict on: selected images, a selected mask counts as its parent image"""
        images = []
        for item in self.selected_items():
            image = item if item.text(1) == 'image' else item.parent()
            if image is not None and image not in images:
                images.append(image)
        return images

    def _update_display(self, force: bool = False):
        """Display the first selected item; skip if it is already displayed unless forced"""
        self.predTargetsChanged.emit(self.selected_images())
        item = self.selected_item()
        if not force and item is self._displayed_item: return
        self._displayed_item = item
        self.itemChange.emit(item)

    def _on_selection_changed(self):
        self._update_display()

    def on_item_select_change(self):
        # display settings changed -> redraw the displayed item
        self._update_display(force=True)
    
    def item_context_menu(self, pos: QPoint):
        item = self.ItemList.itemAt(pos)
        if not item: return

        menu = QMenu(self.ItemList)
        act_remove =  menu.addAction("Remove")
        act = menu.exec_(self.ItemList.mapToGlobal(pos))
        
        # context menu actions
        if act == act_remove:
            parent = item.parent()
            if parent: # child item (predict mask)
                parent.removeChild(item)
                self.ItemList.setCurrentItem(parent)

            else: # top-level item
                idx = self.ItemList.indexOfTopLevelItem(item)
                self.ItemList.takeTopLevelItem(idx)
            self._update_display()

    def export_mask(self, file_path: str):
        item = self.selected_item()
        if item is None:
            return

        item_type    = item.text(1)
        if item_type != 'mask':
            QMessageBox.warning(self, "Warning", f"The selected item is not a mask")
            return 

        mask  = item.data(0, Qt.UserRole)
        try:
            sitk.WriteImage(mask, fileName=file_path)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to save {file_path}:\n{e}")
            return
        QMessageBox.information(self, "Success", f"Result has been saved to {file_path}")



class ImageCache(QObject):
    def __init__(self, max_size=100):
        super().__init__()
        self._cache   = {}
        self._max_size = max_size

    def set(self, key, value):
        # evict oldest if full
        if len(self._cache) >= self._max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = value

    def get(self, key, default=None):
        return self._cache.get(key, default)

    def has(self, key):
        return key in self._cache

    def remove(self, key):
        self._cache.pop(key, None)

    def clear(self):
        self._cache.clear()

    def size(self):
        return len(self._cache)


    

class SliceViewer(QWidget):
    def __init__(self, 
                 parent = None, 
                 display_storage: DisplayItemStorage = None,
                 display_setting_storage: DisplaySettingStorage = None):
        
        super().__init__(parent)

        self.DisplayStorage        = display_storage
        self.DisplaySettingStorage = display_setting_storage

        self.image_cache = ImageCache(max_size=20)

        self.label = QLabel("Select DICOM folder")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(400,400)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.setText("Select Image to Display")
        self.label.setStyleSheet(
            """
            QLabel{
                border: 1px solid #888;
            }
            """
        )

        self.bar = QProgressBar()
        self.bar.setTextVisible(True)
        self.bar.hide()


        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0,0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.set_index)
        self.slider_index = QLabel("0 / 0")

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.slider, 1)
        slider_layout.addWidget(self.slider_index)

        # color legend for multi-label masks
        self.legend = QLabel()
        self.legend.setTextFormat(Qt.RichText)
        self.legend.hide()

        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.legend)
        layout.addWidget(self.bar)
        layout.addLayout(slider_layout)

        # Some States
        self.stack = []
        self.current_index = 0
        self._current_loader_signal = None

        self.thread = QThreadPool.globalInstance()


    def display_image(self, item:QTreeWidgetItem = None): 
        if item is None:
            self.reset()
            return

        self.bar.show()
        self.bar.setRange(0,0)

        display_setting = self.DisplaySettingStorage.getSetting()
        
        if item.text(1) == 'image':
            image  = item.data(0,Qt.UserRole)
            mask   = None
            labels = None
        else:
            image  = item.parent().data(0,Qt.UserRole)
            mask   = item.data(0, Qt.UserRole)
            labels = item.data(0, Qt.UserRole + 1)

        self.set_legend(labels)

        image_loader = ImageLoaderWorker(image=image,mask=mask, display_setting=display_setting, labels=labels)
        self._current_loader_signal = image_loader.signal # only the latest request may update the view
        image_loader.signal.result.connect(self.on_loader_result_receive)
        image_loader.signal.error.connect(self.on_error)
        image_loader.signal.finished.connect(self.on_load_finish)
        self.thread.start(image_loader)

    def set_legend(self, labels: list = None):
        if not labels:
            self.legend.hide()
            return
        entries = [
            f"<span style='color:rgb({l['color'][0]},{l['color'][1]},{l['color'][2]})'>&#9632;</span> {l['name']}"
            for l in labels
        ]
        self.legend.setText("&nbsp;&nbsp;&nbsp;".join(entries))
        self.legend.show()

    def reset(self):
        self._current_loader_signal = None # drop results of loads still running
        self.set_legend(None)
        self.bar.hide()
        self.label.clear()
        self.stack = []
        self.slider.setEnabled(False)
        self.slider.setRange(0,0)
        self.slider_index.setText("0 / 0")


    def on_loader_result_receive(self, result: list[QPixmap]):
        if self.sender() is not self._current_loader_signal: return # stale load
        self.set_frames(result)


    def set_frames(self, frames):
        if frames is None: 
            self.reset()
            return 
        
        self.stack = frames
        self.slider.setEnabled(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(frames)-1)
        self.slider.setSingleStep(1)
        self.slider.setValue(self.current_index)
        self.set_index(self.current_index)
    
    def set_index(self, i: int):
        if not self.stack: return
        self.current_index = max(0, min(i, len(self.stack) - 1))
        self.slider_index.setText(f"{self.current_index} / {len(self.stack)}")
        
        # Display Image based on Index
        pm = self.stack[self.current_index]
        scaled = pm.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(scaled)
    
    def refresh_pixmap(self):
        if not self.stack:
            self.label.clear()
            return 
        
        # Display Image based on Index
        pm = self.stack[self.current_index]
        scaled = pm.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(scaled)


    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.refresh_pixmap()

    def wheelEvent(self, a0):
        if not self.stack: return

        steps = a0.angleDelta().y() // 120
        if steps != 0:
            new_i = int(np.clip(self.slider.value() + steps , 0 , self.slider.maximum()))
            self.slider.setValue(new_i)
            self.current_index = new_i


    @pyqtSlot()
    def on_load_finish(self):
        if self.sender() is not self._current_loader_signal: return # a newer load is still running
        self.bar.hide()

    @pyqtSlot(str)
    def on_error(self, tb):
        QMessageBox.warning(self, "Warning", tb)


class PredictWidget(QWidget):
    result = pyqtSignal(object)

    def __init__(self, parent, 
                 model_config, 
                 model_storage, 
                 display_storage,):
        
        super().__init__(parent)
        
        # State
        self.model_config   = model_config
        self.ModelStorage   = model_storage
        self.DisplayStorage = display_storage

        self.pred_items    = [] # currently selected images
        self.running_items = [] # images the running prediction belongs to
        self.failed        = [] # (image name, traceback) of images that failed in the running batch


        self.threadpool = QThreadPool.globalInstance()


        layout = QVBoxLayout()

        # Prediction Button
        self.pred_btn = QPushButton("Run Prediction")
        self.pred_btn.clicked.connect(self.on_start_prediction)
        self.pred_btn.setEnabled(False)

        # Prediction Progress Bar
        self.pred_progress_bar = QProgressBar()
        self.pred_progress_bar.hide()

        layout.addWidget(self.pred_btn)
        layout.addWidget(self.pred_progress_bar)
        self.setLayout(layout)

        self.isPredict = False

    # Public Function
    def enable_prediction(self):
        if not self.isPredict and self.pred_items:
            self.pred_btn.setEnabled(True)


    def disable_prediction(self):
        self.pred_btn.setEnabled(False)

    def set_pred_items(self, items: list):
        """Selected images to predict on (masks already mapped to their parent image)"""
        self.pred_items = list(items)
        n = len(self.pred_items)
        self.pred_btn.setText("Run Prediction" if n <= 1 else f"Run Prediction ({n} images)")
        if self.pred_items: self.enable_prediction()
        else:               self.disable_prediction()

    def on_start_prediction(self):
        try:
            if not self.pred_items or self.isPredict: return

            self.running_items = list(self.pred_items)
            self.failed        = []
            self.on_start()

            pred_images    = [item.data(0, Qt.UserRole) for item in self.running_items]
            curr_model     = self.ModelStorage.current_model
            model_name     = curr_model.model_name
            model_function = curr_model.model_function
            model_detail   = curr_model.model_detail


            worker = PredictionWorker(pred_images    = pred_images,
                                      model_name     = model_name,
                                      model_function = model_function,
                                      model_detail   = model_detail)

            worker.signal.error.connect(self.on_error)
            worker.signal.result.connect(self.on_image_result)
            worker.signal.image_error.connect(self.on_image_error)
            worker.signal.progress.connect(self.on_progress)
            worker.signal.finished.connect(self.on_finish)
            self.threadpool.start(worker)

        except Exception as e:
            self.on_error(str(e))

    def on_start(self):
        self.isPredict = True
        self.disable_prediction()
        self.pred_progress_bar.show()
        n = len(self.running_items)
        if n > 1:
            self.pred_progress_bar.setRange(0, n)
            self.pred_progress_bar.setValue(0)
            self.pred_progress_bar.setFormat("%v / %m images")
        else:
            self.pred_progress_bar.setRange(0,0) # busy indicator for a single image


    @pyqtSlot(int, object)
    def on_image_result(self, idx: int, mask_result: MaskResultItem):
        # add each mask as soon as it is ready; only the first image's mask gets selected / displayed
        pred_result = {
            'pred_item': self.running_items[idx],
            'result':    mask_result,
            'select':    idx == 0,
        }
        self.result.emit(pred_result)

    @pyqtSlot(int, str)
    def on_image_error(self, idx: int, tb: str):
        self.failed.append((self.running_items[idx].text(0), tb))

    @pyqtSlot(int)
    def on_progress(self, done: int):
        if len(self.running_items) > 1:
            self.pred_progress_bar.setValue(done)

    @pyqtSlot()
    def on_finish(self):
        n_total  = len(self.running_items)
        failed   = self.failed
        self._end_prediction()

        if not failed:
            QMessageBox.information(self, "Success", "Predict finished!" if n_total == 1
                                    else f"Predict finished for {n_total} images!")
        elif n_total == 1:
            QMessageBox.warning(self, "Warning", failed[0][1])
        else:
            details = "\n\n".join(f"[{name}]\n{tb}" for name, tb in failed)
            QMessageBox.warning(self, "Warning",
                                f"{n_total - len(failed)} of {n_total} images predicted, {len(failed)} failed:\n\n{details}")


    @pyqtSlot(str)
    def on_error(self, tb):
        self._end_prediction()
        QMessageBox.warning(self, "Warning", tb)

    def _end_prediction(self):
        self.isPredict     = False
        self.running_items = []
        self.failed        = []
        self.pred_progress_bar.hide()
        self.enable_prediction()



class RightPanel(QWidget):
    PredictResult = pyqtSignal(dict)

    def __init__(self, parent, 
                 model_config: dict, 
                 model_storage: ModelStorage,
                 display_storage: DisplayItemStorage,
                 display_setting_storage: DisplaySettingStorage):
        
        super().__init__(parent)
        self.model_config   = model_config
        self.DisplayStorage = display_storage
        self.ModelStorage   = model_storage
        self.DisplaySettingStorage = display_setting_storage

        layout = QVBoxLayout(self)

        self.PredWidget = PredictWidget(self, model_config, model_storage , display_storage)
        self.PredWidget.result.connect(self.handle_pred_result)
        layout.addWidget(self.PredWidget)


        self.Viewer = SliceViewer(self, display_storage, display_setting_storage)
        layout.addWidget(self.Viewer)

        self.setLayout(layout)


        # Thread pool
        self.threadpool = QThreadPool.globalInstance()


    def display_image(self, item: QTreeWidgetItem = None):
        self.Viewer.display_image(item)

    def set_pred_items(self, items: list):
        self.PredWidget.set_pred_items(items)


    def handle_pred_result(self, result: dict):
        self.PredictResult.emit(result)





class MenuBarSignal(QObject):
    OpenDicomFolder = pyqtSignal()
    OpenNiftiFolder = pyqtSignal()
    OpenMHDFile     = pyqtSignal()
    ImportMask      = pyqtSignal()
    ExportResult    = pyqtSignal()
    Reset           = pyqtSignal()

    DisplayChange   = pyqtSignal()
    ModelChange     = pyqtSignal()
    
    ShowAbout       = pyqtSignal()


class MenuBar(QMenuBar):
    def __init__(self, parent=None, 
                 cfg = None, 
                 model_storage: ModelStorage = None,
                 display_setting_storage: DisplaySettingStorage = None ):
        
        super().__init__(parent)

        self.signal = MenuBarSignal()
        self.cfg = cfg
        self.ModelStorage = model_storage
        self.DisplaySettingStorage = display_setting_storage
        self.initial_setting = self.DisplaySettingStorage.getSetting()

        self._build_file_menu()
        self._build_model_menu()
        self._build_window_menu()
        self._build_display_menu()
        self._build_axis_menu()
        self._build_help_menu()



    def _build_file_menu(self):
        self.file_menu = self.addMenu("&File")


        self.file_menu.addAction("Open Dicom Folder", self.signal.OpenDicomFolder)
        self.file_menu.addAction("Open Nifti File",   self.signal.OpenNiftiFolder)
        self.file_menu.addSeparator()
        self.file_menu.addAction("Import MHD File",   self.signal.OpenMHDFile)
        self.file_menu.addAction("Import Mask...",    self.signal.ImportMask)
        self.file_menu.addSeparator()
        self.file_menu.addAction("Export Result", self.signal.ExportResult)  
        self.file_menu.addAction("Reset", self.signal.Reset)

    def _build_model_menu(self):
        self.model_menu = self.addMenu("&Model")
        self.model_menu.addAction('Add Model', self._open_add_model_dialog)

        self.model_menu.addSeparator()
        self.model_group = QActionGroup(self)
        self.model_group.setExclusive(True)

        for i , model in enumerate(self.cfg['models'].keys()):
            act_label = self.cfg['models'][model]['function']
            act_data  = ModelInfo(
                model_name = model,
                model_function = act_label,
                model_detail = self.cfg['models'][model]
            )

            action = QAction(act_label, self, checkable=True)
            action.setData(act_data)
            self.model_group.addAction(action)
            self.model_menu.addAction(action)
            if i == 0: action.setChecked(True)

        self.model_group.triggered.connect(self.signal.ModelChange)


    
    def _build_window_menu(self):
        self.window_menu = self.addMenu("&Window Level")
        self.window_group = QActionGroup(self)
        self.window_group.setExclusive(True)

        for i, (key,value) in enumerate(self.cfg['windows'].items()):
            action = QAction(key, self, checkable=True)
            action.setData(value)
            self.window_group.addAction(action)
            self.window_menu.addAction(action)
            if i == 0: action.setChecked(True)
        self.window_group.triggered.connect(self.signal.DisplayChange)

    def _build_display_menu(self):
        

        self.display_menu = self.addMenu("&Display")
        self.overlay_action = QAction("Overlay", self, checkable=True)
        self.overlay_action.setChecked(self.initial_setting.overlay)
        self.overlay_action.triggered.connect(self.signal.DisplayChange)

        self.contour_action = QAction("Contour", self, checkable=True)
        self.contour_action.setChecked(self.initial_setting.contour)
        self.contour_action.triggered.connect(self.signal.DisplayChange)

        self.display_menu.addAction(self.overlay_action)
        self.display_menu.addAction(self.contour_action)
        self.display_menu.addSeparator()
        
        self.display_menu.addAction("Display Settings", self._open_setting_dialog)

    def _build_axis_menu(self): 
        self.axis_menu = self.addMenu("&Axis")
        self.axis_group = QActionGroup(self)
        self.axis_group.setExclusive(True)


        for i, (key,value) in enumerate(self.cfg['axis'].items()):
            action = QAction(key, self, checkable=True)
            action.setData(value)
            self.axis_group.addAction(action)
            self.axis_menu.addAction(action)
            if i==0: action.setChecked(True)

        self.axis_group.triggered.connect(self.signal.DisplayChange)


    def _build_help_menu(self):
        self.help_menu = self.addMenu("&Help")
        self.help_menu.addAction("About", self.signal.ShowAbout)


    def _open_setting_dialog(self):
        dlg = SettingDialog(self, init_cfg=self.cfg['display'])
        res = dlg.exec_()

        if int(res) == int(QDialog.Accepted):
            self.cfg['display'].update(dlg.values())
            self.signal.DisplayChange.emit()


    def _open_add_model_dialog(self):
        dlg = AddModelDialog(self, templates=list(self.cfg['models'].keys()))
        res = dlg.exec_()

        if int(res) == int(QDialog.Accepted):
            res              = dlg.values()
            model_name       = res['name']
            model_function   = res['function']
            model_checkpoint = res['checkpoint']

            # new checkpoint reuses the architecture / preprocessing of an existing model
            model_detail = copy.deepcopy(self.cfg['models'][res['template']])
            model_detail['checkpoint'] = model_checkpoint
            model_detail['function']   = model_function
            self.cfg['models'][model_name] = model_detail

            act_label = res['function']
            act_data  = ModelInfo(
                model_name=model_name,
                model_function=model_function,
                model_detail=model_detail
            )

            action = QAction(act_label, self, checkable=True)  
            action.setData(act_data)
            self.model_group.addAction(action)
            self.model_menu.addAction(action)



    def getDisplaySetting(self) -> DisplaySetting:
        display_setting = DisplaySetting(
            overlay= self.overlay_action.isChecked(),
            contour= self.contour_action.isChecked(),
            axis= self.axis_group.checkedAction().data(),
            window= self.window_group.checkedAction().data(),
            mask_color= self.cfg['display']['mask_color'],
            contour_width= self.cfg['display']['contour_width'],
            mask_alpha=self.cfg['display']['mask_alpha']
        )
        return display_setting
    

    
    def getCurrentModel(self) -> ModelInfo:
        model_data = self.model_group.checkedAction().data()
        return model_data
