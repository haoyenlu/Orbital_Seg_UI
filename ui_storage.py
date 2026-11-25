from PyQt5.QtCore import QObject, pyqtSignal, pyqtProperty

from ui_data import ModelInfo, DisplayItem, DisplaySetting

class ModelStorage(QObject):
    ModelChanged = pyqtSignal(ModelInfo)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = None

    
    def getModel(self):
        return self._model
    
    def setModel(self, m):
        if m is self._model: return 

        self._model = m
        self.ModelChanged.emit(m)
    
    current_model = pyqtProperty(ModelInfo, fget=getModel, fset=setModel, notify=ModelChanged)




class DisplayItemStorage(QObject):
    DisplayItemChanged = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._item = None

    def getItem(self):
        return self._item
    
    def setItem(self, _it):
        self._item = _it
        self.DisplayItemChanged.emit()
    
    current_item = pyqtProperty(DisplayItem, fget=getItem, fset=setItem, notify=DisplayItemChanged)



class DisplaySettingStorage(QObject):
    SettingChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setting = None

    def getSetting(self):
        return self._setting
    
    def setSetting(self, _setting: DisplaySetting):
        self._setting = _setting
        self.SettingChanged.emit()
    
    current_setting = pyqtProperty(DisplaySetting, fget=getSetting, fset=setSetting, notify=SettingChanged)