from PyQt5 import QtWidgets as qw
from PyQt5 import QtCore as qc
from PyQt5 import QtGui as qg


from PyQt5 import QtCore, QtGui, QtWidgets

class ColorButton(qw.QPushButton):
    colorChanged = qc.pyqtSignal(qg.QColor)

    def __init__(self, color: tuple = (255,0,0), parent = None, show_alpha = False):
        super().__init__(parent)
        self._color =  qg.QColor(*color)
        self.show_alpha = show_alpha
        self.setCursor(qc.Qt.CursorShape.PointingHandCursor)
        self._update_style()
        self.clicked.connect(self.choose_color)

    
    def choose_color(self):
        options = qw.QColorDialog.ColorDialogOption.ShowAlphaChannel if self.show_alpha else qw.QColorDialog.ColorDialogOption(0)

        c = qw.QColorDialog.getColor(self._color, self, "Choose Color", options=options)
        if not c.isValid(): return # dialog cancelled
        self.setColor(c)

    def setColor(self, c:qg.QColor):
        self._color = c
        self._update_style()
        self.colorChanged.emit(self._color)


    def _update_style(self):
        r,g,b,a = self._color.getRgb()
        luminance = (0.299 * r + 0.587*g + 0.114*b)
        text = '#000' if luminance > 186 else '#fff'
        self.setText(self._color.name())
        self.setStyleSheet(f"QPushButton {{ background:{ self._color.name(qg.QColor.NameFormat.HexRgb)}; color: {text}; }}")


class SettingDialog(qw.QDialog):
    def __init__(self, parent, init_cfg: dict):
        super().__init__(parent)
        self.setWindowTitle("Display Settings")
        self.parent = parent
        self.init_cfg = init_cfg
        self.cfg = dict(init_cfg) # edit a copy so Cancel leaves the config untouched

        self.contour_width_spin = qw.QDoubleSpinBox()
        self.contour_width_spin.setRange(0.0,3.0)
        self.contour_width_spin.setSingleStep(0.1)
        self.contour_width_spin.setValue(init_cfg['contour_width'])

        self.mask_alpha_spin = qw.QDoubleSpinBox()
        self.mask_alpha_spin.setRange(0,255)
        self.mask_alpha_spin.setSingleStep(1)
        self.mask_alpha_spin.setValue(init_cfg['mask_alpha'])

        self.color_btn = ColorButton(color=init_cfg['mask_color'], show_alpha=False)
        self.color_btn.colorChanged.connect(self.on_color_changed)

        form = qw.QFormLayout()
        form.addRow("Mask Color:", self.color_btn)
        form.addRow("Contour Width:", self.contour_width_spin)
        form.addRow("Mask Alpha:", self.mask_alpha_spin)

        self.buttons = qw.QDialogButtonBox(
            qw.QDialogButtonBox.StandardButton.Cancel | 
            qw.QDialogButtonBox.StandardButton.Apply 
        )
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(qw.QDialogButtonBox.StandardButton.Apply).clicked.connect(self._on_apply)

        vl = qw.QVBoxLayout(self)
        vl.addLayout(form)
        vl.addWidget(self.buttons)
        self.setLayout(vl)

    def values(self):
        return self.cfg

    @qc.pyqtSlot(qg.QColor)
    def on_color_changed(self, color: qg.QColor):
        r,g,b,a = color.getRgb()
        self.cfg['mask_color'] = (r,g,b)

    def _on_apply(self):
        self.cfg['contour_width'] = self.contour_width_spin.value()
        self.cfg['mask_alpha']    = self.mask_alpha_spin.value()
        self.accept()




class AddModelDialog(qw.QDialog):
    def __init__(self, parent, templates: list = None):
        super().__init__(parent)
        self.setWindowTitle("Add Model")
        self.parent = parent
        self.name_edit = qw.QLineEdit()
        self.function_edit = qw.QLineEdit()
        self.checkpoint_edit = qw.QLineEdit()
        self.template_combo = qw.QComboBox()
        self.template_combo.addItems(templates or [])

        form_layout = qw.QFormLayout()
        form_layout.addRow('Name:', self.name_edit)
        form_layout.addRow('Function:', self.function_edit)
        form_layout.addRow('Checkpoint:', self.checkpoint_edit)
        form_layout.addRow('Based on:', self.template_combo)



        self.dialog_buttons = qw.QDialogButtonBox(
            qw.QDialogButtonBox.StandardButton.Cancel | 
            qw.QDialogButtonBox.StandardButton.Apply 
        )
        self.dialog_buttons.rejected.connect(self.reject)
        self.dialog_buttons.button(qw.QDialogButtonBox.StandardButton.Apply).clicked.connect(self._on_apply)
        self.dialog_buttons.button(qw.QDialogButtonBox.StandardButton.Apply).setText('Add')
        
        
        vbox = qw.QVBoxLayout()
        vbox.addLayout(form_layout)
        vbox.addWidget(self.dialog_buttons)
        self.setLayout(vbox)

    def values(self):
        return {
            'name':  self.name_edit.text(),
            'function': self.function_edit.text(),
            'checkpoint': self.checkpoint_edit.text(),
            'template': self.template_combo.currentText()
        }


    def _on_apply(self):
        v = self.values()
        if not v['name'] or not v['function'] or not v['checkpoint'] or not v['template']:
            qw.QMessageBox.warning(self, "Warning", "Name, Function, Checkpoint and Based on are required")
            return
        self.accept()



class AboutDialog(qw.QDialog):
    def __init__(self,
                 app_name: str,
                 version: str,
                 website: str,
                 description: str,
                 parent = None):
        

        super().__init__(parent)
        self.setWindowTitle(f"About {app_name}")
        self.setModal(True)
        self.setMinimumWidth(420)


        # --- Icon (optional: inherits from parent app icon if available) ---
        icon_label = QtWidgets.QLabel()
        icon = self.windowIcon() or (parent.windowIcon() if parent else QtGui.QIcon())
        if not icon.isNull():
            pix = icon.pixmap(64, 64)
            icon_label.setPixmap(pix)
        else:
            icon_label.setVisible(False)


        # --- Title & Version ---
        title_lbl = QtWidgets.QLabel(f"<b style='font-size:14pt'>{app_name}</b>")
        ver_lbl = QtWidgets.QLabel(f"Version: <code>{version}</code>")


        title_ver_box = QtWidgets.QVBoxLayout()
        title_ver_box.addWidget(title_lbl)
        title_ver_box.addWidget(ver_lbl)
        title_ver_box.addStretch(1)


        top = QtWidgets.QHBoxLayout()
        top.addWidget(icon_label)
        top.addLayout(title_ver_box)
        top.addStretch(1)


        # --- Description (rich text allowed) ---
        desc = QtWidgets.QTextBrowser()
        desc.setOpenExternalLinks(True)
        desc.setReadOnly(True)
        desc.setFrameShape(QtWidgets.QFrame.NoFrame)
        desc.setHtml(
        f"""
        <p>{description}</p>
        <p>Website: <a href='{website}'>{website}</a></p>
        <p>&copy; {QtCore.QDate.currentDate().year()} {app_name} Team</p>
        """
        )


        # --- Optional details (Qt & Python versions) ---
        details = QtWidgets.QTextEdit()
        details.setReadOnly(True)
        details.setFixedHeight(90)
        details.setText(
        "\n".join(
        [
        f"Qt Version: {QtCore.QT_VERSION_STR}",
        f"PyQt Version: {QtCore.PYQT_VERSION_STR}",
        f"Python: {QtCore.PYTHON_VERSION_STR if hasattr(QtCore, 'PYTHON_VERSION_STR') else QtCore.qVersion()}",
        f"Platform: {QtCore.QSysInfo.prettyProductName()} ({QtCore.QSysInfo.currentCpuArchitecture()})",
        ]
        )
        )


        # --- Buttons ---
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        btns.accepted.connect(self.accept)


        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addSpacing(8)
        lay.addWidget(desc)
        lay.addWidget(details)
        lay.addWidget(btns)