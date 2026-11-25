# PyInstaller spec for SegUI
#   build:  venv3.10\Scripts\python -m PyInstaller SegUI.spec --noconfirm
#   output: dist\SegUI\SegUI.exe  (+ _internal\, config.yaml, checkpoints\ copied next to it)
#
# One-folder build: a 3 GB one-file exe would unpack itself to %TEMP% on every launch (slow start).
# config.yaml and checkpoints\ stay outside the exe so models / settings can be changed without rebuilding.
import os
import shutil
import yaml
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = (
    ['vSegnets', 'nnunet_infer']
    + collect_submodules('models')            # keras classes loaded by name from config.yaml ('vSegnets.R2UNet', ...)
    + collect_submodules('monai.inferers')
    + collect_submodules('monai.transforms')
)

a = Analysis(
    ['ui.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=['tensorboard', 'IPython', 'jupyter', 'notebook', 'pytest', 'tkinter'],
    noarchive=False,
)
# PyQt5 ships an old MSVC runtime (14.26) in Qt5\bin, which PyQt5's runtime hook puts on the DLL path before
# torch loads -> "WinError 1114 ... c10.dll". Drop it; Qt runs fine on the newer runtime collected at the top level.
OLD_QT_RUNTIME = {'msvcp140.dll', 'msvcp140_1.dll', 'vcruntime140.dll', 'vcruntime140_1.dll'}
a.binaries = [
    b for b in a.binaries
    if not (b[0].replace('\\', '/').lower().startswith('pyqt5/qt5/bin/') and os.path.basename(b[0]).lower() in OLD_QT_RUNTIME)
]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SegUI',
    console=False,   # windowed; stdout/stderr go to SegUI.log next to the exe
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name='SegUI', upx=False)

# files read at runtime from the exe folder
out = os.path.join(DISTPATH, 'SegUI')
shutil.copy2('config.yaml', out)
if os.environ.get('SEGUI_COPY_CHECKPOINTS', '1') == '1':
    # only the checkpoints config.yaml uses (file, or a whole nnU-Net results folder)
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)
    for model in cfg['models'].values():
        src = model['checkpoint']
        dst = os.path.join(out, src)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
            shutil.copy2(src, dst)
