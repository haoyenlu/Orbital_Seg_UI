"""Image / mask list in the left panel (widgets are created off-screen, nothing is shown)"""
import numpy as np
import pytest
from PyQt5.QtCore import Qt

from conftest import make_image
from ui_data import ListItem
from ui_panel import LeftPanel
from ui_storage import DisplayItemStorage


@pytest.fixture
def panel(qtbot):
    p = LeftPanel(display_storage=DisplayItemStorage())
    qtbot.addWidget(p)
    return p


def _image(value=0):
    return make_image(np.full((4, 5, 6), value, np.int16))


def _add(panel, name):
    panel.add_item(ListItem(name=name, data=_image(), type='image', path=f'E:/data/{name}.nii.gz'))
    return panel.ItemList.topLevelItem(panel.ItemList.topLevelItemCount() - 1)


def test_add_item_selects_and_displays_it(qtbot, panel):
    with qtbot.waitSignal(panel.itemChange) as blocker:
        item = _add(panel, 'case_1')
    assert blocker.args == [item]
    assert item.text(1) == 'image'
    assert item.toolTip(0) == 'E:/data/case_1.nii.gz'


def test_reset_emits_none(qtbot, panel):
    """Regression: itemChange(None) used to crash (signal was typed QTreeWidgetItem)"""
    _add(panel, 'case_1')
    with qtbot.waitSignal(panel.itemChange) as blocker:
        panel.reset()
    assert blocker.args == [None]
    assert panel.ItemList.topLevelItemCount() == 0


def test_duplicate_names_get_suffix(panel):
    names = [_add(panel, 'case_1').text(0) for _ in range(3)]
    assert names == ['case_1', 'case_1 (2)', 'case_1 (3)']


def test_add_mask_child(panel):
    image  = _add(panel, 'case_1')
    labels = [{'value': 1, 'name': 'Nerve', 'color': [255, 255, 0]}]
    mask   = _image(1)
    panel.add_mask_child(image, 'Nerve', mask, labels)

    child = image.child(0)
    assert child.text(1) == 'mask'
    assert child.data(0, Qt.UserRole) is mask
    assert child.data(0, Qt.UserRole + 1) == labels
    assert image.isExpanded()
    assert panel.selected_item() is child


def test_selected_images_for_prediction(qtbot, panel):
    a, b, c = (_add(panel, n) for n in ('a', 'b', 'c'))
    panel.add_mask_child(a, 'Fat', _image(1), select=False)

    targets = []
    panel.predTargetsChanged.connect(targets.append)

    panel.ItemList.clearSelection()
    for item in (c, a.child(0), b):                                      # selection order does not matter
        item.setSelected(True)

    assert panel.selected_images() == [a, b, c]                          # list order; mask -> its image
    assert panel.selected_item() is a.child(0)                           # first selected item is displayed
    assert targets[-1] == [a, b, c]
