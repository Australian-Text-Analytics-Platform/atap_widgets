import re

import pytest

from atap_widgets.plotting import ConversationPlot


# bokeh palettes only have 256 colours so make sure we can generate more
@pytest.mark.parametrize("n", [2, 3, 10, 15, 20, 100, 240, 256, 260])
def test_get_palettes(n):
    hex_pattern = re.compile("#[a-f0-9]{3,6}")
    palette = ConversationPlot._get_categorical_palette(n)
    assert len(palette) == n
    assert all(hex_pattern.match(colour) for colour in palette)
