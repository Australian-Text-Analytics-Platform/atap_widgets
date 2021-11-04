from typing import Callable

from bokeh import layouts
from bokeh.plotting import figure, show
from bokeh.transform import linear_cmap, factor_cmap
from bokeh.themes import Theme
from bokeh import models
from bokeh import palettes
import numpy as np
import pandas as pd

from conversation import Conversation

BLANK_PLOT_THEME = Theme(
    json={
        "attrs": {
            "Figure": {
                "background_fill_color": "#FFFFFF",
                "border_fill_color": "#FFFFFF",
                "outline_line_color": "#000000",
            },
            "Axis": {
                "axis_line_color": None,
            },
            "Grid": {"grid_line_color": None},
            "Title": {"text_color": "black"},
        }
    }
)

WORD_WRAP_CELL_TEMPLATE = """
<span style="word-wrap: break-word; max-width: 600px; 
      display: inline-block; white-space: normal;">
<%= value %>
</span>
"""


WORD_WRAP_FORMATTER = models.widgets.tables.HTMLTemplateFormatter(
    template=WORD_WRAP_CELL_TEMPLATE
)

PLOT_HELP_TEXT = """
<h3>Similarity plot</h3>
    
<p>
Click an item on the diagonal to view it in the table below.
Click anywhere on the background to deselect it.
</p>
"""


class ConversationPlot:
    # TODO: Just setting these as constants for now,
    #   can allow passing a dict of params for more customisation later
    width = 800
    height = 800

    def __init__(self, conversation: Conversation, tile_padding: int = 1):
        self.tile_padding = tile_padding

        long_data = self._get_long_data(conversation)
        self.n_speakers = conversation.n_speakers
        self.speakers = conversation.get_speaker_names()
        self.diagonal_datasource = self._get_diagonal_datasource(
            conversation, long_data=long_data
        )
        self.similarity_datasource = self._get_similarity_datasource(
            long_data=long_data
        )

    def _get_diagonal_datasource(
        self, conversation: Conversation, long_data: pd.DataFrame
    ) -> models.ColumnDataSource:
        """
        Get the data for the texts on the diagonal, as a bokeh ColumnDataSource
        """
        text_data = long_data.query("x_index == y_index").copy()
        text_data["text"] = text_data["x_index"].map(conversation.data["text"])
        return models.ColumnDataSource(text_data)

    def _get_similarity_datasource(
        self, long_data: pd.DataFrame
    ) -> models.ColumnDataSource:
        """
        Get the data for the similarity cells in the lower triangle
        """
        return models.ColumnDataSource(long_data.query("x_index < y_index"))

    def _get_long_data(self, conversation: Conversation):
        """
        Convert the similarity matrix to long form for plotting
        """
        # Info about each text, for merging with similarity data
        text_info = pd.DataFrame(
            {
                "n_words": conversation.data["spacy_doc"].map(len),
                "speaker": conversation.data["speaker"],
            }
        )
        # Log-scale number of words for rectangle sizes
        text_info["size"] = np.log(text_info["n_words"])
        # Position is half the width of the current tile plus half the width,
        #   then we add tile_padding between each
        text_info["position"] = (
            (text_info["size"] / 2) + (text_info["size"].shift(1, fill_value=0) / 2)
        ).cumsum() + (np.arange(len(text_info)) * self.tile_padding)

        similarity = (
            pd.DataFrame(conversation.similarity_matrix()).melt(
                ignore_index=False, var_name="y_index", value_name="similarity"
            )
            # Copy index into x_index column
            .assign(x_index=lambda df: df.index)
            # Don't need the upper triangle
            .query("x_index <= y_index")
        )
        # Copy x index
        similarity["x_index"] = similarity.index
        # Get x properties
        similarity["width"] = similarity["x_index"].map(text_info["size"])
        similarity["x_position"] = similarity["x_index"].map(text_info["position"])
        similarity["x_speaker"] = similarity["x_index"].map(text_info["speaker"])
        similarity["n_words"] = similarity["x_index"].map(text_info["n_words"])
        # Get y properties
        similarity["height"] = similarity["y_index"].map(text_info["size"])
        similarity["y_position"] = similarity["y_index"].map(text_info["position"])
        similarity["y_speaker"] = similarity["y_index"].map(text_info["speaker"])

        return similarity

    def _create_text_table(self):
        """
        Create the text table widget that shows information for clicked/selected
        tiles
        """
        columns = [
            models.TableColumn(field="x_speaker", title="speaker"),
            models.TableColumn(
                field="text", title="text", formatter=WORD_WRAP_FORMATTER
            ),
        ]
        # Start off showing nothing
        empty_view = models.CDSView(
            source=self.diagonal_datasource, filters=[models.IndexFilter([])]
        )
        table = models.DataTable(
            source=self.diagonal_datasource,
            columns=columns,
            autosize_mode="fit_columns",
            sizing_mode="stretch_width",
            row_height=100,
            view=empty_view,
        )
        return table

    def _get_categorical_palette(
        self, n_colours: int, palette_name: str = "Category10"
    ):
        """
        Return a palette of n_colours
        """
        # bokeh doesn't have 2-colour palettes, just use the 3-colour
        palette_n = max(n_colours, 3)
        return palettes.d3[palette_name][palette_n][:n_colours]

    def show_plot(self):
        plot_func = self.create_plot_function()
        try:
            show(plot_func)
        except AssertionError:
            raise RuntimeError(
                "No bokeh output detected. Make sure you run "
                "bokeh.io.output_notebook() at the top of your notebook"
                "to enable output"
            )

    def create_plot_function(self) -> Callable:
        """
        In order to have interactivity controlled by Python callbacks,
        bokeh requires us to wrap the plot creation in a function that
        accepts a "document" argument.
        """

        def _add_plot_tools(plot):
            click_to_select = models.TapTool()
            hover_similarity = models.HoverTool(
                names=["similarity_tiles"], tooltips=[("Similarity", "@similarity")]
            )
            hover_text = models.HoverTool(
                names=["text_tiles"], tooltips=[("Text", "@text")]
            )
            plot.add_tools(hover_similarity, hover_text, click_to_select)
            plot.toolbar.active_tap = click_to_select

        def plot_func(doc):
            doc.theme = BLANK_PLOT_THEME

            # Table for viewing texts
            text_table = self._create_text_table()

            def _set_table_filter(attr, old, new):
                text_table.view = models.CDSView(
                    filters=[models.IndexFilter(new)], source=self.diagonal_datasource
                )

            self.diagonal_datasource.selected.on_change("indices", _set_table_filter)

            ## Main plot
            plot = figure(width=self.width, height=self.height, aspect_scale=1.0)
            # Plot similarity tiles
            similarity_colour_mapper = linear_cmap("similarity", "Viridis256", 0.0, 1.0)
            plot.rect(
                x="x_position",
                y="y_position",
                width="width",
                height="height",
                fill_color=similarity_colour_mapper,
                source=self.similarity_datasource,
                name="similarity_tiles",
            )
            # Colour bar/legend for similarity
            colour_bar = models.ColorBar(
                color_mapper=similarity_colour_mapper["transform"]
            )
            plot.add_layout(colour_bar, "right")

            # Plot diagonal/text tiles
            speaker_colours = self._get_categorical_palette(self.n_speakers)
            speaker_cmap = factor_cmap(
                "x_speaker", palette=speaker_colours, factors=self.speakers
            )
            plot.rect(
                x="x_position",
                y="y_position",
                fill_color=speaker_cmap,
                source=self.diagonal_datasource,
                name="text_tiles",
            )

            # Options/style
            plot.y_range.flipped = True
            plot.axis.major_tick_line_color = None
            plot.axis.minor_tick_line_color = None
            plot.axis.major_label_text_color = None

            # Interactivity/plot tools
            _add_plot_tools(plot)

            # Create layout
            doc.add_root(
                layouts.column(
                    layouts.row(models.Div(text=PLOT_HELP_TEXT, width=400)),
                    layouts.row(plot),
                    layouts.row(models.Div(text="<h3>Selected text</h3>")),
                    layouts.row(text_table),
                    sizing_mode="stretch_width",
                )
            )

        return plot_func
