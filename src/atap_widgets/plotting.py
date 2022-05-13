import warnings
from typing import Callable
from typing import Optional

import numpy as np
import pandas as pd
from bokeh import layouts
from bokeh import models
from bokeh import palettes
from bokeh.plotting import figure
from bokeh.plotting import show
from bokeh.themes import Theme
from bokeh.transform import factor_cmap

from .conversation import ConceptSimilarityModel
from .conversation import Conversation
from .utils import _get_remote_jupyter_proxy_url
from .utils import _is_binder

DEFAULT_BOKEH_PORT = 48_246

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


PLOT_HELP_TEXT = """
<h3>Similarity plot</h3>

<p>
Click an item on the diagonal to view it in the table below.
Click anywhere on the background to deselect it.
</p>
"""


def _get_word_wrap_formatter():
    """
    We can't store this formatter as a constant or it causes
    issues in Jupyter notebooks to do with reusing
    models
    """
    return models.widgets.tables.HTMLTemplateFormatter(template=WORD_WRAP_CELL_TEMPLATE)


class ConversationPlot:
    """
    Create an interactive conversation plot from a Conversation object.

    Args:
        conversation: A Conversation object.
        similarity_matrix: An optional matrix of similarity scores between each
            turn in the conversation. If this isn't provided, spacy's default
            similarity calculation for documents (cosine similarity between
            average word embeddings) will be used.
            The index and column names for the matrix must match the text_id from
            the conversation.
        similarity_model: A ConceptSimilarityModel instance that will be used
            to identify the individual terms in common between turns.
        options: A dictionary of options for the visual style of the plot. See
            ConversationPlot.DEFAULT_OPTIONS for the available options.
    """

    DEFAULT_OPTIONS = {"width": 800, "height": 800, "tile_padding": 1}

    def __init__(
        self,
        conversation: Conversation,
        similarity_matrix: Optional[pd.DataFrame] = None,
        similarity_model: Optional[ConceptSimilarityModel] = None,
        options: Optional[dict] = None,
    ):
        self.options = self.DEFAULT_OPTIONS.copy()
        if options is not None:
            self.options.update(options)

        self.similarity_model = similarity_model
        long_data = self._get_long_data(conversation, similarity_matrix)
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
        Get the data for the similarity cells in the lower triangle,
        including calculating coordinates for the triangles which
        will be displayed.
        """
        similarity_data = long_data.query("x_index < y_index").copy()
        half_width = similarity_data["width"] / 2
        half_height = similarity_data["height"] / 2
        triangle_coords = pd.DataFrame(index=similarity_data.index)
        triangle_coords["left"] = similarity_data["x_position"] - half_width
        triangle_coords["right"] = similarity_data["x_position"] + half_width
        # We flip the y-axis for the plot, so subtract to get to the top
        triangle_coords["top"] = similarity_data["y_position"] - half_height
        triangle_coords["bottom"] = similarity_data["y_position"] + half_height

        # Concepts in common
        def _get_concepts(row):
            concepts = common_concepts[(row["x_index"], row["y_index"])]
            return ", ".join(concepts)

        if self.similarity_model is not None:
            common_concepts = self.similarity_model.get_common_concepts()
            similarity_data["concepts"] = similarity_data.apply(
                _get_concepts, axis="columns"
            )

        # Having some issues converting nested pandas cols to ColumnDataSource,
        #   convert to dict before we try to convert to ColumnDataSource
        columns = ["x_index", "y_index", "x_speaker", "y_speaker", "similarity"]
        if self.similarity_model is not None:
            columns.append("concepts")
        similarity_dict = similarity_data[columns].to_dict(orient="list")

        # Get upper triangle coordinates: bottom left, top left, top right
        # Use df.values.tolist() to quickly create nested lists from df rows
        # bokeh's multi_polygon plot function actually needs triply-nested lists,
        # because it allows for plotting polygons with multiple holes
        similarity_dict["upper_triangle_x"] = [
            [[row]]
            for row in triangle_coords[["left", "left", "right"]].values.tolist()
        ]
        similarity_dict["upper_triangle_y"] = [
            [[row]] for row in triangle_coords[["bottom", "top", "top"]].values.tolist()
        ]

        # Get lower triangle coordinates: bottom left, top right, bottom right
        similarity_dict["lower_triangle_x"] = [
            [[row]]
            for row in triangle_coords[["left", "right", "right"]].values.tolist()
        ]
        similarity_dict["lower_triangle_y"] = [
            [[row]]
            for row in triangle_coords[["bottom", "top", "bottom"]].values.tolist()
        ]

        return models.ColumnDataSource(similarity_dict)

    def _get_long_data(
        self,
        conversation: Conversation,
        similarity_matrix: Optional[pd.DataFrame] = None,
    ):
        """
        Convert the similarity matrix to long form for plotting.
        """
        if similarity_matrix is None:
            similarity_matrix = pd.DataFrame(conversation.vector_similarity_matrix())
        in_current_plot = conversation.data["text_id"].isin(similarity_matrix.index)
        current_data = conversation.data.loc[in_current_plot, :]
        # Info about each text, for merging with similarity data
        text_info = pd.DataFrame(
            {
                "n_words": current_data["spacy_doc"].map(len),
                "speaker": current_data["speaker"],
            }
        )
        # Log-scale number of words for rectangle sizes
        text_info["size"] = np.log(text_info["n_words"])
        # Position is half the width of the current tile plus half the width,
        #   then we add tile_padding between each
        text_info["position"] = (
            (text_info["size"] / 2) + (text_info["size"].shift(1, fill_value=0) / 2)
        ).cumsum() + (np.arange(len(text_info)) * self.options["tile_padding"])

        similarity = (
            similarity_matrix.melt(
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
                field="text", title="text", formatter=_get_word_wrap_formatter()
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

    def show(self, autodetect_binder: bool = True, **kwargs):
        """
        Show the interactive plot as a Jupyter notebook output.
        Requires bokeh.io.output_notebook() to have been run.

        Args:
           autodetect_binder: Automatically detect if we are running on binder
              and set the URL/proxy for the bokeh server
           kwargs: additional arguments passed to bokeh.io.show()
        """
        plot_func = self.create_plot_function()
        if autodetect_binder:
            if _is_binder():
                kwargs["notebook_url"] = _get_remote_jupyter_proxy_url
        try:
            show(plot_func, **kwargs)
        except AssertionError:
            raise RuntimeError(
                "No bokeh output detected. Make sure you run "
                "bokeh.io.output_notebook() at the top of your notebook"
                " to enable output"
            ) from None

    def show_plot(self):
        warnings.warn("show_plot() has been renamed to show()", DeprecationWarning)
        self.show()

    def create_plot_function(self) -> Callable:
        """
        In order to have interactivity controlled by Python callbacks,
        bokeh requires us to wrap the plot creation in a function that
        accepts a "document" argument.
        """

        def _add_plot_tools(plot):
            similarity_tooltips = [
                ("ID (column)", "@x_index"),
                ("ID (row)", "@y_index"),
                ("Similarity", "@similarity"),
            ]
            if self.similarity_model is not None:
                similarity_tooltips.append(("Shared concepts:", "@concepts"))
            hover_similarity = models.HoverTool(
                names=["similarity_upper", "similarity_lower"],
                tooltips=similarity_tooltips,
            )

            text_tooltips = [("ID", "@x_index"), ("Text", "@text")]
            hover_text = models.HoverTool(names=["text_tiles"], tooltips=text_tooltips)

            click_to_select = models.TapTool()
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

            def _set_table_filter_from_similarity(attr, old, new):
                x_texts = [self.similarity_datasource.data["x_index"][i] for i in new]
                y_texts = [self.similarity_datasource.data["y_index"][i] for i in new]
                matching = [
                    x in x_texts or y in y_texts
                    for (x, y) in zip(
                        self.diagonal_datasource.data["x_index"],
                        self.diagonal_datasource.data["y_index"],
                    )
                ]
                text_table.view = models.CDSView(
                    filters=[models.BooleanFilter(matching)],
                    source=self.diagonal_datasource,
                )

            self.diagonal_datasource.selected.on_change("indices", _set_table_filter)
            self.similarity_datasource.selected.on_change(
                "indices", _set_table_filter_from_similarity
            )

            # Main plot ############
            plot = figure(
                width=self.options["width"],
                height=self.options["height"],
                aspect_scale=1.0,
            )

            # Set up speaker colours
            speaker_colours = self._get_categorical_palette(self.n_speakers)
            speaker_cmap = factor_cmap(
                "x_speaker", palette=speaker_colours, factors=self.speakers
            )
            other_speaker_cmap = factor_cmap(
                "y_speaker", palette=speaker_colours, factors=self.speakers
            )
            # Plot similarity tiles
            plot.multi_polygons(
                xs="upper_triangle_x",
                ys="upper_triangle_y",
                alpha="similarity",
                color=speaker_cmap,
                source=self.similarity_datasource,
                line_width=0,
                name="similarity_upper",
            )
            plot.multi_polygons(
                xs="lower_triangle_x",
                ys="lower_triangle_y",
                alpha="similarity",
                color=other_speaker_cmap,
                source=self.similarity_datasource,
                line_width=0,
                name="similarity_lower",
            )

            # Plot diagonal/text tiles
            plot.rect(
                x="x_position",
                y="y_position",
                color=speaker_cmap,
                source=self.diagonal_datasource,
                name="text_tiles",
                legend_field="x_speaker",
            )

            # Options/style
            plot.y_range.flipped = True
            plot.axis.major_tick_line_color = None
            plot.axis.minor_tick_line_color = None
            plot.axis.major_label_text_color = None
            plot.legend.title = "Speaker"

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
