import math
import re
import uuid
from typing import Union

import ipywidgets
import pandas as pd
import spacy
from IPython.display import display
from textacy.extract import keyword_in_context

SEARCH_CSS_TEMPLATE = """
<style>
#{element_id} table {{
  background-color: #f9f9f9;
}}

#{element_id} tbody tr:nth-child(odd) {{
  background-color: #f9f9f9;
}}

#{element_id} tbody tr:nth-child(even) {{
  background-color: #dddddd;
}}

#{element_id} .atap.search_highlight {{
    background-color: #99CC99;
    padding-left: 0px;
    padding-right: 0px;
}}

#{element_id} .atap.context {{
    color: #444;
    white-space: pre;
}}
#{element_id} .atap.context.context_left {{
    text-align: right;
    padding-right: 0px;
}}
#{element_id} .atap.context.context_right {{
    text-align: left;
    padding-left: 0px;
}}
#{element_id}.search_table th {{
    text-align: left;
}}

#{element_id}.search_table td {{
    border: none;
}}

#{element_id} td.text_id {{
    border-right: 1px solid #bdbdbd;
}}

#{element_id} .atap.regex_error {{
    background-color: #AA7777;
}}
</style>
"""

SEARCH_TABLE_TEMPLATE = """
{css}
<table id="{element_id}" class="atap search_table">
    <thead>
        <tr>
            <th>Document ID</th>
            <th>Result</th>
            <th/>
            <th/>
        </tr>
    </thead>
    <tbody>
        {table_rows}
    </body>
</table>
{n_results} results. A maximum of {max_results} are displayed.
"""

SEARCH_ROW_TEMPLATE = """
<tr>
    <td class="atap text_id">{text_id}</td>
    <td class="atap context context_left">{match[0]}</td>
    <td class="atap search_highlight">{match[1]}</td>
    <td class="atap context context_right">{match[2]}</td>
</tr>
"""

REGEX_ERROR_TEMPLATE = """
{css}
<div id="{element_id}">
<span class="atap regex_error">
Incomplete or incorrect regular expression: {error.msg}
</span>
</div>
"""


def prepare_text_df(
    df: pd.DataFrame,
    text_column: str,
    id_column: str = None,
    language_model: Union[str, spacy.language.Language] = "en_core_web_sm",
):
    """
    Our text processing functions expect a dataframe with
    a few required columns:

    - "text": containing the raw text data
    - "text_id": a unique ID for each text
    - "spacy_doc": the text, processed into a spacy Doc

    prepare_text_df() takes the input df and creates/renames
    columns as necessary to match this format.
    """
    output = df.copy()
    if id_column is None:
        output["text_id"] = pd.Series(range(output.shape[0]), dtype=pd.Int64Dtype)
        id_column = "text_id"
    output = output.rename(columns={text_column: "text", id_column: "text_id"})
    output = output.set_index("text_id", drop=False)

    if isinstance(language_model, str):
        language_model = spacy.load(language_model)
    output["spacy_doc"] = output["text"].map(language_model)
    return output


class SearchWidget:
    def __init__(self, df: pd.DataFrame, results_per_page: int = 20):
        self.data = df
        self.results_per_page = results_per_page
        self.search_table = SearchTable(df=self.data)

    def show(self):
        def display_results(page: int, **kwargs):
            if not kwargs["keyword"]:
                return None
            for attr, value in kwargs.items():
                setattr(self.search_table, attr, value)

            results = self.search_table._get_results()
            # Need at least one page
            n_pages = max(self.search_table._get_total_pages(n_results=len(results)), 1)
            page_input.max = n_pages

            display(ipywidgets.HTML(self.search_table._get_html(page=page)))

        keyword_input = ipywidgets.Text(description="Keyword(s):")
        regex_toggle_input = ipywidgets.Checkbox(
            value=False,
            description="Enable regular expressions",
            disabled=False,
            style={"description_width": "initial"},
        )
        ignore_case_input = ipywidgets.Checkbox(
            value=True, description="Ignore case", disabled=False
        )
        whole_word_input = ipywidgets.Checkbox(
            value=False,
            description="Match whole words",
            disabled=False,
            style={"description_width": "initial"},
        )
        page_input = ipywidgets.BoundedIntText(
            value=1,
            min=1,
            max=1,
            step=1,
            description="Page:",
        )
        window_width_input = ipywidgets.BoundedIntText(
            value=50,
            min=10,
            step=1,
            description="Window size (characters):",
            style={"description_width": "initial"},
        )
        output = ipywidgets.interactive_output(
            display_results,
            {
                "keyword": keyword_input,
                "regex": regex_toggle_input,
                "ignore_case": ignore_case_input,
                "whole_word": whole_word_input,
                "page": page_input,
                "window_width": window_width_input,
            },
        )
        # Excel export
        filename_input = ipywidgets.Text(
            value="",
            placeholder="Enter filename",
            description="Filename (without.xlsx extension)",
            style={"description_width": "initial"},
            disabled=False,
        )
        export_button = ipywidgets.Button(
            description="Export to Excel", disabled=False, button_style="success"
        )

        def _export(widget):
            filename = filename_input.value
            # TODO: raise an error here?
            if not filename:
                return

            if not filename.endswith(".xlsx"):
                filename = filename + ".xlsx"

            self.search_table.to_excel(filename)

        export_button.on_click(_export)
        export_controls = ipywidgets.HBox([filename_input, export_button])

        # Set up layout of widgets
        checkboxes = ipywidgets.HBox(
            [regex_toggle_input, ignore_case_input, whole_word_input]
        )
        checkboxes.layout.justify_content = "flex-start"
        number_inputs = ipywidgets.HBox([page_input, window_width_input])
        number_inputs.layout.justify_content = "flex-start"
        return ipywidgets.VBox(
            [keyword_input, checkboxes, number_inputs, export_controls, output]
        )


class SearchTable:
    def __init__(
        self,
        df: pd.DataFrame,
        keyword: str = "",
        regex: bool = False,
        ignore_case: bool = True,
        whole_word: bool = False,
        results_per_page: int = 20,
        window_width: int = 50,
    ):
        self.df = df
        self.keyword = keyword
        self.regex = regex
        self.ignore_case = ignore_case
        self.whole_word = whole_word
        self.results_per_page = results_per_page
        self.window_width = window_width
        self.element_id = "atap_" + str(uuid.uuid4())
        self.css = SEARCH_CSS_TEMPLATE.format(element_id=self.element_id)

    def _repr_mimebundle_(self, include, exclude):
        return {"text/html": self._get_html()}

    def _get_search_regex(self) -> re.Pattern:
        """
        Compile a regex we can pass to keyword_in_context.
        Will be escaped to only return literal matches if
        self.regex = False.
        Will raise a re.error exception if the regex is invalid
        (useful for showing feedback to the user).

        Only call this at display time because there's a decent
        chance the regex will be invalid while the user is typing,
        and we just handle the exception as part of the display logic.
        """
        if not self.regex:
            regex = re.escape(self.keyword)
        else:
            regex = self.keyword
        if self.whole_word:
            regex = f"\\b{regex}\\b"

        flags = re.IGNORECASE if self.ignore_case else 0
        return re.compile(regex, flags=flags)

    def is_regex_valid(self) -> bool:
        try:
            _ = self._get_search_regex()
            return True
        except re.error:
            return False

    def _get_results(self):
        def _get_matches(doc):
            matches = keyword_in_context(
                doc, keyword=search_regex, window_width=self.window_width
            )
            return list(matches)

        search_regex = self._get_search_regex()
        search_results = self.df["spacy_doc"].apply(_get_matches)
        search_results = search_results.loc[search_results.map(len) > 0].explode()
        search_results.name = "match"
        return search_results

    def _get_total_pages(self, n_results: int) -> int:
        return math.ceil(n_results / self.results_per_page)

    def _get_table_html(self, search_results: pd.Series, n_total: int) -> str:
        table_rows = "\n".join(
            SEARCH_ROW_TEMPLATE.format(text_id=text_id, match=match).replace(r"\n", "")
            for text_id, match in search_results.iteritems()
        )

        html = SEARCH_TABLE_TEMPLATE.format(
            css=self.css,
            element_id=self.element_id,
            table_rows=table_rows,
            n_results=n_total,
            max_results=self.results_per_page,
        )
        return html

    def _get_error_html(self, error) -> str:
        return REGEX_ERROR_TEMPLATE.format(
            css=self.css, element_id=self.element_id, error=error
        )

    def _get_html(self, page: int = 1):
        regex_valid = self.is_regex_valid()
        if not regex_valid:
            try:
                self._get_search_regex()
            except re.error as error:
                return self._get_error_html(error)

        start_index = (page - 1) * self.results_per_page
        end_index = (page * self.results_per_page) - 1
        results = self._get_results()
        n_total = len(results)
        results = results.iloc[start_index:end_index]
        return self._get_table_html(results, n_total=n_total)

    def to_dataframe(self) -> pd.DataFrame:
        results = self._get_results()
        results_df = results.to_frame()
        # Use apply(pd.Series) to unpack nested lists into columns
        results_df[["left_context", "match", "right_context"]] = results_df[
            "match"
        ].apply(pd.Series)
        results_df.index.name = "text_id"
        results_df.reset_index(inplace=True)
        # Reorder columns
        results_df = results_df[["text_id", "left_context", "match", "right_context"]]
        return results_df

    def to_excel(self, filename: str, max_col_width: int = 100):
        def _set_col_widths():
            for column in df:
                col_width = max(df[column].astype(str).map(len).max(), len(column))
                col_width = min(col_width, max_col_width)
                col_index = df.columns.get_loc(column)
                writer.sheets["results"].set_column(col_index, col_index, col_width)

        df = self.to_dataframe()
        # Need xlsxwriter for (relatively) easy setting of column widths
        writer = pd.ExcelWriter(filename, engine="xlsxwriter")
        df.to_excel(writer, index=False, sheet_name="results")
        # Formatting
        right_align = writer.book.add_format()
        right_align.set_align("right")
        left_align = writer.book.add_format()
        left_align.set_align("left")
        writer.sheets["results"].set_column("B:B", None, right_align)
        writer.sheets["results"].set_column("D:D", None, left_align)
        _set_col_widths()

        writer.save()
