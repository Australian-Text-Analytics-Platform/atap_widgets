import math
import re
from unittest import result
import uuid
from typing import Union

import ipywidgets
import pandas as pd
import spacy
from IPython.display import display
from textacy.extract import keyword_in_context
import dask.bag as db
import re


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
    background-color: #CC9999;
}}
</style>
"""

SEARCH_TABLE_TEMPLATE = """
{css}
<table id="{element_id}" class="atap search_table">
    <thead>
        <tr>
            <th>Document ID</th>
            <th>Result / Context</th>
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
    <td class="atap context context_left">{left_context}</td>
    <td class="atap search_highlight">{match}</td>
    <td class="atap context context_right">{right_context}</td>
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


class NoResultsError(Exception):
    pass


def prepare_text_df(
    df: pd.DataFrame,
    text_column: str = "text",
    id_column: str = None,
    language_model: Union[str, spacy.language.Language] = "en_core_web_sm",
) -> pd.DataFrame:
    """
    Our text processing functions expect a dataframe with
    a few required columns:

    - "text": containing the raw text data
    - "text_id": a unique ID for each text
    - "spacy_doc": the text, processed into a spacy Doc

    prepare_text_df() takes an input DataFrame, df, and creates/renames
    columns as necessary to match this format.

    Args:
        df: Input dataframe containing multiple texts.
        text_column: The current name of the text column in df
        id_column: The current column name of the unique identifier for each text
            in df. If not given, numeric IDs will be generated for each text.
        language_model: The name of a spacy model like "en_core_web_sm", or a
            spacy language model instance.
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

class DataIngest():
    #
    """
    Run before prepare_text_df. Caters for different datastructures, file types and enables grouping of text column by chunk input:
        - loads either via csv or existing dataframe
        - Splits data into chunks that are used for grouping
    input:
        chunk:  (Optional) int. Group text by chunks representing number of rows that feeds into individual spacy docs.
        df_input: (Optional) Dataframe. Datframe where a column with "text" exists.
        path: file path to a csv that will be read into a dataframe. Neccessary that there is column with "text"
        type: type of underlying file to read. i.e. csv, txt or existing dataframe    
    """ 
    def __init__(self,path:str = "",type:str = "csv",chunk : int = 1,df_input = None,re_symbol_txt = ":") -> None:
        self.file_location = path
        self.type = type
        self.chunk = chunk
        self.df_input = df_input
        self.re_symbol_txt = re_symbol_txt
        self.data = self.read_data() #chunk iterator
        self.grouped_data = self.process()
        self.app = None
        
    def read_data(self):
        """ Cater for different file types / structures
        """
        if self.type == "csv":
            return pd.read_csv(self.file_location,chunksize = self.chunk)
        elif self.type == "dataframe":
            return self.chunk_a_dataframe(self.df_input)
        elif self.type == "txt":
            df = self.read_txt()
            return self.chunk_a_dataframe(df)

    def chunk_a_dataframe(self,df):
        if df is not None: 
            list_df = [df[i:i+self.chunk] for i in range(0,df.shape[0],self.chunk)]
            iter_df = iter(list_df)
            return iter_df # passthrough input dataframe as an iterable that can be chunked / split
        else:
            raise ValueError("type dataframe is enabled, but df_input is None ")

    def read_txt(self):
        #symbol = re.compile(self.re_symbol_txt)
        #b = db.read_text(self.file_location).str.strip().str.split(symbol) #works in jupyter not here?
        b = db.read_text(self.file_location).str.strip().str.split(self.re_symbol_txt)
        b = b.filter(lambda r : len(r) > 1) 
        def flatten(record):
            return {
                'person': record[0],
                'text': record[1],
            }
        df = b.map(flatten).to_dataframe().compute()
        return(df)
    
    def process(self):

        """ Interates over chunks to create a column for grouping
        """
        total = pd.DataFrame()
        for i, df_chunk in enumerate(self.data):
            new = df_chunk.copy()
            new['chunk'] = i
            total = pd.concat([total,new])
        self.data = total
        return total
    
    def get_grouped_data(self):
        return self.group_by_chunk(self.grouped_data)
    
    def get_chunked_data(self):
        return self.grouped_data 

    def get_original_data(self):
        return self.data

    def group_by_chunk(self,df):
        return df.groupby(['chunk'])['text'].apply(','.join).reset_index()

    def appify(self,language_model: Union[str, spacy.language.Language] = "en_core_web_sm"):
        data = self.get_grouped_data()
        prepared_df = prepare_text_df(data,language_model=language_model)
        #widget = ConcordanceWidget(prepared_df) #This works but try combine widget with pandas stlying
        widget = DataIngestWidget(prepared_df) #dev
        widget.show()
        self.app = widget
        return widget

    def get_current_match_and_row(self,speaker_column = "TEST"):
        #Gets current results from within widget
        #Note if the data has been chunked row is the chunked row
        try:
            results = self.app.search_table.to_dataframe()
            return results
        except:
            return None
    def reference_original_data(results):
        results.text_id
        results.match
        
class ConcordanceWidget:
    """
    Interactive widget for searching and displaying concordance
    results
    """

    def __init__(self, df: pd.DataFrame, results_per_page: int = 20):
        """
        Args:
            df: DataFrame containing texts, as returned by prepare_text_df()
            results_per_page: How many search results to show at a time
        """
        self.data = df
        self.results_per_page = results_per_page
        self.search_table = ConcordanceTable(df=self.data)

    def show(self):
        """
        Display the interactive widget
        """

        def display_results(page: int, **kwargs):
            if not kwargs["keyword"]:
                return None
            for attr, value in kwargs.items():
                setattr(self.search_table, attr, value)

            try:
                # This will either be results or a regex error message
                html = self.search_table._get_html(page=page)
                if self.search_table.is_regex_valid():
                    results = self.search_table._get_results()
                    # Need at least one page
                    n_pages = max(
                        self.search_table._get_total_pages(n_results=len(results)), 1
                    )
                    page_input.max = n_pages
            except NoResultsError:
                n_pages = 1

            display(ipywidgets.HTML(html))

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
            step=10,
            max = 500,
            description="Window size (characters):",
            style={"description_width": "initial"},
        )
        sort_input = ipywidgets.Dropdown(
            options=["text_id", "left_context", "right_context"],
            value="text_id",
            description="Sort by:",
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
                "sort": sort_input,
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
            description="Export to Excel",
            disabled=False,
            button_style="success",
            tooltip="The excel file will be saved to the same location as the "
            "notebook on the server. Use the Jupyter Lab sidebar to access "
            "and download it.",
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
            [
                keyword_input,
                checkboxes,
                number_inputs,
                sort_input,
                export_controls,
                output,
            ]
        )


class ConcordanceTable:
    """
    Search for matches in a text (using plain-text or regular expressions),
    and display them in a HTML table.

    The arguments given to the constructor set up the table with
    initial data and search settings. In order
    to allow interactive use, these attributes can be changed
    after initialization and the updated values will be used
    next time a search is triggered.

    Args:
        df: DataFrame of texts, as returned by prepare_text_df().
        keyword: Word, phrase or regular expression to search for.
        regex: Use regular expression matching?
        ignore_case: If False, searches are case-sensitive.
        whole_word: Only return matches where the search term matches
           whole words (with space or punctuation either side)
        results_per_page: Number of results to show at a time.
        window_width: Number of characters to show either side of a match.
        sort: Sort by 'text_id', 'left_context' or 'right_context'.
        historic_utterances: Positive Integer representing number of previous lines to give context.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        keyword: str = "",
        regex: bool = False,
        ignore_case: bool = True,
        whole_word: bool = False,
        results_per_page: int = 20,
        window_width: int = 50,
        sort: str = "text_id"
    ):
        self.df = df
        self.keyword = keyword
        self.regex = regex
        self.ignore_case = ignore_case
        self.whole_word = whole_word
        self.results_per_page = results_per_page
        self.window_width = window_width
        self.sort = sort
        self.element_id = "atap_" + str(uuid.uuid4())
        self.css = SEARCH_CSS_TEMPLATE.format(element_id=self.element_id)
        

    def _repr_mimebundle_(self, include, exclude):
        """
        Define _repr_mimebundle_() so the default behaviour in Jupyter
        notebooks is to view the HTML table.
        """
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
        """
        Test if the current keyword is a valid regular expression.
        """
        try:
            _ = self._get_search_regex()
            return True
        except re.error:
            return False

    def _get_results(self) -> pd.DataFrame:
        """
        Return a Series of matches, with text_id as the index. Each element
        is a match returned by keyword_in_context(), i.e. a left_context,
        match, right_context tuple. 
        """

        def _get_matches(doc):
            matches = keyword_in_context(
                doc, keyword=search_regex, window_width=self.window_width
            )
            return list(matches)

        search_regex = self._get_search_regex()
        search_results = self.df["spacy_doc"].apply(_get_matches)
        search_results = search_results.loc[search_results.map(len) > 0].explode()
        search_results.name = "match"

        if len(search_results) == 0:
            raise NoResultsError("No results found.")

        results_df = search_results.to_frame()
        # Use apply(pd.Series) to unpack nested lists into columns
        results_df[["left_context", "match", "right_context"]] = results_df[
            "match"
        ].apply(pd.Series)
        results_df.index.name = "text_id"
        results_df.reset_index(inplace=True)
        # Reorder columns
        results_df = results_df[["text_id", "left_context", "match", "right_context"]]
        # Sort
        if self.sort == "text_id":
            results_df = results_df.sort_values(by="text_id")
        elif self.sort in ("left_context", "right_context"):
            results_df = self.sort_by_context(results_df, context=self.sort)
        else:
            raise ValueError(
                f"Invalid sort value {self.sort}: should be 'text_id',"
                " 'left_context' or 'right_context'"
            )
        return results_df

    def _get_total_pages(self, n_results: int) -> int:
        return math.ceil(n_results / self.results_per_page)

    def _get_table_html(self, search_results: pd.DataFrame, n_total: int) -> str:
        table_rows = "\n".join(
            SEARCH_ROW_TEMPLATE.format(**row).replace(r"\n", "")
            for index, row in search_results.iterrows()
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
        try:
            results = self._get_results()
            n_total = len(results)
            results = results.iloc[start_index:end_index]
        except NoResultsError:
            return "No results found. Try a different search term"

        return self._get_table_html(results, n_total=n_total)

    def _get_html2(self, page: int = 1):
        """ pandas styling equivalent
        """
        regex_valid = self.is_regex_valid()
        if not regex_valid:
            try:
                self._get_search_regex()
            except re.error as error:
                return self._get_error_html(error)

        start_index = (page - 1) * self.results_per_page
        end_index = (page * self.results_per_page) - 1
        try:
            results = self._get_results()
            n_total = len(results)
            results = results.iloc[start_index:end_index]
        except NoResultsError:
            return "No results found. Try a different search term"

        cell_hover = {  'selector': 'td:hover','props': [('background-color', '#ffffb3')]}
        index_names = {'selector': '.index_name','props': 'font-style: italic; color: darkgrey; font-weight:normal;'}
        headers = {'selector': 'th:not(.index_name)','props': 'background-color: #000066; color: white;'}
        html = results.style.set_table_styles([cell_hover, index_names, headers]).to_html()
        return html

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a dataframe returning all the matches for the current
        keyword.
        """
        return self._get_results()

    def to_excel(self, filename: str, max_col_width: int = 100):
        """
        Export matches for the current keyword to Excel.

        Args:
            filename: Path to write the Excel file to.
            max_col_width: Maximum width of Excel columns. We try to automatically
                set the widths of columns to fit their content, but only
                up to this maximum.
        """

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
 
    @staticmethod
    def sort_by_context(results: pd.DataFrame, context: str = "left_context"):
        """
        Sort concordance results by either the left context or right
        context. For left context, this means sorting by the preceding
        word, then the word before that, etc.
        """

        def get_reversed_words(s: pd.Series):
            return s.str.strip().str.lower().str.split(r"\s").map(lambda x: x[::-1])

        if context == "left_context":
            return results.sort_values(by=context, key=get_reversed_words)
        elif context == "right_context":
            return results.sort_values(
                by=context, key=lambda x: x.str.strip().str.lower()
            )
        else:
            raise ValueError(
                "Invalid context option: should be 'left_context' or 'right_context'"
            )

class DataIngestWidget:
    """
    New widget to go with pandas stlying approach
    DOESNT WORK AT MOMENT
    """

    def __init__(self, df: pd.DataFrame, results_per_page: int = 20):
        """
        Args:
            df: DataFrame containing texts, as returned by prepare_text_df()
            results_per_page: How many search results to show at a time
        """
        self.data = df
        self.results_per_page = results_per_page
        self.search_table = ConcordanceTable(df=self.data)

    def show(self):
        """
        Display the interactive widget
        """

        def display_results(page: int, **kwargs):
            if not kwargs["keyword"]:
                return None
            for attr, value in kwargs.items():
                setattr(self.search_table, attr, value)

            try:
                # CHANGE SOMEWHERE HERE!!!!
                html = self.search_table._get_html2(page=page) #THIS IS HALF WRITE (attached to ConcordanceTable )
                if self.search_table.is_regex_valid():
                    results = self.search_table._get_results()
                    # Need at least one page
                    n_pages = max(
                        self.search_table._get_total_pages(n_results=len(results)), 1
                    )
                    page_input.max = n_pages
            except NoResultsError:
                n_pages = 1

            display(ipywidgets.HTML(html))
            return html

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
            step=10,
            max = 500,
            description="Window size (characters):",
            style={"description_width": "initial"},
        )
        sort_input = ipywidgets.Dropdown(
            options=["text_id", "left_context", "right_context"],
            value="text_id",
            description="Sort by:",
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
                "sort": sort_input,
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
            description="Export to Excel",
            disabled=False,
            button_style="success",
            tooltip="The excel file will be saved to the same location as the "
            "notebook on the server. Use the Jupyter Lab sidebar to access "
            "and download it.",
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
            [
                keyword_input,
                checkboxes,
                number_inputs,
                sort_input,
                export_controls,
                output,
            ]
        )