"""Module for formatting output data in HTML. FROM polars: https://github.com/pola-rs/polars/blob/main/py-polars/polars/dataframe/_html.py"""
from __future__ import annotations

import polars as pl
import os
from textwrap import dedent
from typing import TYPE_CHECKING, Iterable

import html
from itertools import groupby

if TYPE_CHECKING:
    from types import TracebackType

    from polars import DataFrame


class Tag:
    """Class for representing an HTML tag."""

    def __init__(
            self,
            elements: list[str],
            tag: str,
            attributes: dict[str, str] | None = None,
    ):
        self.tag = tag
        self.elements = elements
        self.attributes = attributes

    def __enter__(self) -> None:
        if self.attributes is not None:
            s = f"<{self.tag} "
            for k, v in self.attributes.items():
                s += f'{k}="{v}" '
            s = f"{s.rstrip()}>"
            self.elements.append(s)
        else:
            self.elements.append(f"<{self.tag}>")

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
    ) -> None:
        self.elements.append(f"</{self.tag}>")


# ----------------------------------------------------------------------

class HTMLFormatter:
    """Class for creating and styling HTML summaries in table format."""

    def __init__(
            self,
            header_data: pl.DataFrame,
            body_data: pl.DataFrame,
    ) -> None:
        self.elements: list[str] = []

        self.header_rows: list[tuple[str, ...]] = header_data.rows()
        self.body_rows: list[tuple[str, ...]] = body_data.rows()

    def write_header(self) -> None:

        with Tag(self.elements, 'thead'):

            for header in self.header_rows:
                with Tag(self.elements, 'tr'):
                    for content, group in groupby(header):
                        count = len(list(group))

                        attributes = {
                            'colspan': str(count)
                        } if count > 1 else {}

                        with Tag(self.elements, 'th', attributes):
                            self.elements.append(content)

    def write_body(self) -> None:
        for row in self.body_rows:
            with Tag(self.elements, 'tr'):
                for cell in row:
                    with Tag(self.elements, 'td'):
                        self.elements.append(cell)

    def write_style(self) -> None:
        ...

    def render(
            self,
            title: str = None,
            caption: str = None,
    ) -> list[str]:
        """Render the HTML table with optional title and caption."""

        with Tag(self.elements, "table"):
            if title:
                self.elements.insert(
                    0,
                    f"<caption style='caption-side: top; text-align: center;'>{title}</caption>"
                )

            self.write_header()
            with Tag(self.elements, 'tbody'):
                self.write_body()

            if caption:
                self.elements.append(
                    f"<caption style='caption-side: bottom; text-align: left;'>{caption}</caption>")

        # insert style at 0
        self.write_style()

        return self.elements


# class HTMLSummary(HTMLFormatter):
#     def __init__(self, header_data: pl.DataFrame, body_data: pl.DataFrame):
#         super().__init__(header_data=header_data, body_data=body_data)
#
#     def write_style(self):
#         """Generate and append the CSS style for the table."""
#         style = """
#         <style>
#             table {
#                 border-collapse: collapse;
#                 width: 100%;
#                 margin-bottom: 20px; /* Space for the caption at the bottom */
#             }
#             th, td {
#                 border: 1px solid #dddddd;
#                 text-align: center;
#                 padding: 8px;
#             }
#             th {
#                 background-color: #f2f2f2;
#                 border-bottom: 2px solid #000;
#             }
#             caption {
#                 padding: 8px;
#                 caption-side: bottom;
#                 text-align: left;
#                 font-weight: bold;
#             }
#             .caption-top {
#                 caption-side: top;
#                 text-align: center;
#             }
#         </style>
#         """
#         self.elements.insert(0, style)


class HTMLSummary(HTMLFormatter):

    def __init__(self,
                 header_data,
                 body_data
                 ):

        super().__init__(header_data, body_data)

    def write_body(self):

        for i, row in enumerate(self.body_rows):

            with Tag(self.elements, 'tr'):
                with Tag(self.elements, 'td', {"class": 'coef'}):
                    self.elements.append(row[0])

                for cell in row[1:]:
                    with Tag(self.elements, 'td'):
                        self.elements.append(cell)

    def write_style(self):

        style = """
            table {
              border-collapse: separate;
                  border-spacing: 2px;
            }
        
            td {
              border: 1px solid transparent;
              padding: 2px; 
              text-align: center;
            }
        
            th {
              border: 1px solid #dddddd;
              padding: 8px;
              background-color: #f2f2f2;
              border-bottom: 2px solid #000; 
            }

  th:first-child {
    border-bottom: none; 
  }
  
thead {
  border-bottom: 5px solid #000;
}


            .coef {
              text-align: left; 
            }
            """

        self.elements.insert(0, f"<style>{style}</style>")


# Define headers and body rows for the table
headers = [
    ("DEP VAR", "y", "y", "z"),
    ("GROUP", "A", "B", "B"),
    ("", "(1)", "(2)", "(3)")
]
headers = pl.DataFrame(list(zip(*headers)))

body = [
    ("coef 1", "1", "2", "3"),
    ("", "(123)", "(123)", "(123)"),
    ("coef 2", "a", "b", "c"),
    ("", "(123)", "(123)", "(123)"),

]
body = pl.DataFrame(list(zip(*body)))

# Usage example
summary = HTMLSummary(headers, body)

html = " ".join(summary.render())

from pathlib import Path

# Step 1: Define the path to the HTML file
html_file_path = Path('/Users/bernardodionisi/Desktop/check.html')
html_file_path.write_text(html)

# # ----------------------------------------------------------------------
# class HTMLFormatter:
#     def __init__(
#             self,
#             df: DataFrame,
#             *,
#             max_cols: int = 75,
#             max_rows: int = 40,
#             from_series: bool = False,
#     ):
#         self.df = df
#         self.elements: list[str] = []
#         self.max_cols = max_cols
#         self.max_rows = max_rows
#         self.series = from_series
#         self.row_idx: Iterable[int]
#         self.col_idx: Iterable[int]
#
#         if max_rows < df.height:
#             self.row_idx = [
#                 *list(range(max_rows // 2)),
#                 -1,
#                 *list(range(df.height - max_rows // 2, df.height)),
#             ]
#         else:
#             self.row_idx = range(df.height)
#         if max_cols < df.width:
#             self.col_idx = [
#                 *list(range(max_cols // 2)),
#                 -1,
#                 *list(range(df.width - max_cols // 2, df.width)),
#             ]
#         else:
#             self.col_idx = range(df.width)
#
#     def write_header(self) -> None:
#         """Write the header of an HTML table."""
#         with Tag(self.elements, "thead"):
#             if not bool(int(os.environ.get("POLARS_FMT_TABLE_HIDE_COLUMN_NAMES", "0"))):
#                 with Tag(self.elements, "tr"):
#                     columns = self.df.columns
#                     for c in self.col_idx:
#                         with Tag(self.elements, "th"):
#                             if c == -1:
#                                 self.elements.append("&hellip;")
#                             else:
#                                 self.elements.append(html.escape(columns[c]))
#             if not bool(
#                     int(os.environ.get("POLARS_FMT_TABLE_HIDE_COLUMN_DATA_TYPES", "0"))
#             ):
#                 with Tag(self.elements, "tr"):
#                     dtypes = self.df._df.dtype_strings()
#                     for c in self.col_idx:
#                         with Tag(self.elements, "td"):
#                             if c == -1:
#                                 self.elements.append("&hellip;")
#                             else:
#                                 self.elements.append(dtypes[c])
#
#     def write_body(self) -> None:
#         """Write the body of an HTML table."""
#         str_lengths = int(os.environ.get("POLARS_FMT_STR_LEN", "15"))
#         with Tag(self.elements, "tbody"):
#             for r in self.row_idx:
#                 with Tag(self.elements, "tr"):
#                     for c in self.col_idx:
#                         with Tag(self.elements, "td"):
#                             if r == -1 or c == -1:
#                                 self.elements.append("&hellip;")
#                             else:
#                                 series = self.df[:, c]
#                                 self.elements.append(
#                                     html.escape(series._s.get_fmt(r, str_lengths))
#                                 )
#
#     def write(self, inner: str) -> None:
#         """Append a raw string to the inner HTML."""
#         self.elements.append(inner)
#
#     def render(self) -> list[str]:
#         """Return the lines needed to render a HTML table."""
#         if not bool(
#                 int(
#                     os.environ.get("POLARS_FMT_TABLE_HIDE_DATAFRAME_SHAPE_INFORMATION", "0")
#                 )
#         ):
#             # format frame/series shape with '_' thousand-separators
#             s = self.df.shape
#             shape = f"({s[0]:_},)" if self.series else f"({s[0]:_}, {s[1]:_})"
#
#             self.elements.append(f"<small>shape: {shape}</small>")
#
#         with Tag(
#                 # be careful changing the CSS class ref here...
#                 # ref: https://github.com/pola-rs/polars/issues/7443
#                 self.elements,
#                 "table",
#                 {"border": "1", "class": "dataframe"},
#         ):
#             self.write_header()
#             self.write_body()
#         return self.elements
#
#
# class NotebookFormatter(HTMLFormatter):
#     """
#     Class for formatting output data in HTML for display in Jupyter Notebooks.
#
#     This class is intended for functionality specific to DataFrame._repr_html_().
#
#     """
#
#     def write_style(self) -> None:
#         style = """\
#             <style>
#             .dataframe > thead > tr > th,
#             .dataframe > tbody > tr > td {
#               text-align: right;
#               white-space: pre-wrap;
#             }
#             </style>
#         """
#         self.write(dedent(style))
#
#     def render(self) -> list[str]:
#         """Return the lines needed to render a HTML table."""
#         with Tag(self.elements, "div"):
#             self.write_style()
#             super().render()
#         return self.elements


"""

<style>
            table {
              border-collapse: separate;
              border-spacing: 2px;
/*              width: 200;*/
            }
        
        tbody td {
  padding: 0px; /* smaller padding */
               text-align: center;
                                    font-size: 10px;
}

/*            td {
              border: 0px solid transparent;
              text-align: center;
                                    font-size: 10px;

            }
        */
              th:first-child {
                      font-size: 5px;
                                    text-align: left; 

            }


            th:not(:first-child) {
/*              border: 1px solid #dddddd;*/
              padding: 4px;
                                    font-size: 11px;

/*              background-color: #f2f2f2;*/
              border-bottom: 1px solid #000; 
            }



thead tr.last-row th {
  border-bottom: none;
}


.divider {
  border-top: 1px solid #000;
}
            .coef {
              text-align: left; 
            }
            </style> <table> <thead > <tr> <tr> <th> DEP VAR </th> <th colspan="2"> y </th> <th> z </th> </tr> <tr> <th> GROUP </th> <th> Anna </th> <th colspan="2"> Bernardo </th> </tr> <tr class="last-row"> <th>  </th> <th> (1) </th> <th> (2) </th> <th> (3) </th> </tr> 

            </tr>

             </thead> 

              <td colspan="100%"><div class="divider"></div></td>
<tbody >

             <tr colspan="100%" style="background-color: red;"> <td class="coef"> coef 1 </td> <td> 1 </td> <td> 2 </td> <td> 3 </td> </tr> <tr> <td class="coef">  </td> <td> (123) </td> <td> (123) </td> <td> (123) </td> </tr> <tr> <td class="coef"> coef 2 </td> <td> a </td> <td> b </td> <td> c </td> </tr> <tr> <td class="coef">  </td> <td> (123) </td> <td> (123) </td> <td> (123) </td> </tr> 

           </thead> </table>
"""
