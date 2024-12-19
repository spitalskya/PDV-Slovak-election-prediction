import os
import re

import pandas as pd
from docling.document_converter import DocumentConverter
from tqdm import tqdm

# def main():
# parse the documents in the folder
converter = DocumentConverter()
pdfs = list(map(lambda x: "src/focus_scraping/FOCUS_pdf/" + x, os.listdir("src/focus_scraping/FOCUS_pdf")))
result = converter.convert(pdfs[101])
# for i in result.document.tables:
#     print(i.export_to_dataframe())
# exit()
result = converter.convert_all(pdfs)
# extract them to list
tables = []
for table in tqdm(result, total=len(pdfs)):
    try:
        temp = table.document.tables[1].export_to_dataframe()
    except IndexError:
        temp = table.document.tables[0].export_to_dataframe()

    # cutting only relevant columns and renaming
    cols = temp.columns
    temp.rename(
        {
            cols[0]: "political_party",
            cols[1]: "potentional_voters",
            cols[2]: "confidence_interval_95_perc",
        },
        axis="columns",
        inplace=True,
    )
    tables.append(
        temp[["political_party", "potentional_voters", "confidence_interval_95_perc"]]
    )

# adding year and month
for pdf, table in zip(pdfs, tables):
    matches = re.search(r"(\d{4})_([a-zA-Z]+)", pdf)
    if not matches:
        raise KeyError("Regex did not match!")

    year, month = matches.groups()

    table["year"] = [year] * len(table)
    table["month"] = [month] * len(table)

# put all the tables together
concatenated = pd.concat(tables).reset_index()

# # fix anomalies -> Madarska strana on multiple lines
# concatenated.iloc[1429]["political_party"] = (
#     concatenated.iloc[1429]["political_party"]
#     + f" {concatenated.iloc[1430]['political_party']}"
# )  # type: ignore
# concatenated.iloc[1165]["political_party"] = (
#     concatenated.iloc[1165]["political_party"]
#     + f" {concatenated.iloc[1166]['political_party']}"
# )  # type: ignore
# concatenated.drop([1430, 1166], inplace=True)

concatenated.to_csv("src/focus_scraping/pdf_data_extraction/focus.csv", index=False)


# if __name__ == "__main__":
#     main()
