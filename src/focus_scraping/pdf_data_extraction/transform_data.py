import pandas as pd
from tqdm import tqdm

df = pd.read_csv("src/focus_scraping/pdf_data_extraction/focus_to_election.csv")
df["date"] = pd.to_datetime(df["date"])
df[~df.isin(["Centrum - periféria", "Mesto - vidiek"]).any(axis=1)]
df.replace(r".*-.*", "", inplace=True, regex=True)


cols = df["date"].unique()
cols = cols[cols.argsort()]


table = []
for i in tqdm(df["political_party"].unique()):
    values = [i]
    for j in cols:
        value = df.loc[(df["political_party"] == i) & (df["date"] == j)][
            "potentional_voters"
        ]
        try:
            values.append(
                float(value.iloc[0][: value.iloc[0].find("%")].replace(",", "."))
            )
        except Exception:
            values.append(float(0))
    table.append(values)

cols = ["political_party"] + list(cols.strftime("%Y-%m-%d"))
data = pd.DataFrame(table, columns=cols)

data.to_csv("src/focus_scraping/pdf_data_extraction/focus_final.csv")
