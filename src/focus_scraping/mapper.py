import pandas as pd
import json

excel_path = "data\\raw\\election\\2023\\Volby 2023.xlsx"

df = pd.read_excel(excel_path)

df. columns = ["Polotický subjekt","Počet platných hlasov", "Podiel platných hlasov    v %"]

json_path = "src\\focus_scraping\\pdf_data_extraction\\mapper.json"

with open(json_path, 'r', encoding='utf-8') as file:
    party_mapping = json.load(file)

df["Polotický subjekt"] = df["Polotický subjekt"].map(party_mapping).fillna("unmapped")

output_path = "data\\raw\\election\\2023\\Volby 2023_Remapped.xlsx"
df.to_excel(output_path,index=False)