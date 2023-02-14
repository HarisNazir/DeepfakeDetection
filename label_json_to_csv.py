import pandas as pd

for i in range(4):
    path_json = "data/metadata.json"
    path_csv = "data/metadata.csv"
    print(path_csv)
    print(path_json)
    read_json = pd.read_json(path_json)
    df = pd.DataFrame(read_json)
    df_2 = pd.DataFrame(df.transpose())
    df_2.to_csv(path_csv)
    read_csv = pd.read_csv(path_csv)
    read_csv.columns = ["URI","label","original","split"]
    read_csv.to_csv(path_csv,index=False)
    print(read_csv.head(5))

