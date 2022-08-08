# merge annotations
import pandas as pd
import itertools
from config import *
sets = [
    "800",
    "800_1600",
    "1600_2400",
    "2400_3200"
]
annotators=[
    "lars",
    "linusb",
    "ls",
    "luke"
]
files = [f"{s}_{ann}.csv" for s, ann in itertools.product(sets, annotators)]
#print(files)
def read_csv(ann, s):
  df = pd.read_csv(f"{DATA_DIR}/annotated/{s}_{ann}.csv")
  return df.rename(columns={"Unnamed: 0": "id"})
dfs = { ann: { s: read_csv(ann, s) for s in sets} for ann in annotators }
#dfs[annotators[0]]
#print(dfs)
#list(map(lambda a: list(map(lambda df: len(df), a.values())), dfs.values()))
#list(map(lambda a: list(map(lambda df: df.reset_index().columns, a.values())), dfs.values()))
#list(map(lambda a: list(a.keys()), dfs.values()))
list(map(lambda a: list(map(lambda df: len(df), a.values())), dfs.values()))
#pd.set_option('display.max_rows', None)
# concat vertically
catted = [pd.concat(set_dfs.values(), ignore_index=True) for set_dfs in dfs.values()]
print(catted[3]["humor"][60])
# merge horizontally (columns)
merged = catted[0]
merged['humor'] += catted[1]['humor'] + catted[2]['humor'] + catted[3]['humor']
merged['humor'] = (merged['humor'] - len(catted))/len(catted)
print(min(merged["humor"]))
print(max(merged["humor"]))

#merged.to_csv(f"{DATA_DIR}/humor/quirky_annotated.csv")
#top = merged.sort_values(by="humor", ascending=False).loc[:, ["humor", "title"]]
#top.to_csv(f"{DATA_DIR}/humor/top_quirky.csv")
#top

import matplotlib.pyplot as plt
plt.hist(merged["humor"], bins=8)