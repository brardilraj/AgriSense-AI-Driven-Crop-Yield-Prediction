import pandas as pd

yield_df = pd.read_csv("data/Tamilnadu agriculture yield data.csv")
rain_df = pd.read_csv("data/rainfall_data.csv")

print(yield_df.head())
print(rain_df.head())
print(yield_df.columns)
print(rain_df.columns)
