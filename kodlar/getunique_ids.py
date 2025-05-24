import pyarrow.dataset as ds
import pandas as pd

parafilepath = "MobilityDataMay2024.paraquet"
dataset = ds.dataset(parafilepath, format="parquet")

scanner = dataset.scanner(batch_size=100_000)

unique_ids = set()

for record_batch in scanner.to_batches():
    df_chunk = record_batch.to_pandas()
    if "device_aid" in df_chunk.columns:
        unique_ids.update(df_chunk["device_aid"].dropna().unique())

unique_ids_list = sorted(unique_ids)
df_unique = pd.DataFrame(unique_ids_list, columns=["device_aid"])

df_unique.to_csv("unique_device_ids.csv", index=False)
print("CSV dosyası kaydedildi: unique_device_ids.csv")
