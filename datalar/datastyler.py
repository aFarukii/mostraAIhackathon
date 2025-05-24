import pandas as pd

df=pd.read_csv('venue_analysis_results.csv')
df=df.drop(['Kahve_Puani','Veteriner_Puani'],axis=1)

df.to_csv('sondatalar/venue_analysis_result.csv')