import pandas as pd

df = pd.read_excel(r'C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\projects\NarcCataplexyQuestionnaire\data\raw\data for paper - annotated_giocrm.xlsx')

df_pp = pd.read_excel(r'C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\projects\NarcCataplexyQuestionnaire\data\pproc\SSQDX_pp_manually_annotated.xlsx')


df['included'] = 1  # Everyone included by default

# Filter preprocessed frame to those you're excluding
filter_pp = (df_pp['source'] == 'ssqdx') & (df_pp['cataplexy_clear_cut'] == 2)
df_pp = df_pp.loc[filter_pp, :]

# Use the columns that are in both dataframes to merge/match
common_columns = list(df_pp.columns.intersection(df.columns))

# Make sure there is at least one common column
if len(common_columns) == 0:
    raise ValueError("No common columns to merge on.")

for col in common_columns:
    if col in df.columns and col in df_pp.columns:
        df[col] = df[col].astype(str)
        df_pp[col] = df_pp[col].astype(str)

# Find rows in df that match those in df_pp (on common columns)
matching_rows = df.merge(df_pp[common_columns], how='inner', on=common_columns)

# Set 'included = 0' in df for those matched rows
df.loc[df.index.isin(matching_rows.index), 'included'] = 0