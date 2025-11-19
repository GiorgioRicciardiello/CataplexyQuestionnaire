"""
Two data sources are used the pre-processing. The sources must be merged into a single datataset.

1. Pre-process the SSQDX dataset
2. Pre-process the SSQ dataset
3. Merge the two datasets

This updated scripts uses the created ICSD criteria for NT1 and NT2 directy from the dataset


Columns must be homogenized
"""
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import pandas as pd
from config.config import config
import ast
from config.SSI_Digitial_Questionnaire import key_mapping
import re
from typing import List, Dict, Union, Any, Tuple, Optional
from sklearn.experimental import enable_iterative_imputer  # needed to use IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


def visualize_table(df: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
    """
    Count the unique pair combinations in a dataframe within the grouped by columns.
    :param df: Input DataFrame
    :param group_by: List of column names to group by
    :return: DataFrame showing counts of unique combinations
    """
    df_copy = df.copy()
    print("Distribution before modification:")

    # Only fill NaN with 'NaN' in object (string) columns to avoid dtype issues
    df_plot_before = df_copy.copy()
    for col in df_plot_before.select_dtypes(include='object'):
        df_plot_before[col] = df_plot_before[col].fillna('NaN')

    grouped_counts_before = df_plot_before.groupby(group_by).size().reset_index(name='Counts')

    print(tabulate(grouped_counts_before, headers='keys', tablefmt='grid'))
    print(f'Remaining Rows: {df_copy.shape[0]}')
    return grouped_counts_before

#%% Functions for the okun dataset
def pre_process_okun_dataset(dataset_path:pathlib.Path) -> pd.DataFrame:
    """
    Pre-process of the SSQDX dataset.
    :param dataset_path:
    :return:
    """

    df_okun = pd.read_excel(dataset_path)
    tab = visualize_table(df=df_okun,
                          group_by=['DQB1*0602', 'NT1 ICSD3 - TR'])

    # %% Work on the dx dataset
    df_okun.replace(to_replace='.', value=np.nan, inplace=True)
    df_okun.rename(columns={'# NAPS': 'NAPS',
                             'DQB1*0602': 'DQB10602'}, inplace=True)
    df_okun['source'] = 'okun'
    df_okun['sex'] = df_okun['sex'].map({'F': 0, 'M': 1})
    df_okun['BMI'] = df_okun['BMI'].round(1)

    # %% rename columns
    # SSI_Digital_Questionnaire, these columns are of muscle weakness that we do not use
    # eg., dreaming, sleeping, bladder controsl, fainting, fall prevention,
    mapper_q = [
        'Q:86',
        'Q:87',
        'Q:88',
        'Q:89',
        'Q:90',
        'Q:91',
        'Q:92',
        'Q:93',
        'Q:94',
        'Q:101',
        'Q:113',
        'Q:137',
        'Q:140'
    ]
    df_okun.drop(columns=mapper_q, inplace=True)

    df_okun['NT1 ICSD3 - TR'] = df_okun['NT1 ICSD3 - TR'].replace({'not meeting criteria': 'not meet criteria'})
    df_okun['NT2 ICSD3 - TR'] = df_okun['NT2 ICSD3 - TR'].replace({'not meeting criteria': 'not meet criteria'})

    tab = visualize_table(df=df_okun,
                          group_by=['source', 'DQB10602', 'NT1 ICSD3 - TR', 'NT2 ICSD3 - TR'])
    tab.to_csv('tab_okun_target_raw.csv', index=False)

    df_okun = df_okun.drop(columns=[
        'Referral',
        'Recruit',
        'Dx3',
        'Dx',
        # 'MSLT (<=8)',
        # 'SOREMP (>=2)',
        # 'MSLTAGE',
        # 'SE',
        # 'REMLAT (< 15)',
    ])
    df_okun.rename(mapper={

    })
    # df_okun.columns = map(str.lower, df_okun.columns)
    df_okun = df_okun.rename(columns={'SEX2': 'SEX',
                                      'MSLT (<=8)': 'MSLT',
                                      'SOREMP (>=2)': 'SOREMP',
                                      'MSLTAGE': 'MSLTAGE',
                                      'SE': 'SE',
                                      'REMLAT (< 15)': 'REMLAT'}
                             )

    df_okun[target] = df_okun[target].replace({'no data': np.nan,
                                                'control': 0,
                                               '1': 1,
                                               '0': 0})

    df_okun[target_nt2] = df_okun[target_nt2].replace({'no data': np.nan,
                                                'control': 0,
                                               '1': 1,
                                               '0': 0})

    return df_okun

def pre_process_anic_dataset(dataset_path: pathlib.Path) -> pd.DataFrame:
    """

    :param dataset_path:
    :return:
    """

    def expand_series(series: pd.Series, col_name: str) -> pd.DataFrame:
        """
        Expand the multiple response answers where a single cell in the format 111101 is parsed as a dataframe e.g.,
        ['1', '1', '1', '1', '0', '1']. For all the rows.

        Row order is preserve. A sparse dataframe is create
        :param series:
        :param col_name:
        :return:  sparse dataframe
        """
        df_exp_sparse = pd.DataFrame(series.apply(lambda x: list(x)).tolist())
        # mirror the index order
        df_exp_sparse.index = series.index
        # set the columns
        df_exp_sparse.columns = [f'{col_name}_exp_{i}' for i in range(1, df_exp_sparse.shape[1] + 1)]
        return df_exp_sparse

    def is_integer(string):
        try:
            int(string)
            return True
        except ValueError:
            return False

    def remove_non_numeric(value):
        if isinstance(value, str):
            return re.sub(r'\D', '', value)
        return value

    def string_to_numeric(value):
        if pd.isna(value):
            return value
        if isinstance(value, str):
            # Remove non-numeric characters
            numeric_string = re.sub(r'\D', '', value)
            try:
                # Convert to numeric value using literal_eval
                return ast.literal_eval(numeric_string)
            except (ValueError, SyntaxError):
                # If conversion fails, return the numeric string
                return numeric_string
        return value

    def contains_string_or_datetime(series):
        """Function to check if a column contains any string or datetime values"""
        return series.apply(lambda x: isinstance(x, (str, pd.Timestamp))).any()

    df_anic = pd.read_excel(dataset_path)

    #  rename columns and drop unwanted
    df_anic.drop(columns=['Name'], inplace=True)
    df_anic.rename(columns={"Pt's Last Name": 'name_last',
                           'Full name (Last, First)': 'name',
                           "Pt's First Name": 'name_first',
                           "DOB": "date_of_birth",
                           "AGE": "age",
                           "Gender": "gender",
                           "PLACE OF BIRTH": "place_birth",
                           "ETHNIC": "ethnicity",
                           "Completed (date)": "completed",
                           "A. Clear-Cut Cataplexy": "cataplexy_clear_cut",
                           "B. Possibly": "possibility",
                           "C. Narcolepsy": "narcolepsy",
                           "D. Other Sleep Disorder": "d_other_sleep_disorder",
                           "D1. Name of Other Disorder": "d_one",
                           "D2. Name of Other Disorder": "d_two",
                           "D3. Name of Other Disorder": "d_three",
                           },
                  inplace=True)
    #  PHI formatting
    df_anic['name'] = df_anic['name_first'] + ' ' + df_anic['name_last']
    df_anic = df_anic.drop(columns=['name_first', 'name_last'])
    df_anic['gender'] = df_anic['gender'].replace({'M': 1, 'F': 0})
    df_anic['gender'] = df_anic['gender'].astype(int)
    df_anic['place_birth'] = df_anic['place_birth'].str.lstrip()  # .replace(' ', '')
    df_anic['place_birth'] = df_anic['place_birth'].str.strip()  # .replace(' ', '')
    # %% errors
    df_anic.replace('????', np.nan, inplace=True)
    df_anic['97'] = df_anic['97'].replace({'19-30': 19.5})
    df_anic['95'] = df_anic['95'].replace({'18-25': 18.15})
    df_anic['46'] = df_anic['46'].replace({'late30s-ear40s': 35,
                        '16-26': 23})
    df_anic.loc[df_anic['65a'] == '2,1', '65a'] = 1
    df_anic['65a'] = df_anic['65a'].astype(int)
    # df_anic['83a'].astype(int)
    index_to_drop = df_anic.loc[(df_anic['54b'] == '0 ?') | (df_anic['54b'] == 9)].index
    df_anic.drop(index=index_to_drop, inplace=True)
    # there is an extra zero in:
    df_anic.loc[956, '60b'] = '000009'  # before was 0000090'. Results extra column full of None in the next code block
    df_anic.loc[740, '64b'] = '900000'  # before " '900000"
    df_anic.loc[933, '72b'] = '011000'  # before '0110000'

    # sparse the dataset - Emotions and muscle weakness
    # Note: The new columns indexes make reference to the dictionary mw_experiences in the config folder
    # Select columns that match the pattern using regular expression
    pattern = r'^\d+[ab]$'
    col_emot_mw = df_anic.filter(regex=pattern).columns
    col_emot_mw_multi_response = [col for col in col_emot_mw if
                                  df_anic[col].apply(lambda x: isinstance(x, str)).any()]
    col_emot_mw_multi_response = [col for col in col_emot_mw_multi_response if col.endswith('b')]
    col_emot_mw_multi_response = [col for col in col_emot_mw_multi_response if
                                  ast.literal_eval(col.split('b')[0]) < 80]
    # remove the ' symbol that not all cells have
    df_anic[col_emot_mw_multi_response] = df_anic[col_emot_mw_multi_response].replace("'", '', regex=True)
    # all same format
    df_anic[col_emot_mw_multi_response] = df_anic[col_emot_mw_multi_response].astype(str)
    # expand the cells and insert the slice of the new frame
    for col in col_emot_mw_multi_response:
        df_tmp = expand_series(series=df_anic[col], col_name=col)
        column_index = df_anic.columns.get_loc(col)
        # squezze in the middle the new columns
        df_anic = pd.concat([df_anic.iloc[:, :column_index], df_tmp, df_anic.iloc[:, column_index + 1:]], axis=1)
    # set as integer the leading columns that indicate that yes/ no response
    pattern = r'^\d+[a]$'
    col_emot_mw_yn = df_anic.filter(regex=pattern).columns
    col_emot_mw_yn = [col for col in col_emot_mw_yn if ast.literal_eval(col.split('a')[0]) < 80]
    df_anic[col_emot_mw_yn] = df_anic[col_emot_mw_yn].astype(int)

    # # re-order the columns
    # columns = ['name'] + [col for col in df_anic.columns if col != 'name']
    # df_anic = df_anic[columns]
    # df_anic.reset_index(drop=True, inplace=True)

    # data type formatting
    col_unwanted = ['102a', '102b', '103a', '103b',
                    '103c', '103d', '103e', '103f',
                    '103g', '103h', '103i', '103j']
    df_anic.drop(columns=col_unwanted, inplace=True)

    col_keep_format = ["PtKey",
                        "name",
                        "to DB file order",
                        "DbID",
                        "ID1",
                        "Alternate Name",
                        "GWAS ID",
                        "date_of_birth",
                        "age",
                        "gender",
                        "place_birth",
                        "bmi",
                        "ethnicity",
                        "completed",
                        "CSF Hcrt concentration crude (1)",
                        "Dx (1)",
                        "MSLT MSL (1)",
                        "MSLT #SOREM (1)",
                        "DQB1*0602",
                        "DQB1*0602 Typing Date",
                        "NT2 ICSD3 - TR",
                        "NT1 ICSD3 - TR",
                        "cataplexy_clear_cut",
                        "possibility",
                        "C. Narcolepsy - NOT RELIABLE",
                        "d_other_sleep_disorder",
                        "d_one",
                        "d_two",
                        "d_three",]

    for col in df_anic.columns:
        if col in col_keep_format:
            continue
        # print(col)
        df_anic[col] = df_anic[col].apply(remove_non_numeric)
    df_anic.replace('', np.nan, inplace=True)

    for col in df_anic.columns:
        if col in col_keep_format:
            continue
        # print(col)
        df_anic[col] = df_anic[col].apply(string_to_numeric)
    df_anic = df_anic.round(2)

    # convert as integers the columns
    # Get list of columns that do not contain any NaN values and do not have any string or datetime values
    columns_without_nan = [col for col in df_anic.columns if
                           not df_anic[col].isna().any() and not contains_string_or_datetime(df_anic[col])]

    df_anic[columns_without_nan] = df_anic[columns_without_nan].astype(int)

    # rename all columns as strings
    pattern = r'^\d.*[a-zA-Z]$'  # string starts with a number and ends with any letter
    compiled_pattern = re.compile(pattern)
    columns = [col if isinstance(col, str) else str(col) for col in df_anic.columns]
    df_anic.columns = columns

    for col in df_anic.columns:
        if is_integer(col):
            col_int = int(col)
            if col_int in key_mapping.keys():
                df_anic.rename(columns={col: f'{col}_{key_mapping[col_int]}'}, inplace=True)
        if isinstance(col, str) and col in df_anic.columns:
            # check if the string starts with a number and ends with any letter
            if compiled_pattern.match(col):
                num_col = col[0:2]
                ramification = col[2]
                columns_starting_with_num_col = [col for col in df_anic.columns if str(col).startswith(num_col)]
                new_col_pattern = {col: f'{col}_{key_mapping[int(num_col)].replace("_", "-")}' for col in
                                   columns_starting_with_num_col}
                df_anic.rename(mapper=new_col_pattern, inplace=True, axis=1)

    # Convert the 'epworth' columns to integers, ignoring NaNs
    ess_columns = df_anic.columns.str.contains('epworth')
    df_anic.loc[:, ess_columns] = df_anic.loc[:, ess_columns].astype(float)
    df_anic.loc[:, ess_columns] = df_anic.loc[:, ess_columns].replace([9, 8], np.nan)

    ess_columns = df_anic.columns[df_anic.columns.str.contains('epworth')].tolist()
    # remove the wrongly labeled epworth question
    ess_columns = [col for col in ess_columns if col != '12_epworth_daytime_sleepiness']
    print("ESS columns:", ess_columns)

    val_max_dict = {}
    for col in ess_columns:
        val_max_dict[col] = df_anic[col].max()

    for val in ess_columns:
        print(f'{val}: {df_anic[val].value_counts().to_dict()}')
        print(f'\t\t max:{max(df_anic[val].value_counts().to_dict().keys())}')

    # ignore numbers that are used to mark missinges
    df_anic['epworth_score'] = df_anic.loc[:, ess_columns].sum(skipna=True, axis=1)

    print(f'Distribution ESS Score: \n {df_anic["epworth_score"].describe()}')

    df_anic.columns = map(str.lower, df_anic.columns)
    df_anic['bmi'] = pd.to_numeric(df_anic['bmi'], errors='coerce').round(1)
    df_anic['age'] = pd.to_numeric(df_anic['age'], errors='coerce').round(0)

    # drop unnamed columns
    df_anic = df_anic.drop(columns=df_anic.filter(regex='^unnamed:').columns)
    df_anic['source'] =  'anic'
    col_first = [
        'source',
        'ptkey',
        'gwas id',
        'name',
        'age',
        'gender',
        'date_of_birth',
        'bmi',
        'place_birth',
        'csf hcrt concentration crude (1)',
        'ethnicity',
        # 'completed',
        'dqb1*0602',
        # 'dqb1*0602 typing date',
        # 'cataplexy_clear_cut',
        # 'possibility',
        # 'narcolepsy',
        'nt1 icsd3 - tr',
        'nt2 icsd3 - tr'
    ]
    col_rest = [col for col in df_anic.columns if col not in col_first]
    cols = col_first + col_rest
    # assert len(cols) == df_anic.shape[1]
    df_anic = df_anic[cols]
    df_anic.reset_index(drop=True, inplace=True)
    df_anic = df_anic.drop(columns=[
        # 'csf hcrt concentration crude (1)',
        'dx (1)',
        # 'mslt #sorem (1)',  # sorem (1)
        'cataplexy_clear_cut',
        'possibility',
        'c. narcolepsy - not reliable'
    ])

    df_anic.rename(columns={
        'dqb1*0602': 'DQB10602',
        'nt1 icsd3 - tr': 'NT1 ICSD3 - TR',
        'nt2 icsd3 - tr': 'NT2 ICSD3 - TR',
        'mslt msl (1)': 'MSLT',
        'mslt #sorem (1)': 'SOREMP',
        # 'dqb1*0602 typing date': 'DQB10602 date',
    }, inplace=True)

    df_anic[target] = df_anic[target].replace({'no data': np.nan,
                                                'control': 0,
                                               '1': 1,
                                               '0': 0})

    df_anic[target_nt2] = df_anic[target_nt2].replace({'no data': np.nan,
                                                'control': 0,
                                               '1': 1,
                                               '0': 0})
    # SOREMP reported as 4 of 5, 2 of 3, ... parsed to the fist integer
    # df_anic['SOREMP'] = df_anic['SOREMP'].str.extract(r'(\d+)').astype(float)

    df_anic = df_anic.loc[~df_anic[target].isna(), :]

    tab = visualize_table(df=df_anic,
                          group_by=['source', 'DQB10602', 'NT1 ICSD3 - TR', 'NT2 ICSD3 - TR'])
    tab.to_csv('tab_df_anic_target_raw.csv', index=False)


    # check the unique values per columns
    # for col in df_anic.columns:
    #     print(col)
    #     if df_anic[col].nunique() > 10 or col == 'ethnicity':
    #         continue
    #     if col in ['NT2 ICSD3 - TR', 'NT1 ICSD3 - TR', 'mslt']:
    #         continue
    #     value_counts = df_anic[col].value_counts().to_dict()
    #     value_counts = dict(sorted(value_counts.items()))  # Sort the dictionary by key
    #     print(f'{col}: \n\t{value_counts}')

    return df_anic


# %% Functions for merging

def plot_histograms(df1, col1, df2, col2, title):
    """
    Plots histograms of two different columns from two different DataFrames side by side using Seaborn.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    col1 (str): The column name from the first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.
    col2 (str): The column name from the second DataFrame.
    title (str): The title of the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(title, fontsize=16)

    # Plot histogram for the first DataFrame
    sns.histplot(df1[col1].dropna(), kde=True, ax=axes[0])
    axes[0].grid(0.7)

    # Plot histogram for the second DataFrame
    sns.histplot(df2[col2].dropna(), kde=True, ax=axes[1])
    axes[1].grid(0.7)
    plt.tight_layout()
    plt.show()


def make_new_col(row, columns_to_check: list[str]) -> int:
    return 1 if any(row[col] == 1 for col in columns_to_check) else 0

def quest76map(row):
    if pd.isna(row):
        return 0
    if 3 >= row > 0:
        return 1
    else:
        return row

def create_sleep_complaint(row) -> float:
    subset = row[['44_age_aware_sleepiness',
                  '45_sleepiness_severity_since_age',
                  '46_most_severe_sleepiness_age']]
    filtered_subset = subset[(subset >= 9) & (subset <= 99)]
    if filtered_subset.shape[0] == 1:
        return filtered_subset.iloc[0]
    elif filtered_subset.shape[0] > 1:
        return filtered_subset.mean().round(0)
    else:
        return np.nan

def set_to_zero_except_one(x):
    return 1 if x == 1 else 0

def wrangle_target_combinations(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame], Union[pd.DataFrame, Any]]:
    """
    Cleans and processes a DataFrame with narcolepsy, cataplexy, and DQB1*06:02 data to create target combinations
    and ensure distribution alignment.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing columns 'narcolepsy', 'cataplexy_clear_cut', 'DQB10602'.

    Returns:
    - tuple: Verification results, processed DataFrame.
    """

    def visualize_table(df: pd.DataFrame,
                        group_by: List[str]) -> pd.DataFrame:
        """
        Count the unique pair combinations ina dataframe within the grouped by columns
        :param df:
        :param group_by:
        :return:
        """
        df_copy = df.copy()
        print("Distribution before modification:")
        df_plot_before = df_copy.fillna('NaN')
        grouped_counts_before = df_plot_before.groupby(group_by).size().reset_index(
            name='Counts')
        print(tabulate(grouped_counts_before, headers='keys', tablefmt='grid'))
        print(f'Remaining Rows: {df_copy.shape[0]}')
        return grouped_counts_before

    def verify_dqb_distribution(df: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        """
        Verifies if the distribution of DQB1*06:02 in narcolepsy, NT1, and non-NT1 cases aligns with expected standards:
        - 98% of NT1 cases (cataplexy_clear_cut = 1) should be DQB1*06:02 positive.
        - 23% of non-NT1 cases (cataplexy_clear_cut = 0) should be DQB1*06:02 positive and 77% negative.
        - Narcolepsy group as a whole should show DQB1*06:02 positivity aligned with NT1 and non-NT1 distributions.

        Parameters:
        - df (pd.DataFrame): DataFrame with columns 'narcolepsy', 'cataplexy_clear_cut', and 'DQB10602'.

        Returns:
        - Tuple[bool, pd.DataFrame]: A tuple containing a boolean indicating if the verification passed,
                                     and a DataFrame with the results and discrepancies.
        """
        # Results dictionary with structured keys
        results = {
            'NT1 & Narc Cases (DQB1*06:02+)': {
                'Expected_Percentage': 98,
                'Actual_Percentage': None,
                'Meets_Requirement': None
            },
            'Non-NT1 & Narc Cases (DQB1*06:02+)': {
                'Expected_Percentage': 23,
                'Actual_Percentage': None,
                'Meets_Requirement': None
            },
            'Non-NT1 & Narc Cases (DQB1*06:02-)': {
                'Expected_Percentage': 77,
                'Actual_Percentage': None,
                'Meets_Requirement': None
            },
            'Overall Narcolepsy Group (DQB1*06:02+)': {
                'Expected_Percentage': None,
                'Actual_Percentage': None,
                'Meets_Requirement': None
            }
        }

        # ---- NT1 Case Verification ----
        nt1_narc_pos_cases = df.loc[(df['cataplexy_clear_cut'] == 1) & (df['narcolepsy'] == 'narcolepsy')]
        dqb_positive_nt1_narc = nt1_narc_pos_cases['DQB10602'].value_counts(normalize=True).get(1, 0) * 100
        results['NT1 & Narc Cases (DQB1*06:02+)']['Actual_Percentage'] = dqb_positive_nt1_narc
        results['NT1 & Narc Cases (DQB1*06:02+)']['Meets_Requirement'] = dqb_positive_nt1_narc >= 98

        # ---- Non-NT1 Case Verification ----
        non_nt1_cases_narc_pos = df.loc[(df['cataplexy_clear_cut'] == 0) & (df['narcolepsy'] == 'narcolepsy')]
        dqb_positive_non_nt1_narc_pos = non_nt1_cases_narc_pos['DQB10602'].value_counts(normalize=True).get(1, 0) * 100
        dqb_negative_non_nt1_narc_pos = non_nt1_cases_narc_pos['DQB10602'].value_counts(normalize=True).get(0, 0) * 100
        results['Non-NT1 & Narc Cases (DQB1*06:02+)']['Actual_Percentage'] = dqb_positive_non_nt1_narc_pos
        results['Non-NT1 & Narc Cases (DQB1*06:02+)']['Meets_Requirement'] = abs(
            dqb_positive_non_nt1_narc_pos - 23) <= 1
        results['Non-NT1 & Narc Cases (DQB1*06:02-)']['Actual_Percentage'] = dqb_negative_non_nt1_narc_pos
        results['Non-NT1 & Narc Cases (DQB1*06:02-)']['Meets_Requirement'] = abs(
            dqb_negative_non_nt1_narc_pos - 77) <= 1

        # ---- Overall Narcolepsy Group Verification ----
        narcolepsy_cases = df.loc[(df['narcolepsy'] == 'narcolepsy') | (df['narcolepsy'] == 'factual narcolepsy')]
        dqb_positive_narcolepsy = narcolepsy_cases['DQB10602'].value_counts(normalize=True).get(1, 0) * 100
        weighted_expected = (98 * len(nt1_narc_pos_cases) + 23 * len(non_nt1_cases_narc_pos)) / max(
            len(narcolepsy_cases),
            1)
        results['Overall Narcolepsy Group (DQB1*06:02+)']['Expected_Percentage'] = weighted_expected
        results['Overall Narcolepsy Group (DQB1*06:02+)']['Actual_Percentage'] = dqb_positive_narcolepsy
        results['Overall Narcolepsy Group (DQB1*06:02+)']['Meets_Requirement'] = abs(
            dqb_positive_narcolepsy - weighted_expected) <= 1

        # Convert results to DataFrame
        results_df = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={'index': 'Category'})

        # Determine if all requirements are met
        verification_passed = all(results_df['Meets_Requirement'])

        # Display the results
        print("Verification of DQB1*06:02 Distribution Requirements:")
        print(tabulate(results_df, headers='keys', tablefmt='grid'))

        return verification_passed, results_df

    tabs = {}
    tab = visualize_table(df=df,
                          group_by=['source', 'narcolepsy', 'cataplexy_clear_cut', 'DQB10602'])
    tabs['tab_zero'] = tab
    # ---- Step 1: Remove Rows with All Missing Values in Key Columns ----
    # Drop rows where 'narcolepsy', 'cataplexy_clear_cut', and 'DQB10602' are all NaN.
    df = df.dropna(subset=['narcolepsy', 'cataplexy_clear_cut', 'DQB10602'], how='all')
    df = df.dropna(subset=['narcolepsy', 'DQB10602'], how='all')

    # ---- Step 2: Drop Rows with Undefined Narcolepsy and Missing NT1 and DQB10602 ----
    # Remove rows where 'narcolepsy' is 'undefined' and both 'cataplexy_clear_cut' and 'DQB10602' are NaN.
    nans_drop = df[(df['narcolepsy'] == 'undefined') &
                   (df['cataplexy_clear_cut'].isna()) &
                   (df['DQB10602'].isna())].index
    df = df.drop(nans_drop)
    tab_one = visualize_table(df=df, group_by=['source', 'narcolepsy', 'cataplexy_clear_cut', 'DQB10602'])
    tabs['tab_one'] = tab_one
    # ---- Step 3: Remove Rows with Defined NT1 but Missing DQB10602 ----
    # Drop rows where 'narcolepsy' is confirmed, 'cataplexy_clear_cut' is defined, but 'DQB10602' is missing.
    indexes_to_drop = df.loc[(df['narcolepsy'] == 'narcolepsy') &
                             (~df['cataplexy_clear_cut'].isna()) &
                             (df['DQB10602'].isna())].index
    df = df.drop(indexes_to_drop)
    tab_two = visualize_table(df=df, group_by=['source', 'narcolepsy', 'cataplexy_clear_cut', 'DQB10602'])
    tabs['tab_two'] = tab_two
    # ---- Step 4: Apply Control Group Rule for 23% DQB1*06:02 Positive, 77% Negative ----
    # For non-cataplexy, non-narcoleptic subjects, assign 23% as DQB1*06:02 positive, and the remaining 77% as negative.
    df_controls = df.loc[(df['narcolepsy'] == 'non-narcoleptic') &
                         (df['cataplexy_clear_cut'] == 0) &
                         (df['DQB10602'].isna()), :]
    nt1_sample_size = int(df_controls.shape[0] * 0.23)
    nt1_indices = df_controls.sample(nt1_sample_size, random_state=42).index
    df.loc[nt1_indices, 'DQB10602'] = 1
    nt1_neg_indices = list(set(df_controls.index) - set(nt1_indices))
    df.loc[nt1_neg_indices, 'DQB10602'] = 0
    tab_three = visualize_table(df, group_by=['source', 'narcolepsy', 'cataplexy_clear_cut', 'DQB10602'])
    tabs['tab_three'] = tab_three
    # ---- Step 5: Assign NT1 for Narcolepsy with DQB1*06:02 Positive ----
    # For narcolepsy patients with unknown cataplexy but DQB1*06:02 positive, assign NT1 status.
    df.loc[(df['narcolepsy'] == 'narcolepsy') &
           (df['cataplexy_clear_cut'].isna()) &
           (df['DQB10602'] == 1), 'cataplexy_clear_cut'] = 1

    # ---- Step 6: Assign Non-NT1 for Non-Narcoleptic with DQB1*06:02 Status ----
    # Set 'cataplexy_clear_cut' to 0 for non-narcoleptic subjects with either DQB1*06:02 positive or negative.
    df.loc[(df['narcolepsy'] == 'non-narcoleptic') &
           (df['cataplexy_clear_cut'].isna()) &
           (df['DQB10602'] == 0), 'cataplexy_clear_cut'] = 0
    df.loc[(df['narcolepsy'] == 'non-narcoleptic') &
           (df['cataplexy_clear_cut'].isna()) &
           (df['DQB10602'] == 1), 'cataplexy_clear_cut'] = 0
    tab_four = visualize_table(df=df, group_by=['source', 'narcolepsy', 'cataplexy_clear_cut', 'DQB10602'])
    tabs['tab_four'] = tab_four
    # ---- Step 7: Create 'Pseudo Narcolepsy' Class for Certain NT1 and Factual Narcolepsy Cases ----
    # Label certain cases as 'pseudo narcolepsy' (e.g., factual narcolepsy cases or NT1 without DQB1*06:02 positivity).
    pseudo_nt1_cases_one = df.loc[(df['narcolepsy'] == 'narcolepsy') &
                                  (df['cataplexy_clear_cut'] == 1) &
                                  (df['DQB10602'] == 0), 'cataplexy_clear_cut'].index
    pseudo_nt1_cases_two = df.loc[df['narcolepsy'] == 'factual narcolepsy', 'cataplexy_clear_cut'].index
    pseudo_nt1_cases = [*pseudo_nt1_cases_one] + [*pseudo_nt1_cases_two]
    df.loc[pseudo_nt1_cases, 'cataplexy_clear_cut'] = 2
    df.loc[pseudo_nt1_cases, 'narcolepsy'] = 'pseudo narcolepsy'
    tab_five = visualize_table(df=df, group_by=['narcolepsy', 'cataplexy_clear_cut', 'DQB10602'])
    tabs['tab_five'] = tab_five
    # ---- Step 8: Verify Distribution of DQB1*06:02 in Final Data ----
    # Check if the final distribution meets expected standards for NT1 and non-NT1 cases.
    df.loc[df['narcolepsy'] != 'pseudo narcolepsy', :]
    verification_passed, verification_results = verify_dqb_distribution(
        df.loc[df['narcolepsy'] != 'pseudo narcolepsy', :])

    for name, frame in tabs.items():
        frame.to_csv(f'{name}.csv', index=False)

    return verification_results, tabs, df

def compare_imputation(df_original:pd.DataFrame,
                       df_imputed:pd.DataFrame, covariates:Dict[str, str]) -> pd.DataFrame:
    """
    Compare columns in original and imputed DataFrames, showing changes
    in value counts or descriptive statistics based on covariate type.

    Parameters:
    - df_original (pd.DataFrame): Original DataFrame with NaN values.
    - df_imputed (pd.DataFrame): Imputed DataFrame without NaN values.
    - covariates (dict): Dictionary specifying column types ('continuous' or 'ordinal').

    Returns:
    - pd.DataFrame: Comparison DataFrame with statistics before and after imputation.
    """
    comparison_data = []

    for col, col_type in covariates.items():
        if col_type == 'continuous':
            # Collect statistics for continuous variables
            before_stats = df_original[col].describe()
            after_stats = df_imputed[col].describe()
            comparison_data.append({
                'Column': col,
                'Type': 'continuous',
                'Before_Mean': before_stats['mean'],
                'After_Mean': after_stats['mean'],
                'Before_Std': before_stats['std'],
                'After_Std': after_stats['std'],
                'Before_Min': before_stats['min'],
                'After_Min': after_stats['min'],
                'Before_Max': before_stats['max'],
                'After_Max': after_stats['max']
            })

        elif col_type == 'ordinal':
            # Collect value counts for ordinal/categorical variables
            before_counts = df_original[col].value_counts(dropna=False).to_dict()
            after_counts = df_imputed[col].value_counts().to_dict()
            comparison_data.append({
                'Column': col,
                'Type': 'ordinal',
                'Before_ValueCounts': before_counts,
                'After_ValueCounts': after_counts
            })

    # Convert comparison data into a DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

# %% Main
if __name__ == "__main__":
    PLOT = False
    target = 'NT1 ICSD3 - TR'
    target_nt2 = target.replace('1', '2')
    # %% Pre-process SSQDX dataset
    df_okun = pre_process_okun_dataset(dataset_path=config.get('data_raw_files').get('okun'))
    # %% Pre-process SSQ DATASET
    df_anic = pre_process_anic_dataset(dataset_path=config.get('data_raw_files').get('anic'))
    # df_anic['narcolepsy'] = df_anic['narcolepsy'].map({1: 'narcolepsy', 0: 'non-narcolepsy'})

    # %% rename csf column
    col_csf = 'csf hcrt concentration crude (1)'
    if not col_csf in df_anic.columns:
        df_anic[col_csf] = np.nan
        print(f'CFS Anic: \n{df_anic[col_csf].describe()}')

    # okun does not ha e csf data
    if not col_csf in df_okun.columns:
        df_okun[col_csf] = np.nan
        print(f'CFS Okun: \n{df_okun[col_csf].describe()}')



    # %% Merge the two dataset
    # Mapping columns of SSQ HLA
    emotions_interest = ["LAUGHING", "QUICKVERBAL", "ANGER"]
    count_array = [str(i) for i in range(54, 74)]
    narc_columns_a = [col for narcolepsy in count_array for col in df_anic if col.startswith(narcolepsy + 'a')]

    # POSEMOT  cataplexy triggered by positive emotions
    columns_posemot = [
        '54a_laughing-cataplexy',
        '56a_excitement-cataplexy',
        '58a_happy-memory-cataplexy',
        '60a_quick-response-cataplexy',
        '66a_elation-cataplexy',
        '63a_sexual-intercourse-cataplexy',
        '70a_exciting-game-cataplexy',
        '72a_joke-cataplexy',
    ]
    df_anic['POSEMOT'] = df_anic.apply(make_new_col, axis=1, args=(columns_posemot,))
    # DISNOCSLEEP  disturbed nocturnal sleep (patients complains of poor sleep at night)
    columns_dinocsleep = [
        '25_difficulty_falling_asleep_ever',
        '26_current_difficulty_falling_asleep',
    ]
    df_anic['DISNOCSLEEP'] = df_anic.apply(make_new_col, axis=1, args=(columns_dinocsleep,))
    # NEGEMOT  cataplexy triggered by negative emotions, anger, embarrassment , stress etc (composite of multiple
    # answers as OR)
    columns_negemot = [
        '55a_anger-cataplexy',
        '61a_embarrassment-cataplexy',
        '67a_stress-cataplexy',
        '68a_startle-cataplexy',
        '69a_tension-cataplexy'
    ]
    df_anic['NEGEMOT'] = df_anic.apply(make_new_col, axis=1, args=(columns_negemot,))
    # NDEMOT  CATAPLEXY triggered when remember an emotional moment (the only one I am less sure)
    columns_ndemot = [
        '55a_anger-cataplexy',
        '61a_embarrassment-cataplexy',
        '62a_disciplining-children-cataplexy',
        '67a_stress-cataplexy',
        '69a_tension-cataplexy',
        '73a_emotional-moment-cataplexy'
    ]
    df_anic['NDEMOT'] = df_anic.apply(make_new_col, axis=1, args=(columns_ndemot,))
    columns_movedemot = [
        '59a_emotional-memory-cataplexy',
    ]
    df_anic['MOVEDEMOT'] = df_anic.apply(make_new_col, axis=1, args=(columns_movedemot,))
    for col in ['POSEMOT', 'NEGEMOT', 'NDEMOT', 'DISNOCSLEEP']:
        print(f'{col}: \n{df_anic[col].value_counts()}')

    print(df_anic['DQB10602'].value_counts())

    #  Transform column values so they match both sets
    mapper = {
        'Race': 'ethnicity',
        'sex': 'gender',
        'Age': 'age',
        'BMI': 'bmi',
        'ESS': 'epworth_score',
        'NAPS': '39_nap_frequency',
        'SLEEPIONSET': '46_age_sleep_complaint',
        # 'CATCODE': '',
        'DURATION': '84_muscle_weakness_duration',
        'FREQ': '85_muscle_weakness_frequency',
        'ONSET': '95_first_muscle_weakness_age',
        'INJURED': '101_injured_during_episode',
        # 'MEDCATA': '',
        'MSLT': 'MSLT',
        'SOREMP': 'SOREMP',
        # 'MSLTAGE': '',
        # 'DQB10602': '',
        'DISNOCSLEEP': 'DISNOCSLEEP',
        'POSEMOT': 'POSEMOT',
        'NEGEMOT': 'NEGEMOT',
        'NDEMOT': 'NDEMOT',
        'MOVEDEMOT': 'MOVEDEMOT',
        target:target,
        target_nt2: target_nt2,
        'csf': col_csf,
        'source': 'source'
    }
    mapper_inv = {val: key for key, val in mapper.items()}
    df_okun = df_okun.replace(to_replace='.', value=np.nan)
    df_okun['ESS'] = df_okun['ESS'].astype(float)
    # clip age to the SSQHLA
    df_okun['Age'] = df_okun['Age'].astype(float)
    df_anic[df_anic['age'] < 9] = np.nan

    df_okun['sex'] = df_okun['sex'].replace({'M': 1, 'F': 0})
    df_anic['gender'] = df_anic['gender'].replace({9: np.nan})

    df_okun['Race'] = df_okun['Race'].replace({'Caucasian ': 'Caucasian'})
    df_anic['ethnicity'] = df_anic['ethnicity'].replace({
        'Cauc': 'Caucasian',
        'Latino': 'Latino',
        '9': np.nan,
    })

    # df_okun['NAPS'].unique()
    df_anic.loc[df_anic['39_nap_frequency'] > 100, '39_nap_frequency'] = np.nan
    df_okun['NAPS'].value_counts()
    df_anic['39_nap_frequency'].value_counts()

    # %% visualization naps: start
    if PLOT:
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 6))
        ax = sns.countplot(
            x="39_nap_frequency",
            hue=target,
            data=df_anic,
            palette="Set2"  # optional: choose any seaborn palette
        )
        plt.xlabel("Naps")
        plt.ylabel("Count")
        plt.title("Counts of 39_nap_frequency, Clustered by Target Category")
        plt.legend(title="Target")
        plt.tight_layout()
        plt.show()

        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 4))
        ax = sns.countplot(
            x="NAPS",
            hue=target,
            data=df_okun,
            palette="Set2"  # optional: choose any seaborn palette
        )
        plt.xlabel("Naps")
        plt.ylabel("Count")
        plt.title("Counts of Naps, Clustered by Target Category")
        plt.legend(title="Target")
        plt.tight_layout()
        plt.show()


    # %% visualizatio naps: end
    # df_okun['NAPS'].unique()
    df_anic['45_sleepiness_severity_since_age'].value_counts()
    # age sleep complaints columns are not perfect, feature engineer by combining and estimating
    df_okun['SLEEPIONSET'] = df_okun['SLEEPIONSET'].replace({'33': 33})

    df_anic['46_age_sleep_complaint'] = df_anic.apply(create_sleep_complaint, axis=1)

    if PLOT:
        sns.histplot(df_anic['46_age_sleep_complaint'].dropna(), kde=True)
        plt.grid(alpha=0.7)
        plt.tight_layout()
        plt.show()
        plot_histograms(df1=df_okun,
                        df2=df_anic,
                        col1=f'SLEEPIONSET',
                        col2=f'46_age_sleep_complaint',
                        title=f'SSQHLA SLEEPIONSET {df_okun.shape[0]} \nVs \nSSQ 46_age_sleep_complaint {df_anic.shape[0]}')

    # Duration is ordinal -> 5-30sec, 13sec-2min, 2-10min, 10min++ -> 0,1,2,3
    df_okun['DURATION'].unique()
    df_anic['84_muscle_weakness_duration'] = df_anic['84_muscle_weakness_duration'].map({0: 0, 1: 1, 2: 2, 3: 3})
    if PLOT:
        sns.histplot(df_anic['84_muscle_weakness_duration'].dropna(), kde=True)
        plt.grid(alpha=0.7)
        plt.tight_layout()
        plt.show()

    # frequency, in the anic is okay is they are not much since the majority are controls
    # df_okun['FREQ'].unique()
    df_anic['85_muscle_weakness_frequency'] = df_anic['85_muscle_weakness_frequency'].map({0: 0, 1: 1, 2: 2, 3: 3})
    if PLOT:
        sns.histplot(df_anic['85_muscle_weakness_frequency'].dropna(), kde=True)
        plt.grid(alpha=0.7)
        plt.tight_layout()
        plt.show()

    df_okun['ONSET'].unique()
    df_anic['95_first_muscle_weakness_age'].unique()
    df_anic.loc[df_anic['95_first_muscle_weakness_age'] > 100, '95_first_muscle_weakness_age'] = np.nan
    if PLOT:
        sns.histplot(df_anic['95_first_muscle_weakness_age'].dropna(), kde=True)
        plt.grid(alpha=0.7)
        plt.tight_layout()
        plt.show()

    # df_okun['INJURED'].unique()
    df_anic['101_injured_during_episode'].unique()
    df_anic.loc[df_anic['101_injured_during_episode'] > 2, '101_injured_during_episode'] = np.nan
    if PLOT:
        sns.histplot(df_anic['101_injured_during_episode'].dropna(), kde=True)
        plt.grid(alpha=0.7)
        plt.tight_layout()
        plt.show()

    if PLOT:
        # comapre the columns side-by-side in a single dataframe and making histogram plots
        df_comparison_dist = pd.DataFrame()
        for col_anichla, col_anic in mapper.items():
            desc_anichla = df_okun[col_anichla].describe()
            desc_anic = df_anic[col_anic].describe()
            combined_desc = pd.concat([desc_anichla, desc_anic], axis=1)
            combined_desc.columns = ['anichla', 'anic']
            combined_desc['columns'] = f'{col_anichla}, {col_anic}'
            df_comparison_dist = pd.concat([df_comparison_dist, combined_desc])

        for col_anichla, col_anic in mapper.items():
            plot_histograms(df1=df_okun,
                            df2=df_anic,
                            col1=f'{col_anichla}',
                            col2=f'{col_anic}',
                            title=f'SSQHLA {col_anichla} {df_okun.loc[~df_okun[col_anichla].isna(), col_anichla].shape[0]} '
                                  f'\nVs \nSSQ {col_anic} {df_anic.loc[~df_anic[col_anic].isna(), col_anic].shape[0]}')
    # slice the frame with the columns of interest (intersection between SSQ and SSQHLA)
    df_anic_to_merge = df_anic[[*mapper.values()]].copy()
    # include the multiple target columns
    df_anic_to_merge['DQB10602'] = df_anic['DQB10602']
    # df_anic_to_merge['narcolepsy'] = df_anic['narcolepsy']
    # df_anic_to_merge['target'] = df_anic['target']
    # df_anic_to_merge['cataplexy_clear_cut'] = df_anic['cataplexy_clear_cut']
    df_anic_to_merge['SOREMP'] = df_anic_to_merge['SOREMP'].str.extract(r'(\d+)\s+of')[0].astype('Int64')

    print(df_anic_to_merge['DQB10602'].value_counts())
    print(df_anic_to_merge[target].value_counts())
    # print(df_anic_to_merge['target'].value_counts())
    # %% Emotions and muscle weakness equivalence (Mapping between the two datasets)
    emotions_anic_anichla = {
        '56a_excitement-cataplexy': 'EXCITED',
        '73a_emotional-moment-cataplexy': 'MOVEDEMOT',
        '62a_disciplining-children-cataplexy': 'DISCIPLINE',
        '65a_post-athletic-activities-cataplexy': 'AFTATHLETIC',
        '72a_joke-cataplexy': 'JOKING',
        '61a_embarrassment-cataplexy': 'EMBARRAS',
        '55a_anger-cataplexy': 'ANGER',
        '63a_sexual-intercourse-cataplexy': 'SEX',
        '70a_exciting-game-cataplexy': 'PLAYGAME',
        '54a_laughing-cataplexy': 'LAUGHING',
        '68a_startle-cataplexy': 'STARTLED',
        '66a_elation-cataplexy': 'ELATED',
        '59a_emotional-memory-cataplexy': 'EMOTIONAL',
        '60a_quick-response-cataplexy': 'QUICKVERBAL',
        '64a_athletic-activities-cataplexy': 'DURATHLETIC',
        '71a_romantic-moment-cataplexy': 'ROMANTIC',
        '67a_stress-cataplexy': 'STRESSED',
        '69a_tension-cataplexy': 'TENSE',
        '58a_happy-memory-cataplexy': 'HAPPY',
        '57a_surprise-cataplexy': 'SURPRISED'
    }
    # df_emotions = pd.DataFrame.from_dict(emotions_anic_anichla, orient='index', columns=['Emotion'])
    # df_emotions = df_emotions.reset_index().rename(columns={'index': 'Event'})

    for emotion_anic_, emotions_anic_anichla_ in emotions_anic_anichla.items():
        df_anic_to_merge[emotions_anic_anichla_] = df_anic[emotion_anic_]

    df_anic_to_merge['JAW'] = df_anic[f'{77}_{key_mapping.get(77)}']
    df_anic_to_merge['KNEES'] = df_anic[f'{76}_{key_mapping.get(76)}'].apply(quest76map)
    df_anic_to_merge['HEAD'] = df_anic[f'{78}_{key_mapping.get(78)}']
    df_anic_to_merge['HAND'] = df_anic[f'{79}_{key_mapping.get(79)}']


    # falling experience with experience 6
    col_exp_falling = [col for col in df_anic.columns if '_exp_6' in col]
    df_anic_to_merge['FALL'] = df_anic[col_exp_falling].any(axis=1).astype(int)

    # rename the columns
    df_anic_to_merge.rename(mapper_inv, inplace=True, axis=1)
    df_anic_to_merge['DQB10602'].value_counts()
    # %% include the missing columns (not measure from the questionnaire)
    # df_okun.shape
    # df_anic_to_merge.shape
    for col in df_okun.columns:
        if not col in df_anic_to_merge.columns:
            # print(col)
            df_anic_to_merge[col] = np.nan
    # %% complete the merge
    df_data = pd.concat([df_okun, df_anic_to_merge], axis=0)
    df_data.reset_index(inplace=True, drop=True)
    tab_one = visualize_table(df=df_data, group_by=['source', 'DQB10602', target])
    tab_one = visualize_table(df=df_okun, group_by=['source', 'DQB10602', target])

    # df_data.rename(columns={'DQB10602': 'target'}, inplace=True)
    if PLOT:
        palette = sns.color_palette("Set2",
                                    n_colors=len(df_data['source'].unique()))
        # Create a 2x2 grid, but adjust the layout so the third plot spans both columns in the second row
        fig, ax = plt.subplots(nrows=2,
                               ncols=2,
                               figsize=(10, 8),
                               sharey=False,
                               gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
        # Top-left: DQB10602
        sns.countplot(data=df_data, x='DQB10602', hue='source', palette=palette, ax=ax[0, 0])
        ax[0, 0].set_title(f'DQB10602 In Complete Dataset\n{df_data.DQB10602.value_counts().to_dict()}')
        ax[0, 0].grid(alpha=.7)
        # Top-right: Narcolepsy
        sns.countplot(data=df_data, x='cataplexy_clear_cut', hue='source', palette=palette, ax=ax[0, 1])

        ax[0, 1].set_title(f'Merged DQB and Narcolepsy In Complete Dataset\n'
                            f'{df_data["cataplexy_clear_cut"].value_counts().to_dict()}')
        ax[0, 1].grid(alpha=.7)
        # Span the third plot across both columns in the second row
        ax_bottom = fig.add_subplot(2, 1, 2)
        sns.countplot(data=df_data, x='narcolepsy', hue='source', palette=palette, ax=ax_bottom)
        ax_bottom.set_title(f'Narcolepsy In Complete Dataset\n{df_data.narcolepsy.value_counts().to_dict()}')
        ax_bottom.grid(alpha=.7)
        # Remove the unused axes
        ax[1, 0].remove()  # Remove the empty subplot
        ax[1, 1].remove()  # Remove the empty subplot
        plt.tight_layout()
        plt.show()


    # %% Check data types on emotions and muscle weaknesses
    # keep them as binary responses yes/no
    emotions_anichla = {
        "LAUGHING", "EXCITED", "HAPPY", "QUICKVERBAL", "SEX", "ELATED", "PLAYGAME",
        "JOKING", "NEGEMOT", "ANGER", "EMBARRAS", "DISCIPLINE", "STRESSED", "TENSE",
        "NDEMOT", "SURPRISED", "EMOTIONAL", "DURATHLETIC", "AFTATHLETIC", "STARTLED",
        "ROMANTIC", "MOVEDEMOT"
    }
    # muscle weakness
    mw_anichla = {'KNEES', 'JAW', 'HEAD', 'HAND', 'SPEECH'}

    for col in emotions_anichla:
        # print(f'\n{col}: {df_data[col].value_counts()}')
        df_data[col] = df_data[col].apply(set_to_zero_except_one)

    for col in mw_anichla:
        # print(f'\n{col}: {df_data[col].value_counts()}')
        df_data[col] = df_data[col].apply(set_to_zero_except_one)

    # %% organize columns
    cols_head = [
        "Race",
        "Ethnicity",
        "sex",
        "Age",
        "BMI",
    ]
    # cols_tail = [
    #     "CATCODE",
    #     "MEDCATA",
    #     "MSLT",
    #     "SOREMP",
    #     "MSLTAGE",
    #     "SE",
    #     "REMLAT",
    #     "RDI",
    #     "PLMIND",
    #     "Dx",
    #     "contains_A_to_F",
    #     "narcolepsy",
    #     "cataplexy_clear_cut",
    #     "DQB10602",
    #     'source'
    # ]
    cols_tail = ['CATCODE',
                 'MEDCATA',
                 'RDI',
                 'PLMIND',
                 # 'Dx',
                 'DQB10602',
                 'source',
                 'MSLT',
                 'SOREMP',
                 'csf',
                 target,
                 target_nt2
                 ]

    col_middle = [col for col in df_data.columns if not (col in cols_head or col in cols_tail)]
    columns = cols_head + col_middle + cols_tail
    df_data = df_data[columns]

    # keep the binary nature of the variable
    df_data['DISNOCSLEEP'] = df_data['DISNOCSLEEP'].clip(upper=1)

    # df_data.to_excel('anic_okun_data.xlsx', index=False)
    df_data = df_data.loc[~df_data['source'].isna(), :]
    df_data = df_data.loc[~df_data[target].isna(), :]
    df_data = df_data.loc[df_data[target].isin([0, 1]), :]
    # df_data = df_data.loc[~df_data[target].isna(), :]
    df_data = df_data[~df_data['DQB10602'].isna()]

    df_data.reset_index(inplace=True, drop=True)


    tab = visualize_table(df=df_data,
                          group_by=['source', 'DQB10602', target, target_nt2]) #

    tab['DQB10602'] = tab['DQB10602'].replace({1: '+', 0:'-'})
    tab[target] = tab[target].replace({1: 'case',
                                       0:'control'})
    tab[target_nt2] = tab[target_nt2].replace({1: 'case',
                                               0:'control'})


    tab.to_csv(config.get('results_path')['results'].joinpath('tab_okun_target_raw.csv'), index=False)


    # %% remove the non meeting criteria for the target
    # NT1 ICSD3 - TR
    # 0                    929
    # 1                    289
    # not meet criteria    211
    # Name: count, dtype: int64

    # NT2 ICSD3 - TR
    # 0                    1173
    # not meet criteria     201
    # 1                      54
    # Name: count, dtype: int64

    #%% save dataset
    df_data.to_csv(config.get('data_pre_proc_files').get('anic_okun'), index=False)







