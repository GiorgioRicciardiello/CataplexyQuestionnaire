"""
Use the preprocess dataset and generate the table one.
"""
from config.config import config
from config.SSI_Digitial_Questionnaire import variable_definitions
import numpy as np
import pandas as pd
from library.table_one import MakeTableOne
from typing import Dict, List, Optional
from scipy.stats import fisher_exact, spearmanr, mannwhitneyu, shapiro, ttest_ind, chi2_contingency
from tabulate import tabulate
from statsmodels.stats.multitest import multipletests


def stats_test_binary_symptoms(
    data: pd.DataFrame,
    columns: List[str],
    strata_col: str = 'NT1',
    SHOW: Optional[bool] = False
) -> pd.DataFrame:
    """
    Perform Chi-Square Test or Fisher's Exact Test for binary values (1 as “yes”)
    comparing the distribution between two groups.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing binary symptom responses and the group column.
    columns : List[str]
        Columns to perform the statistical test on.
    strata_col : str, default 'NT1'
        Column name for the grouping variable (must have exactly two unique values).
    SHOW : bool, default False
        If True, print the 2×2 contingency table for each test.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'Variable' (the symptom name)
        - '{strata_col}={group0}' (percent “yes” in group0 and raw counts)
        - '{strata_col}={group1}' (percent “yes” in group1 and raw counts)
        - 'p-value' (numeric p-value)
        - 'p-value formatted' (formatted string, 4 decimal places or scientific if < 1e-4)
        - 'Effect Size (Odds Ratio)'
        - 'Stat Method' (“Chi-Square Test” or “Fisher's Exact Test”)
    """
    # Ensure exactly two strata levels
    unique_strata = data[strata_col].dropna().unique()
    if len(unique_strata) != 2:
        raise ValueError(f"Strata column '{strata_col}' must have exactly 2 unique values.")
    group0, group1 = np.sort(unique_strata)

    results = []

    for col in columns:
        if col == strata_col:
            continue

        # Subset to non-missing rows for this column and the strata column
        df = data[[col, strata_col]].dropna().copy()

        # Skip if column is not strictly binary 0/1
        if set(df[col].unique()) != {0, 1}:
            continue

        # Count total N in each stratum
        n_controls = (df[strata_col] == group0).sum()
        n_cases    = (df[strata_col] == group1).sum()

        ci_low, ci_high, odds_ratio = np.nan, np.nan, np.nan
        # If either stratum has no observations, skip
        if n_controls == 0 or n_cases == 0:
            results.append({
                'Variable': str(col),
                f'{strata_col}={group1}': f'{n_cases}',
                f'{strata_col}={group0}': f'{n_controls}',
                'p-value': np.nan,
                'p-value formatted': np.nan,
                'Effect Size (Odds Ratio)': np.nan,
                # '95% CI': f"[{ci_low:.2f}, {ci_high:.2f}]" if not np.isnan(ci_low) else np.nan,
                'Effect Size (Odds Ratio 95% CI)': f'{round(odds_ratio, 3)} [{ci_low:.2f}, {ci_high:.2f}]',

                'Stat Method': 'None'
            })

        # Count “yes” (=1) in each stratum
        controls_yes = df.loc[df[strata_col] == group0, col].sum()
        cases_yes    = df.loc[df[strata_col] == group1, col].sum()

        # Build 2×2 table as:
        #           |  “no”  |  “yes”
        # --------------------------------
        # controls  |   a    |   b
        # cases     |   c    |   d
        a = n_controls - controls_yes  # controls who answered 0 (“no”)
        b = controls_yes               # controls who answered 1 (“yes”)
        c = n_cases    - cases_yes     # cases who answered 0 (“no”)
        d = cases_yes                  # cases who answered 1 (“yes”)

        table = [[a, b],
                 [c, d]]

        if SHOW:
            headers = [f"{col} = 0", f"{col} = 1"]
            row_labels = [f"{strata_col}={group0}", f"{strata_col}={group1}"]
            labeled_table = [[row_labels[i], row[0], row[1]] for i, row in enumerate(table)]
            print(tabulate(labeled_table, headers=["Group", *headers], tablefmt="grid"))

        # Compute Chi‐Square expected counts
        chi2_stat, p_chi2, dof, expected = chi2_contingency(table)
        use_fisher = (expected.min() < 5)

        if use_fisher:
            odds_ratio, p_value = fisher_exact(table, alternative='two-sided')
            method = "Fisher's Exact Test"
        else:
            p_value = p_chi2
            # Compute odds ratio only if b*c != 0
            odds_ratio = (a * d) / (b * c) if (b * c) != 0 else np.nan
            method = "Chi-Square Test"

        if all(x > 0 for x in [a, b, c, d]):
            se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
            log_or = np.log(odds_ratio)
            ci_low = np.exp(log_or - 1.96 * se)
            ci_high = np.exp(log_or + 1.96 * se)

        # Percent “yes” in each stratum
        pct_controls = 100 * b / n_controls
        pct_cases    = 100 * d / n_cases

        # Format p‐value string
        if p_value >= 0.0001:
            p_fmt = f"{p_value:.4f}"
        else:
            p_fmt = f"{p_value:.4e}"

        results.append({
            'Variable': str(col),
            f'{strata_col}={group1}': f'{pct_cases:.1f}% ({d}/{n_cases})',
            f'{strata_col}={group0}': f'{pct_controls:.1f}% ({b}/{n_controls})',
            'p-value': p_value,
            'p-value formatted': p_fmt,
            # 'Effect Size (Odds Ratio 95% CI)': round(odds_ratio, 3) if not np.isnan(odds_ratio) else np.nan,
            'Effect Size (Odds Ratio 95% CI)': f'{round(odds_ratio, 3)} [{ci_low:.2f}, {ci_high:.2f}]',
            # '95% CI': f"[{ci_low:.2f}, {ci_high:.2f}]" if not np.isnan(ci_low) else np.nan,
            'Stat Method': method
        })
    return pd.DataFrame(results)

def stats_test_continuous(
    data: pd.DataFrame,
    columns: List[str],
    strata_col: str = 'NT1',
    SHOW: Optional[bool] = False
) -> pd.DataFrame:
    """
    Perform statistical tests for continuous variables between two groups.
    - First, test each group's values for normality via Shapiro-Wilk.
    - If both groups are approximately normal, run an independent t-test (unequal variance)
      and compute Cohen's d.
    - Otherwise, run a two-sided Mann-Whitney U test and compute rank-biserial r.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing continuous variables and a grouping column.
    columns : List[str]
        List of column names to test.
    strata_col : str, default 'NT1'
        Name of the binary grouping column (must have exactly two unique, non-NaN values).
    SHOW : bool, default False
        If True, print summary statistics, p-values, and effect sizes for each variable.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row corresponds to one tested column and contains:
          - 'Variable'
          - '{strata_col}={group0} Mean (SD)'
          - '{strata_col}={group1} Mean (SD)'
          - 'n {strata_col}={group0}' (sample size in group0)
          - 'n {strata_col}={group1}' (sample size in group1)
          - 'p-value' (numeric)
          - 'p-value formatted' (string with 4 decimals or scientific if < 1e-4)
          - 'Effect Size' (Cohen's d or rank-biserial r)
          - 'Stat Method' ('Independent t-test' or 'Mann-Whitney U test')
    """

    def _rank_biserial(group0, group1):
        # Perform U‐test (two‐sided)
        U, p = mannwhitneyu(group0, group1, alternative='two-sided')
        n0, n1 = len(group0), len(group1)
        # Convert U to Z
        mu_U = n0 * n1 / 2
        sigma_U = np.sqrt(n0 * n1 * (n0 + n1 + 1) / 12)
        Z = (U - mu_U) / sigma_U
        # r_rb = Z / sqrt(N)
        return Z / np.sqrt(n0 + n1), p

    def _cliffs_delta(group0, group1):
        n0, n1 = len(group0), len(group1)
        gt = 0
        lt = 0
        for x in group0:
            for y in group1:
                if x > y:
                    gt += 1
                elif x < y:
                    lt += 1
        return (gt - lt) / (n0 * n1)


    results = []

    # Identify and sort the two unique strata values
    unique_groups = data[strata_col].dropna().unique()
    if len(unique_groups) != 2:
        raise ValueError(f"Strata column '{strata_col}' must have exactly 2 unique values.")
    group0, group1 = np.sort(unique_groups)

    for col in columns:
        if col == strata_col:
            continue

        # Keep only rows where both col and strata_col are non-missing
        df = data[[col, strata_col]].dropna().copy()

        # Extract values for each group
        group0_vals = df.loc[df[strata_col] == group0, col].to_numpy()
        group1_vals = df.loc[df[strata_col] == group1, col].to_numpy()

        # Count observations
        n0, n1 = len(group0_vals), len(group1_vals)

        # Descriptive statistics (use ddof=1 for sample SD)
        mean0, std0 = (np.nan, np.nan) if n0 == 0 else (group0_vals.mean(), group0_vals.std(ddof=1))
        mean1, std1 = (np.nan, np.nan) if n1 == 0 else (group1_vals.mean(), group1_vals.std(ddof=1))

        # If either group has fewer than 10 observations, report NaNs and continue
        if n0 < 10 or n1 < 10:
            results.append({
                'Variable': col,
                f'{strata_col}={group0} Mean (SD)': f'{mean0:.2f}\u00B1{std0:.2f} ({n0})',
                f'{strata_col}={group1} Mean (SD)': f'{mean1:.2f}\u00B1{std1:.2f} ({n1})',
                # 'n ' + f'{strata_col}={group0}': n0,
                # 'n ' + f'{strata_col}={group1}': n1,
                'p-value': np.nan,
                'p-value formatted': np.nan,
                'Effect Size': np.nan,
                'Stat Method': np.nan
            })
            continue

        # Normality check (Shapiro-Wilk). If Shapiro fails or raises error, treat as non-normal.
        try:
            normal0 = shapiro(group0_vals).pvalue > 0.05
        except:
            normal0 = False
        try:
            normal1 = shapiro(group1_vals).pvalue > 0.05
        except:
            normal1 = False

        # Initialize placeholders
        p_value = np.nan
        effect_size = np.nan
        stat_method = ''

        if normal0 and normal1:
            # Independent (Welch's) t-test
            stat_method = 'Independent t-test'
            t_stat, p_value = ttest_ind(group0_vals, group1_vals, equal_var=False)

            # Cohen's d (using pooled standard deviation)
            s_pooled = np.sqrt(
                (((n0 - 1) * std0**2) + ((n1 - 1) * std1**2)) / (n0 + n1 - 2)
            )
            cohen_d = (mean1 - mean0) / s_pooled if s_pooled > 0 else np.nan
            effect_size = round(cohen_d, 3)

        else:
            # Mann-Whitney U test
            stat_method = 'Mann-Whitney U test'
            r_rb, p_value = _rank_biserial(group0_vals, group1_vals)
            effect_size = round(r_rb, 3)

        # Format p-value string
        p_fmt = f"{p_value:.4f}" if p_value >= 0.0001 else f"{p_value:.4e}"

        results.append({
            'Variable': col,
            f'{strata_col}={group0} Mean (SD)': f'{mean0:.2f}\u00B1{std0:.2f} ({n0})',
            f'{strata_col}={group1} Mean (SD)': f'{mean1:.2f}\u00B1{std1:.2f} ({n1})',
            # 'n ' + f'{strata_col}={group0}': n0,
            # 'n ' + f'{strata_col}={group1}': n1,
            'p-value': p_value,
            'p-value formatted': p_fmt,
            'Effect Size': effect_size,
            'Stat Method': stat_method
        })

        if SHOW:
            print(
                f"{col}: {stat_method} | "
                f"{strata_col}={group0} → {mean0:.2f}\u00B1{std0:.2f} (n={n0}); "
                f"{strata_col}={group1} → {mean1:.2f}\u00B1{std1:.2f} (n={n1}); "
                f"p = {p_fmt}; ES = {effect_size}"
            )

    return pd.DataFrame(results)





# def stats_test_binary_symptoms(data: pd.DataFrame,
#                                columns: List[str],
#                                strata_col: str = 'NT1',
#                                SHOW: Optional[bool] = False
#                                ) -> pd.DataFrame:
#     """
#     Perform Chi-Square Test or Fisher's Exact Test for binary values (1 as positive response)
#     comparing the distribution between two groups.
#
#     Parameters
#     ----------
#     data : pd.DataFrame
#         DataFrame containing binary symptom responses and the group column.
#     columns : List[str]
#         Columns to perform the statistical test on.
#     strata_col : str, default 'NT1'
#         Column name for the grouping variable.
#     SHOW : bool, default False
#         Print the 2x2 contingency table for each test.
#
#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with counts, percentages, p-value, effect size (odds ratio), and test method used.
#     """
#     results = []
#
#     # Define a helper lambda to compute count and percentage for a boolean Series.
#     get_counts_rates = lambda cond: (cond.sum(), cond.mean() * 100)
#
#     if not len(data[strata_col].unique()) == 2:
#         raise ValueError(f'Strata column {strata_col} must have exactly 2 groups')
#
#     group0, group1 = np.sort(data[strata_col].unique())
#
#     for col in columns:
#         if col == strata_col:
#             continue
#
#         df = data[[col, strata_col]].dropna().copy()
#         N = df.shape[0]
#         n_controls = df[df[strata_col] == group0].shape[0]
#         n_cases = df[df[strata_col] == group1].shape[0]
#
#         # Ensure binary values
#         unique_vals = set(df[col].unique())
#         if unique_vals != {0, 1}:
#             continue
#
#         # Count responses for each group
#         grp0 = (df[df[strata_col] == group0][col] == 0)
#         grp1 = (df[df[strata_col] == group1][col] == 1)
#
#         group0_n, group0_rate = get_counts_rates(grp0)
#         group1_n, group1_rate = get_counts_rates(grp1)
#         total_n, total_rate = get_counts_rates(df[col] == 1)
#
#         df.loc[df[strata_col] == group0, col].sum()
#
#         # Create 2x2 contingency table
#         # Create contingency table
#         a = group0_n
#         b = df[df[strata_col] == group0].shape[0] - group0_n
#         c = group1_n
#         d = df[df[strata_col] == group1].shape[0] - group1_n
#         table = [[a, b], [c, d]]
#
#         # Display table if SHOW is True
#         if SHOW:
#             headers = [f"{col} {group0}", f"{col} {group1}"]
#             row_labels = [f"{strata_col} {group0}", f"{strata_col} {group1}"]
#             table_with_labels = [[row_labels[i]] + row for i, row in enumerate(table)]
#             headers_with_labels = ["Group"] + headers
#             print(tabulate(table_with_labels, headers=headers_with_labels, tablefmt="grid"))
#
#         # Compute expected counts for Chi-Square condition
#         chi2_stat, p_chi2, dof, expected = chi2_contingency(table)
#         expected_min = expected.min()
#
#         if expected_min < 5:
#             # Use Fisher's Exact Test if any expected count is <5
#             odds_ratio, p_value = fisher_exact(table, alternative='two-sided')
#             test_method = "Fisher's Exact Test"
#         else:
#             # Use Chi-Square Test and compute OR manually
#             p_value = p_chi2
#             try:
#                 odds_ratio = (a * d) / (b * c) if b * c != 0 else np.nan
#             except ZeroDivisionError:
#                 odds_ratio = np.nan
#             test_method = "Chi-Square Test"
#
#         # Store results
#         res = {
#             'Variable': col,
#             f'{strata_col} {group0}': f'{round(group0_rate, 1)}% ({n_controls})',
#             f'{strata_col} {group1}': f'{round(group1_rate, 1)}% ({n_cases})',
#             'Total N, (%)': f'{round(total_n, 1)} ({round(total_rate, 1)})',
#             'p-value': p_value,
#             'p-value formatted': f"{p_value:.4f}" if p_value >= 0.0001 else "<0.0001",
#             'Effect Size (Odds Ratio)': round(odds_ratio, 3) if odds_ratio is not None else "N/A",
#             'Stat Method': test_method
#         }
#         # res = {
#         #     'Variable': col,
#         #     f'{strata_col} {group0} N, (%)': f'{round(group0_n, 1)} ({round(group0_rate, 1)})',
#         #     f'{strata_col} {group1} N, (%)': f'{round(group1_n, 1)} ({round(group1_rate, 1)})',
#         #     'Total N, (%)': f'{round(total_n, 1)} ({round(total_rate, 1)})',
#         #     'p-value': p_value,
#         #     'p-value formatted': f"{p_value:.4f}" if p_value >= 0.0001 else "<0.0001",
#         #     'Effect Size (Odds Ratio)': round(odds_ratio, 3) if odds_ratio is not None else "N/A",
#         #     'Stat Method': test_method
#         # }
#
#
#         results.append(res)
#
#     return pd.DataFrame(results)

if __name__ == '__main__':
    df_data = pd.read_csv(config.get('data_pre_proc_files').get('anic_okun'))
    # df_data.rename(columns={'cataplexy_clear_cut': 'NT1'}, inplace=True)
    target = 'NT1 ICSD3 - TR'
    target_nt2 = target.replace('1', '2')
    # Use only cases and controls
    # df_data = df_data[df_data[target].isin({0, 1})]

    df_nans = df_data.isna().sum() * 100 / df_data.shape[0]
    df_nans = df_nans.reset_index(name='nan_percent')
    df_nans.sort_values(by='nan_percent', ascending=False, inplace=True)
    df_nans['nan_percent'] = df_nans['nan_percent'] .round(2)
    # define the columns dtypes
    # categorical_var = ['sex',
    #                    'DQB10602',
    #                    'DURATION',
    #                    'HALLUC',
    #                    'SP',
    #                    'DISNOCSLEEP',
    #                    # 'Race',
    #                    'MEDCATA']
    # categorical_var = [var for var in categorical_var if var in df_data]
    # categorical_var = list( np.sort(categorical_var))
    continuous_var = ['Age',
                      'BMI',
                      'ESS',
                      'NAPS',
                      'SLEEPIONSET',
                      'ONSET',
                      'SPONSET',
                      'HHONSET',
                      'MSLT',
                      'MSLTAGE',
                      'SOREMP',
                      'SE',  # sleep latency
                      'REMLAT'
                      ]
    continuous_var = [var for var in continuous_var if var in df_data]

    # columns = list(set(categorical_var + continuous_var + [target]))
    df_soremp_cases = df_data.loc[(df_data['NT1 ICSD3 - TR'] == 1) &
                          (~df_data['SOREMP'].isna()), ['NT1 ICSD3 - TR', 'SOREMP']]

    df_soremp_cases_mean = df_soremp_cases['SOREMP'].mean().round(3)

    df_soremp_cases_std = np.round(df_soremp_cases['SOREMP'].std(), 3)

    n_cases = df_data.loc[df_data['NT1 ICSD3 - TR'] == 1].shape[0]


    soremp_n_percent = np.round((df_soremp_cases.shape[0]/n_cases) * 100, 2)
    #%%
    # make_tab_one = MakeTableOne(df_data,
    #                             categorical_var=categorical_var,
    #                             continuous_var=continuous_var,
    #                             strata=target)
    # df_tab_one = make_tab_one.create_table()
    # df_tab_one = make_tab_one.group_variables_table(df=df_tab_one)
    #
    # for col in columns:
    #     print(df_data.groupby(by=target)[col].count())
    # var = 'HHONSET'

    # %% compute statistics of significance different groups
    categorical_var = ['AFTATHLETIC',
                       'ANGER',
                       'DISCIPLINE',
                       'DISNOCSLEEP',
                       'DQB10602',
                       'DURATHLETIC',
                       'DURATION',
                       'ELATED',
                       'EMBARRAS',
                       'EMOTIONAL',
                       'EXCITED',
                       'HALLUC',
                       'HAND',
                       'HAPPY',
                       'HEAD',
                       'JAW',
                       'JOKING',
                       'KNEES',
                       'LAUGHING',
                       'MEDCATA',
                       'MOVEDEMOT',
                       'PLAYGAME',
                       'QUICKVERBAL',
                       'ROMANTIC',
                       'SEX',
                       'SP',
                       'SPEECH',
                       'STARTLED',
                       'STRESSED',
                       'TENSE',
                       'sex']
    df_stats_bin = stats_test_binary_symptoms(data=df_data,
                               columns=categorical_var,
                               strata_col=target,
                                SHOW=True)

    df_stats_bin['Variable'] = df_stats_bin['Variable'].replace({key: f'{val} (yes)' for key, val in variable_definitions.items()})

    df_stats_bin = df_stats_bin.sort_values(by='Variable', ascending=True)

    df_stats_cont = stats_test_continuous(
                data=df_data,
                columns=list(continuous_var),
                strata_col=target,
                SHOW=True)

    df_stats_cont['Variable'] = df_stats_cont['Variable'].replace(variable_definitions)

    column_mapper    = {
        'Variable': 'Variable',
        'NT1 ICSD3 - TR=1 Mean (SD)': 'Cases',
        'NT1 ICSD3 - TR=0 Mean (SD)': 'Controls',
        'p-value': 'p-value',
        'p-value formatted': 'p-value formatted',
        'Effect Size (Odds Ratio 95% CI)': 'Effect Size',
        'Stat Method': 'Stat Method',

        'NT1 ICSD3 - TR=1': 'Cases',
        'NT1 ICSD3 - TR=0': 'Controls',

    }

    df_stats_bin.rename(columns=column_mapper, inplace=True)
    df_stats_cont.rename(columns=column_mapper, inplace=True)

    df_tab_one = pd.concat([df_stats_bin, df_stats_cont], axis=0)
    df_tab_one.reset_index(drop=True, inplace=True)
    # %% correct for multiple comparisosn test

    # Apply Benjamini-Hochberg FDR correction
    p_val_idx = df_tab_one[~df_tab_one['p-value'].isna()].index
    raw_p_array = np.array(df_tab_one.loc[p_val_idx,'p-value'] , dtype=float)
    _, p_fdr_array, _, _ = multipletests(raw_p_array, alpha=0.05, method='fdr_bh')

    # Insert FDR-adjusted p-values back into df_results
    df_tab_one['p-value FDR'] = '-'
    df_tab_one.loc[p_val_idx, 'p-value FDR'] = p_fdr_array
    # Format the FDR-adjusted p-values
    formatted_fdr = [
        (f"{pv:.4f}" if pv >= 0.0001 else f"{pv:.4e}") for pv in p_fdr_array
    ]
    df_tab_one.loc[p_val_idx, 'p-value FDR formatted'] = formatted_fdr

    # %% organize the rows
    desired_order = [
        'Age',
        'BMI',
        'Gender (Male) (yes)',

        'Age sleep complaints',
        'ESS Score',
        'Naps',

        'MSLT Age',
        ' MSLT',
        'HLA-DQB1*06:02 (yes)',
        'REM latency',
        'Sleep latency',
        'SOREMP',
        'After athletic activities (yes)',
        'Angry (yes)',
        'Cataplexy medication (yes)',
        'Discipline children (yes)',
        'Disturbed nocturnal sleep (yes)',
        'During athletic activities (yes)',
        'During sexual intercourse (yes)',
        'Elated (yes)',
        'Embarrassed (yes)',
        'Excited (yes)',
        'Hallucinations (yes)',
        'Hallucinations Age Onset',
        'Have a romantic thought or moment (yes)',
        'Hear or tell a joke (yes)',
        'Laugh (yes)',
        'Moved by something emotional (yes)',
        'Playing an exciting game (yes)',
        'Quick response cataplexy (yes)',
        'Remember a happy moment (yes)',
        'Remember an emotional moment (yes)',
        'Sleep paralysis (yes)',
        'Sleep paralysis age onset',
        'Startled (yes)',
        'Stressed (yes)',
        'Tense (yes)',
        
        'Muscle weakness age onset',
        'Muscle weakness head and shoulder dropping (yes)',
        'Muscle weakness in hand and arms (yes)',
        'Muscle weakness jaw sagging (yes)',
        'Muscle weakness legs and knees (yes)',
        'Muscle weakness, speech becomes slurred (yes)',
     ]

    assert len(desired_order) == df_tab_one.shape[0]
    # missing = [item for item in desired_order if item not in df_tab_one['Variable'].values]
    df_tab_one = df_tab_one.set_index('Variable') \
        .reindex(desired_order) \
        .reset_index()

    # %% Inlcude the annotation
    def _annotate_name(row):
        base = row["Variable"]
        if row["Stat Method"] == "Mann-Whitney U test":
            return f"{base}^Δ"
        elif row["Stat Method"] == "Chi-Square Test":
            return f"{base}^‡"
        else:
            return f"{base}"


    df_tab_one["Variable"] = df_tab_one.apply(_annotate_name, axis=1)

    # df_tab_one = df_tab_one.sort_values(by='Variable', ascending=True).reset_index(drop=True)

    # %% save the tables
    df_tab_one.to_csv(config.get('results_path').get('results').joinpath('table_one_reduced.csv'), index=False,)
    # df_stats_bin.to_csv(config.get('results_path').get('results').joinpath('stats_bin.csv'), index=False,)
    # df_stats_cont.to_csv(config.get('results_path').get('results').joinpath('stats_cont.csv'), index=False,)


    # %% merge the table one with the statistics
    # df_tab_one_cont = pd.merge(df_tab_one,
    #                            df_stats_cont,
    #                            left_on='variable',
    #                            right_on='Variable',
    #                            how='left')
    # df_tab_one = pd.merge(df_tab_one, df_stats_bin, on='Variable', how='left')
    #
    #

