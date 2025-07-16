"""
Classs dedicated to create teh table one of the paper

Author: Giorgio Ricciardiello
contact: giocrm@stanford.edu
Year: 2024

"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List
from tabulate import tabulate
import re
class MakeTableOne:
    def __init__(self, df: pd.DataFrame,
                 continuous_var: list[str],
                 categorical_var: list[str],
                 strata: Optional[str]=None):
        """
        Class to easily create a table one from the dataframe.
        Important: The columns must not be of mixed data types e.g., strings and numeric. Only one type is accepted
        nans are ignored in all counts
        :param df:
        :param continuous_var:
        :param categorical_var:
        :param strata:

        usage:

            continuous_var = ['Age', 'BMI', 'ESS']
            categorical_var = ['sex', 'DQB10602', 'hla_positive']
            column_groups = 'Race'

            tableone_constructor = MakeTableOne(df=df_race,
                                            continuous_var=continuous_var,
                                            categorical_var=categorical_var,
                                            strata=column_groups)

            table_one = tableone_constructor.create_table()

                      variable      Caucasian  ...         Mixed       Other
            0            Count            512  ...             9           1
            1              Age  45.17 (16.39)  ...  31.44 (8.13)  30.0 (nan)
            2              BMI     28.0 (6.6)  ...  23.26 (3.63)  30.3 (nan)
            3              ESS   16.92 (4.59)  ...  16.78 (2.91)  16.0 (nan)
            4           sex__0   274 (53.52%)  ...    7 (77.78%)    0 (0.0%)
            5           sex__1   238 (46.48%)  ...    2 (22.22%)  1 (100.0%)
            6      DQB10602__0   225 (43.95%)  ...    3 (33.33%)  1 (100.0%)
            7      DQB10602__1    278 (54.3%)  ...    6 (66.67%)    0 (0.0%)
            8  hla_positive__0   161 (31.45%)  ...      0 (0.0%)  1 (100.0%)
            9  hla_positive__1   351 (68.55%)  ...    9 (100.0%)    0 (0.0%)

        """
        self.df = df
        self.continuous_var = continuous_var
        self.categorical_var = categorical_var
        self.strata = 'SingleDistributionTable' if strata is None else strata
        self.index = continuous_var + categorical_var
        self._check_columns_input()
        self.index_categorical = None
        self.index_mapper = {}
        self.tableone = self._create_empty_table()

    def create_table(self) -> pd.DataFrame:
        """Pipeline to create the table"""
        self._populate_count()
        self._populate_continuous_columns()
        self._populate_categorical_columns()
        if not self.index_mapper is None:
            self._re_map_indexes_to_strings()

        print(tabulate(self.tableone,
                       headers=[*self.tableone.columns],
                       showindex=False,
                       tablefmt="fancy_grid"))
        return self.tableone

    def _check_columns_input(self):
        """Check if all the inputs are in the dataset"""
        for col in self.index:
            if '__' in col:
                raise ValueError(f"Column {col} has __ in it's name, this will break the code."
                                 f" Please rename the column without the __ ")
            if not col in self.df.columns:
                raise ValueError(f'Column {col} is not in the dataframe')

    def _create_categorical_row_indexes(self) -> list[str]:
        """
        Categorical, ordinal, discrete columns will have one row per unique value with their count
        spawned over the columns. Here we create the row indexes for each of these columns
        :return:
        """
        index_categorical = []
        for cat_col in self.categorical_var:
            index_categorical.extend(self._create_index(self.df, col=cat_col))
        return index_categorical

    def _create_empty_table(self) -> pd.DataFrame:
        """
        Create the empty table one. The columns are the unique values of the strata and the rows are the
        continuous variables + the categorical variables expanded to all their unique values
        :return:
        """
        self.index_categorical = self._create_categorical_row_indexes()
        indexes = ['Count'] + self.continuous_var + self.index_categorical
        if self.strata != 'SingleDistributionTable':
            tableone = pd.DataFrame(index=indexes, columns=self.df[self.strata].unique())
        else:
            tableone = pd.DataFrame(index=indexes, columns=[self.strata])
        tableone.index.name = 'variable'
        tableone.reset_index(inplace=True)
        return tableone

    def _populate_count(self):
        if self.strata == 'SingleDistributionTable':
            self.tableone.loc[self.tableone['variable'] == 'Count', self.strata] = self.df.shape[0]
        else:
            for column_group_ in self.df[self.strata].unique():
                self.tableone.loc[self.tableone['variable'] == 'Count', column_group_] = (
                    self.df[self.df[self.strata] == column_group_].shape)[0]

    def _populate_continuous_columns(self):
        """
        Strata are the columns. Moves through the columns and calculates the metrics (mean ± sd) of the variable
        which is a row.
        """
        if self.strata == 'SingleDistributionTable':
            for cont_var in self.continuous_var:
                self.tableone.loc[self.tableone['variable'] == cont_var,
                self.strata] = self._continuous_var_dist(frame=self.df, col=cont_var)
        else:
            for column_group_ in self.df[self.strata].unique():
                for cont_var in self.continuous_var:
                    self.tableone.loc[self.tableone['variable'] == cont_var,
                    column_group_] = self._continuous_var_dist(frame=self.df[self.df[self.strata] == column_group_],
                                                               col=cont_var)

    def _populate_categorical_columns(self):
        """
        Strata are the columns. Moves through the columns and calculates the metrics (count, percent) of the variable
        which is a row.
        """
        if self.strata == 'SingleDistributionTable':
            for cat_ in self.index_categorical:
                col = cat_.split('__')[0]
                category = cat_.split('__')[1]
                self.tableone.loc[self.tableone['variable'] == cat_, self.strata] = self._categorical_var_dist(
                    frame=self.df, col=col, category=int(float(category)))
        else:
            for column_group_ in self.df[self.strata].unique():
                for cat_ in self.index_categorical:
                    col = cat_.split('__')[0]
                    category = cat_.split('__')[1]
                    self.tableone.loc[self.tableone['variable'] == cat_, column_group_] = self._categorical_var_dist(
                        frame=self.df[self.df[self.strata] == column_group_],
                        col=col,
                        category=int(float(category)))

    def _create_index(self, frame: pd.DataFrame,
                      col: str,
                      prefix: Optional[str] = None) -> list[str]:
        """
        Create the indexes that will be used as rows of table one based on the unique values of the current column.
        We place a prefix to differentiate it from other similar-named rows.
        Eg. gender becomes gender__0, gender__1
        :param frame: dataset
        :param col: column to map as indexes
        :param prefix: rename the column with a more suitable name.
        :return:
        """
        if prefix is None:
            prefix = col
        unique_vals = frame.dropna(subset=[col])[col].unique()
        if all([isinstance(val, str) for val in unique_vals]):
            unique_vals = self._map_index_str_to_int(prefix=prefix, unique_vals=unique_vals)
        elif any([isinstance(val, str) for val in unique_vals]):
            raise ValueError(f'The column {col} is of mixed data types, strings and numeric. Please set to one or the '
                             f'other')
        unique_vals.sort()
        return [f'{prefix}__{int(i)}' for i in unique_vals]

    def _map_index_str_to_int(self, prefix:str, unique_vals:list[str]) -> list[int]:
        """
        If the cell values are strings, we convert them to integers.
        :param prefix:
        :param unique_vals:
        :return:
        """
        self.index_mapper[prefix] = {int_name:name for name, int_name in enumerate(unique_vals)}
        self.df[prefix] = self.df[prefix].map(self.index_mapper[prefix])
        return list(self.index_mapper[prefix].values())

    def _re_map_indexes_to_strings(self):
        """
        re-map the variables that were all indexes to their original string names
        e.g.,
        If the column Race in df had unique values ['Caucasian', 'Black', 'Latino', 'Asian', 'Mixed', 'Other']
        rename the Race__numeric to their respective string names
        'Race__0': 'Caucasian',
         'Race__1': 'Black',
         'Race__2': 'Latino',
         'Race__3': 'Asian',
         'Race__4': 'Mixed',
         'Race__5': 'Other'}
        :return:
        """
        for preffix in self.index_mapper.keys():
            mapper = {f'{preffix}__{numeric}':string for string, numeric in self.index_mapper.get(preffix).items()}
            self.tableone['variable'] = self.tableone.variable.replace(mapper)

    @staticmethod
    def _continuous_var_dist(frame: pd.DataFrame,
                             col: str,
                             decimal: Optional[int] = 2) -> str:
        # cell = (f'({frame[col].count()})\n'
        #         f'{np.round(frame[col].mean(), decimal)}'
        #         u'\u00B1'
        #         f'{np.round(frame[col].std(), decimal)}')
        cell = (f'{np.round(frame[col].mean(), decimal)}'
                u'\u00B1'
                f'{np.round(frame[col].std(), decimal)}'
                f'({frame[col].count()})')
        return cell

    @staticmethod
    def _categorical_var_dist(
            frame: pd.DataFrame,
            col: str,
            category: Optional[Union[int, str]] = None,
            decimal: int = 2
    ) -> Union[str, List[str]]:
        """
        Computes counts and percentages for categorical variables in a DataFrame column.

        Parameters:
        - frame: DataFrame containing the data.
        - col: Column name for which to compute stats.
        - category: Optional specific category to report stats for.
        - decimal: Number of decimal places for percentages.

        Returns:
        - A string (if category specified) or list of strings (for all categories),
          formatted as: "count/valid_n (percent%) [missing=X]"
        """

        total_n = frame.shape[0]
        non_nan_series = frame[col].dropna()
        non_nan_count = non_nan_series.shape[0]
        missing = total_n - non_nan_count

        if category is not None:
            count = (frame[col] == category).sum()
            if non_nan_count > 0:
                percent = np.round((count / non_nan_count) * 100, decimal)
                # return f"({non_nan_count})\n {count}/{non_nan_count} ({percent}%)"
                return f"{percent}%({non_nan_count})"
            else:
                # all values are nans
                # return f"{count}/0 (-%) [missing={missing}]"
                return f"-% ({count}/0)]"
        else:
            counts = non_nan_series.value_counts().sort_index()
            results = []
            for level, count in counts.items():
                percent = np.round((count / non_nan_count) * 100, decimal)
                # results.append(f"{level}: {count}/{non_nan_count} ({percent}%) [missing={missing}]")
                results.append(f"{level}: {percent}\%({non_nan_count})")
            return results


    # def _categorical_var_dist(frame: pd.DataFrame,
    #                           col: str,
    #                           category: Optional[int] = None,
    #                           decimal: Optional[int] = 2,
    #                           ) -> Union[str, list]:
    #     """
    #     Count the number of occurrences in a column, giving he number of evens and the percentage. Used for category columns
    #
    #     :param frame: dataframe from there to compute the count on the columns
    #     :param col: column to compute the calculation of the count
    #     :param category: if we want to count a specific category of the categories
    #     :param decimal: decimal point to show in the table
    #     :return:
    #     """
    #     if category is not None:
    #         count = frame.loc[frame[col] == category, col].shape[0]
    #         non_nan_count = frame[col].count()  # frame.loc[~frame[col].isna()].shape[0]  # use the non-nan count
    #         if frame.shape[0] == 0:
    #             return f'0'
    #         else:
    #             if non_nan_count > 0:
    #                 cell = (f'{count}/{non_nan_count}'
    #                         f'({np.round((count / non_nan_count) * 100, decimal)}%)')
    #             else:
    #                 cell = f'{count} (-%)'
    #
    #         return cell
    #     else:
    #         # return the count ordered by the index
    #         count = frame[col].value_counts()
    #         count = count.sort_index()
    #         non_nan_count = frame.loc[~frame[col].isna()].shape[0]  # use the non-nan count
    #
    #         if count.shape[0] == 1:
    #             # binary data so counting the ones
    #             if non_nan_count > 0:
    #                 cell = (f'{count[1]} '
    #                         f'({np.round((count[1] / non_nan_count) * 100, decimal)}%)'
    #                         f'{[non_nan_count]}')
    #             else:
    #                 cell = f'{count} (0%)'
    #
    #             return cell
    #         else:
    #             if non_nan_count > 0:
    #                 cell = [f'{count_} (0%) {[non_nan_count]}' for count_ in count]
    #             else:
    #                 cell = [f'{count_} (0%) {[count]}' for count_ in count]
    #             return cell

    def remove_reference_categories(self):
        """
        Remove the reference group or zero group from the categories
        :return:
        """
        rmv_idx = self.tableone[self.tableone['variable'].str.contains("__0")].index
        self.tableone.drop(index=rmv_idx, inplace=True)

    def get_table(self):
        return self.tableone

    @staticmethod
    def group_variables_table(df: pd.DataFrame) -> pd.DataFrame:
        """
        Group the variable into their responses e.g.,
        rows
        cir_0200__1,
        cir_0200__2,
        cir_0200__3 becomes:

        cir_0200
            1,
            2,
            3
        :param df:
        :return:
        """
        # Create an empty DataFrame to store the transformed data
        df_transformed = pd.DataFrame(columns=df.columns)

        # Track previous group to avoid duplicate headers
        prev_group = None

        for i, row in df.iterrows():
            # Extract the prefix if the variable matches <name>__<int>
            prefix_match = row['variable'].split('__')[0] if '__' in row['variable'] else None

            # Check if a header row is needed
            if prefix_match and prefix_match != prev_group:
                # Insert a header row for the prefix
                prefix_row = [prefix_match] + ['--'] * (len(df.columns)-1)
                header_row = pd.DataFrame([prefix_row], columns=df.columns)
                df_transformed = pd.concat([df_transformed, header_row], ignore_index=True)
                prev_group = prefix_match

            # Add the current row, removing suffix if it's part of a group
            if prefix_match:
                row['variable'] = row['variable'].replace(f"{prefix_match}__", "")

            df_transformed = pd.concat([df_transformed, pd.DataFrame([row])], ignore_index=True)

        return df_transformed

    @staticmethod
    def merge_levels_others(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        List of columns to sum, excluding the first element
        :param df:
        :param columns:
        :return:
        """
        col_head_index = df.loc[df['variable'] == columns[0]].index[0]
        columns = columns[1:]

        def extract_and_sum_percentages(column_values):
            """
            Iterate over the column values, and it will sum the percentage count str(<int> (<float>%))
            :param column_values:
            :return:
            """
            # Initialize totals for counts outside and inside parentheses
            count_total = 0.0
            percent_total = 0.0

            for value in column_values:
                # Extract the numbers outside and inside the parentheses
                match = re.match(r"(\d+(?:\.\d+)?)\s*\((\d+(?:\.\d+)?)%\)", str(value))
                if match:
                    count_value = float(match.group(1))
                    percent_value = float(match.group(2))

                    # Sum these extracted values
                    count_total += count_value
                    percent_total += percent_value

            # Format the result as a string similar to the original format
            return f"{count_total} ({percent_total:.2f}%)"

        df_subset = df.loc[df['variable'].isin(columns), :]
        rows_other = {}
        for col_ in df.columns:
            if col_ == 'variable':
                continue
            rows_other[col_] = extract_and_sum_percentages(df_subset[col_])

        final_row = pd.DataFrame([rows_other], columns=df_subset.columns)
        final_row['variable'] = ' Other'
        # Reconstruct the dataframe with 'Other' row inserted below the column header row
        df_updated = df.copy()
        df_updated = df_updated.drop(index=df_subset.index)
        df_updated = pd.concat([df_updated.iloc[:col_head_index + 1],
                                final_row,
                                df_updated.iloc[col_head_index + 1:]],
                               ignore_index=True)
        return df_updated