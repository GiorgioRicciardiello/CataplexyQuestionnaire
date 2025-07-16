import pathlib
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dataclasses import dataclass
from typing import Optional

@dataclass
class Datasets:
    def __init__(self,
                 train_X: pd.DataFrame,
                 train_y: pd.Series,
                 test_X: Optional[pd.DataFrame] = None,
                 test_y: Optional[pd.Series] = None,
                 valid_X: Optional[pd.DataFrame] = None,
                 valid_y: Optional[pd.Series] = None):
        self.train_X = train_X
        self.valid_X = valid_X
        self.test_X = test_X
        self.train_y = train_y
        self.valid_y = valid_y
        self.test_y = test_y
        self._create_splits_dict()

    def _create_splits_dict(self):
        self.splits = {
            'train_X': self.train_X,
            'valid_X': self.valid_X,
            'test_X': self.test_X,
            'train_y': self.train_y,
            'valid_y': self.valid_y,
            'test_y': self.test_y
        }

    def plot_stratified_distribution(self,
                                     output_path: Union[pathlib.Path, None]=None,
                                     show_plot:Optional[bool]=True,
                                     save_plot:Optional[bool]=True,
                                     ):
        """
        Plot the stratified target (categorical/ordinal) as a bar plot. The  x axis contains the train, validation, and
        test split. Each x-ticks has the bar of the count of each class in the split
        :return:
        """
        splits_target = {key: item for key, item in self.splits.items() if 'y' in key and item is not None}
        splits_count = {}
        for lbl_, split_ in splits_target.items():
            splits_count[lbl_] = split_.value_counts().to_dict()
        # Sorting each inner dictionary by its keys
        splits_count = {outer_k: dict(sorted(outer_v.items())) for outer_k, outer_v in splits_count.items()}

        df_splits_count = pd.DataFrame(splits_count)
        df_melted = df_splits_count.reset_index().melt(id_vars='index',
                                                       var_name='split',
                                                       value_name='count')
        df_melted.rename(columns={'index': 'class'},
                         inplace=True)

        # Now we can create a seaborn barplot with splits on the x-axis and count on the y-axis
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_melted,
                    x='split',
                    y='count',
                    hue='class')
        plt.title('Counts of Classes across Different Splits')
        plt.xlabel('Split')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(0.7)
        if save_plot and output_path is not None:
            plt.savefig(output_path.joinpath('Distribution_Model.png'), dpi=300)
        if show_plot:
            plt.show()

    def plot_distribution(self):
        """
        Plot how the train, validation, and test sets are distributed
        :return:
        """
        # Calculate the percentage of each class in the datasets
        train_percentages = self.train_y.value_counts(normalize=True) * 100
        valid_percentages = self.valid_y.value_counts(normalize=True) * 100
        test_percentages = self.test_y.value_counts(normalize=True) * 100

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 7))
        index = np.arange(len(train_percentages))
        bar_width = 0.25

        bar1 = ax.bar(index, train_percentages, bar_width, label='Train')
        bar2 = ax.bar(index + bar_width, valid_percentages, bar_width, label='Validation')
        bar3 = ax.bar(index + 2 * bar_width, test_percentages, bar_width, label='Test')

        ax.set_xlabel('Class')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Percentage of patients in each class per data split - Stratified')
        ax.set_xticks(index + bar_width)
        ax.set_xticklabels(train_percentages.index)
        ax.legend()

        # Adding the percentages on top of the bars
        for bar in bar1 + bar2 + bar3:
            height = bar.get_height()
            ax.annotate('%.2f%%' % height,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def get_shape(self):
        return {
            'train_X': self.train_X.shape,
            'train_y': self.train_y.shape,
            'test_X': self.test_X.shape if self.test_X is not None else None,
            'test_y': self.test_y.shape if self.test_y is not None else None,
            'valid_X': self.valid_X.shape if self.valid_X is not None else None,
            'valid_y': self.valid_y.shape if self.valid_y is not None else None,
        }

    def get_info(self):
        return {
            'train_X' : self.train_X.info(),
            'valid_X' : self.valid_X.info() if self.valid_X is not None else None,
            'test_X' : self.test_X.info(),
            'train_y' : "Series size : " + str(self.train_y.size),
            'valid_y' : "Series size : " + str(self.valid_y.size) if self.valid_X is not None else 'None',
            'test_y' : "Series size : " + str(self.test_y.size)
        }

    def get_describe(self):
        return {
            'train_X': self.train_X.describe(),
            'valid_X': self.valid_X.describe() if self.valid_X is not None else None,
            'test_X': self.test_X.describe(),
            'train_y': self.train_y.describe(),
            'valid_y': self.valid_y.describe() if self.valid_y is not None else None,
            'test_y': self.test_y.describe()
        }

    def plot_distribution_2(self, layout: Optional[str] = 'vertical_stack', palette: Optional[str] = 'Set2'):
        """
        Plot distribution of target column for different splits.

        :param layout: Layout of the subplots, either 'stacked' or 'side_by_side'.
        :param palette: Seaborn color palette to use for the plots.
        """
        # Determine the number of subplots based on the layout
        if layout == 'side_by_side':
            nrows, ncols = 1, 3
            figsize = (18, 6)  # Wider figure for side-by-side layout
        elif layout == 'vertical_stack':
            nrows, ncols = 3, 1
            figsize = (6, 12)  # Taller figure for stacked layout
        else:
            raise ValueError("Invalid layout. Choose 'vertical_stack' or 'side_by_side'.")

        # Create subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        # Plot for each split
        splits = [('Train', self.train_y), ('Validation', self.valid_y), ('Test', self.test_y)]
        for idx, (split, data) in enumerate(splits):
            if data is not None:
                sns.histplot(data, kde=True, ax=axes[idx], palette=palette)
                axes[idx].set_title(f'{split} ({data.shape[0]})')
            else:
                axes[idx].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
                axes[idx].set_title(f'{split} (No Data)')
                axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()

    def count_nans(self):
        nan_count = {
            'train_X' : self.train_X.isna().sum(),
            'valid_X' : self.valid_X.isna().sum(),
            'test_X' : self.test_X.isna().sum(),
            'train_y' : self.train_y.isna().sum(),
            'valid_y' : self.valid_y.isna().sum(),
            'test_y' : self.test_y.isna().sum()
        }

        return nan_count