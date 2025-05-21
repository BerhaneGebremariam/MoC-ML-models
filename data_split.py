from __future__ import annotations
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from explanatory_analysis import SpDataCleaner
import typing as tp

#****************************************************************************************************************
# the categorical variables in the data are first converted to dummies so that both RF and ANN models are trained on the same data set,
## NB:
#*** as an alternative Embedding layers could be used in ANN models to handle categorical
#*** variables(the impact of this method on the performance of the ANN models is not considered for this study).
# this study aims to lay a groundwork for future studies that will compare the performance of different models on the same data set structure.

sp = SpDataCleaner()
data_frame = sp.to_dummies()

#****************************************************************************************************************

class TrainValTestSplitter:
    """

    The main objective of this class is to split the data into train, validation and test sets.
    These splits are designed in such a way that they are reproducible so that the RF and ANN models can be trained on the same data sets.
    Besides in order to compare the performance of models between random validation,grouped test and grouped validation,
    grouped test the test set is reproducible and both models are tested in the same data set
    [for comparison between (M3RF,M5RF),(M4RF,M6RF),(M9RF,M11RF),(M10RF,M12RF),
    (M3ANN,M5ANN),(M4ANN,M6ANN),(M9ANN,M11ANN),(M10ANN,M12ANN)].
    """


    def __init__(self, random_state: int = 42, df: pd.DataFrame = data_frame, target_col: str = 'choice', test_size: float = 0.2,
                 val_size: float = 0.25):
        self.df = df
        self.target_col = target_col
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.train_val_df, self.test_df = self.unique_test()

    def random_split(self)-> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Randomly split the data into train, validation and test sets for a given split sizes.

        :return: dataframes and series for train, validation and test sets
        """
        X = self.df.drop(columns=[self.target_col], axis=1)
        y = self.df[self.target_col]
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=self.test_size,
                                                                    random_state=self.random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=self.val_size,
                                                          random_state=self.random_state)

        X_val = X_val.drop(columns=['ResponseId'], axis=1)
        X_test = X_test.drop(columns=['ResponseId'], axis=1)
        return X_train_val, X_train, X_val, X_test, y_train_val, y_train, y_val, y_test

    def unique_test(self, group_col: str = 'ResponseId')-> tp.Tuple[pd.DataFrame, pd.DataFrame]:
        """
        this is a function used to have the same test data set for both grouped_tes_random_val and
        grouped_test_grouped_val models.
        :param group_col: for this study individual responses are grouped by ResponseId
        :return:
        """
        unique_ind = list(self.df[group_col].unique())

        random.seed(self.random_state)
        np.random.seed(self.random_state)
        test_ind = np.random.choice(unique_ind, int(len(unique_ind) * self.test_size), replace=False)
        test_df = self.df[self.df[group_col].isin(test_ind)]

        train_val_df = self.df[~self.df[group_col].isin(test_ind)]

        return train_val_df, test_df

    def grouped_test_random_val(self)-> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        This method prepares randomly selected train and validation sets and a grouped test set from the unique_test method.
        :return:
        """
        train_val_df = self.train_val_df
        test_df = self.test_df
        X_train_val = train_val_df.drop(columns=[self.target_col], axis=1)
        y_train_val = train_val_df[self.target_col]
        X_test = test_df.drop(columns=[self.target_col,'ResponseId'], axis=1)
        y_test = test_df[self.target_col]
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=self.val_size,
                                                          random_state=self.random_state)
        X_val= X_val.drop(columns=['ResponseId'], axis=1)
        return X_train_val, X_train, X_val, X_test, y_train_val, y_train, y_val, y_test

    def grouped_test_grouped_val(self, group_col: str = 'ResponseId')-> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        This method prepares grouped train, validation and test sets.
        :param group_col: for this study individual responses are grouped by ResponseId
        :return:
        """
        train_val_df = self.train_val_df
        test_df = self.test_df

        unique_ind = list(train_val_df[group_col].unique())
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        val_ind = np.random.choice(unique_ind, int(len(unique_ind) * self.val_size), replace=False)
        val_df = train_val_df[train_val_df[group_col].isin(val_ind)]
        train_df = train_val_df[~train_val_df[group_col].isin(val_ind)]

        X_train_val = train_val_df.drop(columns=[self.target_col], axis=1)
        y_train_val = train_val_df[self.target_col]
        X_train = train_df.drop(columns=[self.target_col], axis=1)
        y_train = train_df[self.target_col]
        X_val = val_df.drop(columns=[self.target_col,'ResponseId'], axis=1)
        y_val = val_df[self.target_col]
        X_test = test_df.drop(columns=[self.target_col,'ResponseId'], axis=1)
        y_test = test_df[self.target_col]

        return X_train_val, X_train, X_val, X_test, y_train_val, y_train, y_val, y_test


def data_splits(random_state:int = 42) -> tp.List[tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]]:
    """
    This function is used to generate the different data splits for the RF and ANN models.
    :param random_state: for reproducibility purposes
    :return: list of data split dataframes and series
    """

    random.seed(random_state)
    np.random.seed(random_state)
    splitter = TrainValTestSplitter(random_state=random_state)
    random_split = splitter.random_split()
    grouped_test_random_val = splitter.grouped_test_random_val()
    grouped_test_grouped_val = splitter.grouped_test_grouped_val()

    return [random_split, grouped_test_random_val, grouped_test_grouped_val]


def main():
    data_splits()


if __name__ == '__main__':
    main()
