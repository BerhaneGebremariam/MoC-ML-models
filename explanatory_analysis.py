from __future__ import annotations
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import typing as tp

class SpDataCleaner:
    """
    This class is used to clean the data from the moco_spdata.csv file.
    The data is read from the file and the following steps are performed:
    - Subtract 1 from the column 'choice' so that the RF and ANN models can be trained on the same data(doesn't affect the analysis but the ANN model needs the target variable to start from 0)
    - Drop columns that are not needed for the analysis
    - Fill missing values in the columns 'hh_children' and 'p_outofhome' so that they can be used for the analysis instead of dropping them
    - Drop columns that are not needed for the analysis
    - Convert columns with categorical values to dummy variables
    - Replace values in the columns that are not allowed in the column names such as '\', ' ', '/', '-'
    - Return the cleaned data frame
    """

    def __init__(self, file_name: str = 'moco_spdata.csv'):
        self.df = pd.read_csv(file_name)
        self.columns_to_drop: tp.List[str] = ['Phase', 'exp_name', 'p_dateofbirth_1', 'cs', 'dc', 'dc_name', 'choice_name',
                                           'p_homeloc', 'p_risk', 'p_climate', 'p_justice_self', 'p_justice_others',
                                           'mob_activity_work', 'mob_activity_leisure', 'mob_activity_errand',
                                           'mob_distclass_work', 'mob_distclass_leisure', 'mob_distclass_errand',
                                           'mob_modeaccess_1', 'mob_modeaccess_2', 'mob_modeaccess_3',
                                           'mob_modeaccess_4', 'mob_modeaccess_5', 'mob_modeaccess_6',
                                           'mob_modeaccess_7','mob_modeusage_car', 'mob_modeusage_bicycle', 'mob_modeusage_pt',
                                          'mob_modeusage_walk']

        self.col_to_dummies: tp.List[str] = ['trip_purp', 'quality_b', 'weather', 'p_gender', 'p_working', 'p_educ',
                                          'p_inc', 'p_homemunich', 'p_driverslicense', 'p_ptmobtool', 'p_nineeuro',
                                          'hh_size', 'hh_caravail', 'exam_question', 'hh_children', 'p_outofhome',
                                          ]
        self.choice_from_zero()
        self.fill_na()
        self.drop_columns()


    def choice_from_zero(self)-> None:
        self.df['choice'] = self.df['choice'] - 1

    def fill_na(self):
        self.df['hh_children'] = self.df['hh_children'].fillna('keine_angabe')
        self.df['p_outofhome'] = self.df['p_outofhome'].fillna('keine_angabe')


    def replace_values(self)-> None:
        to_replace = ["\\", " ", "/", "-"]
        for i in to_replace:
            col = list(self.df.columns)
            new_col = [x.replace(i, "_") for x in col]
            self.df.columns = new_col

    def drop_columns(self)-> None:
        self.df.drop(columns=self.columns_to_drop, inplace=True)

    def to_dummies(self)-> pd.DataFrame:

        self.df = pd.get_dummies(self.df, columns=self.col_to_dummies, drop_first=True,dtype='int64')
        self.replace_values()
        return self.df


class SpDataVisualizer:
    """
    This class is used to visualize the cleaned moco spdata.
    """

    def __init__(self, sp: SpDataCleaner = SpDataCleaner()):
        self.sp = sp
        self.df = sp.df

    def general_styling(self)-> None:
        sns.set_style('white')
        sns.set_context('notebook')
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['figure.titlesize'] = 14
        #plt.figure(figsize=(10, 6))
        plt.tight_layout()


    def count_plot(self, column: str = 'p_driverslicense',hue: str= 'choice', palette: str = 'viridis')-> None:
        self.general_styling()
        sns.countplot(x=column, data=self.df, palette=palette, order=self.df[column].value_counts().index, alpha=0.7, edgecolor='black', linewidth=1, saturation=0.8, dodge=True, orient='v', hue=hue)
        plt.show()

    def box_plot(self, x: str = 'choice', y: str = 'p_age', x_label: str = 'Mode Choice', y_label: str = 'Age',hue: str = 'exp_nr',palette: str = 'viridis',):
        self.general_styling()
        sns.boxplot(x=x, y=y, data=self.df, palette=palette, hue=hue)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        #customize legend content
        plt.show()

    def corr_plot(self, range_1: int = -20, range_2: int = -3)-> None:
        sp = self.sp
        sp.to_dummies()
        sp.replace_values()
        df = sp.df
        df = df.drop(columns=['ResponseId'])
        self.general_styling()
        print(df.corr()['choice'].sort_values()[range_1:range_2])
        df.corr()['choice'].sort_values()[range_1:range_2].plot(kind='bar')
        sns.heatmap(df.corr(), annot=False, cmap='viridis')
        # increase the size of the borders so that the whole heatmap is visible and the labels are not cut off
        plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
        plt.show()

    def joint_plot(self, y: str = 'trip_dist', x: str = 'choice')-> None:
        self.general_styling()
        sns.jointplot(x=x, y=y, data=self.df)
        plt.show()



def main():
    sp_vis = SpDataVisualizer()
    sp_vis.corr_plot()


if __name__ == '__main__':

   main()

