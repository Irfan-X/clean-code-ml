import unittest
import pandas as pd
import numpy as np
from src.workshop import add, impute_nans
from pandas.testing import assert_frame_equal
# from preprocessing import add_derived_title, categorize_column, add_is_alone_column, impute_nans


class TestWorkshop(unittest.TestCase):
    def test_add_1_should_return_2(self):
        ## Triple 'A'
        # Arrange
        expected = 2
        # Act
        actual = add(1, 1)
        # Assert
        self.assertEqual(expected, actual)

    def test_df_should_equal_itself(self):
        # Arrange
        df_1 = pd.DataFrame({
            'column_1': [1, 2, 3]
        })
        df_2 = pd.DataFrame({
            'column_1': [1, 2, 3]
        })
        # Assert
        assert_frame_equal(df_1, df_2)

    def test_impute_nans_should_fill_nans_median_value(self):
        # Arrange
        df_1 = pd.DataFrame({
            'some_column': [1, np.nan]
        })
        expected = pd.DataFrame({
            'some_column': [1., 1.]
        })
        # Act
        actual = impute_nans(df_1, columns=['some_column'])
        # Assert
        assert_frame_equal(expected, actual)
