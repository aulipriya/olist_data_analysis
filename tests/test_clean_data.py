import numpy as np
import pandas as pd
from src.clean_data import DataCleaner
import pytest

class TestDataCleaner:
    def test_convert_dtypes_converts_date_column(self):
        df = pd.DataFrame({'purchase_date': ["2023-01-01", "2023-12-31"]})
        result = DataCleaner()._convert_dtypes(df)
        assert result['purchase_date'].dtype == 'datetime64[ns]'

    def test_convert_dtypes_converts_object_to_category(self):
        df = pd.DataFrame({'status': ["enabled", "disabled", "running"]})
        result = DataCleaner()._convert_dtypes(df)
        assert result['status'].dtype == 'category'

    def test_convert_dtypes_downcasts_small_int_to_uint8(self):
        df = pd.DataFrame({"count": [1, 5, 200]})
        result = DataCleaner()._convert_dtypes(df)
        assert result["count"].dtype == "uint8"

    def test_drop_missing_drops_high_null_columns(self):
        df = pd.DataFrame({'purchase_date': [None, None, None, None, None, None, None, None, None, 1], 'id': list(range(10))})
        result = DataCleaner()._drop_missing_value_columns(df)
        assert 'purchase_date' not in result.columns

    def test_drop_missing_drops_low_null_columns(self):
        df = pd.DataFrame({'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'score': [4, 4, 4, 4, 5, 5, 4, 5, None, None]})
        result = DataCleaner()._drop_missing_value_columns(df)
        assert 'score' in result.columns

    def test_drop_missing_drops_rows_in_important_column(self):
        data = {'id': list(range(100)), 'score': [float(i) for i in range(100)]}
        data['score'][5] = None
        df = pd.DataFrame(data)
        result = DataCleaner()._drop_missing_value_columns(df)
        assert len(result) == 99

    def test_fix_date_converts_valid_date_to_datetime64(self):
        data = pd.DataFrame({'purchase_date': ["2023-01-01", "2023-12-31"]})
        result = DataCleaner()._fix_datetime_columns(data, ['purchase_date'])
        assert result['purchase_date'].dtype == 'datetime64[ns]'

    def test_fix_date_converts_invalid_date_column_to_nat(self):
        data = pd.DataFrame({'purchase_date': ['2023-01-01', '2023-12-31', 'baddate']})
        result = DataCleaner()._fix_datetime_columns(data, ['purchase_date'])
        assert pd.isna(result['purchase_date'].iloc[2])

    def test_run_cleaning_writes_output_files(self, tmp_path):
        input_dir = tmp_path / 'input'
        input_dir.mkdir()
        pd.DataFrame({"id": [1, 2], "status": ['active', 'inactive']}).to_csv(input_dir / 'orders.csv', index=False)
        pd.DataFrame({"id": [3, 4], "status": ['active', 'inactive']}).to_csv(input_dir / 'customers.csv', index=False)

        output_dir = tmp_path / 'cleaned'

        cleaner = DataCleaner(data_directory=str(input_dir))
        cleaner.run_cleaning(output_directory=str(output_dir))
        output_files = list(output_dir.glob('*.csv'))
        assert len(output_files) == 2
        assert (output_dir / "orders_cleaned.csv").exists()
        assert (output_dir / "customers_cleaned.csv").exists()


