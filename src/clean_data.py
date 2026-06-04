import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# Configure logging for data cleaning operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    A comprehensive data cleaning class for Olist e-commerce dataset.

    This class provides methods for cleaning, validating, and transforming
    multiple CSV files with standardized approaches.
    """

    def __init__(self, data_directory: str = 'data/archive', verbose: bool = True):
        """
        Initialize the DataCleaner.

        Args:
            data_directory (str): Path to directory containing CSV files
            verbose (bool): Whether to print detailed cleaning information
        """
        self.data_directory = Path(data_directory)
        self.verbose = verbose
        self.cleaned_data = {}
        self.cleaning_report = {}

    def _log_info(self, message: str) -> None:
        """Log information if verbose mode is enabled."""
        if self.verbose:
            logger.info(message)

    def _fix_datetime_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Convert specified columns to datetime format.

        Args:
            df (pd.DataFrame): Input dataframe
            columns (List[str]): List of column names to convert

        Returns:
            pd.DataFrame: DataFrame with converted datetime columns
        """
        df_copy = df.copy()
        for column in columns:
            if column in df_copy.columns:
                original_null_count = df_copy[column].isnull().sum()
                df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce')
                new_null_count = df_copy[column].isnull().sum()

                if new_null_count > original_null_count:
                    self._log_info(f"Warning: {new_null_count - original_null_count} values became NaT in column '{column}'")

        return df_copy

    def _check_missing_values(self, df: pd.DataFrame, name: str) -> Dict[str, int]:
        """
        Check and report missing values in dataframe.

        Args:
            df (pd.DataFrame): Input dataframe
            name (str): Name of the dataset

        Returns:
            Dict[str, int]: Dictionary of column names and missing value counts
        """
        missing_values = df.isnull().sum().to_dict()

        if self.verbose:
            self._log_info(f"Missing values in {name}:")
            for col, count in missing_values.items():
                if count > 0:
                    percentage = (count / len(df)) * 100
                    print(f"  {col}: {count} ({percentage:.2f}%)")

        return missing_values

    def _check_data_types(self, df: pd.DataFrame, name: str) -> Dict[str, str]:
        """
        Check and report data types in dataframe.

        Args:
            df (pd.DataFrame): Input dataframe
            name (str): Name of the dataset

        Returns:
            Dict[str, str]: Dictionary of column names and their data types
        """
        data_types = df.dtypes.astype(str).to_dict()

        if self.verbose:
            self._log_info(f"Data types in {name}:")
            for col, dtype in data_types.items():
                print(f"  {col}: {dtype}")

        return data_types

    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically convert data types for optimization.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: DataFrame with optimized data types
        """
        df_copy = df.copy()

        # Identify and convert datetime columns
        date_cols = [c for c in df_copy.columns if any(keyword in c.lower()
                    for keyword in ['date', 'timestamp', '_at'])]
        df_copy = self._fix_datetime_columns(df_copy, date_cols)

        # Convert low-cardinality object columns to category
        for column in df_copy.columns:
            if (df_copy[column].dtype == 'object' and
                df_copy[column].nunique() < 10 and
                df_copy[column].nunique() > 1):
                df_copy[column] = df_copy[column].astype('category')
                self._log_info(f"Converted '{column}' to category type")

        # Convert numeric columns to more efficient types
        for column in df_copy.select_dtypes(include=['int64']).columns:
            col_min = df_copy[column].min()
            col_max = df_copy[column].max()

            if col_min >= 0:  # Unsigned integers
                if col_max <= 255:
                    df_copy[column] = df_copy[column].astype('uint8')
                elif col_max <= 65535:
                    df_copy[column] = df_copy[column].astype('uint16')
                elif col_max <= 4294967295:
                    df_copy[column] = df_copy[column].astype('uint32')
            else:  # Signed integers
                if col_min >= -128 and col_max <= 127:
                    df_copy[column] = df_copy[column].astype('int8')
                elif col_min >= -32768 and col_max <= 32767:
                    df_copy[column] = df_copy[column].astype('int16')
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    df_copy[column] = df_copy[column].astype('int32')

        return df_copy

    def _drop_missing_value_columns(self, df: pd.DataFrame,
                                   high_missing_threshold: float = 0.9,
                                   low_missing_threshold: float = 0.05) -> pd.DataFrame:
        """
        Drop columns and rows based on missing value thresholds.

        Args:
            df (pd.DataFrame): Input dataframe
            high_missing_threshold (float): Threshold for dropping columns with high missing values
            low_missing_threshold (float): Threshold for dropping rows with missing values

        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df_copy = df.copy()
        original_shape = df_copy.shape

        # Drop columns with high percentage of missing values
        columns_to_drop = df_copy.columns[df_copy.isnull().mean() >= high_missing_threshold]
        if len(columns_to_drop) > 0:
            self._log_info(f"Dropping columns with >{high_missing_threshold*100}% missing values: {list(columns_to_drop)}")
            df_copy = df_copy.drop(columns=columns_to_drop)

        # Drop rows with missing values in important columns (low missing percentage)
        important_columns = df_copy.columns[df_copy.isnull().mean() < low_missing_threshold]
        if len(important_columns) > 0:
            rows_before = len(df_copy)
            df_copy = df_copy.dropna(subset=list(important_columns))
            rows_dropped = rows_before - len(df_copy)
            if rows_dropped > 0:
                self._log_info(f"Dropped {rows_dropped} rows with missing values in important columns")

        final_shape = df_copy.shape
        self._log_info(f"Shape change: {original_shape} → {final_shape}")

        return df_copy

    def clean_single_dataset(self, file_path: Path,
                           high_missing_threshold: float = 0.9,
                           low_missing_threshold: float = 0.05) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean a single dataset file.

        Args:
            file_path (Path): Path to the CSV file
            high_missing_threshold (float): Threshold for dropping columns
            low_missing_threshold (float): Threshold for dropping rows

        Returns:
            Tuple[pd.DataFrame, Dict]: Cleaned dataframe and cleaning report
        """
        name = file_path.stem
        self._log_info(f"Starting to clean dataset: {name}")

        # Load data
        df = pd.read_csv(file_path)
        original_shape = df.shape

        # Initialize cleaning report
        report = {
            'original_shape': original_shape,
            'original_dtypes': self._check_data_types(df, f"{name} (original)"),
            'original_missing': self._check_missing_values(df, f"{name} (original)")
        }

        # Convert data types
        df = self._convert_dtypes(df)
        report['converted_dtypes'] = self._check_data_types(df, f"{name} (converted)")

        # Handle missing values
        df = self._drop_missing_value_columns(df, high_missing_threshold, low_missing_threshold)
        report['final_missing'] = self._check_missing_values(df, f"{name} (final)")
        report['final_shape'] = df.shape

        # Calculate memory usage improvement
        memory_before = sum(report['original_dtypes'][col] == 'object' for col in df.columns) * 8  # Rough estimate
        memory_after = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        report['memory_usage_mb'] = memory_after

        self._log_info(f"Completed cleaning {name}: {original_shape} → {df.shape}")

        return df, report

    def clean_all_datasets(self,
                          high_missing_threshold: float = 0.9,
                          low_missing_threshold: float = 0.05) -> Dict[str, pd.DataFrame]:
        """
        Clean all CSV files in the data directory.

        Args:
            high_missing_threshold (float): Threshold for dropping columns
            low_missing_threshold (float): Threshold for dropping rows

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of cleaned dataframes
        """
        if not self.data_directory.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_directory}")

        csv_files = list(self.data_directory.glob('*.csv'))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_directory}")

        self._log_info(f"Found {len(csv_files)} CSV files to clean")

        for csv_file in csv_files:
            name = csv_file.stem
            try:
                cleaned_df, report = self.clean_single_dataset(
                    csv_file, high_missing_threshold, low_missing_threshold
                )
                self.cleaned_data[name] = cleaned_df
                self.cleaning_report[name] = report

            except Exception as e:
                logger.error(f"Error cleaning {name}: {str(e)}")
                continue

        self._log_info(f"Successfully cleaned {len(self.cleaned_data)} datasets")
        return self.cleaned_data

    def get_cleaning_summary(self) -> pd.DataFrame:
        """
        Get a summary of all cleaning operations performed.

        Returns:
            pd.DataFrame: Summary of cleaning operations
        """
        if not self.cleaning_report:
            return pd.DataFrame()

        summary_data = []
        for name, report in self.cleaning_report.items():
            summary_data.append({
                'dataset': name,
                'original_rows': report['original_shape'][0],
                'original_cols': report['original_shape'][1],
                'final_rows': report['final_shape'][0],
                'final_cols': report['final_shape'][1],
                'rows_dropped': report['original_shape'][0] - report['final_shape'][0],
                'cols_dropped': report['original_shape'][1] - report['final_shape'][1],
                'memory_usage_mb': report.get('memory_usage_mb', 0)
            })

        return pd.DataFrame(summary_data)

    def save_cleaned_data(self, output_directory: str = '../data/cleaned') -> None:
        """
        Save all cleaned datasets to CSV files.

        Args:
            output_directory (str): Directory to save cleaned files
        """
        output_path = Path(output_directory)
        output_path.mkdir(exist_ok=True)

        for name, df in self.cleaned_data.items():
            output_file = output_path / f"{name}_cleaned.csv"
            df.to_csv(output_file, index=False)
            self._log_info(f"Saved cleaned {name} to {output_file}")

    def run_cleaning(self, output_directory: str = '../data/cleaned') -> None:
        self.clean_all_datasets()
        self._log_info("\n" + "=" * 50)
        self._log_info("CLEANING SUMMARY")
        self._log_info("=" * 50)
        summary = self.get_cleaning_summary()
        self._log_info(summary.to_string(index=False))
        self.save_cleaned_data(output_directory)


def main():
    """Main function to demonstrate DataCleaner usage."""
    # Initialize the data cleaner
    cleaner = DataCleaner(data_directory='../data/archive', verbose=True)

    cleaner.run_cleaning()



if __name__ == "__main__":
    main()


