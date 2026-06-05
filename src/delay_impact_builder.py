import pandas as pd
import  logging
from pathlib import Path

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class DelayImpactBuilder:
    """
    A comprehensive analytics builder for analyzing the impact of delivery delays on customer satisfaction and financial outcomes in the Olist e-commerce dataset.
    """
    def __init__(self, analytics_folder_path:str, cleaned_folder_path:str, verbose:bool=False):
        self.analytics_folder_path = Path(analytics_folder_path)
        self.cleaned_folder_path = Path(cleaned_folder_path)
        self.verbose = verbose
        self.order_facts = None
        self.order_reviews = None
        self.order_payments = None
        self.delay_impact = None
        self.delay_bucket_analysis = None
        self.order_facts_reviews_revenue = None

    def _log(self, message:str):
        """Utility method for logging messages if verbose mode is enabled."""
        if self.verbose:
            logger.info(message)

    def _load_data(self):
        """Load order facts, reviews, and payments data from CSV files."""
        self._log("Loading order facts, reviews, and payments data...")
        self.order_facts = pd.read_csv(self.analytics_folder_path / 'order_fact_table.csv')
        self.order_reviews = pd.read_csv(self.cleaned_folder_path / 'olist_order_reviews_dataset_cleaned.csv')
        self.order_payments = pd.read_csv(self.cleaned_folder_path / 'olist_order_payments_dataset_cleaned.csv')
        self._log("Data loaded successfully.")


    def _build_order_review_revenue_table(self)-> pd.DataFrame:
        """Build review metrics by merging order facts with reviews and calculating average ratings and review counts."""
        if self.order_facts is None or self.order_reviews is None:
            raise Exception("Order facts or reviews data not loaded.")
        self._log("Building delay impact metrics...")


        order_facts_reviews_revenue = (
            self.order_facts.merge(self.order_reviews[['order_id', 'review_score']], on='order_id', how='left')
            .merge(self.order_payments[['order_id', 'payment_value']], on='order_id', how='left')

        )
        self.order_facts_reviews_revenue = order_facts_reviews_revenue
        return order_facts_reviews_revenue

    def _create_delay_impact_metrics(self)-> pd.DataFrame:
        """Calculate average review scores and total revenue for delayed vs on-time deliveries."""
        if self.order_facts_reviews_revenue is None:
            raise Exception("Delay impact data not built yet.")
        self._log("Calculating delay impact metrics...")
        df = self.order_facts_reviews_revenue.copy()
        delay_impact = (
            df.groupby('is_delayed')
            .agg(
                avg_review_score = ('review_score', 'mean'),
                avg_revenue = ('payment_value', 'mean'),
                total_revenue = ('payment_value', 'sum'),
                total_order = ('order_id', 'nunique')
            )
        ).reset_index()
        self.delay_impact = delay_impact
        return delay_impact

    def _create_delay_bucket_analysis(self)-> pd.DataFrame:
        """Classify orders into delay buckets and analyze review scores and revenue by bucket."""
        if self.order_facts_reviews_revenue is None:
            raise Exception("Delay impact data not built yet.")
        self._log("Performing delay bucket analysis...")
        df = self.order_facts_reviews_revenue.copy()
        df['delay_bucket'] = pd.cut(
            df['delivery_delay_days'],
            bins=[-100, 0, 3, 7, 30, float('inf')],
            labels=['On Time', '1-3 Days', '4-7 Days', '8-30 Days', '30+ Days']
        )

        bucket_metrics = (
            df.groupby('delay_bucket')
            .agg(
                avg_review_score = ('review_score', 'mean'),
                avg_revenue = ('payment_value', 'mean'),
                total_revenue = ('payment_value', 'sum'),
                total_order = ('order_id', 'nunique')
            )
        ).reset_index()
        self.delay_bucket_analysis = bucket_metrics
        return bucket_metrics


    def _save_delay_impact(self) -> None:
        """Save the delay impact analysis results to a CSV file."""
        if self.order_facts_reviews_revenue is None or self.delay_bucket_analysis is None or self.delay_impact is None:
            raise Exception("Delay impact data not built yet.")
        self._log("Saving delay impact analysis results...")
        self.order_facts_reviews_revenue.to_csv(self.analytics_folder_path / 'order_facts_reviews_revenue.csv', index=False)
        self.delay_impact.to_csv(self.analytics_folder_path / 'delay_impact.csv', index=False)
        self.delay_bucket_analysis.to_csv(self.analytics_folder_path / 'delay_bucket_analysis.csv', index=False)
        self._log("Delay impact analysis results saved successfully.")

    def run(self) -> None:
        self._load_data()
        self._build_order_review_revenue_table()
        delay_metrics = self._create_delay_impact_metrics()
        self._log("*" * 50)
        self._log(delay_metrics)
        self._log("*" * 50)
        bucket_metrics = self._create_delay_bucket_analysis()
        self._log(bucket_metrics)
        self._save_delay_impact()


def main():
    analytics_folder_path = '../data/analytics/'
    cleaned_folder_path = '../data/cleaned/'
    builder = DelayImpactBuilder(analytics_folder_path, cleaned_folder_path, verbose=True)
    builder.run()

if __name__ == "__main__":
    main()

