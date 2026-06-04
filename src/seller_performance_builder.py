import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SellerPerformanceBuilder:
    """
    A comprehensive analytics builder for aggregating and analyzing Olist seller performance data.

    """
    def __init__(self, analytics_data_folder: str, verbose: bool=False):
        self.analytics_data_folder = Path(analytics_data_folder)
        self.verbose = verbose
        self.order_facts = None
        self.seller_performance = None
        self.seller_bucket = None


    def _log(self, message: str):
        if self.verbose:
            logger.info(message)

    def _load_order_facts(self):
        """Load cleaned order facts data from CSV file."""
        self._log("Loading order facts data...")
        self.order_facts = pd.read_csv(self.analytics_data_folder / 'order_facts_reviews_revenue.csv')
        self._log("Order facts data loaded successfully.")

    def _add_seller_performance_metrics(self)-> pd.DataFrame:
        """Build seller performance metrics from the order facts table."""
        if self.order_facts is None:
            raise Exception("Order facts data not loaded.")
        self._log("Building seller performance metrics...")

        seller_performance = (
            self.order_facts.groupby('seller_id')
            .agg(
                total_orders = ('order_id', 'nunique'),
                avg_delay_days = ('delivery_delay_days', 'mean'),
                median_delay_days = ('delivery_delay_days', 'median'),
                delay_std = ('delivery_delay_days', 'std'),
                p90_delay = ('delivery_delay_days', lambda x: x.quantile(0.9)),
                percentage_late_deliveries = ('is_delayed', 'mean'),
                on_time_delivery_rate = ('is_delayed', lambda x: 1 - x.mean()),
                total_revenue = ('payment_value', 'sum'),
                average_revenue_per_order = ('payment_value', 'mean'),
                average_review_score = ('review_score', 'mean')

            )
        ).reset_index()
        seller_performance['percentage_late_deliveries'] *= 100
        seller_performance['on_time_delivery_rate'] *= 100
        self.seller_performance = seller_performance
        self._log("Seller performance metrics built successfully.")
        return seller_performance



    def _classify_seller_risk(self):
        if self.seller_performance is None:
            raise Exception("Seller performance data not built yet.")
        self._log("Classifying seller risk levels...")
        df = self.seller_performance.copy()
        df['seller_risk'] = pd.cut(
            df['percentage_late_deliveries'],
            bins=[-1, 10, 30, 100],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        self.seller_performance = df
        return df

    def _build_seller_bucket_analysis_table(self, seller_performance: pd.DataFrame):
        df = seller_performance.copy()
        print(df.columns)
        seller_risk = (
            df
            .groupby("seller_risk")
            .agg(
                avg_revenue=("total_revenue", "mean"),
                avg_review=("average_review_score", "mean"),
                avg_delay=("percentage_late_deliveries", "mean"),
                seller_count=("seller_id", "count")
            )
            .reset_index()
        )
        seller_risk["Revenue"] = seller_risk["avg_revenue"] / seller_risk["avg_revenue"].max()
        seller_risk["Review Score"] = seller_risk["avg_review"] / seller_risk["avg_review"].max()
        seller_risk["Delay Rate"] = seller_risk["avg_delay"] / seller_risk["avg_delay"].max()

        self.seller_bucket = seller_risk
        self._log("Bucket analysis table built successfully.")
        return seller_risk

    def _validate_seller_performance(self)-> dict:
        """Validate the seller performance data."""
        if self.seller_performance is None:
            raise Exception("Seller performance data not built yet.")
        self._log("Validating seller performance data...")
        checks = {
            "Total sellers": self.seller_performance['seller_id'].nunique(),
            "High risk seller Percentage": (
                self.seller_performance['seller_risk'] == 'High Risk'
            ).mean() * 100
        }

        for key, value in checks.items():
            self._log(f"{key}: {value}")

        return checks

    def _save_seller_performance(
            self,
            output_dir: str = "../data/analytics"
    ) -> None:
        """Persist seller performance table."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        output_file = output_path / "seller_performance_table.csv"
        self.seller_performance.to_csv(output_file, index=False)

        self._log(f"Saved seller performance table to {output_file}")

    def _save_seller_bucket(self, output_dir:str ="../data/analytics") -> None:
        """Persist seller bucket table."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        output_file = output_path / "seller_bucket_table.csv"
        self.seller_bucket.to_csv(output_file, index=False)
        self._log(f"Saved seller bucket table to {output_file}")

    def run(self) -> None:
        self._load_order_facts()
        self._add_seller_performance_metrics()
        seller_performance = self._classify_seller_risk()
        self._build_seller_bucket_analysis_table(seller_performance)

        validation = self._validate_seller_performance()

        self._log("\n" + "=" * 50)
        self._log("SELLER PERFORMANCE VALIDATION SUMMARY")
        self._log("=" * 50)
        for k, v in validation.items():
            self._log(f"{k}: {v}")

        self._save_seller_performance(str(self.analytics_data_folder))
        self._save_seller_bucket(str(self.analytics_data_folder))

        self._log("\nSeller performance analysis completed successfully!")

def main():
    seller_performance_builder = SellerPerformanceBuilder(
        analytics_data_folder="../data/analytics",
        verbose=True
    )

    seller_performance_builder.run()


if __name__ == "__main__":
    main()
