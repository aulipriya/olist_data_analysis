import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalyticsBuilder:
    """
    A comprehensive analytics builder for aggregating and analyzing Olist e-commerce data.

    """
    def __init__(self, clean_data_folder: str, output_folder: str = '../data/analytics', verbose: bool=False):
        self.clean_data_folder = Path(clean_data_folder)
        self.output_folder = output_folder
        self.verbose = verbose
        self.orders = None
        self.customers = None
        self.order_items = None
        self.order_facts = None
        self.products = None
        self.order_products_facts = None

    def _log(self, message: str):
        if self.verbose:
            logger.info(message)

    def _load_data(self):
        """Load cleaned data from CSV files."""
        self._log("Loading cleaned data...")
        self.orders = pd.read_csv(self.clean_data_folder / 'olist_orders_dataset_cleaned.csv')
        self.customers = pd.read_csv(self.clean_data_folder / 'olist_customers_dataset_cleaned.csv')
        self.order_items = pd.read_csv(self.clean_data_folder / 'olist_order_items_dataset_cleaned.csv')
        self.products = pd.read_csv(self.clean_data_folder / 'olist_products_dataset_cleaned.csv')
        self._log("Data loaded successfully.")

    def _build_order_facts(self)-> pd.DataFrame:
        """Build the order facts table by merging orders, customers, and order items."""
        self._log("Building order facts table...")
        order_facts = (
            self.orders
            .merge(self.customers, how='left', on='customer_id')
            .merge(self.order_items, how='left', on='order_id')
        )
        self.order_facts = order_facts
        self._log("Order facts table built successfully.")
        return order_facts

    def _build_order_products_facts(self) -> pd.DataFrame:
        """Build the order products facts table by merging order facts with products."""
        if self.order_items is None:
            raise ValueError("Order items table not built yet")
        self._log("Building order products facts table...")
        order_products_facts = (
            self.order_items
            .merge(self.products, how='left', on='product_id')
        )
        self._log("Order products facts table built successfully.")
        self.order_products_facts = order_products_facts
        return order_products_facts


    def _add_delay_metrics(self):
        """Add delay metrics to the order facts table."""
        if self.order_facts is None:
            raise ValueError("Order fact table not built yet")
        self._log("Engineering delivery features")

        self.order_facts['order_approval_delay'] = (
            pd.to_datetime(self.order_facts['order_approved_at']) -
            pd.to_datetime(self.order_facts['order_purchase_timestamp'])
        ).dt.days

        self.order_facts['delivery_days'] = (
            pd.to_datetime(self.order_facts['order_delivered_customer_date']) -
            pd.to_datetime(self.order_facts['order_purchase_timestamp'])
        ).dt.days
        self.order_facts['estimated_delivery_days'] = (
                pd.to_datetime(self.order_facts['order_estimated_delivery_date']) -
                pd.to_datetime(self.order_facts['order_purchase_timestamp'])).dt.days
        self.order_facts['delivery_delay_days'] = (self.order_facts['delivery_days'] -
                                                   self.order_facts['estimated_delivery_days'])

        self.order_facts['is_delayed'] = self.order_facts['delivery_delay_days'] > 0
        self.order_facts['is_early'] = self.order_facts['delivery_delay_days'] < 0

        negative_delivery_days = (self.order_facts['delivery_days'] < 0).sum()
        if negative_delivery_days > 0:
            self._log(f"DATA QUALITY ISSUE: {negative_delivery_days} orders have negative delivery_days")

        return self.order_facts


    def validate_order_facts(self):
        """Validate the order facts table for missing values and data types."""
        if self.order_facts is None:
            raise ValueError("Order fact table not built yet")
        self._log("Validating order facts table...")
        checks ={
            'number_of_rows': len(self.order_facts),
            'unique_orders': self.order_facts['order_id'].nunique(),
            'multiseller_orders': self.order_facts.groupby('order_id')['seller_id']
            .nunique().gt(1).sum(),
            'delayed_orders_percentage': round(self.order_facts['is_delayed'].mean() * 100, 2),
        }
        for check, value in checks.items():
            self._log(f"{check}: {value}")
        self._log("Validation completed.")


    def validate_order_products_facts(self):
        """Validate the order products facts table for missing values and data types."""
        self._log("Validating order products facts table...")
        checks = {
            'number_of_rows': len(self.order_products_facts),
            'unique_order_ids': self.order_products_facts['order_id'].nunique(),
            'unique_product_ids': self.order_products_facts['product_id'].nunique(),
        }
        for check, value in checks.items():
            self._log(f"{check}: {value}")
        self._log("Validation completed.")

    def save_order_fact(self, output_path: str = "../data/analytics") -> None:
        """Persist the analytical dataset."""
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / "order_fact_table.csv"
        self.order_facts.to_csv(output_file, index=False)

        self._log(f"Saved order fact table to {output_file}")


    def save_order_products_fact(self, output_path: str = "../data/analytics") -> None:
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "order_products_table.csv"
        self.order_products_facts.to_csv(output_file, index=False)
        self._log(f"Saved order products fact table to {output_file}")

    def run(self) -> None:
        self._load_data()
        self._build_order_facts()
        self._add_delay_metrics()
        self.validate_order_facts()
        self.save_order_fact(self.output_folder)

        self._build_order_products_facts()
        self.validate_order_products_facts()
        self.save_order_products_fact(self.output_folder)

def main():
    analytics_builder = AnalyticsBuilder('../data/cleaned', verbose=True)
    analytics_builder.run()


if __name__ == "__main__":
    main()

