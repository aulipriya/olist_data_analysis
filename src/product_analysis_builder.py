import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductAnalysisBuilder:
    """
    A builder class for analyzing product-level data in the Olist e-commerce dataset.
    This class focuses on building a comprehensive product analysis table that includes key metrics such as total sales, average review scores.
    """
    def __init__(self, clean_data_folder: str, analytics_data_folder:str, verbose: bool=False):
        self.clean_data_folder = Path(clean_data_folder)
        self.analytics_data_folder = Path(analytics_data_folder)
        self.verbose = verbose
        self.order_products = None
        self.order_reviews = None
        self.order_revenue = None
        self.product_analysis = None
        self.product_category_analysis = None

    def _log(self, message: str):
        if self.verbose:
            logger.info(message)

    def _load_data(self):
        """Load cleaned data from CSV files."""
        self._log("Loading cleaned data...")
        self.order_products = pd.read_csv(self.analytics_data_folder / 'order_products_table.csv')
        self.order_reviews = pd.read_csv(self.clean_data_folder / 'olist_order_reviews_dataset_cleaned.csv')
        self.order_revenue  =pd.read_csv(self.clean_data_folder / 'olist_order_payments_dataset_cleaned.csv')
        self._log("Data loaded successfully.")

    def build_product_analysis(self) -> pd.DataFrame:
        """Build the product analysis table by merging products with order items and calculating key metrics."""
        if self.order_products is None or self.order_revenue  is None or self.order_reviews is None:
            raise ValueError("Data not loaded yet")

        self._log("Building product analysis table...")
        unique_products = self.order_products[['product_id', 'product_category_name', 'price']].drop_duplicates(
            'product_id')
        df = self.order_products.copy()
        # Item-level revenue
        df["item_revenue"] = df["price"] + df["freight_value"]
        product_analysis = (
             df
            .merge(self.order_reviews[['order_id', 'review_score']], on='order_id', how='left')
            .groupby('product_id')
            .agg(
                total_revenue = ('item_revenue', 'sum'),
                average_review_score= ('review_score', 'mean'),
                total_sales = ('order_id', 'count'),
                average_price = ('price', 'mean')
        )
        .reset_index().merge(unique_products, on='product_id', how='left')
        )

        self._log("Product analysis table built successfully.")
        self.product_analysis = product_analysis
        return product_analysis


    def _add_product_segment_product_analysis(self):
        if self.product_analysis is None:
            raise ValueError("Data not loaded yet")
        df = self.product_analysis.copy()

        revenue_threshold = df["total_revenue"].median()
        review_threshold = df["average_review_score"].mean()

        def classify(row):
            if row["total_revenue"] >= revenue_threshold and row["average_review_score"] >= review_threshold:
                return "High Revenue & High Review"
            elif row["total_revenue"] >= revenue_threshold:
                return "High Revenue but Low Review"
            elif row["average_review_score"] >= review_threshold:
                return "Low Revenue but High Review"
            else:
                return "Low Revenue & Low Review"


        df["product_segment"] = df.apply(classify, axis=1)
        self.product_analysis = df
        self._log("Product segment data added successfully.")
        return df


    def build_product_category_analysis(self) -> pd.DataFrame:
        """Build the product category analysis table by aggregating product-level metrics to the category level."""
        if self.product_analysis is None:
            raise ValueError("Product analysis table not built yet")

        self._log("Building product category analysis table...")
        product_category_analysis = (
            self.product_analysis
            .groupby('product_category_name')
            .agg(
                total_category_revenue = ('total_revenue', 'sum'),
                average_review_score= ('average_review_score', 'mean'),
                total_sales = ('total_sales', 'sum'),
                total_products = ('product_id', 'count')
            )
            .reset_index()
        )
        product_category_analysis["revenue_share"] = (
                product_category_analysis["total_category_revenue"] /
                product_category_analysis["total_category_revenue"].sum()

        )

        product_category_analysis["revenue_per_sale"] = (
                product_category_analysis["total_category_revenue"] /
                product_category_analysis["total_sales"]
        )

        segment_counts = (
            self.product_analysis
            .pivot_table(
                index="product_category_name",
                columns="product_segment",
                values="product_id",
                aggfunc="count",
                fill_value=0
            )
            .reset_index()
        )

        # Merge
        product_category_analysis = product_category_analysis.merge(
            segment_counts,
            on="product_category_name",
            how="left"
        )

        risky_col = "Low Revenue & Low Review"
        star_col = "High Revenue & High Review"

        if risky_col in product_category_analysis.columns:
            product_category_analysis["risky_product_pct"] = (
                    product_category_analysis[risky_col] /
                    product_category_analysis["total_products"]
            )

        if star_col in product_category_analysis.columns:
            product_category_analysis["star_product_pct"] = (
                    product_category_analysis[star_col] /
                    product_category_analysis["total_products"]
            )
        segment_mapping = {
            "High Revenue & High Review": "Star",
            "High Revenue but Low Review": "Risky",
            "Low Revenue & Low Review": "Low Value",
            "Low Revenue but High Review": "Opportunity"
        }

        product_category_analysis.rename(columns=segment_mapping, inplace=True)
        segment_cols = ["Star", "Risky", "Opportunity", "Low Value"]

        for col in segment_cols:
            product_category_analysis[col] = product_category_analysis[col] / product_category_analysis["total_products"]

        self.product_category_analysis = product_category_analysis
        self._log("Product category analysis table built successfully.")
        return product_category_analysis


    def _save_product_analysis(self, product_analysis: pd.DataFrame):
        """Save the product analysis table to a CSV file."""
        output_path = self.analytics_data_folder / 'product_analysis_table.csv'
        self._log(f"Saving product analysis table to {output_path}...")
        product_analysis.to_csv(output_path, index=False)
        self._log("Product analysis table saved successfully.")

    def _save_product_category_analysis(self, product_category_analysis: pd.DataFrame):
        """Save the product category analysis table to a CSV file."""
        output_path = self.analytics_data_folder / 'product_category_analysis_table.csv'
        self._log(f"Saving product category analysis table to {output_path}...")
        product_category_analysis.to_csv(output_path, index=False)
        self._log("Product category analysis table saved successfully.")

    def run(self):
        self._load_data()
        self.build_product_analysis()
        product_analysis = self._add_product_segment_product_analysis()
        self._save_product_analysis(product_analysis)
        product_category_analysis = self.build_product_category_analysis()
        self._save_product_category_analysis(product_category_analysis)


def main():
    clean_data_folder = "../data/cleaned"
    analytics_data_folder = "../data/analytics"
    builder = ProductAnalysisBuilder(clean_data_folder, analytics_data_folder, verbose=True)
    builder.run()

if __name__ == "__main__":
    main()
