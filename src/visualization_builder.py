import pandas as pd
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VisualizationBuilder:
    """
    A comprehensive analytics builder for creating visualizations to analyze the impact of delivery delays on customer satisfaction and financial outcomes in the Olist e-commerce dataset.
    """
    def __init__(self, analytics_folder_path: str, result_path: str, verbose: bool=False):
        self.analytics_folder_path = Path(analytics_folder_path)
        self.result_path = Path(result_path)
        self.delay_impact = None
        self.seller_performance = None
        self.delay_bucket_analysis = None
        self.order_facts_reviews_revenue = None
        self.product_analysis = None
        self.product_category_analysis = None
        self.seller_performance_bucket = None
        self.verbose = verbose

    def _log(self, message: str):
        """Utility method for logging messages if verbose mode is enabled."""
        if self.verbose:
            logger.info(message)

    def _load_data(self):
        self._log("Loading data...")
        self.delay_impact = pd.read_csv(self.analytics_folder_path / "delay_impact.csv")
        self.seller_performance = pd.read_csv(self.analytics_folder_path / "seller_performance_table.csv")
        self.delay_bucket_analysis = pd.read_csv(self.analytics_folder_path / "delay_bucket_analysis.csv")
        self.product_analysis = pd.read_csv(self.analytics_folder_path / "product_analysis_table.csv")
        self.order_facts_reviews_revenue = pd.read_csv(self.analytics_folder_path / "order_facts_reviews_revenue.csv")
        self.product_category_analysis = pd.read_csv(self.analytics_folder_path / "product_category_analysis_table.csv")
        self.seller_performance_bucket = pd.read_csv(self.analytics_folder_path / "seller_bucket_table.csv")
        self._log("Data loaded successfully.")

    def _create_barplot(self, data, x, y, hue=None, palette="colorblind", title="", xlabel="", ylabel="", save_path=None):
        plt.figure(figsize=(10, 6))
        sns.barplot(data=data, x=x, y=y, palette=palette, hue=hue)
        plt.title(title, fontsize=20, weight="bold")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # Add value labels
        for i, v in enumerate(data[y]):
            if isinstance(v, float):
                plt.text(i, v + 0.02 * max(data[y]), f"{v:.2f}", ha="center")
        plt.tight_layout()
        sns.despine()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def _plot_review_score_by_delay(self):
        self._log("Plotting average review score by delay status...")
        # Map boolean to readable labels
        self.delay_impact["delay_label"] = self.delay_impact["is_delayed"].map({
            False: "On-Time Delivery",
            True: "Delayed Delivery"
        })
        colors = ["#4CAF50", "#E53935"]
        self._create_barplot(
            data=self.delay_impact,
            x="delay_label",
            y="avg_review_score",
            palette=colors,
            title="Review Score Drops If The Order Is Delayed",
            xlabel="",
            ylabel="Average Review Score",
            save_path=self.result_path / "avg_review_score_by_delay.jpg"
        )

    def _plot_revenue_by_delay(self):
        self._log("Plotting average revenue by delay status...")
        self.delay_impact["delay_label"] = self.delay_impact["is_delayed"].map({
            False: "On-Time Delivery",
            True: "Delayed Delivery"
        })
        colors = ["#4CAF50", "#E53935"]
        self._create_barplot(
            data=self.delay_impact,
            x="delay_label",
            y="avg_revenue",
            palette=colors,
            title="Revenue Shows No Impact From Delays",
            xlabel="",
            ylabel="Average Revenue",
            save_path=self.result_path / "avg_revenue_by_delay.jpg"
        )

    def _plot_delay_bucket_analysis(self):
        self._log("Plotting delay bucket analysis...")
        colors = ["#2E7D32", "#F9A825", "#FB8C00", "#E53935", "#8E0000"]
        self._create_barplot(
            data= self.delay_bucket_analysis,
            x='delay_bucket',
            y= 'avg_review_score',
            title="Review Score Drops More As Delay Increases",
            palette=colors,
            xlabel="",
            ylabel="Average Review Score",
            save_path=self.result_path / "avg_review_score_by_delay_bucket.jpg"
        )
        self._create_barplot(
            data= self.delay_bucket_analysis,
            x='delay_bucket',
            y= 'avg_revenue',
            title="Revenue Shows Low Impact by Delay",
            palette=colors,
            xlabel="Delay Bucket",
            ylabel="Average Revenue",
            save_path=self.result_path / "avg_revenue_by_delay_bucket.jpg"
        )
        self._create_barplot(
            data= self.delay_bucket_analysis,
            x='delay_bucket',
            y='total_order',
            title="Total number of orders that were delayed",
            palette=colors,
            xlabel="",
            ylabel="Total Order",
            save_path=self.result_path / "total_order_by_delay_bucket.jpg"
        )


    def plot_seller_delay_distribution(self):
        self._log("Plotting seller delay distribution")

        plt.figure()
        sns.histplot(self.seller_performance["percentage_late_deliveries"], bins=25, kde=True, palette="YlOrRd")

        # Highlight risk threshold
        plt.axvline(
            x=20,
            color="red",
            linestyle="--",
            linewidth=2,
            label="High Risk Threshold"
        )

        plt.title("Most Sellers Deliver Reliably, but a Small Group Has High Delay Rates")
        plt.xlabel("Percentage of Delayed Orders per Seller")
        plt.ylabel("Number of Sellers")

        plt.legend()

        sns.despine()
        plt.tight_layout()

        plt.tight_layout()
        plt.savefig(self.result_path / "seller_delay_distribution.jpg")
        plt.close()

    def plot_risk_category_distribution(self):
        self._log("Plotting seller risk categories")

        plt.figure()
        sns.countplot(data=self.seller_performance, x="seller_risk")
        plt.title("Very Small Group of Sellers in the high risk category")

        plt.tight_layout()
        plt.savefig(self.result_path / "seller_risk_categories.jpg")
        plt.close()

    def plot_top_risky_sellers(self):
        self._log("Plotting top risky sellers")

        df = (
            self.seller_performance
            .sort_values("percentage_late_deliveries", ascending=False)
            .head(10)
        )

        self._create_barplot(
            data = df,
            x= 'percentage_late_deliveries',
            y= 'seller_id',
            title="Top 10 High-Risk Sellers",
            xlabel="Percentage of Delayed Orders",
            ylabel="Seller ID",
            save_path=self.result_path / "top_risky_sellers.jpg"
        )

    def plot_seller_risk_vs_revenue(self):

        df = self.seller_performance.copy()

        plt.figure(figsize=(10, 6))

        sns.scatterplot(
            data=df,
            x="percentage_late_deliveries",
            y="total_revenue",
            hue="seller_risk",
            palette=["#4CAF50", "#F9A825", "#E53935"],
            size="total_orders",
            sizes=(20, 200)
        )
        sns.despine()
        plt.tight_layout()

        plt.savefig(self.result_path / "seller_risk_vs_revenue.jpg")
        plt.close()



    def plot_seller_performance_summary(self):
        df = self.seller_performance_bucket.copy()
        plot_df = df.melt(
            id_vars="seller_risk",
            value_vars=["Revenue", "Review Score", "Delay Rate"],
            var_name="Metric",
            value_name="Value"
        )
        self._create_barplot(
            data= plot_df,
            x="seller_risk",
            y="Value",
            hue="Metric",
            title="High-Risk Sellers Generate Revenue but Hurt Customer Experience",
            xlabel="Seller Risk Category",
            ylabel="Normalized Comparison",
            save_path=self.result_path / "seller_risk_summary.jpg"
        )

    def plot_revenue_vs_reviews_of_products(self):

        df = self.product_analysis.copy()

        plt.figure(figsize=(10, 6))

        sns.scatterplot(
            data=df,
            x="total_revenue",
            y="average_review_score",
            size="total_sales",
            sizes=(20, 200),
            alpha=0.6
        )

        # Add quadrant lines
        plt.axhline(df["average_review_score"].mean(), linestyle="--", color="grey")
        plt.axvline(df["total_revenue"].mean(), linestyle="--", color="grey")

        plt.title(
            "Do High-Revenue Products Also Deliver High Customer Satisfaction?",
            fontsize=16,
            weight="bold"
        )

        plt.xlabel("Total Revenue")
        plt.ylabel("Average Review Score")

        sns.despine()
        plt.tight_layout()

        plt.savefig(self.result_path / "revenue_vs_reviews.jpg")
        plt.close()


    def plot_product_revenue_review_summary(self):
        df = self.product_analysis.copy()
        plt.figure(figsize=(10, 6))

        sns.countplot(
            data=df,
            y="product_segment",
            order=df["product_segment"].value_counts().index,
            palette=[
                "#4CAF50",  # best
                "#E53935",  # risky
                "#F9A825",  # opportunity
                "#BDBDBD"  # low
            ]
        )

        plt.title(
            "Not All High-Revenue Products Deliver High Customer Satisfaction",
            fontsize=16,
            weight="bold"
        )

        plt.xlabel("Number of Products")
        plt.ylabel("Product Segment")

        sns.despine()
        plt.tight_layout()

        plt.savefig(self.result_path / "product_segments.jpg")
        plt.close()


    def plot_top_revenue_by_product_category(self):
        self._log("Plotting top revenue by product category")

        df = (
            self.product_category_analysis
            .sort_values("total_category_revenue", ascending=False)
            .head(5)
        )

        self._create_barplot(
            data = df,
            x= 'product_category_name',
            y= 'total_category_revenue',
            title="Top 5 Revenue-Generating Product Categories",
            xlabel="Product Category",
            ylabel="Total Revenue Generated By Each Category",
            save_path=self.result_path / "top_revenue_product_categories.jpg"
        )

    def plot_revenue_vs_reviews_of_product_categories(self):

        df = self.product_category_analysis.copy()

        plt.figure(figsize=(10, 6))

        sns.scatterplot(
            data=df,
            x="total_category_revenue",
            y="average_review_score",
            size="total_sales",
            sizes=(20, 200),
            alpha=0.6
        )

        # Add quadrant lines
        plt.axhline(df["average_review_score"].mean(), linestyle="--", color="grey")
        plt.axvline(df["total_category_revenue"].mean(), linestyle="--", color="grey")

        plt.title(
            "Do High-Revenue Products Also Deliver High Customer Satisfaction?",
            fontsize=16,
            weight="bold"
        )

        plt.xlabel("Total Revenue")
        plt.ylabel("Average Review Score")

        sns.despine()
        plt.tight_layout()

        plt.savefig(self.result_path / "revenue_vs_reviews_product_category.jpg")
        plt.close()

    def plot_product_category_segments(self):
        df = self.product_category_analysis.copy()
        # Focus on top categories (important)
        df = df.sort_values("total_category_revenue", ascending=False).head(10)

        categories = df["product_category_name"]

        plt.figure(figsize=(12, 7))

        bottom = [0] * len(df)

        colors = {
            "Star": "#4CAF50",  # green
            "Opportunity": "#F9A825",  # yellow
            "Low Value": "#BDBDBD",  # grey
            "Risky": "#E53935"  # red
        }
        segment_cols = ["Star", "Risky", "Opportunity", "Low Value"]

        for segment in segment_cols:
            plt.barh(
                categories,
                df[segment],
                left=bottom,
                color=colors[segment],
                label=segment
            )
            bottom = [i + j for i, j in zip(bottom, df[segment])]

        plt.title(
            "Some Categories Depend Heavily on Risky Products",
            fontsize=16,
            weight="bold"
        )

        plt.xlabel("Product Mix (%)")
        plt.ylabel("Product Category")

        plt.legend(title="Product Segment")

        sns.despine()
        plt.tight_layout()

        plt.savefig(self.result_path / "category_segment_mix.jpg")
        plt.close()

    def run(self):
        self._load_data()
        self._plot_review_score_by_delay()
        self._plot_revenue_by_delay()
        self._plot_delay_bucket_analysis()
        self.plot_seller_delay_distribution()
        self.plot_risk_category_distribution()
        self.plot_top_risky_sellers()
        self.plot_top_revenue_by_product_category()
        self.plot_seller_risk_vs_revenue()
        self.plot_seller_performance_summary()
        self.plot_revenue_vs_reviews_of_products()
        self.plot_product_revenue_review_summary()
        self.plot_revenue_vs_reviews_of_product_categories()
        self.plot_product_category_segments()



def main():
    analytics_folder_path = '../data/analytics/'
    result_path = '../results/'
    builder = VisualizationBuilder(analytics_folder_path, result_path, verbose=True)
    builder.run()

if __name__ == "__main__":
    main()



