from airflow.sdk import dag, task
from datetime import datetime, timedelta
from src.clean_data import DataCleaner
from src.analytics_builder import AnalyticsBuilder
from src.delay_impact_builder import DelayImpactBuilder
from src.seller_performance_builder import SellerPerformanceBuilder
from src.product_analysis_builder import ProductAnalysisBuilder
from src.visualization_builder import VisualizationBuilder

@dag(
    dag_id="analysis_pipeline",
    default_args={"owner": "Astro", "retries": 3},
    start_date=datetime(2026, 4, 29),
    schedule="@once",
    catchup=False,
)
def analysis_pipeline():
    @task()
    def clean_data():
        cleaner = DataCleaner('include/data/archive')
        cleaner.run_cleaning('include/data/cleaned')

    @task()
    def build_analytics():
        builder = AnalyticsBuilder('include/data/cleaned', 'include/data/analytics', verbose=True)
        builder.run()

    @task()
    def delay_impact_builder():
        builder = DelayImpactBuilder('include/data/analytics', 'include/data/cleaned', verbose=True)
        builder.run()


    @task()
    def seller_performance_analyzer():
        builder = SellerPerformanceBuilder('include/data/analytics', verbose=True)
        builder.run()


    @task()
    def product_analysis():
        builder = ProductAnalysisBuilder('include/data/cleaned', 'include/data/analytics', verbose=True)
        builder.run()

    @task()
    def visualization_builder():
        builder = VisualizationBuilder('include/data/analytics', 'include/results', verbose=True)
        builder.run()

    clean = clean_data()
    analytics = build_analytics()
    delay = delay_impact_builder()
    seller = seller_performance_analyzer()
    product = product_analysis()
    viz = visualization_builder()

    clean >> analytics >> delay >> [seller, product] >> viz


analysis_pipeline()
