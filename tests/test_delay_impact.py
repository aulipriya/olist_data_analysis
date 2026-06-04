import pandas as pd
import pytest

from src.delay_impact_builder import DelayImpactBuilder


class TestDelayImpactBuilder:
    @pytest.fixture
    def builder(self):
        dib = DelayImpactBuilder(analytics_folder_path="dummy", cleaned_folder_path="dummy")
        dib.order_facts = pd.DataFrame({
            "order_id": ["o1", "o2", "o3"],
            "is_delayed": [True, True, False],
            "delivery_delay_days": [2, 5, -2],
        })
        dib.order_reviews = pd.DataFrame({
            "order_id": ["o1", "o2", "o3"],
            "review_score": [2.0, 4.0, 5.0],
        })
        dib.order_payments = pd.DataFrame({
            "order_id": ["o1", "o2", "o3"],
            "payment_value": [100.0, 200.0, 150.0],
        })
        dib.order_facts_reviews_revenue = pd.DataFrame({
            "order_id": ["o1", "o2", "o3"],
            "is_delayed": [True, True, False],
            "delivery_delay_days": [2, 5, -2],
            "review_score": [2.0, 4.0, 5.0],
            "payment_value": [100.0, 200.0, 150.0],
        })
        return dib

    def test_build_order_review_revenue_table_raises_if_order_facts_missing(self):
        dib = DelayImpactBuilder(analytics_folder_path="dummy", cleaned_folder_path="dummy")
        dib.order_reviews = pd.DataFrame({"order_id": ["o1"], "review_score": [4]})
        dib.order_payments = pd.DataFrame({"order_id": ["o1"], "payment_value": [50.0]})
        with pytest.raises(Exception, match="Order facts or reviews data not loaded."):
            dib._build_order_review_revenue_table()

    def test_build_order_review_revenue_table_raises_if_order_reviews_missing(self):
        dib = DelayImpactBuilder(analytics_folder_path="dummy", cleaned_folder_path="dummy")
        dib.order_facts = pd.DataFrame({"order_id": ["o1"], "is_delayed": [True]})
        dib.order_payments = pd.DataFrame({"order_id": ["o1"], "payment_value": [50.0]})
        with pytest.raises(Exception, match="Order facts or reviews data not loaded."):
            dib._build_order_review_revenue_table()

    def test_build_order_review_revenue_table_has_expected_columns(self, builder):
        result = builder._build_order_review_revenue_table()
        expected_columns = {"order_id", "is_delayed", "delivery_delay_days", "review_score", "payment_value"}
        assert expected_columns.issubset(set(result.columns))

    def test_create_delay_impact_metrics_raises_if_merged_table_missing(self):
        dib = DelayImpactBuilder(analytics_folder_path="dummy", cleaned_folder_path="dummy")
        with pytest.raises(Exception, match="Delay impact data not built yet."):
            dib._create_delay_impact_metrics()

    def test_create_delay_impact_metrics_has_expected_columns(self, builder):
        result = builder._create_delay_impact_metrics()
        expected_columns = {"is_delayed", "avg_review_score", "avg_revenue", "total_revenue", "total_order"}
        assert expected_columns.issubset(set(result.columns))

    def test_create_delay_impact_metrics_avg_review_score(self, builder):
        result = builder._create_delay_impact_metrics()
        delayed_row = result[result["is_delayed"] == True].iloc[0]
        on_time_row = result[result["is_delayed"] == False].iloc[0]
        assert delayed_row["avg_review_score"] == pytest.approx(3.0)   # (2 + 4) / 2
        assert on_time_row["avg_review_score"] == pytest.approx(5.0)

    def test_create_delay_impact_metrics_revenue_aggregates(self, builder):
        result = builder._create_delay_impact_metrics()
        delayed_row = result[result["is_delayed"] == True].iloc[0]
        on_time_row = result[result["is_delayed"] == False].iloc[0]
        assert delayed_row["avg_revenue"] == pytest.approx(150.0)    # (100 + 200) / 2
        assert delayed_row["total_revenue"] == pytest.approx(300.0)  # 100 + 200
        assert on_time_row["avg_revenue"] == pytest.approx(150.0)
        assert on_time_row["total_revenue"] == pytest.approx(150.0)

    def test_create_delay_impact_metrics_total_order_count(self, builder):
        result = builder._create_delay_impact_metrics()
        delayed_row = result[result["is_delayed"] == True].iloc[0]
        on_time_row = result[result["is_delayed"] == False].iloc[0]
        assert delayed_row["total_order"] == 2
        assert on_time_row["total_order"] == 1

    def test_create_delay_bucket_analysis_raises_if_merged_table_missing(self):
        dib = DelayImpactBuilder(analytics_folder_path="dummy", cleaned_folder_path="dummy")
        with pytest.raises(Exception, match="Delay impact data not built yet."):
            dib._create_delay_bucket_analysis()

    def test_create_delay_bucket_analysis_has_expected_columns(self, builder):
        result = builder._create_delay_bucket_analysis()
        expected_columns = {"delay_bucket", "avg_review_score", "avg_revenue", "total_revenue", "total_order"}
        assert expected_columns.issubset(set(result.columns))

    def test_create_delay_bucket_analysis_assigns_correct_buckets(self, builder):
        result = builder._create_delay_bucket_analysis()
        occupied = result[result["total_order"] > 0]["delay_bucket"].astype(str).tolist()
        assert "On Time" in occupied    # o3: delay=-2
        assert "1-3 Days" in occupied   # o1: delay=2
        assert "4-7 Days" in occupied   # o2: delay=5

    def test_create_delay_bucket_analysis_aggregates_per_bucket(self, builder):
        result = builder._create_delay_bucket_analysis()
        on_time = result[result["delay_bucket"] == "On Time"].iloc[0]
        one_to_three = result[result["delay_bucket"] == "1-3 Days"].iloc[0]
        assert on_time["avg_review_score"] == pytest.approx(5.0)
        assert on_time["total_revenue"] == pytest.approx(150.0)
        assert on_time["total_order"] == 1
        assert one_to_three["avg_review_score"] == pytest.approx(2.0)
        assert one_to_three["total_revenue"] == pytest.approx(100.0)
        assert one_to_three["total_order"] == 1

    @pytest.fixture
    def tmp_builder(self, tmp_path):
        analytics_dir = tmp_path / "analytics"
        cleaned_dir = tmp_path / "cleaned"
        analytics_dir.mkdir()
        cleaned_dir.mkdir()

        pd.DataFrame({
            "order_id": ["o1", "o2", "o3"],
            "is_delayed": [True, True, False],
            "delivery_delay_days": [2, 5, -2],
        }).to_csv(analytics_dir / "order_fact_table.csv", index=False)

        pd.DataFrame({
            "order_id": ["o1", "o2", "o3"],
            "review_score": [2.0, 4.0, 5.0],
        }).to_csv(cleaned_dir / "olist_order_reviews_dataset_cleaned.csv", index=False)

        pd.DataFrame({
            "order_id": ["o1", "o2", "o3"],
            "payment_value": [100.0, 200.0, 150.0],
        }).to_csv(cleaned_dir / "olist_order_payments_dataset_cleaned.csv", index=False)

        return DelayImpactBuilder(
            analytics_folder_path=str(analytics_dir),
            cleaned_folder_path=str(cleaned_dir),
        ), analytics_dir

    def test_run_populates_all_instance_attributes(self, tmp_builder):
        dib, _ = tmp_builder
        dib.run()
        assert dib.order_facts_reviews_revenue is not None
        assert dib.delay_impact is not None
        assert dib.delay_bucket_analysis is not None

    def test_run_creates_output_files(self, tmp_builder):
        dib, analytics_dir = tmp_builder
        dib.run()
        assert (analytics_dir / "order_facts_reviews_revenue.csv").exists()
        assert (analytics_dir / "delay_impact.csv").exists()
        assert (analytics_dir / "delay_bucket_analysis.csv").exists()