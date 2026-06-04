import pandas as pd
import pytest

from src.seller_performance_builder import SellerPerformanceBuilder


class TestSellerPerformanceBuilder:
    @pytest.fixture
    def builder(self):
        spb = SellerPerformanceBuilder(analytics_data_folder="dummy")
        spb.order_facts = pd.DataFrame({
            "order_id": ["o1", "o2", "o3", "o4"],
            "seller_id": ["s1", "s1", "s2", "s2"],
            "is_delayed": [True, True, False, False],
            "delivery_delay_days": [2, 5, -1, -2],
            "payment_value": [100.0, 200.0, 150.0, 100.0],
            "review_score": [3.0, 2.0, 5.0, 4.0],
        })
        spb.seller_performance = pd.DataFrame({
            "seller_id": ["s1", "s2"],
            "total_orders": [2, 2],
            "avg_delay_days": [3.5, -1.5],
            "median_delay_days": [3.5, -1.5],
            "delay_std": [2.12, 0.71],
            "p90_delay": [4.7, -1.1],
            "percentage_late_deliveries": [100.0, 0.0],
            "on_time_delivery_rate": [0.0, 100.0],
            "total_revenue": [300.0, 250.0],
            "average_revenue_per_order": [150.0, 125.0],
            "average_review_score": [2.5, 4.5],
            "seller_risk": pd.Categorical(
                ["High Risk", "Low Risk"],
                categories=["Low Risk", "Medium Risk", "High Risk"]
            ),
        })
        return spb

    # --- _add_seller_performance_metrics ---

    def test_add_seller_performance_metrics_raises_if_order_facts_missing(self):
        spb = SellerPerformanceBuilder(analytics_data_folder="dummy")
        with pytest.raises(Exception, match="Order facts data not loaded."):
            spb._add_seller_performance_metrics()

    def test_add_seller_performance_metrics_has_expected_columns(self, builder):
        result = builder._add_seller_performance_metrics()
        expected_columns = {
            "seller_id", "total_orders", "avg_delay_days", "median_delay_days",
            "delay_std", "p90_delay", "percentage_late_deliveries",
            "on_time_delivery_rate", "total_revenue", "average_revenue_per_order",
            "average_review_score",
        }
        assert expected_columns.issubset(set(result.columns))

    def test_add_seller_performance_metrics_total_orders(self, builder):
        result = builder._add_seller_performance_metrics()
        s1 = result[result["seller_id"] == "s1"].iloc[0]
        s2 = result[result["seller_id"] == "s2"].iloc[0]
        assert s1["total_orders"] == 2
        assert s2["total_orders"] == 2

    def test_add_seller_performance_metrics_percentage_late_is_0_to_100(self, builder):
        result = builder._add_seller_performance_metrics()
        s1 = result[result["seller_id"] == "s1"].iloc[0]
        s2 = result[result["seller_id"] == "s2"].iloc[0]
        assert s1["percentage_late_deliveries"] == pytest.approx(100.0)  # both delayed
        assert s2["percentage_late_deliveries"] == pytest.approx(0.0)    # none delayed

    def test_add_seller_performance_metrics_on_time_delivery_rate(self, builder):
        result = builder._add_seller_performance_metrics()
        s1 = result[result["seller_id"] == "s1"].iloc[0]
        s2 = result[result["seller_id"] == "s2"].iloc[0]
        assert s1["on_time_delivery_rate"] == pytest.approx(0.0)
        assert s2["on_time_delivery_rate"] == pytest.approx(100.0)

    # --- _classify_seller_risk ---

    def test_classify_seller_risk_raises_if_seller_performance_missing(self):
        spb = SellerPerformanceBuilder(analytics_data_folder="dummy")
        with pytest.raises(Exception, match="Seller performance data not built yet."):
            spb._classify_seller_risk()

    def test_classify_seller_risk_adds_seller_risk_column(self, builder):
        result = builder._classify_seller_risk()
        assert "seller_risk" in result.columns

    def test_classify_seller_risk_correct_labels(self, builder):
        builder.seller_performance["seller_risk"] = None  # clear pre-set risk
        result = builder._classify_seller_risk()
        s1 = result[result["seller_id"] == "s1"].iloc[0]
        s2 = result[result["seller_id"] == "s2"].iloc[0]
        assert str(s1["seller_risk"]) == "High Risk"   # percentage_late=100
        assert str(s2["seller_risk"]) == "Low Risk"    # percentage_late=0

    # --- _build_seller_bucket_analysis_table ---

    def test_build_seller_bucket_analysis_table_has_expected_columns(self, builder):
        result = builder._build_seller_bucket_analysis_table(builder.seller_performance)
        expected_columns = {
            "seller_risk", "avg_revenue", "avg_review", "avg_delay",
            "seller_count", "Revenue", "Review Score", "Delay Rate",
        }
        assert expected_columns.issubset(set(result.columns))

    def test_build_seller_bucket_analysis_table_normalized_columns_max_is_1(self, builder):
        result = builder._build_seller_bucket_analysis_table(builder.seller_performance)
        occupied = result[result["seller_count"] > 0]
        assert occupied["Revenue"].max() == pytest.approx(1.0)
        assert occupied["Review Score"].max() == pytest.approx(1.0)
        assert occupied["Delay Rate"].max() == pytest.approx(1.0)

    def test_build_seller_bucket_analysis_table_normalized_columns_in_range(self, builder):
        result = builder._build_seller_bucket_analysis_table(builder.seller_performance)
        for col in ["Revenue", "Review Score", "Delay Rate"]:
            assert result[col].dropna().between(0, 1).all()

    # --- _validate_seller_performance ---

    def test_validate_seller_performance_raises_if_seller_performance_missing(self):
        spb = SellerPerformanceBuilder(analytics_data_folder="dummy")
        with pytest.raises(Exception, match="Seller performance data not built yet."):
            spb._validate_seller_performance()

    def test_validate_seller_performance_returns_expected_keys(self, builder):
        result = builder._validate_seller_performance()
        assert "Total sellers" in result
        assert "High risk seller Percentage" in result

    def test_validate_seller_performance_total_sellers_count(self, builder):
        result = builder._validate_seller_performance()
        assert result["Total sellers"] == 2

    def test_validate_seller_performance_high_risk_percentage(self, builder):
        result = builder._validate_seller_performance()
        assert result["High risk seller Percentage"] == pytest.approx(50.0)  # 1 of 2 sellers

    # --- run ---

    @pytest.fixture
    def tmp_builder(self, tmp_path):
        analytics_dir = tmp_path / "analytics"
        analytics_dir.mkdir()

        pd.DataFrame({
            "order_id": ["o1", "o2", "o3", "o4"],
            "seller_id": ["s1", "s1", "s2", "s2"],
            "is_delayed": [True, True, False, False],
            "delivery_delay_days": [2, 5, -1, -2],
            "payment_value": [100.0, 200.0, 150.0, 100.0],
            "review_score": [3.0, 2.0, 5.0, 4.0],
        }).to_csv(analytics_dir / "order_facts_reviews_revenue.csv", index=False)

        return SellerPerformanceBuilder(analytics_data_folder=str(analytics_dir)), analytics_dir

    def test_run_populates_all_instance_attributes(self, tmp_builder):
        spb, _ = tmp_builder
        spb.run()
        assert spb.order_facts is not None
        assert spb.seller_performance is not None
        assert spb.seller_bucket is not None

    def test_run_creates_output_files(self, tmp_builder):
        spb, analytics_dir = tmp_builder
        spb.run()
        assert (analytics_dir / "seller_performance_table.csv").exists()
        assert (analytics_dir / "seller_bucket_table.csv").exists()
