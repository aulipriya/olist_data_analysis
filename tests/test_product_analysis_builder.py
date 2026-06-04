import pandas as pd
import pytest

from src.product_analysis_builder import ProductAnalysisBuilder


class TestProductAnalysisBuilder:
    @pytest.fixture
    def builder(self):
        pab = ProductAnalysisBuilder(clean_data_folder="dummy", analytics_data_folder="dummy")
        # 4 products across 2 categories, covering all 4 segments
        pab.order_products = pd.DataFrame({
            "order_id": ["o1", "o2", "o3", "o4"],
            "product_id": ["p1", "p2", "p3", "p4"],
            "product_category_name": ["electronics", "electronics", "books", "books"],
            "price": [200.0, 200.0, 50.0, 50.0],
            "freight_value": [0.0, 0.0, 0.0, 0.0],
        })
        pab.order_reviews = pd.DataFrame({
            "order_id": ["o1", "o2", "o3", "o4"],
            "review_score": [5.0, 2.0, 5.0, 2.0],
        })
        pab.order_revenue = pd.DataFrame({
            "order_id": ["o1", "o2", "o3", "o4"],
            "payment_value": [200.0, 200.0, 50.0, 50.0],
        })
        # pre-set product_analysis with all 4 segments for downstream tests
        # revenue_threshold=median([200,200,50,50])=125, review_threshold=mean([5,2,5,2])=3.5
        pab.product_analysis = pd.DataFrame({
            "product_id": ["p1", "p2", "p3", "p4"],
            "total_revenue": [200.0, 200.0, 50.0, 50.0],
            "average_review_score": [5.0, 2.0, 5.0, 2.0],
            "total_sales": [1, 1, 1, 1],
            "average_price": [200.0, 200.0, 50.0, 50.0],
            "product_category_name": ["electronics", "electronics", "books", "books"],
            "product_segment": [
                "High Revenue & High Review",
                "High Revenue but Low Review",
                "Low Revenue but High Review",
                "Low Revenue & Low Review",
            ],
        })
        return pab

    # --- build_product_analysis ---

    def test_build_product_analysis_raises_if_data_not_loaded(self):
        pab = ProductAnalysisBuilder(clean_data_folder="dummy", analytics_data_folder="dummy")
        with pytest.raises(ValueError, match="Data not loaded yet"):
            pab.build_product_analysis()

    def test_build_product_analysis_has_expected_columns(self, builder):
        result = builder.build_product_analysis()
        expected = {
            "product_id", "total_revenue", "average_review_score",
            "total_sales", "average_price", "product_category_name",
        }
        assert expected.issubset(set(result.columns))

    def test_build_product_analysis_item_revenue_includes_freight(self):
        pab = ProductAnalysisBuilder(clean_data_folder="dummy", analytics_data_folder="dummy")
        pab.order_products = pd.DataFrame({
            "order_id": ["o1"],
            "product_id": ["p1"],
            "product_category_name": ["electronics"],
            "price": [100.0],
            "freight_value": [20.0],
        })
        pab.order_reviews = pd.DataFrame({"order_id": ["o1"], "review_score": [5.0]})
        pab.order_revenue = pd.DataFrame({"order_id": ["o1"], "payment_value": [120.0]})
        result = pab.build_product_analysis()
        assert result.iloc[0]["total_revenue"] == pytest.approx(120.0)  # 100 + 20

    def test_build_product_analysis_correct_total_sales(self, builder):
        result = builder.build_product_analysis()
        for pid in ["p1", "p2", "p3", "p4"]:
            assert result[result["product_id"] == pid].iloc[0]["total_sales"] == 1

    def test_build_product_analysis_correct_avg_review_score(self, builder):
        result = builder.build_product_analysis()
        assert result[result["product_id"] == "p1"].iloc[0]["average_review_score"] == pytest.approx(5.0)
        assert result[result["product_id"] == "p2"].iloc[0]["average_review_score"] == pytest.approx(2.0)

    # --- _add_product_segment_product_analysis ---

    def test_add_product_segment_raises_if_product_analysis_missing(self):
        pab = ProductAnalysisBuilder(clean_data_folder="dummy", analytics_data_folder="dummy")
        with pytest.raises(ValueError, match="Data not loaded yet"):
            pab._add_product_segment_product_analysis()

    def test_add_product_segment_adds_column(self, builder):
        result = builder._add_product_segment_product_analysis()
        assert "product_segment" in result.columns

    def test_add_product_segment_correct_labels(self, builder):
        result = builder._add_product_segment_product_analysis()
        # revenue_threshold = median([200,200,50,50]) = 125
        # review_threshold  = mean([5,2,5,2]) = 3.5
        assert result[result["product_id"] == "p1"].iloc[0]["product_segment"] == "High Revenue & High Review"
        assert result[result["product_id"] == "p2"].iloc[0]["product_segment"] == "High Revenue but Low Review"
        assert result[result["product_id"] == "p3"].iloc[0]["product_segment"] == "Low Revenue but High Review"
        assert result[result["product_id"] == "p4"].iloc[0]["product_segment"] == "Low Revenue & Low Review"

    # --- build_product_category_analysis ---

    def test_build_product_category_analysis_raises_if_product_analysis_missing(self):
        pab = ProductAnalysisBuilder(clean_data_folder="dummy", analytics_data_folder="dummy")
        with pytest.raises(ValueError, match="Product analysis table not built yet"):
            pab.build_product_category_analysis()

    def test_build_product_category_analysis_has_expected_columns(self, builder):
        result = builder.build_product_category_analysis()
        expected = {
            "product_category_name", "total_category_revenue", "average_review_score",
            "total_sales", "total_products", "revenue_share", "revenue_per_sale",
        }
        assert expected.issubset(set(result.columns))

    def test_build_product_category_analysis_revenue_share_sums_to_1(self, builder):
        result = builder.build_product_category_analysis()
        assert result["revenue_share"].sum() == pytest.approx(1.0)

    def test_build_product_category_analysis_correct_total_revenue(self, builder):
        result = builder.build_product_category_analysis()
        electronics = result[result["product_category_name"] == "electronics"].iloc[0]
        books = result[result["product_category_name"] == "books"].iloc[0]
        assert electronics["total_category_revenue"] == pytest.approx(400.0)  # p1 + p2
        assert books["total_category_revenue"] == pytest.approx(100.0)        # p3 + p4

    def test_build_product_category_analysis_segment_proportions_in_range(self, builder):
        result = builder.build_product_category_analysis()
        for col in ["Star", "Risky", "Opportunity", "Low Value"]:
            assert result[col].between(0, 1).all()

    # --- run ---

    @pytest.fixture
    def tmp_builder(self, tmp_path):
        analytics_dir = tmp_path / "analytics"
        cleaned_dir = tmp_path / "cleaned"
        analytics_dir.mkdir()
        cleaned_dir.mkdir()

        pd.DataFrame({
            "order_id": ["o1", "o2", "o3", "o4"],
            "product_id": ["p1", "p2", "p3", "p4"],
            "product_category_name": ["electronics", "electronics", "books", "books"],
            "price": [200.0, 200.0, 50.0, 50.0],
            "freight_value": [0.0, 0.0, 0.0, 0.0],
        }).to_csv(analytics_dir / "order_products_table.csv", index=False)

        pd.DataFrame({
            "order_id": ["o1", "o2", "o3", "o4"],
            "review_score": [5.0, 2.0, 5.0, 2.0],
        }).to_csv(cleaned_dir / "olist_order_reviews_dataset_cleaned.csv", index=False)

        pd.DataFrame({
            "order_id": ["o1", "o2", "o3", "o4"],
            "payment_value": [200.0, 200.0, 50.0, 50.0],
        }).to_csv(cleaned_dir / "olist_order_payments_dataset_cleaned.csv", index=False)

        return ProductAnalysisBuilder(
            clean_data_folder=str(cleaned_dir),
            analytics_data_folder=str(analytics_dir),
        ), analytics_dir

    def test_run_populates_all_instance_attributes(self, tmp_builder):
        pab, _ = tmp_builder
        pab.run()
        assert pab.order_products is not None
        assert pab.product_analysis is not None
        assert pab.product_category_analysis is not None

    def test_run_creates_output_files(self, tmp_builder):
        pab, analytics_dir = tmp_builder
        pab.run()
        assert (analytics_dir / "product_analysis_table.csv").exists()
        assert (analytics_dir / "product_category_analysis_table.csv").exists()