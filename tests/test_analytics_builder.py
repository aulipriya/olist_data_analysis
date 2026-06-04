import pandas as pd
import pytest

from src.analytics_builder import AnalyticsBuilder


class TestAnalyticsBuilder:
    @pytest.fixture
    def builder(self):
        ab = AnalyticsBuilder(clean_data_folder="dummy")
        ab.orders = pd.DataFrame({
            "order_id": ["o1", "o2"],
            "customer_id": ["c1", "c2"],
            "order_purchase_timestamp": ["2023-01-01", "2023-01-05"],
            "order_approved_at": ["2023-01-02", "2023-01-06"],
            "order_delivered_customer_date": ["2023-01-10", "2023-01-08"],
            "order_estimated_delivery_date": ["2023-01-08", "2023-01-10"],
        })

        ab.customers = pd.DataFrame({
            "customer_id": ["c1", "c2"],
            "customer_city": ["sao paulo", "rio"],
        })

        ab.order_items = pd.DataFrame({
            "order_id": ["o1", "o2"],
            "product_id": ["p1", "p2"],
            "seller_id": ["s1", "s2"],
            "price": [100.0, 200.0],
        })

        ab.products = pd.DataFrame({
            "product_id": ["p1", "p2"],
            "product_category": ["electronics", "books"],
        })

        return ab

    def test_build_order_facts_stores_on_instance(self, builder):
        builder._build_order_facts()
        assert builder.order_facts is not None

    def test_build_order_facts_has_columns_from_all_tables(self, builder):
        builder._build_order_facts()
        assert "order_id" in builder.order_facts.columns
        assert "customer_city" in builder.order_facts.columns
        assert "price" in builder.order_facts.columns

    def test_add_delay_metrics_raises_if_order_facts_missing(self):
        ab = AnalyticsBuilder(clean_data_folder="dummy")
        with pytest.raises(ValueError):
            ab._add_delay_metrics()

    def test_add_delay_metrics_order_approval_delay(self, builder):
        builder._build_order_facts()
        builder._add_delay_metrics()
        # o1: approved Jan 2 - purchased Jan 1 = 1 day
        # o2: approved Jan 6 - purchased Jan 5 = 1 day
        assert builder.order_facts["order_approval_delay"].iloc[0] == 1
        assert builder.order_facts["order_approval_delay"].iloc[1] == 1

    def test_add_delay_metrics_delivery_days(self, builder):
        builder._build_order_facts()
        builder._add_delay_metrics()
        # o1: delivered Jan 10 - purchased Jan 1 = 9 days
        # o2: delivered Jan 8  - purchased Jan 5 = 3 days
        assert builder.order_facts["delivery_days"].iloc[0] == 9
        assert builder.order_facts["delivery_days"].iloc[1] == 3

    def test_add_delay_metrics_estimated_delivery_days(self, builder):
        builder._build_order_facts()
        builder._add_delay_metrics()
        # o1: estimated Jan 8  - purchased Jan 1 = 7 days
        # o2: estimated Jan 10 - purchased Jan 5 = 5 days
        assert builder.order_facts["estimated_delivery_days"].iloc[0] == 7
        assert builder.order_facts["estimated_delivery_days"].iloc[1] == 5

    def test_add_delay_metrics_is_delayed_flag(self, builder):
        builder._build_order_facts()
        builder._add_delay_metrics()
        # o1: delivered 2 days after estimate → delayed
        # o2: delivered 2 days before estimate → not delayed
        assert builder.order_facts["is_delayed"].iloc[0] == True
        assert builder.order_facts["is_delayed"].iloc[1] == False

    def test_save_order_fact_creates_file(self, builder, tmp_path):
        builder._build_order_facts()
        builder._add_delay_metrics()
        builder.save_order_fact(str(tmp_path))
        assert (tmp_path / "order_fact_table.csv").exists()

    def test_save_order_fact_file_is_readable(self, builder, tmp_path):
        builder._build_order_facts()
        builder._add_delay_metrics()
        builder.save_order_fact(str(tmp_path))
        saved = pd.read_csv(tmp_path / "order_fact_table.csv")
        assert len(saved) == len(builder.order_facts)

    def test_build_order_products_facts_raises_if_order_items_missing(self):
        ab = AnalyticsBuilder(clean_data_folder="dummy")
        with pytest.raises(ValueError):
            ab._build_order_products_facts()

    def test_build_order_products_facts_stores_on_instance(self, builder):
        builder._build_order_products_facts()
        assert builder.order_products_facts is not None

    def test_build_order_products_facts_has_columns_from_both_tables(self, builder):
        builder._build_order_products_facts()
        assert "price" in builder.order_products_facts.columns
        assert "product_category" in builder.order_products_facts.columns

    def test_save_order_products_fact_creates_file(self, builder, tmp_path):
        builder._build_order_products_facts()
        builder.save_order_products_fact(str(tmp_path))
        assert (tmp_path / "order_products_table.csv").exists()

    def test_save_order_products_fact_file_is_readable(self, builder, tmp_path):
        builder._build_order_products_facts()
        builder.save_order_products_fact(str(tmp_path))
        saved = pd.read_csv(tmp_path / "order_products_table.csv")
        assert len(saved) == len(builder.order_products_facts)
