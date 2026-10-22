"""
Inventory analysis helpers for the Supply Chain & Inventory Analytics Dashboard.

Current-stock metrics (inventory value, stock status, low stock) describe the
snapshot in products.csv. Movement metrics (stock out, dispatched units) describe
a historical period and respond to the date filter.

All data is synthetic and for portfolio demonstration only.
"""

import math
from pathlib import Path

import pandas as pd

# Date of the stock snapshot stored in products.csv
SNAPSHOT_DATE = pd.Timestamp("2025-12-31")

STATUS_ORDER = ["Healthy Stock", "Low Stock", "Out of Stock"]

# A product with more than this many months of cover is treated as slow-moving
SLOW_MOVING_COVER_MONTHS = 12


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_data(data_dir):
    """Read the four CSV files and parse the date columns.

    Raises FileNotFoundError if a file is missing (the app shows a friendly message).
    """
    data_dir = Path(data_dir)
    products = pd.read_csv(data_dir / "products.csv")
    suppliers = pd.read_csv(data_dir / "suppliers.csv")
    purchase_orders = pd.read_csv(
        data_dir / "purchase_orders.csv",
        parse_dates=["Order_Date", "Expected_Delivery_Date", "Actual_Delivery_Date"],
    )
    movements = pd.read_csv(data_dir / "inventory_movements.csv", parse_dates=["Date"])
    return {
        "products": products,
        "suppliers": suppliers,
        "purchase_orders": purchase_orders,
        "inventory_movements": movements,
    }


# ---------------------------------------------------------------------------
# Stock status and inventory value (current snapshot)
# ---------------------------------------------------------------------------
def classify_stock_status(current_stock, reorder_point):
    """Out of Stock = 0 units; Low Stock = above 0 and at/below reorder point; otherwise Healthy Stock."""
    status = pd.Series("Healthy Stock", index=current_stock.index, dtype="object")
    status[(current_stock > 0) & (current_stock <= reorder_point)] = "Low Stock"
    status[current_stock <= 0] = "Out of Stock"
    return status


def suggested_reorder_quantity(current_stock, reorder_point):
    """Simple rule, not a forecast.

    For Low Stock and Out of Stock products, order enough to bring stock back up to
    2 x Reorder Point, rounded up to the next multiple of 5. Healthy products get 0.
    """
    shortfall = (2 * reorder_point - current_stock).clip(lower=0)
    rounded = (shortfall / 5).apply(math.ceil) * 5
    needs_reorder = current_stock <= reorder_point
    return rounded.where(needs_reorder, 0).astype(int)


def add_inventory_fields(products):
    """Add Inventory_Value, Stock_Status and Suggested_Reorder_Qty to the product table."""
    result = products.copy()
    result["Inventory_Value"] = (result["Current_Stock"] * result["Unit_Cost"]).round(2)
    result["Stock_Status"] = classify_stock_status(result["Current_Stock"], result["Reorder_Point"])
    result["Suggested_Reorder_Qty"] = suggested_reorder_quantity(result["Current_Stock"], result["Reorder_Point"])
    return result


def total_inventory_value(products):
    """Inventory Value = Current Stock x Unit Cost, summed over products."""
    if products.empty:
        return 0.0
    return float((products["Current_Stock"] * products["Unit_Cost"]).sum())


def count_low_stock(products):
    """Low-stock products: Current Stock at or below Reorder Point (includes out-of-stock products)."""
    if products.empty:
        return 0
    return int((products["Current_Stock"] <= products["Reorder_Point"]).sum())


def inventory_value_by_category(products):
    """Inventory value per category, largest first."""
    result = products.groupby("Category", as_index=False)["Inventory_Value"].sum()
    return result.sort_values("Inventory_Value", ascending=False).reset_index(drop=True)


def top_products_by_value(products, n=10):
    """The n products holding the most inventory value."""
    columns = ["Product_ID", "Product_Name", "Category", "Current_Stock", "Inventory_Value"]
    return products.sort_values("Inventory_Value", ascending=False).head(n)[columns].reset_index(drop=True)


def stock_status_counts(products):
    """Number of products per stock status (all three statuses always present)."""
    counts = products["Stock_Status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
    return counts.rename_axis("Stock_Status").reset_index(name="Products")


def inventory_by_warehouse(products):
    """Units on hand and value per warehouse and stock status."""
    result = products.groupby(["Warehouse", "Stock_Status"], as_index=False).agg(
        Units_On_Hand=("Current_Stock", "sum"),
        Inventory_Value=("Inventory_Value", "sum"),
        Products=("Product_ID", "count"),
    )
    return result


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def filter_products(products, categories, supplier_ids, warehouses):
    """Keep products matching the selected categories, suppliers and warehouses."""
    mask = (
        products["Category"].isin(categories)
        & products["Supplier_ID"].isin(supplier_ids)
        & products["Warehouse"].isin(warehouses)
    )
    return products[mask].copy()


def filter_movements(movements, products, start_date, end_date, categories, supplier_ids, warehouses):
    """Filter movements by date and by the product attributes (category, supplier, warehouse)."""
    product_info = products[["Product_ID", "Product_Name", "Category", "Supplier_ID"]]
    merged = movements.merge(product_info, on="Product_ID", how="left")
    mask = (
        merged["Date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
        & merged["Category"].isin(categories)
        & merged["Supplier_ID"].isin(supplier_ids)
        & merged["Warehouse"].isin(warehouses)
    )
    return merged[mask].copy()


# ---------------------------------------------------------------------------
# Movement and demand analysis (selected period)
# ---------------------------------------------------------------------------
def active_product_count(movements):
    """Active products = products with at least one stock movement in the selected period."""
    if movements.empty:
        return 0
    return int(movements["Product_ID"].nunique())


def monthly_stock_out(movements, start_date, end_date):
    """Units dispatched (Stock Out) per month, with zero-filled gaps."""
    months = pd.date_range(pd.Timestamp(start_date).to_period("M").to_timestamp(), pd.Timestamp(end_date), freq="MS")
    stock_out = movements[movements["Movement_Type"] == "Stock Out"]
    if stock_out.empty:
        return pd.DataFrame({"Month": months, "Units_Dispatched": 0, "Dispatches": 0})
    month_key = stock_out["Date"].dt.to_period("M").dt.to_timestamp()
    grouped = stock_out.groupby(month_key).agg(Units_Dispatched=("Quantity", "sum"), Dispatches=("Movement_ID", "count"))
    grouped = grouped.reindex(months, fill_value=0)
    grouped.index.name = "Month"
    return grouped.reset_index()


def top_dispatched_products(movements, n=10):
    """Products with the most units dispatched in the selected period."""
    stock_out = movements[movements["Movement_Type"] == "Stock Out"]
    if stock_out.empty:
        return pd.DataFrame(columns=["Product_ID", "Product_Name", "Category", "Units_Dispatched", "Dispatches"])
    result = stock_out.groupby(["Product_ID", "Product_Name", "Category"], as_index=False).agg(
        Units_Dispatched=("Quantity", "sum"), Dispatches=("Movement_ID", "count"))
    return result.sort_values("Units_Dispatched", ascending=False).head(n).reset_index(drop=True)


def months_in_period(start_date, end_date):
    """Length of the selected period in months (at least 1)."""
    days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
    return max(days / 30.44, 1.0)


def cover_table(products, movements, start_date, end_date):
    """Add average monthly dispatch and months of cover to every product.

    Months of cover = Current Stock / average monthly units dispatched in the selected period.
    It is NaN when nothing was dispatched in the period.
    """
    table = products.copy()
    stock_out = movements[movements["Movement_Type"] == "Stock Out"]
    units = stock_out.groupby("Product_ID")["Quantity"].sum()
    months = months_in_period(start_date, end_date)
    table["Avg_Monthly_Dispatch"] = table["Product_ID"].map(units).fillna(0) / months
    table["Months_Of_Cover"] = table["Current_Stock"] / table["Avg_Monthly_Dispatch"].where(table["Avg_Monthly_Dispatch"] > 0)
    return table


def slow_moving_products(products, movements, start_date, end_date, cover_months=SLOW_MOVING_COVER_MONTHS):
    """Products in stock with more than `cover_months` of cover, or with no dispatches in the period."""
    table = cover_table(products, movements, start_date, end_date)
    in_stock = table[table["Current_Stock"] > 0]
    slow = (in_stock["Months_Of_Cover"] > cover_months) | in_stock["Months_Of_Cover"].isna()
    return in_stock[slow].sort_values("Inventory_Value", ascending=False).reset_index(drop=True)


def status_mix_by_category(products):
    """Number of products per category and stock status (counts), riskiest categories first."""
    counts = products.pivot_table(index="Category", columns="Stock_Status", values="Product_ID",
                                  aggfunc="count", fill_value=0).reindex(columns=STATUS_ORDER, fill_value=0)
    at_risk_share = (counts["Low Stock"] + counts["Out of Stock"]) / counts.sum(axis=1)
    return counts.loc[at_risk_share.sort_values(ascending=False).index]


def seasonality_index(movements):
    """Average units dispatched per calendar month, divided by the category's monthly average.

    1.0 = an average month for that category; 1.3 = 30% above average.
    """
    stock_out = movements[movements["Movement_Type"] == "Stock Out"]
    if stock_out.empty:
        return pd.DataFrame(columns=range(1, 13))
    month_key = stock_out["Date"].dt.to_period("M").dt.to_timestamp()
    monthly = stock_out.groupby(["Category", month_key])["Quantity"].sum().reset_index()
    monthly["Month_Number"] = monthly["Date"].dt.month
    average = monthly.groupby(["Category", "Month_Number"])["Quantity"].mean().unstack()
    average = average.reindex(columns=range(1, 13))
    return average.div(average.mean(axis=1), axis=0)
