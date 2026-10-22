"""
Procurement and supplier performance helpers for the Supply Chain & Inventory Analytics Dashboard.

Definitions used everywhere in the project
------------------------------------------
* Procurement Spend = ordered value (Quantity Ordered x Unit Cost) of purchase-order
  lines, excluding cancelled orders. It is a commitment, not cash paid or goods received.
* A completed delivery is a purchase order with an Actual Delivery Date.
  Delivery metrics count each purchase order once (not once per line).
* On time = Actual Delivery Date is on or before the Expected Delivery Date.
* Lead time = days from Order Date to Actual Delivery Date (completed deliveries only).
* Delay days = Actual - Expected delivery date, averaged over late deliveries only.

All data is synthetic and for portfolio demonstration only.
"""

import math

import pandas as pd

# Suppliers below this on-time rate are marked "Needs review"
REVIEW_THRESHOLD = 75.0
# Suppliers with fewer completed orders than this are marked "Low volume" (too few orders to judge)
MIN_ORDERS_FOR_RATING = 5

SCORECARD_COLUMNS = [
    "Supplier_ID", "Supplier_Name", "Region", "Completed_Orders", "Total_Procurement_Value",
    "Avg_Lead_Time_Days", "On_Time_Orders", "On_Time_Rate", "On_Time_CI_Low", "On_Time_CI_High",
    "Late_Orders", "Avg_Delay_Days", "Performance_Flag",
]


# ---------------------------------------------------------------------------
# Preparation and filtering
# ---------------------------------------------------------------------------
def prepare_purchase_orders(purchase_orders):
    """Add lead time, delay and on-time columns (empty for orders not yet delivered)."""
    result = purchase_orders.copy()
    delivered = result["Actual_Delivery_Date"].notna()
    result["Is_Completed"] = delivered
    result["Lead_Time_Days"] = (result["Actual_Delivery_Date"] - result["Order_Date"]).dt.days
    result["Delay_Days"] = (result["Actual_Delivery_Date"] - result["Expected_Delivery_Date"]).dt.days
    result["Is_On_Time"] = delivered & (result["Delay_Days"] <= 0)
    result["Order_Month"] = result["Order_Date"].dt.to_period("M").dt.to_timestamp()
    return result


def filter_purchase_orders(prepared_po, products, suppliers, start_date, end_date,
                           categories, supplier_ids, warehouses, statuses):
    """Attach product and supplier details, then apply the sidebar filters (Order Date range)."""
    product_info = products[["Product_ID", "Product_Name", "Category", "Warehouse"]]
    supplier_info = suppliers[["Supplier_ID", "Supplier_Name", "Region"]]
    merged = prepared_po.merge(product_info, on="Product_ID", how="left")
    merged = merged.merge(supplier_info, on="Supplier_ID", how="left")
    mask = (
        merged["Order_Date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
        & merged["Category"].isin(categories)
        & merged["Supplier_ID"].isin(supplier_ids)
        & merged["Warehouse"].isin(warehouses)
        & merged["Delivery_Status"].isin(statuses)
    )
    return merged[mask].copy()


def completed_orders(po):
    """One row per completed purchase order (a PO delivers all its lines together)."""
    return po[po["Is_Completed"]].drop_duplicates("PO_ID")


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
def procurement_spend(po):
    """Ordered value of non-cancelled purchase-order lines."""
    if po.empty:
        return 0.0
    return float(po.loc[po["Delivery_Status"] != "Cancelled", "Total_Cost"].sum())


def on_time_rate(po):
    """On-time completed deliveries / completed deliveries x 100. None if nothing is completed."""
    done = completed_orders(po)
    if done.empty:
        return None
    return float(done["Is_On_Time"].mean() * 100)


def average_lead_time(po):
    """Average days from order to actual receipt over completed deliveries. None if nothing is completed."""
    done = completed_orders(po)
    if done.empty:
        return None
    return float(done["Lead_Time_Days"].mean())


# ---------------------------------------------------------------------------
# Procurement views
# ---------------------------------------------------------------------------
def monthly_spend(po):
    """Ordered value per order month (cancelled orders excluded)."""
    active = po[po["Delivery_Status"] != "Cancelled"]
    result = active.groupby("Order_Month", as_index=False)["Total_Cost"].sum()
    return result.rename(columns={"Total_Cost": "Procurement_Spend"})


def spend_by_supplier(po):
    """Ordered value per supplier, largest first."""
    active = po[po["Delivery_Status"] != "Cancelled"]
    result = active.groupby("Supplier_Name", as_index=False)["Total_Cost"].sum()
    result = result.rename(columns={"Total_Cost": "Procurement_Spend"})
    return result.sort_values("Procurement_Spend", ascending=False).reset_index(drop=True)


def status_distribution(po):
    """Number of purchase orders per delivery status (each PO counted once)."""
    orders = po.drop_duplicates("PO_ID")
    result = orders.groupby("Delivery_Status", as_index=False)["PO_ID"].count()
    return result.rename(columns={"PO_ID": "Purchase_Orders"}).sort_values("Purchase_Orders", ascending=False)


def monthly_spend_by_category(po):
    """Ordered value per order month and product category (cancelled orders excluded)."""
    active = po[po["Delivery_Status"] != "Cancelled"]
    return active.pivot_table(index="Order_Month", columns="Category", values="Total_Cost",
                              aggfunc="sum", fill_value=0.0)


def quarterly_status_mix(po):
    """Number of purchase orders per order quarter and delivery status."""
    orders = po.drop_duplicates("PO_ID")
    quarter = orders["Order_Date"].dt.year.astype(str) + " Q" + orders["Order_Date"].dt.quarter.astype(str)
    return orders.groupby([quarter, orders["Delivery_Status"]]).size().unstack(fill_value=0)


# ---------------------------------------------------------------------------
# Supplier scorecard
# ---------------------------------------------------------------------------
def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval for a rate, in percent. Useful because each supplier has few orders."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (centre - half_width) * 100, (centre + half_width) * 100



def flag_performance(on_time_rate_value, completed, overall_rate):
    """Label a supplier from its on-time rate (rules are documented in the README)."""
    if completed < MIN_ORDERS_FOR_RATING or pd.isna(on_time_rate_value):
        return "Low volume"
    if on_time_rate_value < REVIEW_THRESHOLD:
        return "Needs review"
    if overall_rate is not None and on_time_rate_value < overall_rate:
        return "Below average"
    return "On track"


def supplier_scorecard(po, suppliers):
    """One row per supplier with delivery performance and procurement value."""
    base = suppliers[suppliers["Supplier_ID"].isin(po["Supplier_ID"])][["Supplier_ID", "Supplier_Name", "Region"]]
    if base.empty:
        return pd.DataFrame(columns=SCORECARD_COLUMNS)

    active = po[po["Delivery_Status"] != "Cancelled"]
    value = active.groupby("Supplier_ID")["Total_Cost"].sum().rename("Total_Procurement_Value")

    done = completed_orders(po)
    delivery = done.groupby("Supplier_ID").agg(
        Completed_Orders=("PO_ID", "count"),
        Avg_Lead_Time_Days=("Lead_Time_Days", "mean"),
        On_Time_Orders=("Is_On_Time", "sum"),
        On_Time_Rate=("Is_On_Time", "mean"),
    )
    delivery["On_Time_Rate"] = delivery["On_Time_Rate"] * 100

    late = done[done["Delay_Days"] > 0]
    late_stats = late.groupby("Supplier_ID").agg(Late_Orders=("PO_ID", "count"), Avg_Delay_Days=("Delay_Days", "mean"))

    scorecard = (base.set_index("Supplier_ID")
                 .join(value).join(delivery).join(late_stats)
                 .reset_index())
    scorecard["Total_Procurement_Value"] = scorecard["Total_Procurement_Value"].fillna(0.0)
    scorecard["Completed_Orders"] = scorecard["Completed_Orders"].fillna(0).astype(int)
    scorecard["Late_Orders"] = scorecard["Late_Orders"].fillna(0).astype(int)
    scorecard["On_Time_Orders"] = scorecard["On_Time_Orders"].fillna(0).astype(int)
    intervals = [wilson_interval(k, n) for k, n in zip(scorecard["On_Time_Orders"], scorecard["Completed_Orders"])]
    scorecard["On_Time_CI_Low"] = [low for low, _ in intervals]
    scorecard["On_Time_CI_High"] = [high for _, high in intervals]
    # no late orders -> 0 average delay (only when there are completed orders to judge)
    has_completed = scorecard["Completed_Orders"] > 0
    scorecard.loc[has_completed, "Avg_Delay_Days"] = scorecard.loc[has_completed, "Avg_Delay_Days"].fillna(0.0)

    overall = on_time_rate(po)
    scorecard["Performance_Flag"] = [
        flag_performance(rate, completed, overall)
        for rate, completed in zip(scorecard["On_Time_Rate"], scorecard["Completed_Orders"])
    ]
    return scorecard[SCORECARD_COLUMNS].sort_values("On_Time_Rate", ascending=False, na_position="last").reset_index(drop=True)
