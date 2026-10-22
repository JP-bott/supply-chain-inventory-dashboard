"""
Validate the synthetic datasets in data/.

Checks primary keys, relationships, purchase-order maths, delivery status logic,
and that Current_Stock reconciles with opening balances and inventory movements.

Usage (from the project root):
    python scripts/validate_data.py

Exit code is 0 when every check passes, 1 otherwise.
"""

import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SNAPSHOT_DATE = pd.Timestamp("2025-12-31")

results = []


def check(name, passed, detail=""):
    """Record and print one check result."""
    results.append(bool(passed))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def main():
    files = ["products", "suppliers", "purchase_orders", "inventory_movements", "opening_balances"]
    data = {}
    for name in files:
        path = DATA_DIR / f"{name}.csv"
        check(f"{name}.csv exists and loads", path.exists())
        if not path.exists():
            return 1
        data[name] = pd.read_csv(path)

    products, suppliers = data["products"], data["suppliers"]
    po, mov, opening = data["purchase_orders"], data["inventory_movements"], data["opening_balances"]
    for col in ["Order_Date", "Expected_Delivery_Date", "Actual_Delivery_Date"]:
        po[col] = pd.to_datetime(po[col])
    mov["Date"] = pd.to_datetime(mov["Date"])

    # --- sizes ---
    check("150-250 products", 150 <= len(products) <= 250, len(products))
    check("15-25 suppliers", 15 <= len(suppliers) <= 25, len(suppliers))
    check("1,000-1,500 PO lines", 1000 <= len(po) <= 1500, len(po))
    check("3,000-5,000 movements", 3000 <= len(mov) <= 5000, len(mov))

    # --- primary keys ---
    check("Product_ID unique", products["Product_ID"].is_unique)
    check("Supplier_ID unique", suppliers["Supplier_ID"].is_unique)
    check("PO_Line_ID unique", po["PO_Line_ID"].is_unique)
    check("Movement_ID unique", mov["Movement_ID"].is_unique)
    check("No duplicate product within a PO", not po.duplicated(["PO_ID", "Product_ID"]).any())

    # --- foreign keys ---
    check("products.Supplier_ID in suppliers", products["Supplier_ID"].isin(suppliers["Supplier_ID"]).all())
    check("PO.Supplier_ID in suppliers", po["Supplier_ID"].isin(suppliers["Supplier_ID"]).all())
    check("PO.Product_ID in products", po["Product_ID"].isin(products["Product_ID"]).all())
    check("movements.Product_ID in products", mov["Product_ID"].isin(products["Product_ID"]).all())
    check("opening_balances covers every product once",
          opening["Product_ID"].is_unique and set(opening["Product_ID"]) == set(products["Product_ID"]))
    supplier_of_product = products.set_index("Product_ID")["Supplier_ID"]
    check("PO supplier matches the product's supplier",
          (po["Product_ID"].map(supplier_of_product) == po["Supplier_ID"]).all())
    warehouse_of_product = products.set_index("Product_ID")["Warehouse"]
    check("Movement warehouse matches product warehouse",
          (mov["Product_ID"].map(warehouse_of_product) == mov["Warehouse"]).all())

    # --- missing values ---
    required = {"products": products, "suppliers": suppliers}
    check("No missing values in products / suppliers", all(not df.isna().any().any() for df in required.values()))
    check("No missing values in movements", not mov.isna().any().any())
    po_required = po.drop(columns=["Actual_Delivery_Date"])
    check("PO has no missing values except Actual_Delivery_Date", not po_required.isna().any().any())

    # --- value ranges ---
    check("All stock, cost and price values are valid",
          (products["Current_Stock"] >= 0).all() and (products["Unit_Cost"] > 0).all()
          and (products["Selling_Price"] > products["Unit_Cost"]).all())
    check("Safety_Stock < Reorder_Point", (products["Safety_Stock"] < products["Reorder_Point"]).all())
    check("Dates inside Jan 2024 - Dec 2025",
          po["Order_Date"].between("2024-01-01", "2025-12-31").all()
          and mov["Date"].between("2024-01-01", "2025-12-31").all())

    # --- purchase-order maths ---
    recomputed = (po["Quantity_Ordered"] * po["Unit_Cost"]).round(2)
    check("Total_Cost = Quantity_Ordered x Unit_Cost", (recomputed - po["Total_Cost"]).abs().max() < 0.005)
    check("Expected delivery is after order date", (po["Expected_Delivery_Date"] > po["Order_Date"]).all())
    delivered = po["Actual_Delivery_Date"].notna()
    check("Actual delivery is after order date", (po.loc[delivered, "Actual_Delivery_Date"] > po.loc[delivered, "Order_Date"]).all())
    check("No delivery dated after the snapshot date", (po.loc[delivered, "Actual_Delivery_Date"] <= SNAPSHOT_DATE).all())

    # --- delivery status logic ---
    status = po["Delivery_Status"]
    on_time = delivered & (po["Actual_Delivery_Date"] <= po["Expected_Delivery_Date"])
    late = delivered & (po["Actual_Delivery_Date"] > po["Expected_Delivery_Date"])
    check("'Delivered On Time' matches dates", (status == "Delivered On Time").equals(on_time))
    check("'Delivered Late' matches dates", (status == "Delivered Late").equals(late))
    open_lines = ~delivered & (status != "Cancelled")
    check("Open lines are 'Overdue' or 'In Transit' by expected date",
          ((status == "Overdue") == (open_lines & (po["Expected_Delivery_Date"] < SNAPSHOT_DATE))).all()
          and ((status == "In Transit") == (open_lines & (po["Expected_Delivery_Date"] >= SNAPSHOT_DATE))).all())
    check("Cancelled lines have no delivery", ((status == "Cancelled") <= ~delivered).all())
    check("Quantity_Received = ordered when delivered, else 0",
          (po["Quantity_Received"] == po["Quantity_Ordered"].where(delivered, 0)).all())
    check("PO header fields are consistent across lines",
          po.groupby("PO_ID")[["Supplier_ID", "Order_Date", "Expected_Delivery_Date", "Delivery_Status"]].nunique().max().max() == 1)

    # --- receipts are counted once ---
    stock_in = mov[mov["Movement_Type"] == "Stock In"]
    received_lines = po[po["Quantity_Received"] > 0]
    check("Every Stock In references a received PO line, once",
          stock_in["Reference_ID"].is_unique and set(stock_in["Reference_ID"]) == set(received_lines["PO_Line_ID"]))
    merged = stock_in.merge(received_lines, left_on="Reference_ID", right_on="PO_Line_ID", suffixes=("_mov", "_po"))
    check("Stock In quantity, product and date match the PO line",
          (merged["Quantity"] == merged["Quantity_Received"]).all()
          and (merged["Product_ID_mov"] == merged["Product_ID_po"]).all()
          and (merged["Date"] == merged["Actual_Delivery_Date"]).all())
    check("Movement types are valid", set(mov["Movement_Type"]) <= {"Stock In", "Stock Out", "Adjustment"})
    check("Stock In / Stock Out quantities are positive",
          (mov.loc[mov["Movement_Type"] != "Adjustment", "Quantity"] > 0).all())

    # --- stock reconciliation ---
    signed = mov["Quantity"].where(mov["Movement_Type"] != "Stock Out", -mov["Quantity"])
    net_change = signed.groupby(mov["Product_ID"]).sum()
    reconciled = products.set_index("Product_ID")["Current_Stock"]
    expected = opening.set_index("Product_ID")["Opening_Stock"].add(net_change, fill_value=0)
    check("Current_Stock = Opening + In - Out + Adjustments (all products)",
          (expected.reindex(reconciled.index) == reconciled).all())

    # running balance must never go negative
    ordered = mov.assign(_signed=signed, _prio=mov["Movement_Type"].map({"Stock In": 0, "Adjustment": 1, "Stock Out": 2}))
    ordered = ordered.sort_values(["Product_ID", "Date", "_prio"])
    running = ordered.groupby("Product_ID")["_signed"].cumsum() + ordered["Product_ID"].map(opening.set_index("Product_ID")["Opening_Stock"])
    check("Running stock balance never negative", (running >= 0).all(), f"minimum {running.min()}")

    # --- stock status classification ---
    stock, rop = products["Current_Stock"], products["Reorder_Point"]
    status_counts = pd.Series("Healthy Stock", index=products.index)
    status_counts[(stock > 0) & (stock <= rop)] = "Low Stock"
    status_counts[stock == 0] = "Out of Stock"
    counts = status_counts.value_counts()
    check("All three stock statuses are present", set(counts.index) == {"Healthy Stock", "Low Stock", "Out of Stock"},
          counts.to_dict())

    # --- KPI cross-check against the analysis modules ---
    sys.path.insert(0, str(DATA_DIR.parent))
    from src import inventory_analysis as inv
    from src import supplier_analysis as sup

    loaded = inv.load_data(DATA_DIR)
    prod_enriched = inv.add_inventory_fields(loaded["products"])
    check("KPI: inventory value matches manual sum",
          abs(inv.total_inventory_value(prod_enriched) - float((stock * products["Unit_Cost"]).sum())) < 0.01)
    check("KPI: low-stock count matches manual count", inv.count_low_stock(prod_enriched) == int((stock <= rop).sum()))

    po_prepared = sup.prepare_purchase_orders(loaded["purchase_orders"])
    manual_spend = float(po.loc[status != "Cancelled", "Total_Cost"].sum())
    check("KPI: procurement spend (ordered value, excl. cancelled)",
          abs(sup.procurement_spend(po_prepared) - manual_spend) < 0.01)

    # On-time rate computed with a plain loop over POs
    on_time_count, completed_count, lead_days = 0, 0, []
    for _, one_po in po[delivered].drop_duplicates("PO_ID").iterrows():
        completed_count += 1
        on_time_count += int(one_po["Actual_Delivery_Date"] <= one_po["Expected_Delivery_Date"])
        lead_days.append((one_po["Actual_Delivery_Date"] - one_po["Order_Date"]).days)
    check("KPI: on-time delivery rate matches loop calculation",
          abs(sup.on_time_rate(po_prepared) - on_time_count / completed_count * 100) < 1e-9)
    check("KPI: average lead time matches loop calculation",
          abs(sup.average_lead_time(po_prepared) - sum(lead_days) / len(lead_days)) < 1e-9)

    scorecard = sup.supplier_scorecard(po_prepared, loaded["suppliers"])
    check("Scorecard completed orders add up to total completed POs", int(scorecard["Completed_Orders"].sum()) == completed_count)
    check("Scorecard procurement value adds up to total spend",
          abs(scorecard["Total_Procurement_Value"].sum() - manual_spend) < 0.01)

    rated = scorecard[scorecard["Completed_Orders"] > 0]
    check("Scorecard 95% intervals contain the observed on-time rate and stay within 0-100%",
          ((rated["On_Time_CI_Low"] <= rated["On_Time_Rate"] + 1e-9) & (rated["On_Time_Rate"] <= rated["On_Time_CI_High"] + 1e-9)
           & (rated["On_Time_CI_Low"] >= 0) & (rated["On_Time_CI_High"] <= 100)).all())
    low, high = sup.wilson_interval(0, 0)
    check("Wilson interval is undefined (NaN) with zero orders", pd.isna(low) and pd.isna(high))

    print()
    passed = sum(results)
    print(f"{passed} of {len(results)} checks passed.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
