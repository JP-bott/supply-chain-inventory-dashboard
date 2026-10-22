"""
Generate the synthetic datasets for the Supply Chain & Inventory Analytics Dashboard.

All data is fictional and created for portfolio demonstration only.
The script uses a fixed random seed, so running it again produces identical files.

Usage (from the project root):
    python scripts/generate_data.py

Output files (in data/):
    suppliers.csv, products.csv, purchase_orders.csv,
    inventory_movements.csv, opening_balances.csv

Inventory logic
---------------
    Current_Stock = Opening_Stock + Stock In - Stock Out + Adjustments

* Stock In rows are created ONLY from purchase-order lines that were received
  (one row per received PO line, dated on the actual delivery date).
* Stock Out rows are customer dispatches. A dispatch is never larger than the
  stock on hand, so the running balance never goes below zero.
* Adjustment rows are signed (negative = shrinkage / damage, positive = count correction).
* Opening stock is calculated per product so the final stock lands on a planned level.
"""

from pathlib import Path

import math

import numpy as np
import pandas as pd

SEED = 42
START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2025-12-31")  # snapshot date: "today" inside the dataset
N_MONTHS = 24

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Product categories: item names, variants, cost range, product count and a
# monthly demand pattern (index 1.0 = average month).
CATEGORY_CONFIG = {
    "Packaging Materials": {
        "items": ["Corrugated Shipping Box", "Bubble Wrap Roll", "Packing Tape", "Stretch Wrap Film",
                  "Kraft Mailer", "Foam Insert", "Pallet Strapping", "Poly Bag"],
        "variants": ["Small", "Medium", "Large", "Heavy-Duty"],
        "cost": (0.30, 12.0), "count": 28,
        "season": [0.85, 0.85, 0.90, 0.90, 0.95, 0.95, 1.00, 1.05, 1.10, 1.20, 1.35, 1.40],
    },
    "Industrial Supplies": {
        "items": ["Hex Bolt Set", "Hydraulic Hose", "Ball Bearing", "Conveyor Belt Segment",
                  "Pipe Fitting", "Steel Bracket", "Gasket Kit", "Drive Chain"],
        "variants": ["Standard", "Heavy-Duty", "Stainless", "Compact"],
        "cost": (2.0, 150.0), "count": 26,
        "season": [0.95, 0.95, 1.00, 1.05, 1.05, 1.05, 1.00, 1.00, 1.00, 1.00, 0.95, 0.95],
    },
    "Office Supplies": {
        "items": ["Copy Paper Ream", "Ballpoint Pens Box", "Stapler", "Filing Folders Pack",
                  "Sticky Notes Pack", "Toner Cartridge", "Binder Clips Box", "Whiteboard Markers"],
        "variants": ["Standard", "Value Pack", "Premium", "Eco"],
        "cost": (0.50, 25.0), "count": 24,
        "season": [1.10, 0.95, 0.95, 0.90, 0.90, 0.90, 1.00, 1.20, 1.20, 1.00, 0.95, 0.95],
    },
    "Electronics Accessories": {
        "items": ["USB-C Cable", "HDMI Cable", "Power Adapter", "Wireless Mouse",
                  "Keyboard", "Cable Organizer", "Surge Protector", "Webcam"],
        "variants": ["Basic", "Standard", "Pro", "Compact"],
        "cost": (2.0, 90.0), "count": 25,
        "season": [0.80, 0.80, 0.85, 0.85, 0.90, 0.90, 0.95, 1.00, 1.00, 1.15, 1.40, 1.40],
    },
    "Safety Equipment": {
        "items": ["Safety Gloves", "Hard Hat", "Safety Goggles", "High-Visibility Vest",
                  "Ear Protection", "Respirator Mask", "Steel-Toe Boots", "First Aid Kit"],
        "variants": ["Small", "Medium", "Large", "Universal"],
        "cost": (3.0, 80.0), "count": 22,
        "season": [0.80, 0.85, 1.00, 1.15, 1.20, 1.20, 1.15, 1.10, 1.00, 0.95, 0.85, 0.80],
    },
    "Cleaning Supplies": {
        "items": ["All-Purpose Cleaner", "Disinfectant Wipes", "Floor Cleaner", "Microfiber Cloths",
                  "Trash Bags", "Hand Soap Refill", "Mop Head", "Glass Cleaner"],
        "variants": ["Standard", "Value Pack", "Bulk Pack", "Commercial Grade"],
        "cost": (1.0, 35.0), "count": 25,
        "season": [1.15, 1.05, 1.10, 1.00, 0.95, 0.90, 0.90, 0.95, 1.00, 1.00, 1.00, 1.10],
    },
    "Tools & Hardware": {
        "items": ["Cordless Drill", "Socket Wrench Set", "Utility Knife", "Measuring Tape",
                  "Screwdriver Set", "Claw Hammer", "Adjustable Wrench", "Work Light"],
        "variants": ["Standard", "Professional", "Compact", "Heavy-Duty"],
        "cost": (3.0, 200.0), "count": 26,
        "season": [0.85, 0.90, 1.10, 1.20, 1.20, 1.10, 1.00, 1.00, 0.95, 0.95, 0.95, 0.85],
    },
    "Food Service Supplies": {
        "items": ["Paper Cups", "Takeout Container", "Disposable Cutlery Set", "Paper Napkins",
                  "Food Wrap Film", "Aluminium Foil Pan", "Cup Lids", "Paper Straws"],
        "variants": ["Small", "Medium", "Large", "Bulk Pack"],
        "cost": (0.20, 20.0), "count": 24,
        "season": [0.90, 0.90, 0.95, 1.00, 1.10, 1.20, 1.20, 1.15, 1.00, 0.95, 1.00, 1.15],
    },
}

# Fictional suppliers (name, product category they supply)
SUPPLIERS = [
    ("Pinecrest Packaging", "Packaging Materials"),
    ("BlueHarbor Containers", "Packaging Materials"),
    ("Alder & Finch Paperworks", "Packaging Materials"),
    ("Ironvale Industrial", "Industrial Supplies"),
    ("Granite Ridge Supply", "Industrial Supplies"),
    ("Kestrel Fasteners", "Industrial Supplies"),
    ("Larkspur Office Goods", "Office Supplies"),
    ("Summit Paper & Stationery", "Office Supplies"),
    ("Voltline Components", "Electronics Accessories"),
    ("Cobalt Peak Electronics", "Electronics Accessories"),
    ("Nimbus Cable Co.", "Electronics Accessories"),
    ("Redwood Safety Gear", "Safety Equipment"),
    ("Beacon Protective Wear", "Safety Equipment"),
    ("Clearwater Chemicals", "Cleaning Supplies"),
    ("Fresh Meadow Hygiene", "Cleaning Supplies"),
    ("Sparkwell Supply", "Cleaning Supplies"),
    ("Forge & Anvil Tools", "Tools & Hardware"),
    ("Ridgeback Hardware", "Tools & Hardware"),
    ("Tallgrass Tool Works", "Tools & Hardware"),
    ("Harvest Table Disposables", "Food Service Supplies"),
    ("Copperpot Foodservice", "Food Service Supplies"),
]

# Typical quoted lead time (days) by supplier region
REGION_LEAD_TIME = {
    "North America": (5, 12),
    "Europe": (14, 24),
    "Asia Pacific": (22, 38),
    "Latin America": (10, 20),
}

WAREHOUSES = ["Northgate DC", "Riverside DC", "Southport DC"]

# Demand classes: expected dispatch events per month, and PO lines per product over 24 months
DEMAND_CLASS = {
    "fast": {"share": 0.22, "events_per_month": 1.3, "po_lines": 13.0},
    "medium": {"share": 0.48, "events_per_month": 0.5, "po_lines": 5.0},
    "slow": {"share": 0.30, "events_per_month": 0.08, "po_lines": 1.5},
}

# Planned end-of-period stock status by demand class: (P(out of stock), P(low stock))
STATUS_MIX = {"fast": (0.06, 0.20), "medium": (0.04, 0.13), "slow": (0.02, 0.07)}


# ---------------------------------------------------------------------------
# Suppliers
# ---------------------------------------------------------------------------
def make_suppliers(rng):
    """Build the supplier table plus hidden reliability profiles used to create delays."""
    rows, profiles = [], {}
    regions = list(REGION_LEAD_TIME)
    on_hold = set(rng.choice(len(SUPPLIERS), size=2, replace=False).tolist())

    for i, (name, category) in enumerate(SUPPLIERS):
        supplier_id = f"SUP-{i + 1:02d}"
        region = str(rng.choice(regions, p=[0.35, 0.20, 0.30, 0.15]))
        low, high = REGION_LEAD_TIME[region]
        lead_time = int(rng.integers(low, high + 1))
        rows.append({
            "Supplier_ID": supplier_id,
            "Supplier_Name": name,
            "Region": region,
            "Supplier_Category": category,
            "Average_Lead_Time_Days": lead_time,
            "Contact_Status": "On Hold" if i in on_hold else "Active",
        })
        # Probability a delivery is on time, and typical number of extra late days.
        on_time_prob = float(np.clip(rng.beta(10, 2.2), 0.55, 0.98))
        if lead_time >= 25:
            on_time_prob -= 0.05  # long-distance shipments slip a little more often
        profiles[supplier_id] = {"on_time_prob": on_time_prob, "mean_extra_delay": float(rng.uniform(1.5, 7.0))}

    return pd.DataFrame(rows), profiles


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------
def make_products(rng, suppliers):
    """Build the product master (without Current_Stock) and a planning table."""
    lead_time_by_supplier = suppliers.set_index("Supplier_ID")["Average_Lead_Time_Days"]
    class_names = list(DEMAND_CLASS)
    class_probs = [DEMAND_CLASS[c]["share"] for c in class_names]

    rows, plan_rows = [], []
    product_number = 1
    for category, cfg in CATEGORY_CONFIG.items():
        combos = [(item, variant) for item in cfg["items"] for variant in cfg["variants"]]
        chosen = rng.choice(len(combos), size=cfg["count"], replace=False)
        supplier_ids = suppliers.loc[suppliers["Supplier_Category"] == category, "Supplier_ID"].tolist()
        supplier_order = rng.permutation(cfg["count"])  # spreads products evenly over suppliers

        for position, combo_index in enumerate(chosen):
            item, variant = combos[combo_index]
            product_id = f"PRD-{product_number:03d}"
            supplier_id = supplier_ids[supplier_order[position] % len(supplier_ids)]
            low, high = cfg["cost"]
            unit_cost = round(float(np.exp(rng.uniform(np.log(low), np.log(high)))), 2)
            selling_price = round(unit_cost * float(rng.uniform(1.20, 1.60)), 2)

            demand_class = str(rng.choice(class_names, p=class_probs))
            events_per_month = DEMAND_CLASS[demand_class]["events_per_month"]
            # cheaper items are dispatched in bigger quantities
            base_size = int(np.clip(round(60 / unit_cost ** 0.35 * rng.uniform(0.6, 1.4)), 3, 250))
            monthly_units = events_per_month * base_size
            lead_time = int(lead_time_by_supplier[supplier_id])

            safety_stock = max(5, int(round(0.4 * monthly_units)))
            reorder_point = max(15, safety_stock + 5, int(round(monthly_units * lead_time / 30 + safety_stock)))

            rows.append({
                "Product_ID": product_id,
                "Product_Name": f"{item} - {variant}",
                "Category": category,
                "Supplier_ID": supplier_id,
                "Unit_Cost": unit_cost,
                "Selling_Price": selling_price,
                "Reorder_Point": reorder_point,
                "Safety_Stock": safety_stock,
                "Warehouse": str(rng.choice(WAREHOUSES, p=[0.40, 0.35, 0.25])),
            })
            plan_rows.append({
                "Product_ID": product_id, "demand_class": demand_class, "base_size": base_size,
                "monthly_units": monthly_units, "events_per_month": events_per_month,
            })
            product_number += 1

    products = pd.DataFrame(rows)
    plan = pd.DataFrame(plan_rows).set_index("Product_ID")
    return products, plan


# ---------------------------------------------------------------------------
# Purchase orders
# ---------------------------------------------------------------------------
def make_purchase_orders(rng, products, plan, suppliers, profiles):
    """Create PO lines. A PO groups lines for one supplier on one date and is delivered as a whole."""
    # 1) decide how many lines each product gets and how large they are
    lines = []
    for _, product in products.iterrows():
        info = plan.loc[product["Product_ID"]]
        n_lines = max(1, int(rng.poisson(DEMAND_CLASS[info["demand_class"]]["po_lines"])))
        expected_units = info["monthly_units"] * N_MONTHS * 1.12  # a little above demand: some POs are cancelled or still open
        for _ in range(n_lines):
            qty = expected_units / n_lines * rng.uniform(0.8, 1.25)
            qty = max(20, int(round(qty / 5) * 5))
            lines.append({"Product_ID": product["Product_ID"], "Supplier_ID": product["Supplier_ID"],
                          "Unit_Cost_Std": product["Unit_Cost"], "Quantity_Ordered": qty})
    lines = pd.DataFrame(lines)

    # 2) give each supplier a set of order dates and spread its product lines over them
    category_of_supplier = suppliers.set_index("Supplier_ID")["Supplier_Category"]
    weekdays = pd.bdate_range(START, END)
    order_dates = pd.Series(pd.NaT, index=lines.index, dtype="datetime64[ns]")
    for supplier_id, group in lines.groupby("Supplier_ID"):
        season = CATEGORY_CONFIG[category_of_supplier[supplier_id]]["season"]
        # orders are placed about a month before the demand peak
        weights = np.array([season[(d.month - 2) % 12] for d in weekdays], dtype=float)
        weights /= weights.sum()
        max_lines_per_product = group.groupby("Product_ID").size().max()
        n_dates = max(int(round(len(group) / 2.2)), int(max_lines_per_product) + 3)
        po_dates = rng.choice(weekdays.values, size=n_dates, replace=False, p=weights)
        for _, product_lines in group.groupby("Product_ID"):
            picked = rng.choice(po_dates, size=len(product_lines), replace=False)
            order_dates.loc[product_lines.index] = pd.to_datetime(picked)
    lines["Order_Date"] = order_dates

    # 3) one PO = one supplier on one order date
    po_table = (lines[["Order_Date", "Supplier_ID"]].drop_duplicates()
                .sort_values(["Order_Date", "Supplier_ID"]).reset_index(drop=True))
    po_table["PO_ID"] = [f"PO-{i:05d}" for i in range(1, len(po_table) + 1)]

    lead_quoted = suppliers.set_index("Supplier_ID")["Average_Lead_Time_Days"]
    expected_list, actual_list, status_list = [], [], []
    for _, po in po_table.iterrows():
        profile = profiles[po["Supplier_ID"]]
        quoted_days = max(3, int(lead_quoted[po["Supplier_ID"]]) + int(rng.integers(-2, 3)))
        expected = po["Order_Date"] + pd.Timedelta(days=quoted_days)

        if rng.random() < 0.02:  # a small share of orders is cancelled
            actual, status = pd.NaT, "Cancelled"
        else:
            if rng.random() < profile["on_time_prob"]:
                delta_days = -int(rng.choice([0, 0, 0, 1, 1, 2]))  # on time or a little early
            else:
                delta_days = 1 + int(rng.poisson(profile["mean_extra_delay"]))  # late
                if rng.random() < 0.08:  # occasional serious disruption adds a long delay
                    delta_days += int(rng.integers(8, 21))
            actual = expected + pd.Timedelta(days=delta_days)
            if actual > END:  # not received yet at the snapshot date
                actual = pd.NaT
                status = "Overdue" if expected < END else "In Transit"
            else:
                status = "Delivered On Time" if actual <= expected else "Delivered Late"
        expected_list.append(expected)
        actual_list.append(actual)
        status_list.append(status)
    po_table["Expected_Delivery_Date"] = expected_list
    po_table["Actual_Delivery_Date"] = pd.to_datetime(actual_list)
    po_table["Delivery_Status"] = status_list

    # 4) build PO lines
    purchase_orders = lines.merge(po_table, on=["Order_Date", "Supplier_ID"], how="left")
    purchase_orders = purchase_orders.sort_values(["PO_ID", "Product_ID"]).reset_index(drop=True)
    purchase_orders["Line_No"] = purchase_orders.groupby("PO_ID").cumcount() + 1
    purchase_orders["PO_Line_ID"] = purchase_orders["PO_ID"] + "-" + purchase_orders["Line_No"].map("{:02d}".format)

    # PO unit cost = standard cost with a small price drift (2025 is slightly more expensive)
    drift = rng.uniform(0.97, 1.04, size=len(purchase_orders))
    inflation = np.where(purchase_orders["Order_Date"].dt.year == 2025, 1.02, 1.0)
    purchase_orders["Unit_Cost"] = (purchase_orders["Unit_Cost_Std"] * drift * inflation).round(2)
    purchase_orders["Total_Cost"] = (purchase_orders["Quantity_Ordered"] * purchase_orders["Unit_Cost"]).round(2)

    received = purchase_orders["Actual_Delivery_Date"].notna()
    purchase_orders["Quantity_Received"] = np.where(received, purchase_orders["Quantity_Ordered"], 0)

    columns = ["PO_ID", "PO_Line_ID", "Order_Date", "Supplier_ID", "Product_ID", "Quantity_Ordered",
               "Unit_Cost", "Total_Cost", "Expected_Delivery_Date", "Actual_Delivery_Date",
               "Delivery_Status", "Quantity_Received"]
    return purchase_orders[columns]


# ---------------------------------------------------------------------------
# Inventory movements
# ---------------------------------------------------------------------------
def make_dispatch_events(rng, products, plan):
    """Random customer dispatches (Stock Out requests) for each product, month by month."""
    category_of = products.set_index("Product_ID")["Category"]
    month_starts = pd.date_range(START, END, freq="MS")
    weekdays_by_month = {m: pd.bdate_range(m, m + pd.offsets.MonthEnd(0)) for m in month_starts}

    events = {pid: [] for pid in products["Product_ID"]}
    for pid, info in plan.iterrows():
        season = CATEGORY_CONFIG[category_of[pid]]["season"]
        for month_start in month_starts:
            growth = 1.06 if month_start.year == 2025 else 1.0
            rate = info["events_per_month"] * season[month_start.month - 1] * growth
            for _ in range(int(rng.poisson(rate))):
                day = rng.choice(weekdays_by_month[month_start].values)
                qty = max(1, int(round(info["base_size"] * rng.lognormal(0, 0.35))))
                events[pid].append((pd.Timestamp(day), 2, "Stock Out", qty, None))
    return events


def make_adjustments(rng, products, n_adjustments=130):
    """Small signed stock adjustments: mostly shrinkage, some count corrections."""
    weekdays = pd.bdate_range(START, END)
    events = {pid: [] for pid in products["Product_ID"]}
    for _ in range(n_adjustments):
        pid = str(rng.choice(products["Product_ID"]))
        day = pd.Timestamp(rng.choice(weekdays.values))
        qty = int(rng.integers(1, 9)) if rng.random() < 0.2 else -int(rng.integers(1, 7))
        events[pid].append((day, 1, "Adjustment", qty, None))
    return events


def run_simulation(opening, events):
    """Replay events in date order. Returns closing stock and the events that actually happened."""
    stock = opening
    accepted = []
    for date, priority, kind, qty, ref in events:
        if kind == "Stock In":
            stock += qty
            accepted.append((date, priority, kind, qty, ref))
        elif kind == "Stock Out":
            shipped = min(qty, stock)  # cannot ship more than is on hand
            if shipped > 0:
                stock -= shipped
                accepted.append((date, priority, kind, shipped, ref))
        else:  # Adjustment (signed)
            change = qty if qty > 0 else -min(-qty, stock)
            if change != 0:
                stock += change
                accepted.append((date, priority, kind, change, ref))
    return stock, accepted


def find_opening_stock(events, target_end, start_from=0):
    """Smallest opening stock (>= start_from) whose closing stock reaches the target.

    Closing stock rises by 0 or 1 for every extra opening unit, so a binary search lands on the target.
    """
    if run_simulation(start_from, events)[0] >= target_end:
        return start_from
    total_out = sum(e[3] for e in events if e[2] == "Stock Out")
    total_adj = sum(abs(e[3]) for e in events if e[2] == "Adjustment")
    low, high = start_from, start_from + target_end + total_out + total_adj + 1
    while low < high:
        mid = (low + high) // 2
        if run_simulation(mid, events)[0] >= target_end:
            high = mid
        else:
            low = mid + 1
    return low


def add_extra_dispatches(rng, events, opening, target_end, base_size):
    """If a product would end far above its planned stock, add extra customer dispatches.

    The extra dispatches are spread over the whole period (not bunched at the end), so they
    do not create an artificial spike in the demand trend.
    """
    burn_days = pd.bdate_range(START, END).values
    for _ in range(8):
        closing = run_simulation(opening, events)[0]
        excess = closing - target_end
        if excess <= 0:
            break
        pieces = int(min(4, max(1, math.ceil(excess / max(base_size, 1)))))
        for qty in rng.multinomial(excess, [1 / pieces] * pieces):
            if qty > 0:
                events.append((pd.Timestamp(rng.choice(burn_days)), 2, "Stock Out", int(qty), None))
        events.sort(key=lambda e: (e[0], e[1]))
    return events


def build_inventory(rng, products, plan, purchase_orders):
    """Create movements and opening balances, and set Current_Stock on the product table."""
    dispatches = make_dispatch_events(rng, products, plan)
    adjustments = make_adjustments(rng, products)

    delivered = purchase_orders[purchase_orders["Quantity_Received"] > 0]
    receipts = {pid: [] for pid in products["Product_ID"]}
    for row in delivered.itertuples():
        # the PO line ID travels with the receipt, so each receipt is traceable and counted once
        receipts[row.Product_ID].append(
            (row.Actual_Delivery_Date, 0, "Stock In", int(row.Quantity_Received), row.PO_Line_ID))

    warehouse_of = products.set_index("Product_ID")["Warehouse"]
    movement_rows, opening_rows, closing_stock = [], [], {}

    for _, product in products.iterrows():
        pid = product["Product_ID"]
        info = plan.loc[pid]
        reorder_point = product["Reorder_Point"]

        # Plan the end-of-period stock level for this product
        p_out, p_low = STATUS_MIX[info["demand_class"]]
        draw = rng.random()
        if draw < p_out:
            status = "out"
        elif draw < p_out + p_low:
            status = "low"
        else:
            status = "healthy"
        if status == "out":
            target_end = int(round(reorder_point * rng.uniform(0.3, 0.9)))  # stock just before the last sale
            upper_end = reorder_point
        elif status == "low":
            target_end = int(rng.integers(max(1, int(0.3 * reorder_point)), reorder_point + 1))
            upper_end = reorder_point
        elif info["demand_class"] == "slow":
            target_end = int(round(reorder_point * rng.uniform(2.0, 4.0)))
            upper_end = int(round(reorder_point * 8.0))  # slow movers can pile up
        else:
            target_end = int(round(reorder_point * rng.uniform(1.15, 2.5)))
            upper_end = int(round(reorder_point * 3.5))

        events = receipts[pid] + dispatches[pid] + adjustments[pid]
        events.sort(key=lambda e: (e[0], e[1]))

        # Realistic opening balance first; adjust only when the product would miss its planned level
        opening = int(round(reorder_point * rng.uniform(1.0, 2.5 if info["demand_class"] != "slow" else 4.0)))
        closing = run_simulation(opening, events)[0]
        if closing < target_end:
            opening = find_opening_stock(events, target_end, start_from=opening)
        elif closing > upper_end:
            events = add_extra_dispatches(rng, events, opening, target_end, info["base_size"])

        if status == "out":
            # A final dispatch larger than the remaining stock empties the shelf.
            last_receipt = max([e[0] for e in receipts[pid]], default=START)
            earliest = max(last_receipt, pd.Timestamp("2025-12-01"))
            days = pd.bdate_range(earliest, END)
            sweep_day = pd.Timestamp(rng.choice(days.values))
            events.append((sweep_day, 3, "Stock Out", 10 ** 6, None))
            events.sort(key=lambda e: (e[0], e[1]))

        final_stock, accepted = run_simulation(opening, events)
        closing_stock[pid] = final_stock
        opening_rows.append({"Product_ID": pid, "Opening_Stock": opening, "As_Of_Date": START.date().isoformat()})

        for date, priority, kind, qty, reference in accepted:
            # Stock In keeps its PO line ID; dispatch / adjustment references are added after sorting
            movement_rows.append({"Date": date, "Product_ID": pid, "Movement_Type": kind, "Quantity": qty,
                                  "Warehouse": warehouse_of[pid], "Reference_ID": reference, "_priority": priority})

    movements = pd.DataFrame(movement_rows).sort_values(["Date", "_priority", "Product_ID"]).reset_index(drop=True)
    movements["Movement_ID"] = [f"MOV-{i:05d}" for i in range(1, len(movements) + 1)]

    # Reference IDs for dispatches (sales orders) and adjustments
    is_out = movements["Movement_Type"] == "Stock Out"
    is_adj = movements["Movement_Type"] == "Adjustment"
    movements.loc[is_out, "Reference_ID"] = [f"SO-{20000 + i}" for i in range(1, is_out.sum() + 1)]
    movements.loc[is_adj, "Reference_ID"] = [f"ADJ-{i:04d}" for i in range(1, is_adj.sum() + 1)]

    movements = movements[["Movement_ID", "Date", "Product_ID", "Movement_Type", "Quantity", "Warehouse", "Reference_ID"]]
    products = products.copy()
    products["Current_Stock"] = products["Product_ID"].map(closing_stock).astype(int)
    return products, movements, pd.DataFrame(opening_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    suppliers, profiles = make_suppliers(rng)
    products, plan = make_products(rng, suppliers)
    purchase_orders = make_purchase_orders(rng, products, plan, suppliers, profiles)
    products, movements, opening = build_inventory(rng, products, plan, purchase_orders)

    product_columns = ["Product_ID", "Product_Name", "Category", "Supplier_ID", "Unit_Cost", "Selling_Price",
                       "Reorder_Point", "Safety_Stock", "Current_Stock", "Warehouse"]
    products = products[product_columns]

    for column in ["Order_Date", "Expected_Delivery_Date", "Actual_Delivery_Date"]:
        purchase_orders[column] = pd.to_datetime(purchase_orders[column]).dt.strftime("%Y-%m-%d")
    movements["Date"] = pd.to_datetime(movements["Date"]).dt.strftime("%Y-%m-%d")

    suppliers.to_csv(DATA_DIR / "suppliers.csv", index=False)
    products.to_csv(DATA_DIR / "products.csv", index=False)
    purchase_orders.to_csv(DATA_DIR / "purchase_orders.csv", index=False)
    movements.to_csv(DATA_DIR / "inventory_movements.csv", index=False)
    opening.to_csv(DATA_DIR / "opening_balances.csv", index=False)

    print("Synthetic data written to", DATA_DIR)
    print(f"  suppliers:              {len(suppliers):>5}")
    print(f"  products:               {len(products):>5}")
    print(f"  purchase order lines:   {len(purchase_orders):>5}  ({purchase_orders['PO_ID'].nunique()} POs)")
    print(f"  inventory movements:    {len(movements):>5}")
    print("Run `python scripts/validate_data.py` to check relationships and stock reconciliation.")


if __name__ == "__main__":
    main()
