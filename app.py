"""
Supply Chain & Inventory Analytics Dashboard
Meridian Distribution Co. (fictional company, synthetic data)

Run with:  streamlit run app.py
"""

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import inventory_analysis as inv
from src import supplier_analysis as sup

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LOGO_SVG = ROOT / "assets" / "meridian_logo.svg"
LOGO_PNG = ROOT / "assets" / "meridian_logo.png"

NAVY = "#172B4D"
TEAL = "#008C95"
LIGHT_BG = "#F5F7FA"
GREEN = "#16A34A"
AMBER = "#D97706"
RED = "#DC2626"
GREY = "#94A3B8"
TEXT = "#334155"
MUTED = "#64748B"
GRID = "#E5E9F0"
RULE = "#D9DFE8"
FONT = "Segoe UI, Helvetica Neue, Arial, sans-serif"

STATUS_COLORS = {"Healthy Stock": GREEN, "Low Stock": AMBER, "Out of Stock": RED}
PO_STATUS_COLORS = {
    "Delivered On Time": GREEN, "Delivered Late": AMBER, "Overdue": RED,
    "In Transit": TEAL, "Cancelled": GREY,
}
PO_STATUS_ORDER = ["Delivered On Time", "Delivered Late", "Overdue", "In Transit", "Cancelled"]
FLAG_COLORS = {"Needs review": RED, "Below average": AMBER, "On track": GREEN, "Low volume": GREY}
# One colour per product category, reused on every chart so a category always looks the same
CATEGORY_PALETTE = ["#172B4D", "#008C95", "#5B7DB1", "#7FC8CC", "#B08D57", "#8A94A6", "#3E6F8E", "#C4CFDC"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(
    page_title="Meridian Supply Chain Dashboard",
    page_icon=str(LOGO_PNG) if LOGO_PNG.exists() else None,
    layout="wide",
)

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {LIGHT_BG}; }}
    .block-container {{ padding-top: 1.4rem; max-width: 1400px; }}
    section[data-testid="stSidebar"] {{ background-color: #FFFFFF; border-right: 1px solid {RULE}; }}
    button[data-baseweb="tab"] {{ font-weight: 600; color: {NAVY}; }}

    .mdc-header {{ background: {NAVY}; padding: 16px 22px; display: flex; align-items: center; gap: 22px;
                   border-radius: 3px 3px 0 0; }}
    .mdc-logo {{ background: #FFFFFF; padding: 7px 12px; flex-shrink: 0; border-radius: 2px; }}
    .mdc-logo img {{ height: 46px; display: block; }}
    .mdc-title {{ color: #FFFFFF; font-family: {FONT}; font-size: 1.55rem; font-weight: 650; line-height: 1.2; }}
    .mdc-meta {{ color: #B8C4D9; font-family: {FONT}; font-size: 0.88rem; margin-top: 3px; }}
    .mdc-foot {{ background: #E6F2F3; color: {NAVY}; font-family: {FONT}; font-size: 0.82rem;
                 padding: 7px 22px; border-radius: 0 0 3px 3px; margin-bottom: 16px; }}

    .strip {{ display: grid; grid-template-columns: repeat(6, 1fr); background: #FFFFFF;
              border: 1px solid {RULE}; font-family: {FONT}; margin-bottom: 18px; }}
    .strip .grp {{ padding: 6px 16px; font-size: 0.78rem; font-weight: 600; color: #FFFFFF; }}
    .strip .grp.snap {{ grid-column: span 2; background: {TEAL}; }}
    .strip .grp.span {{ grid-column: span 4; background: {NAVY}; }}
    .strip .cell {{ padding: 13px 16px 12px 16px; border-right: 1px solid {RULE}; }}
    .strip .cell:last-child {{ border-right: none; }}
    .strip .lbl {{ font-size: 0.8rem; color: {MUTED}; font-weight: 600; }}
    .strip .val {{ font-size: 1.75rem; font-weight: 650; color: {NAVY}; line-height: 1.3;
                   font-variant-numeric: tabular-nums; }}
    .strip .val small {{ font-size: 0.85rem; font-weight: 500; color: {MUTED}; }}
    .strip .note {{ font-size: 0.74rem; color: {MUTED}; line-height: 1.3; }}

    .findings {{ background: #FFFFFF; border: 1px solid {RULE}; font-family: {FONT}; margin-bottom: 18px; }}
    .findings .frow {{ display: grid; grid-template-columns: 3fr 2fr; gap: 24px; padding: 9px 16px;
                       border-top: 1px solid {RULE}; color: {TEXT}; font-size: 0.92rem; }}
    .findings .frow.fhead {{ border-top: none; background: #EEF2F7; color: {NAVY}; font-weight: 650;
                             font-size: 0.8rem; }}
    .findings .check {{ color: {MUTED}; }}

    .heading {{ color: {NAVY}; font-family: {FONT}; font-size: 1.1rem; font-weight: 650;
                border-bottom: 1px solid {RULE}; padding-bottom: 4px; margin: 20px 0 8px 0; }}
    .scope {{ color: {MUTED}; font-family: {FONT}; font-size: 0.84rem; margin: 2px 0 10px 0; }}
    @media (max-width: 900px) {{
      .strip {{ grid-template-columns: repeat(2, 1fr); }}
      .strip .grp.snap, .strip .grp.span {{ grid-column: span 2; }}
      .findings .frow {{ grid-template-columns: 1fr; gap: 4px; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def money(value):
    """Dollars for HTML blocks (&#36; stops Markdown treating $ as math)."""
    return f"&#36;{value:,.0f}"


def percent(value):
    return "n/a" if value is None or pd.isna(value) else f"{value:.1f}%"


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def image_as_data_uri(path):
    mime = "image/svg+xml" if path.suffix == ".svg" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def heading(text):
    st.markdown(f'<div class="heading">{text}</div>', unsafe_allow_html=True)


def scope_note(text):
    st.markdown(f'<div class="scope">{text}</div>', unsafe_allow_html=True)


def kpi_strip(snapshot_label, period_label, cells):
    """One strip of six figures. Teal band = current snapshot, navy band = selected period."""
    parts = [f'<div class="strip"><div class="grp snap">{snapshot_label}</div>'
             f'<div class="grp span">{period_label}</div>']
    for label, value, note in cells:
        parts.append(f'<div class="cell"><div class="lbl">{label}</div><div class="val">{value}</div>'
                     f'<div class="note">{note}</div></div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
def style_figure(fig, title, subtitle=None, height=400):
    text = title
    if subtitle:
        text += f"<br><span style='font-size:12px;font-weight:400;color:{MUTED}'>{subtitle}</span>"
    fig.update_layout(
        title=dict(text=text, x=0, xanchor="left", font=dict(size=16, color=NAVY)),
        height=height, margin=dict(l=10, r=20, t=75 if subtitle else 55, b=20),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family=FONT, size=12, color=TEXT),
        hoverlabel=dict(bgcolor="#FFFFFF", font_size=12, font_family=FONT),
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def empty_figure(title, message="No data for the selected filters"):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=14, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style_figure(fig, title, height=300)


def show_chart(fig, key):
    st.plotly_chart(fig, key=key, theme=None)


def row_height(rows):
    """Height for charts with one row per item, so labels stay readable."""
    return max(320, 28 * rows + 130)


# ---------------------------------------------------------------------------
# Charts: inventory (current snapshot)
# ---------------------------------------------------------------------------
def chart_value_treemap(products):
    title = "Inventory Value by Product Category"
    data = products[products["Inventory_Value"] > 0]
    if data.empty:
        return empty_figure(title)
    totals = data.groupby("Category")["Inventory_Value"].sum()
    ids = list(totals.index) + data["Product_ID"].tolist()
    labels = list(totals.index) + data["Product_Name"].tolist()
    parents = [""] * len(totals) + data["Category"].tolist()
    # "remainder" mode: category tiles are sized by the products inside them
    values = [0] * len(totals) + data["Inventory_Value"].tolist()
    colors = [NAVY] * len(totals) + [STATUS_COLORS[status] for status in data["Stock_Status"]]
    details = ([[f"${value:,.0f} total ({value / totals.sum():.0%} of inventory)", ""] for value in totals]
               + [[f"${value:,.0f}", f"{status}, {int(stock):,} units"]
                  for value, status, stock in zip(data["Inventory_Value"], data["Stock_Status"], data["Current_Stock"])])
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values, branchvalues="remainder",
        marker=dict(colors=colors, line=dict(width=1, color="#FFFFFF")), customdata=details,
        textfont=dict(color="#FFFFFF"), textinfo="label", pathbar=dict(visible=False),
        hovertemplate="<b>%{label}</b><br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
    ))
    return style_figure(fig, title, "Each tile is a product, sized by inventory value. Green = healthy, "
                                   "amber = low stock, red = out of stock", height=470)


def chart_top_products_pareto(products):
    title = "Top 10 Products by Inventory Value"
    if products.empty or products["Inventory_Value"].sum() == 0:
        return empty_figure(title)
    total = products["Inventory_Value"].sum()
    top = inv.top_products_by_value(products)
    top["Cumulative_Share"] = top["Inventory_Value"].cumsum() / total * 100
    labels = [f"${value:,.0f}  ({share:.0f}% cumulative)" for value, share in zip(top["Inventory_Value"], top["Cumulative_Share"])]
    fig = go.Figure(go.Bar(
        x=top["Inventory_Value"], y=top["Product_Name"], orientation="h",
        marker_color=[STATUS_COLORS[status] for status in
                      products.set_index("Product_ID").loc[top["Product_ID"], "Stock_Status"]],
        text=labels, textposition="outside", cliponaxis=False,
        customdata=top[["Category", "Current_Stock"]].values,
        hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>%{customdata[1]:,} units in stock"
                      "<br>Inventory value: %{x:$,.0f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickprefix="$", tickformat=",.0f", range=[0, top["Inventory_Value"].max() * 1.45])
    return style_figure(fig, title, f"The top 10 hold {top['Cumulative_Share'].iloc[-1]:.0f}% of total inventory value. "
                                    "Bar colour is stock status", height=430)


def chart_status_mix(products):
    title = "Stock Status Distribution"
    if products.empty:
        return empty_figure(title)
    counts = inv.status_mix_by_category(products)
    share = counts.div(counts.sum(axis=1), axis=0) * 100
    at_risk = int(counts["Low Stock"].sum() + counts["Out of Stock"].sum())
    fig = go.Figure()
    for status in inv.STATUS_ORDER:
        fig.add_trace(go.Bar(
            name=status, x=share[status], y=share.index, orientation="h", marker_color=STATUS_COLORS[status],
            customdata=counts[status].values,
            hovertemplate="<b>%{y}</b><br>" + status + ": %{customdata} products (%{x:.0f}%)<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(ticksuffix="%", range=[0, 100])
    return style_figure(fig, title, f"Share of products by category. {at_risk} of {len(products)} products are at or "
                                    "below reorder point; riskiest categories first", height=row_height(len(share)) + 40)


def chart_warehouse_heatmap(products):
    title = "Inventory Levels by Warehouse"
    if products.empty:
        return empty_figure(title)
    matrix = products.pivot_table(index="Warehouse", columns="Category", values="Inventory_Value",
                                  aggfunc="sum", fill_value=0)
    units = products.groupby("Warehouse")["Current_Stock"].sum()
    y_labels = [f"{warehouse} ({int(units[warehouse]):,} units)" for warehouse in matrix.index]
    text = [[f"${value:,.0f}" for value in row] for row in matrix.values]
    fig = go.Figure(go.Heatmap(
        z=matrix.values, x=list(matrix.columns), y=y_labels, text=text, texttemplate="%{text}",
        colorscale=[[0, "#EEF3F7"], [1, NAVY]], xgap=2, ygap=2,
        colorbar=dict(title="Value", tickprefix="$", tickformat=",.0f"),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Inventory value: %{z:$,.0f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-25)
    return style_figure(fig, title, "Inventory value by warehouse and category. Unit totals in the row labels", height=330)


# ---------------------------------------------------------------------------
# Charts: procurement (orders placed in the selected period)
# ---------------------------------------------------------------------------
def chart_monthly_spend(po):
    title = "Monthly Procurement Spending"
    if po.empty:
        return empty_figure(title)
    by_category = sup.monthly_spend_by_category(po)
    if by_category.empty:
        return empty_figure(title)
    months = pd.date_range(by_category.index.min(), by_category.index.max(), freq="MS")
    by_category = by_category.reindex(months, fill_value=0.0)
    totals = by_category.sum(axis=1)
    fig = go.Figure()
    for category in by_category.sum().sort_values(ascending=False).index:
        fig.add_trace(go.Bar(
            name=category, x=by_category.index, y=by_category[category], marker_color=category_color(category),
            hovertemplate="%{y:$,.0f}<extra>" + category + "</extra>"))
    fig.add_trace(go.Scatter(
        name="3-month average", x=totals.index, y=totals.rolling(3, min_periods=1).mean(), mode="lines",
        line=dict(color=RED, width=3), hovertemplate="%{y:$,.0f}<extra>3-month average</extra>"))
    fig.update_layout(barmode="stack", hovermode="x unified")
    fig.update_xaxes(tickformat="%b %Y")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return style_figure(fig, title, "Ordered value by order month and category, excluding cancelled orders", height=470)


def chart_supplier_pareto(po):
    title = "Procurement Spend by Supplier"
    data = sup.spend_by_supplier(po) if not po.empty else pd.DataFrame()
    if data.empty or data["Procurement_Spend"].sum() == 0:
        return empty_figure(title)
    total = data["Procurement_Spend"].sum()
    data["Cumulative_Share"] = data["Procurement_Spend"].cumsum() / total * 100
    suppliers_for_80 = int((data["Cumulative_Share"] < 80).sum()) + 1
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Ordered value", x=data["Supplier_Name"], y=data["Procurement_Spend"], marker_color=TEAL,
        hovertemplate="<b>%{x}</b><br>Ordered value: %{y:$,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        name="Cumulative share", x=data["Supplier_Name"], y=data["Cumulative_Share"], yaxis="y2",
        mode="lines+markers", line=dict(color=NAVY, width=2), marker=dict(size=6),
        hovertemplate="<b>%{x}</b><br>Cumulative: %{y:.0f}% of spend<extra></extra>"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 105], ticksuffix="%", showgrid=False))
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y2", y0=80, y1=80,
                  line=dict(color=GREY, dash="dot", width=1))
    fig.update_xaxes(tickangle=-40)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return style_figure(fig, title, f"{suppliers_for_80} of {len(data)} suppliers account for 80% of ordered value "
                                    "(dotted line)", height=450)


def chart_po_status(po):
    title = "Purchase Order Status Distribution"
    if po.empty:
        return empty_figure(title)
    mix = sup.quarterly_status_mix(po)
    fig = go.Figure()
    for status in PO_STATUS_ORDER:
        if status in mix.columns:
            fig.add_trace(go.Bar(
                name=status, x=mix.index, y=mix[status], marker_color=PO_STATUS_COLORS[status],
                hovertemplate="<b>%{x}</b><br>" + status + ": %{y} purchase orders<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Purchase orders")
    totals = po.drop_duplicates("PO_ID")["Delivery_Status"].value_counts()
    summary = ", ".join(f"{int(totals[s])} {s.lower()}" for s in PO_STATUS_ORDER if s in totals.index)
    return style_figure(fig, title, f"By order quarter. Total: {summary}", height=450)


# ---------------------------------------------------------------------------
# Charts: supplier performance
# ---------------------------------------------------------------------------
def chart_on_time_rate(scorecard, overall_rate):
    title = "Supplier On-Time Delivery Rate"
    rated = scorecard.dropna(subset=["On_Time_Rate"]).sort_values("On_Time_Rate")
    if rated.empty:
        return empty_figure(title, "No completed deliveries for the selected filters")
    labels = [f"{name}  (n={n})" for name, n in zip(rated["Supplier_Name"], rated["Completed_Orders"])]
    fig = go.Figure()
    fig.add_vrect(x0=0, x1=sup.REVIEW_THRESHOLD, fillcolor=RED, opacity=0.07, line_width=0, layer="below")
    fig.add_trace(go.Scatter(
        x=rated["On_Time_Rate"], y=labels, mode="markers",
        marker=dict(size=11, color=[FLAG_COLORS[flag] for flag in rated["Performance_Flag"]],
                    line=dict(color="#FFFFFF", width=1)),
        error_x=dict(type="data", symmetric=False, color=GREY, thickness=1.5, width=4,
                     array=(rated["On_Time_CI_High"] - rated["On_Time_Rate"]).tolist(),
                     arrayminus=(rated["On_Time_Rate"] - rated["On_Time_CI_Low"]).tolist()),
        customdata=rated[["Completed_Orders", "On_Time_CI_Low", "On_Time_CI_High", "Performance_Flag"]].values,
        hovertemplate="<b>%{y}</b><br>On time: %{x:.1f}%<br>95% range: %{customdata[1]:.0f}% to %{customdata[2]:.0f}%"
                      "<br>Flag: %{customdata[3]}<extra></extra>"))
    if overall_rate is not None:
        fig.add_vline(x=overall_rate, line_dash="dash", line_color=NAVY,
                      annotation_text=f"Overall {overall_rate:.1f}%", annotation_position="top")
    fig.update_xaxes(range=[0, 104], ticksuffix="%")
    return style_figure(fig, title, "Dot = observed rate, line = 95% range given the number of orders (n). "
                                    f"Red zone = below {sup.REVIEW_THRESHOLD:.0f}%", height=row_height(len(rated)) + 40)


def chart_lead_time(po, suppliers):
    title = "Average Lead Time by Supplier"
    done = sup.completed_orders(po) if not po.empty else pd.DataFrame()
    if done.empty:
        return empty_figure(title, "No completed deliveries for the selected filters")
    order = done.groupby("Supplier_Name")["Lead_Time_Days"].median().sort_values(ascending=False).index.tolist()
    quoted = (done[["Supplier_Name", "Supplier_ID"]].drop_duplicates()
              .merge(suppliers[["Supplier_ID", "Average_Lead_Time_Days"]], on="Supplier_ID", how="left"))
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=done["Lead_Time_Days"], y=done["Supplier_Name"], orientation="h", name="Actual lead time",
        boxmean=True, boxpoints="outliers", marker=dict(color=TEAL, size=4), line=dict(color=TEAL),
        fillcolor="rgba(0,140,149,0.22)",
        hovertemplate="<b>%{y}</b><br>%{x} days<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=quoted["Average_Lead_Time_Days"], y=quoted["Supplier_Name"], mode="markers", name="Quoted lead time",
        marker=dict(symbol="diamond", size=10, color=NAVY, line=dict(color="#FFFFFF", width=1)),
        hovertemplate="<b>%{y}</b><br>Quoted: %{x} days<extra></extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=order, autorange="reversed")
    fig.update_xaxes(title_text="Days from order to receipt")
    return style_figure(fig, title, "Box = middle 50% of orders, line = median, dashed = mean. "
                                    "Diamond = lead time the supplier quotes", height=row_height(len(order)) + 60)


def chart_delay_matrix(scorecard):
    title = "Supplier Delivery Delay Comparison"
    data = scorecard[scorecard["Completed_Orders"] > 0].copy()
    if data.empty:
        return empty_figure(title, "No completed deliveries for the selected filters")
    data["Late_Share"] = data["Late_Orders"] / data["Completed_Orders"] * 100
    biggest = max(data["Total_Procurement_Value"].max(), 1)
    sizes = 10 + 32 * np.sqrt(data["Total_Procurement_Value"] / biggest)
    fig = go.Figure(go.Scatter(
        x=data["Late_Share"], y=data["Avg_Delay_Days"], mode="markers+text", text=data["Supplier_Name"],
        textposition="top center", textfont=dict(size=10, color=TEXT),
        marker=dict(size=sizes, color=[FLAG_COLORS[flag] for flag in data["Performance_Flag"]], opacity=0.8,
                    line=dict(color="#FFFFFF", width=1)),
        customdata=data[["Late_Orders", "Completed_Orders", "Total_Procurement_Value"]].values,
        hovertemplate="<b>%{text}</b><br>Late orders: %{customdata[0]} of %{customdata[1]} (%{x:.0f}%)"
                      "<br>Average delay when late: %{y:.1f} days<br>Ordered value: %{customdata[2]:$,.0f}<extra></extra>"))
    fig.add_vline(x=float(data["Late_Share"].median()), line_dash="dot", line_color=GREY)
    fig.add_hline(y=float(data["Avg_Delay_Days"].median()), line_dash="dot", line_color=GREY)
    fig.add_annotation(xref="paper", yref="paper", x=0.99, y=0.99, showarrow=False, xanchor="right", yanchor="top",
                       text="Late often and by a lot", font=dict(size=11, color=RED))
    fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.01, showarrow=False, xanchor="left", yanchor="bottom",
                       text="Rarely late, short delays", font=dict(size=11, color=GREEN))
    fig.update_xaxes(title_text="Orders delivered late (%)", ticksuffix="%", rangemode="tozero")
    fig.update_yaxes(title_text="Average delay when late (days)", rangemode="tozero")
    return style_figure(fig, title, "Bubble size = ordered value. Dotted lines = median across suppliers. "
                                    "Colour = performance flag", height=520)


# ---------------------------------------------------------------------------
# Charts: demand (stock-out movements in the selected period)
# ---------------------------------------------------------------------------
def chart_stock_out_trend(movements, start, end):
    title = "Monthly Stock-Out Movement Trend"
    data = inv.monthly_stock_out(movements, start, end)
    if data.empty or data["Units_Dispatched"].sum() == 0:
        return empty_figure(title)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Units dispatched", x=data["Month"], y=data["Units_Dispatched"], marker_color="rgba(0,140,149,0.55)",
        customdata=data["Dispatches"].values,
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,} units in %{customdata} dispatches<extra></extra>"))
    fig.add_trace(go.Scatter(
        name="3-month average", x=data["Month"], y=data["Units_Dispatched"].rolling(3, min_periods=1).mean(),
        mode="lines", line=dict(color=NAVY, width=3), hovertemplate="%{y:,.0f} units<extra>3-month average</extra>"))
    subtitle = "Units dispatched per month"
    if len(data) >= 3:
        slope, intercept = np.polyfit(np.arange(len(data)), data["Units_Dispatched"].to_numpy(dtype=float), 1)
        fig.add_trace(go.Scatter(
            name="Linear trend", x=data["Month"], y=intercept + slope * np.arange(len(data)), mode="lines",
            line=dict(color=RED, width=2, dash="dash"), hoverinfo="skip"))
        subtitle += f". Trend: {slope:+,.0f} units per month"
    fig.update_xaxes(tickformat="%b %Y")
    fig.update_yaxes(title_text="Units dispatched", rangemode="tozero")
    return style_figure(fig, title, subtitle, height=430)


def chart_top_dispatched(movements):
    title = "Top 10 Products by Quantity Dispatched"
    data = inv.top_dispatched_products(movements)
    if data.empty:
        return empty_figure(title)
    data["Units_Per_Dispatch"] = data["Units_Dispatched"] / data["Dispatches"]
    fig = go.Figure(go.Bar(
        x=data["Units_Dispatched"], y=data["Product_Name"], orientation="h",
        marker_color=[category_color(category) for category in data["Category"]],
        text=[f"{int(d)} dispatches" for d in data["Dispatches"]], textposition="outside", cliponaxis=False,
        customdata=data[["Category", "Dispatches", "Units_Per_Dispatch"]].values,
        hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>%{x:,} units in %{customdata[1]} dispatches"
                      "<br>%{customdata[2]:.0f} units per dispatch<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Units dispatched", range=[0, data["Units_Dispatched"].max() * 1.25])
    return style_figure(fig, title, "Bar colour = product category (same colours as the procurement chart)", height=430)


def chart_cover_scatter(products, movements, start, end):
    title = "Stock Cover vs. Demand"
    if products.empty:
        return empty_figure(title)
    table = inv.cover_table(products, movements, start, end)
    in_stock = table[table["Current_Stock"] > 0]
    plotted = in_stock[in_stock["Avg_Monthly_Dispatch"] > 0]
    if plotted.empty:
        return empty_figure(title, "No dispatches in the selected period")
    no_demand = len(in_stock) - len(plotted)
    biggest = max(plotted["Inventory_Value"].max(), 1)
    fig = go.Figure()
    for status in ["Healthy Stock", "Low Stock"]:
        part = plotted[plotted["Stock_Status"] == status]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(
            name=status, x=part["Avg_Monthly_Dispatch"], y=part["Months_Of_Cover"], mode="markers",
            marker=dict(size=6 + 24 * np.sqrt(part["Inventory_Value"] / biggest), color=STATUS_COLORS[status],
                        opacity=0.65, line=dict(color="#FFFFFF", width=1)),
            customdata=part[["Product_Name", "Category", "Current_Stock", "Inventory_Value"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]:,} units in stock"
                          "<br>Dispatch rate: %{x:.1f} units a month<br>Cover: %{y:.1f} months"
                          "<br>Inventory value: %{customdata[3]:$,.0f}<extra></extra>"))
    # Reference lines are drawn as traces: shape positions on log axes are easy to get wrong
    x_low, x_high = plotted["Avg_Monthly_Dispatch"].min() * 0.8, plotted["Avg_Monthly_Dispatch"].max() * 1.25
    for level, colour, text in [(inv.SLOW_MOVING_COVER_MONTHS, AMBER, f"{inv.SLOW_MOVING_COVER_MONTHS} months: slow-moving above"),
                                (1, RED, "1 month of cover")]:
        fig.add_trace(go.Scatter(x=[x_low, x_high], y=[level, level], mode="lines", showlegend=False,
                                 line=dict(color=colour, width=1.5, dash="dash"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[x_low], y=[level], mode="text", text=[text], textposition="top right",
                                 showlegend=False, textfont=dict(size=11, color=colour), hoverinfo="skip"))
    fig.update_xaxes(type="log", title_text="Average units dispatched per month (log scale)")
    fig.update_yaxes(type="log", title_text="Months of cover (log scale)")
    return style_figure(fig, title, f"One bubble per product, sized by inventory value. {no_demand} in-stock products with "
                                    "no dispatches and all out-of-stock products are not shown", height=520)


def chart_seasonality(movements):
    title = "Demand Seasonality by Category"
    matrix = inv.seasonality_index(movements)
    if matrix.empty or matrix.isna().all().all():
        return empty_figure(title)
    text = [["" if pd.isna(value) else f"{value:.2f}" for value in row] for row in matrix.values]
    fig = go.Figure(go.Heatmap(
        z=matrix.values, x=MONTH_NAMES, y=list(matrix.index), text=text, texttemplate="%{text}",
        colorscale=[[0, "#EAF4F5"], [0.5, "#FFFFFF"], [1, NAVY]], zmin=0.5, zmax=1.5, xgap=2, ygap=2,
        colorbar=dict(title="Index"),
        hovertemplate="<b>%{y}</b>, %{x}<br>Demand index: %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return style_figure(fig, title, "Average monthly units dispatched divided by the category's own monthly average. "
                                    "1.00 = typical month", height=400)


# ---------------------------------------------------------------------------
# Business findings
# ---------------------------------------------------------------------------
def build_findings(products, po, movements, scorecard, overall_rate, start, end):
    """Return (finding, what to check) pairs calculated from the filtered data."""
    findings = []

    if not products.empty and products["Inventory_Value"].sum() > 0:
        by_category = inv.inventory_value_by_category(products)
        top = by_category.iloc[0]
        share = top["Inventory_Value"] / by_category["Inventory_Value"].sum()
        findings.append((
            f"<b>{top['Category']}</b> holds the most inventory value: {money(top['Inventory_Value'])}, "
            f"{share:.0%} of the current total.",
            "Compare its stock against expected demand before the next purchase cycle."))

    if not products.empty:
        low = inv.count_low_stock(products)
        out = int((products["Stock_Status"] == "Out of Stock").sum())
        findings.append((
            f"<b>{low}</b> of {len(products)} products ({low / len(products):.1%}) are at or below reorder point; "
            f"{out} of them have no stock.",
            "Work through the reorder list in the Inventory tab, out-of-stock items first."))

    spend = sup.spend_by_supplier(po) if not po.empty else pd.DataFrame()
    if not spend.empty and spend["Procurement_Spend"].sum() > 0:
        top = spend.iloc[0]
        share = top["Procurement_Spend"] / spend["Procurement_Spend"].sum()
        findings.append((
            f"<b>{top['Supplier_Name']}</b> received the most orders by value: {money(top['Procurement_Spend'])} "
            f"({share:.0%} of ordered value).",
            "Review price and delivery terms here first; it moves total cost the most."))

    rated = scorecard[scorecard["Performance_Flag"] != "Low volume"] if not scorecard.empty else scorecard
    if overall_rate is not None and not rated.empty:
        below = rated[rated["Performance_Flag"].isin(["Needs review", "Below average"])]
        lowest = rated.sort_values("On_Time_Rate").iloc[0]
        findings.append((
            f"<b>{len(below)} of {len(rated)}</b> rated suppliers deliver on time less often than the overall "
            f"{overall_rate:.1f}%. Lowest: <b>{lowest['Supplier_Name']}</b> at {lowest['On_Time_Rate']:.1f}%.",
            "Look at whether the delays are one-off or a pattern, and whether quoted lead times are realistic."))

    top_dispatched = inv.top_dispatched_products(movements, n=1)
    if not top_dispatched.empty:
        top = top_dispatched.iloc[0]
        findings.append((
            f"<b>{top['Product_Name']}</b> shipped the most units: {int(top['Units_Dispatched']):,} "
            f"across {int(top['Dispatches'])} dispatches.",
            "Confirm its reorder point still matches the current dispatch rate."))

    slow = inv.slow_moving_products(products, movements, start, end) if not products.empty else pd.DataFrame()
    if not slow.empty and products["Inventory_Value"].sum() > 0:
        slow_value = slow["Inventory_Value"].sum()
        by_category = slow.groupby("Category")["Inventory_Value"].sum().sort_values(ascending=False)
        findings.append((
            f"<b>{len(slow)} products</b> have more than {inv.SLOW_MOVING_COVER_MONTHS} months of stock at the "
            f"period's dispatch rate. They tie up {money(slow_value)} "
            f"({slow_value / products['Inventory_Value'].sum():.0%} of inventory value), mostly in {by_category.index[0]}.",
            "Check these before reordering; smaller order sizes may free up cash."))
    return findings


def category_color(category):
    index = CATEGORY_ORDER.index(category) if category in CATEGORY_ORDER else 0
    return CATEGORY_PALETTE[index % len(CATEGORY_PALETTE)]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data
def load_all():
    data = inv.load_data(DATA_DIR)
    products = inv.add_inventory_fields(data["products"])
    purchase_orders = sup.prepare_purchase_orders(data["purchase_orders"])
    return products, data["suppliers"], purchase_orders, data["inventory_movements"]


try:
    products_all, suppliers_all, po_all, movements_all = load_all()
except FileNotFoundError as error:
    st.error(f"Data file not found: {error}. Run `python scripts/generate_data.py` from the project folder first.")
    st.stop()

CATEGORY_ORDER = sorted(products_all["Category"].unique())
data_start = min(po_all["Order_Date"].min(), movements_all["Date"].min()).date()
data_end = max(po_all["Order_Date"].max(), movements_all["Date"].max(), inv.SNAPSHOT_DATE).date()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
logo_html = f'<img src="{image_as_data_uri(LOGO_SVG)}" alt="Meridian Distribution Co. logo">' if LOGO_SVG.exists() else ""
st.markdown(
    '<div class="mdc-header">'
    f'<div class="mdc-logo">{logo_html}</div>'
    '<div><div class="mdc-title">Supply Chain &amp; Inventory Analytics Dashboard</div>'
    f'<div class="mdc-meta">Data from {data_start:%d %b %Y} to {data_end:%d %b %Y}. Stock figures are as of '
    f'{inv.SNAPSHOT_DATE:%d %b %Y}. Tracks stock levels, purchasing, supplier delivery and demand '
    'across three warehouses.</div></div></div>'
    '<div class="mdc-foot">Synthetic data: Meridian Distribution Co. is fictional and every figure here was generated '
    'for portfolio demonstration.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
if LOGO_PNG.exists():
    st.sidebar.image(str(LOGO_PNG))
st.sidebar.markdown("### Filters")

date_selection = st.sidebar.date_input("Date range", value=(data_start, data_end),
                                       min_value=data_start, max_value=data_end)
if isinstance(date_selection, (tuple, list)) and len(date_selection) == 2:
    start_date, end_date = date_selection
else:
    st.sidebar.info("Pick both a start and an end date.")
    st.stop()

category_options = sorted(products_all["Category"].unique())
supplier_options = sorted(suppliers_all["Supplier_Name"])
warehouse_options = sorted(products_all["Warehouse"].unique())
status_options = ["Delivered On Time", "Delivered Late", "In Transit", "Overdue", "Cancelled"]

selected_categories = st.sidebar.multiselect("Product category", category_options, default=category_options)
selected_suppliers = st.sidebar.multiselect("Supplier", supplier_options, default=supplier_options)
selected_warehouses = st.sidebar.multiselect("Warehouse", warehouse_options, default=warehouse_options)
selected_statuses = st.sidebar.multiselect("Delivery status (purchase orders)", status_options, default=status_options)

st.sidebar.caption(
    "The date range filters purchase orders by order date and stock movements by movement date. "
    "Current stock is a single snapshot, so the date range does not change it."
)

selected_supplier_ids = suppliers_all.loc[suppliers_all["Supplier_Name"].isin(selected_suppliers), "Supplier_ID"].tolist()

products = inv.filter_products(products_all, selected_categories, selected_supplier_ids, selected_warehouses)
movements = inv.filter_movements(movements_all, products_all, start_date, end_date,
                                 selected_categories, selected_supplier_ids, selected_warehouses)
purchase_orders = sup.filter_purchase_orders(po_all, products_all, suppliers_all, start_date, end_date,
                                             selected_categories, selected_supplier_ids, selected_warehouses,
                                             selected_statuses)

if products.empty and movements.empty and purchase_orders.empty:
    st.warning("Nothing matches these filters. Widen the date range, or select more categories, suppliers, "
               "warehouses or statuses.")
    st.stop()

scorecard = sup.supplier_scorecard(purchase_orders, suppliers_all)
overall_on_time = sup.on_time_rate(purchase_orders)
period_text = f"{start_date:%d %b %Y} to {end_date:%d %b %Y}"

tab_overview, tab_inventory, tab_procurement, tab_suppliers, tab_demand = st.tabs(
    ["Overview", "Inventory", "Procurement", "Supplier Performance", "Demand Analysis"])

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab_overview:
    low_stock_count = inv.count_low_stock(products)
    out_of_stock_count = int((products["Stock_Status"] == "Out of Stock").sum()) if not products.empty else 0
    lead_time = sup.average_lead_time(purchase_orders)
    lead_time_text = "n/a" if lead_time is None else f"{lead_time:.1f}<small> days</small>"
    kpi_strip(
        f"Stock position, {inv.SNAPSHOT_DATE:%d %b %Y}", f"Activity, {period_text}",
        [
            ("Total Inventory Value", money(inv.total_inventory_value(products)), "Current stock x unit cost"),
            ("Low-Stock Products",
             f'<span style="color:{RED if low_stock_count else GREEN}">{low_stock_count:,}</span>'
             f'<small> of {len(products):,}</small>',
             f"At or below reorder point; {out_of_stock_count} have no stock"),
            ("Total Procurement Spend", money(sup.procurement_spend(purchase_orders)),
             "Ordered value, excluding cancelled orders"),
            ("Active Products", f"{inv.active_product_count(movements):,}", "With any stock movement"),
            ("Supplier On-Time Delivery Rate", percent(overall_on_time), "Completed deliveries, one count per order"),
            ("Average Supplier Lead Time", lead_time_text, "Order date to receipt"),
        ],
    )

    heading("What the data says")
    findings = build_findings(products, purchase_orders, movements, scorecard, overall_on_time, start_date, end_date)
    if findings:
        rows = ['<div class="findings"><div class="frow fhead"><div>Finding</div><div>What to check next</div></div>']
        rows += [f'<div class="frow"><div>{finding}</div><div class="check">{check}</div></div>'
                 for finding, check in findings]
        rows.append("</div>")
        st.markdown("".join(rows), unsafe_allow_html=True)
    else:
        st.info("Not enough data in this selection to draw findings.")

    left, right = st.columns(2)
    with left:
        show_chart(chart_monthly_spend(purchase_orders), "overview_spend")
    with right:
        show_chart(chart_status_mix(products), "overview_status")

# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------
with tab_inventory:
    scope_note(f"Everything on this tab is the stock snapshot of {inv.SNAPSHOT_DATE:%d %b %Y}. "
               "These are not historical balances and the date range does not change them.")
    show_chart(chart_value_treemap(products), "inv_treemap")
    left, right = st.columns(2)
    with left:
        show_chart(chart_top_products_pareto(products), "inv_top_products")
    with right:
        show_chart(chart_status_mix(products), "inv_status")
    show_chart(chart_warehouse_heatmap(products), "inv_warehouse")

    heading("Reorder alerts")
    scope_note("Out of Stock: 0 units. Low Stock: above 0 and at or below the reorder point. "
               "Healthy Stock: above the reorder point.")
    status_counts = inv.stock_status_counts(products).set_index("Stock_Status")["Products"]
    count_columns = st.columns(3)
    for column, status in zip(count_columns, inv.STATUS_ORDER):
        column.metric(status, f"{int(status_counts.get(status, 0)):,}")

    filter_left, filter_right = st.columns([2, 1])
    shown_statuses = filter_left.multiselect("Stock status", inv.STATUS_ORDER, default=["Low Stock", "Out of Stock"],
                                             key="monitor_status")
    search_text = filter_right.text_input("Search product name", key="monitor_search")

    table = products[products["Stock_Status"].isin(shown_statuses)]
    if search_text:
        table = table[table["Product_Name"].str.contains(search_text, case=False, na=False, regex=False)]
    severity = {"Out of Stock": 0, "Low Stock": 1, "Healthy Stock": 2}
    table = table.assign(_severity=table["Stock_Status"].map(severity)).sort_values(
        ["_severity", "Inventory_Value"], ascending=[True, False])
    table = table[["Product_Name", "Category", "Warehouse", "Current_Stock", "Reorder_Point", "Safety_Stock",
                   "Stock_Status", "Inventory_Value", "Suggested_Reorder_Qty"]].rename(columns={
        "Product_Name": "Product", "Current_Stock": "Current Stock", "Reorder_Point": "Reorder Point",
        "Safety_Stock": "Safety Stock", "Stock_Status": "Stock Status", "Inventory_Value": "Inventory Value",
        "Suggested_Reorder_Qty": "Suggested Reorder Qty"})

    if table.empty:
        st.info("No products match this stock status and search text.")
    else:
        st.dataframe(table, hide_index=True, column_config={
            "Inventory Value": st.column_config.NumberColumn(format="$%.2f"),
        })
        st.download_button("Download table as CSV", to_csv_bytes(table), file_name="inventory_monitoring.csv",
                           mime="text/csv", key="download_inventory")
    st.caption("Suggested reorder quantity is a simple rule, not a forecast: for Low Stock and Out of Stock products, "
               "order up to 2 x reorder point, rounded up to the next multiple of 5.")

# ---------------------------------------------------------------------------
# Procurement
# ---------------------------------------------------------------------------
with tab_procurement:
    scope_note(f"Spend is the <b>ordered value</b> (quantity x unit cost) of purchase orders placed {period_text}, "
               "excluding cancelled orders. It is not cash paid or goods received.")
    show_chart(chart_monthly_spend(purchase_orders), "proc_monthly")
    left, right = st.columns(2)
    with left:
        show_chart(chart_supplier_pareto(purchase_orders), "proc_pareto")
    with right:
        show_chart(chart_po_status(purchase_orders), "proc_status")

    with st.expander("Purchase order lines"):
        if purchase_orders.empty:
            st.info("No purchase orders match these filters.")
        else:
            lines = purchase_orders[["PO_ID", "PO_Line_ID", "Order_Date", "Supplier_Name", "Product_Name",
                                     "Quantity_Ordered", "Unit_Cost", "Total_Cost", "Expected_Delivery_Date",
                                     "Actual_Delivery_Date", "Delivery_Status"]].copy()
            for column in ["Order_Date", "Expected_Delivery_Date", "Actual_Delivery_Date"]:
                lines[column] = lines[column].dt.strftime("%Y-%m-%d")
            st.dataframe(lines, hide_index=True)
            st.download_button("Download purchase order lines as CSV", to_csv_bytes(lines),
                               file_name="purchase_order_lines.csv", mime="text/csv", key="download_po")

# ---------------------------------------------------------------------------
# Supplier performance
# ---------------------------------------------------------------------------
with tab_suppliers:
    scope_note(f"Purchase orders placed {period_text}. On time means received on or before the expected date. "
               "Each purchase order counts once, however many lines it has.")
    show_chart(chart_on_time_rate(scorecard, overall_on_time), "sup_on_time")
    left, right = st.columns(2)
    with left:
        show_chart(chart_lead_time(purchase_orders, suppliers_all), "sup_lead_time")
    with right:
        show_chart(chart_delay_matrix(scorecard), "sup_delay")

    heading("Supplier scorecard")
    st.caption(f"Needs review: on-time rate below {sup.REVIEW_THRESHOLD:.0f}%. Below average: under the overall rate. "
               f"Low volume: fewer than {sup.MIN_ORDERS_FOR_RATING} completed orders, too few to rate.")
    if scorecard.empty:
        st.info("No supplier records match these filters.")
    else:
        control_left, control_mid, control_right = st.columns([2, 2, 1])
        flags_shown = control_left.multiselect("Performance flag", list(FLAG_COLORS), default=list(FLAG_COLORS),
                                               key="scorecard_flags")
        sort_options = {
            "On-time rate": "On_Time_Rate", "Average delay days": "Avg_Delay_Days",
            "Average lead time": "Avg_Lead_Time_Days", "Total procurement value": "Total_Procurement_Value",
            "Completed orders": "Completed_Orders", "Supplier name": "Supplier_Name",
        }
        sort_label = control_mid.selectbox("Sort by", list(sort_options), key="scorecard_sort")
        ascending = control_right.checkbox("Ascending", value=False, key="scorecard_ascending")

        view = scorecard[scorecard["Performance_Flag"].isin(flags_shown)]
        view = view.sort_values(sort_options[sort_label], ascending=ascending, na_position="last")
        if view.empty:
            st.info("No suppliers match these performance flags.")
        else:
            export = view.drop(columns=["Supplier_ID", "On_Time_Orders"]).rename(columns={
                "Supplier_Name": "Supplier", "Completed_Orders": "Completed Orders",
                "Total_Procurement_Value": "Total Procurement Value", "Avg_Lead_Time_Days": "Avg Lead Time (days)",
                "On_Time_Rate": "On-Time Rate (%)", "On_Time_CI_Low": "On-Time 95% Low (%)",
                "On_Time_CI_High": "On-Time 95% High (%)", "Late_Orders": "Late Orders",
                "Avg_Delay_Days": "Avg Delay (days, late orders)", "Performance_Flag": "Performance Flag"})
            display = export.copy()
            display["On-Time 95% Range"] = [
                "n/a" if pd.isna(low) else f"{low:.0f}% to {high:.0f}%"
                for low, high in zip(export["On-Time 95% Low (%)"], export["On-Time 95% High (%)"])]
            display = display.drop(columns=["On-Time 95% Low (%)", "On-Time 95% High (%)"])
            st.dataframe(display, hide_index=True, column_config={
                "Total Procurement Value": st.column_config.NumberColumn(format="$%.0f"),
                "Avg Lead Time (days)": st.column_config.NumberColumn(format="%.1f"),
                "On-Time Rate (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                "Avg Delay (days, late orders)": st.column_config.NumberColumn(format="%.1f"),
            })
            st.download_button("Download scorecard as CSV", to_csv_bytes(export), file_name="supplier_scorecard.csv",
                               mime="text/csv", key="download_scorecard")

# ---------------------------------------------------------------------------
# Demand analysis
# ---------------------------------------------------------------------------
with tab_demand:
    scope_note(f"Demand is measured by Stock Out movements (units shipped to customers) from {period_text}.")
    show_chart(chart_stock_out_trend(movements, start_date, end_date), "demand_trend")
    left, right = st.columns(2)
    with left:
        show_chart(chart_top_dispatched(movements), "demand_top")
    with right:
        show_chart(chart_seasonality(movements), "demand_seasonality")
    show_chart(chart_cover_scatter(products, movements, start_date, end_date), "demand_cover")

    heading("Slow-moving products")
    st.caption(f"In stock with more than {inv.SLOW_MOVING_COVER_MONTHS} months of cover (current stock divided by average "
               "monthly units dispatched in the selected period), or with no dispatches in the period.")
    slow = inv.slow_moving_products(products, movements, start_date, end_date) if not products.empty else pd.DataFrame()
    if slow.empty:
        st.info("No slow-moving products in this selection.")
    else:
        slow_table = slow[["Product_Name", "Category", "Warehouse", "Current_Stock", "Avg_Monthly_Dispatch",
                           "Months_Of_Cover", "Inventory_Value"]].rename(columns={
            "Product_Name": "Product", "Current_Stock": "Current Stock",
            "Avg_Monthly_Dispatch": "Avg Monthly Dispatch", "Months_Of_Cover": "Months of Cover",
            "Inventory_Value": "Inventory Value"})
        st.dataframe(slow_table, hide_index=True, column_config={
            "Avg Monthly Dispatch": st.column_config.NumberColumn(format="%.1f"),
            "Months of Cover": st.column_config.NumberColumn(format="%.1f"),
            "Inventory Value": st.column_config.NumberColumn(format="$%.2f"),
        })
        st.download_button("Download slow-moving products as CSV", to_csv_bytes(slow_table),
                           file_name="slow_moving_products.csv", mime="text/csv", key="download_slow")
