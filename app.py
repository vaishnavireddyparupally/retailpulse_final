import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import date, timedelta

st.set_page_config(page_title="RetailPulse | Retail Intelligence", page_icon="🛍️", layout="wide")

# ---------------------------- Demo data ---------------------------------------
@st.cache_data
def make_data(seed=42, n_days=540, n_customers=650):
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    dates = pd.date_range(end=end, periods=n_days, freq="D")
    products = ["Everyday Tee", "Running Shoes", "Coffee Maker", "Wireless Earbuds",
                "Denim Jeans", "Water Bottle", "Backpack", "Yoga Mat"]
    categories = {"Everyday Tee":"Apparel", "Running Shoes":"Footwear", "Coffee Maker":"Home",
                  "Wireless Earbuds":"Electronics", "Denim Jeans":"Apparel", "Water Bottle":"Lifestyle",
                  "Backpack":"Accessories", "Yoga Mat":"Fitness"}
    price_map = {"Everyday Tee":22, "Running Shoes":78, "Coffee Maker":54, "Wireless Earbuds":65,
                 "Denim Jeans":48, "Water Bottle":18, "Backpack":42, "Yoga Mat":30}
    rows = []
    for d in dates:
        weekday_factor = 1.15 if d.dayofweek in (5,6) else 1.0
        seasonal = 1 + 0.16*np.sin(2*np.pi*d.dayofyear/365.25)
        promo = 1.35 if rng.random() < 0.09 else 1.0
        for p in products:
            qty = max(0, int(rng.poisson((8 + price_map[p]/18) * weekday_factor * seasonal * promo)))
            price = price_map[p] * (0.9 if promo > 1 else 1.0)
            rows.append((d, p, categories[p], qty, price, qty*price, bool(promo > 1)))
    sales = pd.DataFrame(rows, columns=["date","product","category","units_sold","unit_price","revenue","promotion"])
    # Customer-level RFM / churn demo features
    last_purchase = rng.integers(1, 180, size=n_customers)
    frequency = rng.poisson(8, size=n_customers) + 1
    spend = np.round(rng.gamma(3.5, 55, size=n_customers), 2)
    tenure = rng.integers(30, 1200, size=n_customers)
    support = rng.poisson(1.2, size=n_customers)
    discount = rng.binomial(1, 0.42, size=n_customers)
    churn_prob = 1/(1+np.exp(-(-2.2 + 0.018*last_purchase - 0.13*frequency + 0.22*support + 0.35*discount)))
    churn = rng.binomial(1, np.clip(churn_prob, 0.02, 0.92))
    customers = pd.DataFrame({
        "customer_id":[f"C{i:04d}" for i in range(1,n_customers+1)],
        "recency_days":last_purchase, "purchase_frequency":frequency,
        "total_spend":spend, "tenure_days":tenure, "support_tickets":support,
        "discount_user":discount, "churned":churn
    })
    inventory = pd.DataFrame({
        "product": products,
        "category":[categories[p] for p in products],
        "stock_on_hand":rng.integers(20,180,size=len(products)),
        "lead_time_days":rng.integers(3,15,size=len(products)),
        "safety_stock":rng.integers(8,28,size=len(products)),
        "unit_cost":[round(price_map[p]*0.48,2) for p in products],
        "unit_price":[price_map[p] for p in products]
    })
    return sales, customers, inventory

sales, customers, inventory = make_data()
products = sales["product"].unique().tolist()

# ---------------------------- Sidebar -----------------------------------------
st.sidebar.title("🛍️ RetailPulse")
st.sidebar.caption("AI-powered retail intelligence")
page = st.sidebar.radio("Navigate", [
    "Executive Overview", "Sales & Demand Forecast", "Customer Segmentation",
    "Churn Prediction", "Inventory Optimization", "Data Explorer", "Model Monitoring"
])
st.sidebar.divider()
st.sidebar.caption("Demo mode: synthetic sample data")
st.sidebar.caption("Use the CSV uploader in Data Explorer to explore your own sales data.")

# ---------------------------- Helpers -----------------------------------------
def money(v):
    return f"${v:,.0f}"

def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta)

def make_forecast(product, horizon=30):
    df = sales[sales["product"] == product].groupby("date", as_index=False)["units_sold"].sum()
    df = df.sort_values("date")
    df["lag_1"] = df["units_sold"].shift(1)
    df["lag_7"] = df["units_sold"].shift(7)
    df["rolling_7"] = df["units_sold"].shift(1).rolling(7).mean()
    df["dow"] = df["date"].dt.dayofweek
    df["doy"] = df["date"].dt.dayofyear
    model_df = df.dropna().copy()
    features = ["lag_1","lag_7","rolling_7","dow","doy"]
    model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=2)
    model.fit(model_df[features], model_df["units_sold"])
    hist = df[["date","units_sold"]].copy()
    vals = list(df["units_sold"].astype(float))
    future_rows = []
    last_date = df["date"].max()
    for step in range(1, horizon+1):
        next_date = last_date + pd.Timedelta(days=step)
        feat = pd.DataFrame([{
            "lag_1": vals[-1], "lag_7": vals[-7] if len(vals)>=7 else vals[-1],
            "rolling_7": float(np.mean(vals[-7:])), "dow": next_date.dayofweek,
            "doy": next_date.dayofyear
        }])
        pred = max(0, float(model.predict(feat[features])[0]))
        vals.append(pred)
        future_rows.append((next_date, pred))
    hist["type"] = "Historical"
    future = pd.DataFrame(future_rows, columns=["date","units_sold"])
    future["type"] = "Forecast"
    return pd.concat([hist, future], ignore_index=True), model, model_df, features

# ---------------------------- Pages -------------------------------------------
if page == "Executive Overview":
    st.title("RetailPulse — Executive Overview")
    st.write("A single view of sales performance, customer health, demand, and inventory.")
    total_revenue = sales.revenue.sum()
    total_units = sales.units_sold.sum()
    avg_order_value = sales.groupby("date").revenue.sum().mean()
    churn_rate = customers.churned.mean()
    stock_value = (inventory.stock_on_hand * inventory.unit_cost).sum()
    metric_row([
        ("Revenue (sample period)", money(total_revenue), "synthetic demo"),
        ("Units sold", f"{total_units:,.0f}", None),
        ("Avg. daily revenue", money(avg_order_value), None),
        ("Customer churn rate", f"{churn_rate:.1%}", None),
        ("Inventory cost value", money(stock_value), None)
    ])
    daily = sales.groupby("date", as_index=False).agg(revenue=("revenue","sum"), units=("units_sold","sum"))
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Revenue trend")
        st.plotly_chart(px.line(daily, x="date", y="revenue", title="Daily revenue", template="plotly_white"), use_container_width=True)
    with c2:
        st.subheader("Revenue by category")
        cat = sales.groupby("category", as_index=False).revenue.sum().sort_values("revenue", ascending=False)
        st.plotly_chart(px.bar(cat, x="category", y="revenue", color="category", title="Category contribution", template="plotly_white"), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Top products")
        top = sales.groupby("product", as_index=False).revenue.sum().nlargest(8, "revenue")
        st.plotly_chart(px.bar(top, x="revenue", y="product", orientation="h", title="Revenue by product", template="plotly_white"), use_container_width=True)
    with c4:
        st.subheader("Customer churn distribution")
        ch = customers.churned.value_counts().rename(index={0:"Active",1:"Churned"}).reset_index()
        ch.columns = ["status","customers"]
        st.plotly_chart(px.pie(ch, names="status", values="customers", hole=0.55, title="Customer status"), use_container_width=True)
    st.info("All figures are generated from synthetic demonstration data. Replace it with validated business data before making decisions.")

elif page == "Sales & Demand Forecast":
    st.title("📈 Sales & Demand Forecasting")
    left, right = st.columns([2,1])
    with left:
        chosen = st.selectbox("Choose product", products)
    with right:
        horizon = st.slider("Forecast horizon (days)", 7, 60, 30)
    forecast, model, model_df, features = make_forecast(chosen, horizon)
    fig = px.line(forecast, x="date", y="units_sold", color="type", markers=False,
                  title=f"{chosen}: historical demand + {horizon}-day forecast", template="plotly_white")
    fig.update_layout(xaxis_title="Date", yaxis_title="Units")
    st.plotly_chart(fig, use_container_width=True)
    # holdout validation on last 20% of prepared data
    cut = int(len(model_df)*0.8)
    X = model_df[features]
    y = model_df["units_sold"]
    val_model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=2)
    val_model.fit(X.iloc[:cut], y.iloc[:cut])
    pred = val_model.predict(X.iloc[cut:])
    mae = mean_absolute_error(y.iloc[cut:], pred)
    rmse = np.sqrt(mean_squared_error(y.iloc[cut:], pred))
    metric_row([("Validation MAE", f"{mae:.2f} units", None), ("Validation RMSE", f"{rmse:.2f} units", None),
                ("Forecast total", f"{forecast[forecast.type=='Forecast'].units_sold.sum():,.0f} units", None)])
    st.caption("Baseline recursive Random Forest forecast using lag, rolling-average, weekday and day-of-year features. This is a demo baseline, not Prophet/LSTM ensemble.")
    st.download_button("Download forecast CSV", forecast.to_csv(index=False).encode(), file_name="retailpulse_demand_forecast.csv", mime="text/csv")

elif page == "Customer Segmentation":
    st.title("👥 Customer Segmentation")
    st.write("Segments customers using recency, purchase frequency, total spend, and tenure.")
    features = ["recency_days","purchase_frequency","total_spend","tenure_days"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(customers[features])
    k = st.slider("Number of segments (K)", 3, 8, 5)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    out = customers.copy()
    out["segment"] = km.fit_predict(X_scaled)
    profile = out.groupby("segment", as_index=False).agg(
        customers=("customer_id","count"), avg_spend=("total_spend","mean"),
        avg_recency=("recency_days","mean"), avg_frequency=("purchase_frequency","mean"),
        churn_rate=("churned","mean"))
    profile["segment_name"] = profile.apply(lambda r: "Loyal / high value" if r.avg_frequency >= out.purchase_frequency.quantile(.75) and r.avg_spend >= out.total_spend.quantile(.65)
                                             else ("At-risk / inactive" if r.avg_recency >= out.recency_days.quantile(.70) else "Developing / regular"), axis=1)
    metric_row([("Segments", str(k), None), ("Customers", f"{len(out):,}", None),
                ("Highest avg spend", money(profile.avg_spend.max()), None)])
    c1,c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.scatter(out, x="recency_days", y="total_spend", color=out.segment.astype(str),
                                   size="purchase_frequency", hover_data=["customer_id"], title="Customer map: recency vs spend",
                                   labels={"color":"Segment"}, template="plotly_white"), use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(profile, x="segment", y="customers", color="segment_name",
                               title="Customers per segment", template="plotly_white"), use_container_width=True)
    st.subheader("Segment profiles")
    st.dataframe(profile.round(2), use_container_width=True)
    st.download_button("Download segmented customers", out.to_csv(index=False).encode(), "customer_segments.csv", "text/csv")

elif page == "Churn Prediction":
    st.title("🚦 Customer Churn Prediction")
    st.write("Train a demo classifier to flag customers who may be at risk of leaving.")
    features = ["recency_days","purchase_frequency","total_spend","tenure_days","support_tickets","discount_user"]
    X = customers[features]
    y = customers["churned"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=160, random_state=42, class_weight="balanced", min_samples_leaf=3)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:,1]
    preds = (proba >= .5).astype(int)
    auc = roc_auc_score(y_test, proba) if y_test.nunique() > 1 else 0.0
    metric_row([("ROC-AUC", f"{auc:.3f}", None), ("Accuracy", f"{accuracy_score(y_test,preds):.1%}", None),
                ("At-risk in full sample", f"{clf.predict_proba(X)[:,1].mean():.1%}", None)])
    scored = customers.copy()
    scored["churn_probability"] = clf.predict_proba(X)[:,1]
    scored["risk_band"] = pd.cut(scored.churn_probability, bins=[-0.001,.35,.65,1.001], labels=["Low","Medium","High"])
    c1,c2 = st.columns(2)
    with c1:
        risk_counts = scored.risk_band.value_counts().reindex(["Low","Medium","High"]).fillna(0).reset_index()
        risk_counts.columns=["risk_band","customers"]
        st.plotly_chart(px.bar(risk_counts, x="risk_band", y="customers", color="risk_band", title="Predicted churn risk", template="plotly_white"), use_container_width=True)
    with c2:
        imp = pd.DataFrame({"feature":features,"importance":clf.feature_importances_}).sort_values("importance")
        st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h", title="Model feature importance", template="plotly_white"), use_container_width=True)
    st.subheader("Customers to contact first")
    st.dataframe(scored.sort_values("churn_probability", ascending=False)[["customer_id"]+features+["churn_probability","risk_band"]].head(25).round(3), use_container_width=True)
    st.download_button("Download churn scores", scored.to_csv(index=False).encode(), "churn_risk_scores.csv", "text/csv")
    st.caption("The demo labels are synthetic and are only for demonstrating the pipeline; validate on historical churn labels before using operationally.")

elif page == "Inventory Optimization":
    st.title("📦 Inventory Optimization")
    st.write("Estimates reorder suggestions from recent average demand, lead time, and safety stock.")
    lookback = st.slider("Demand lookback (days)", 14, 90, 30)
    cutoff = sales.date.max() - pd.Timedelta(days=lookback-1)
    recent = sales[sales.date >= cutoff].groupby("product", as_index=False).agg(avg_daily_demand=("units_sold","mean"), recent_units=("units_sold","sum"))
    inv = inventory.merge(recent, on="product", how="left")
    inv["lead_time_demand"] = inv.avg_daily_demand * inv.lead_time_days
    inv["reorder_point"] = np.ceil(inv.lead_time_demand + inv.safety_stock)
    inv["recommended_order_qty"] = np.maximum(0, np.ceil(inv.reorder_point - inv.stock_on_hand))
    inv["estimated_order_cost"] = inv.recommended_order_qty * inv.unit_cost
    metric_row([("Products reviewed", str(len(inv)), None),
                ("Products below reorder point", str((inv.stock_on_hand < inv.reorder_point).sum()), None),
                ("Suggested units to order", f"{inv.recommended_order_qty.sum():,.0f}", None),
                ("Estimated order cost", money(inv.estimated_order_cost.sum()), None)])
    fig = px.bar(inv.sort_values("recommended_order_qty", ascending=False), x="product", y="recommended_order_qty",
                 color="category", title="Recommended order quantity by product", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(inv[["product","category","stock_on_hand","avg_daily_demand","lead_time_days","safety_stock","reorder_point","recommended_order_qty","unit_cost","estimated_order_cost"]].round(2), use_container_width=True)
    st.download_button("Download reorder plan", inv.to_csv(index=False).encode(), "inventory_reorder_plan.csv", "text/csv")
    st.caption("Simple policy: reorder up to estimated lead-time demand plus safety stock. Add supplier constraints and service-level statistics for production use.")

elif page == "Data Explorer":
    st.title("🧾 Data Explorer & CSV Upload")
    st.write("Upload a CSV to inspect it. For sales data, try to include date, product, units_sold and revenue columns.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            user_df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(user_df):,} rows and {len(user_df.columns)} columns.")
            st.dataframe(user_df.head(100), use_container_width=True)
            c1,c2,c3 = st.columns(3)
            c1.metric("Rows", f"{len(user_df):,}")
            c2.metric("Columns", str(len(user_df.columns)))
            c3.metric("Missing cells", f"{int(user_df.isna().sum().sum()):,}")
            st.subheader("Missing values by column")
            miss = user_df.isna().sum().rename("missing").reset_index().rename(columns={"index":"column"})
            st.plotly_chart(px.bar(miss, x="column", y="missing", title="Missing values", template="plotly_white"), use_container_width=True)
            numeric = user_df.select_dtypes(include=np.number)
            if not numeric.empty:
                st.subheader("Numeric feature distributions")
                col = st.selectbox("Choose numeric column", numeric.columns)
                st.plotly_chart(px.histogram(user_df, x=col, nbins=30, title=f"Distribution of {col}", template="plotly_white"), use_container_width=True)
            st.download_button("Download uploaded data copy", user_df.to_csv(index=False).encode(), "uploaded_data_copy.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    else:
        st.info("No upload yet — showing the built-in demo data.")
        st.subheader("Sales sample")
        st.dataframe(sales.head(30), use_container_width=True)
        st.subheader("Customer sample")
        st.dataframe(customers.head(20), use_container_width=True)
        st.subheader("Inventory sample")
        st.dataframe(inventory, use_container_width=True)

elif page == "Model Monitoring":
    st.title("🩺 Model Monitoring & Data Quality")
    st.write("A lightweight monitoring view for this demo. Production monitoring should log metrics and compare live feature distributions with a baseline.")
    data_quality = pd.DataFrame({
        "check":["Sales rows present","Missing sales values","Negative units","Negative revenue","Customer ID unique"],
        "status":[
            "PASS" if len(sales)>0 else "FAIL",
            "PASS" if sales.isna().sum().sum()==0 else "WARN",
            "PASS" if (sales.units_sold<0).sum()==0 else "WARN",
            "PASS" if (sales.revenue<0).sum()==0 else "WARN",
            "PASS" if customers.customer_id.is_unique else "FAIL"
        ],
        "value":[len(sales), int(sales.isna().sum().sum()), int((sales.units_sold<0).sum()),
                 int((sales.revenue<0).sum()), int(customers.customer_id.nunique())]
    })
    metric_row([("Sales records", f"{len(sales):,}", None),
                ("Missing cells", f"{sales.isna().sum().sum():,}", None),
                ("Distinct products", str(sales["product"].nunique()), None)])
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Data quality checks")
        st.dataframe(data_quality, use_container_width=True)
    with c2:
        st.subheader("Recent vs prior demand")
        pivot = sales.copy()
        maxd = pivot.date.max()
        pivot["period"] = np.where(pivot.date >= maxd-pd.Timedelta(days=29), "Recent 30d",
                                   np.where(pivot.date < maxd-pd.Timedelta(days=59), "Prior period", "Middle"))
        drift = pivot[pivot.period.isin(["Recent 30d","Prior period"])].groupby(["period","category"], as_index=False).units_sold.mean()
        st.plotly_chart(px.bar(drift, x="category", y="units_sold", color="period", barmode="group",
                               title="Average daily units by category", template="plotly_white"), use_container_width=True)
    st.subheader("Monitoring checklist")
    st.markdown("- [x] Basic schema and missing-value checks\n- [x] Demand distribution comparison\n- [ ] Persist run metrics in MLflow\n- [ ] Add Evidently drift reports and alert thresholds\n- [ ] Schedule retraining and CI/CD")
    st.caption("This page demonstrates monitoring concepts; it does not connect to a production telemetry system.")
