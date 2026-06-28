"""
=============================================================
  PROJECT: Sales Forecasting
  Goal   : Predict future sales from historical time-series data
  Author : [Your Name]
  Models : Moving Average Baseline · Linear Trend · Random Forest
  Dataset: Synthetic retail dataset (generated in-script)
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

# ── output folder ──────────────────────────────────────────
OUT = "/mnt/user-data/outputs/sales_forecasting"
os.makedirs(OUT, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════
# 1.  GENERATE REALISTIC RETAIL SALES DATASET
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("  SALES FORECASTING — END-TO-END PROJECT")
print("=" * 60)

print("\n[1/7] Generating synthetic retail sales dataset …")

dates = pd.date_range(start="2019-01-01", end="2023-12-31", freq="W")
n = len(dates)

# --- Components ---
t = np.arange(n)
trend      = 500 + 1.2 * t                                        # upward trend
yearly     = 300 * np.sin(2 * np.pi * t / 52)                    # yearly seasonality
quarterly  = 100 * np.sin(2 * np.pi * t / 13)                    # quarterly bumps
noise      = np.random.normal(0, 80, n)                           # random noise

# Holiday/promo spikes (Black Friday, Christmas, Summer sale)
promo = np.zeros(n)
for i, d in enumerate(dates):
    if d.month == 11 and 22 <= d.day <= 30:   promo[i] += 600    # Black Friday
    if d.month == 12 and d.day >= 15:          promo[i] += 450    # Christmas
    if d.month == 7  and d.day <= 14:          promo[i] += 250    # Summer sale

sales = trend + yearly + quarterly + promo + noise
sales = np.clip(sales, 50, None)   # no negative sales

df = pd.DataFrame({
    "date":       dates,
    "sales":      sales.round(2),
    "promotion":  (promo > 0).astype(int),
    "month":      dates.month,
    "week":       dates.isocalendar().week.astype(int),
    "quarter":    dates.quarter,
    "year":       dates.year,
    "day_of_year":dates.day_of_year,
})

df.set_index("date", inplace=True)

print(f"   Dataset shape : {df.shape}")
print(f"   Date range    : {df.index.min().date()} → {df.index.max().date()}")
print(f"   Sales range   : ${df['sales'].min():.0f} – ${df['sales'].max():.0f}")
print(f"   Promo weeks   : {df['promotion'].sum()}")

# ═══════════════════════════════════════════════════════════
# 2.  EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════
print("\n[2/7] Exploratory Data Analysis …")

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Sales Forecasting — Exploratory Data Analysis", fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# 2a. Full time series
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df.index, df["sales"], color="#2563EB", linewidth=0.9, alpha=0.8, label="Weekly Sales")
ax1.scatter(df[df["promotion"] == 1].index, df[df["promotion"] == 1]["sales"],
            color="#EF4444", s=18, zorder=5, label="Promo Week")
ax1.set_title("Weekly Sales Over Time (2019–2023)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Sales ($)")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# 2b. Monthly avg
ax2 = fig.add_subplot(gs[1, 0])
monthly_avg = df.groupby("month")["sales"].mean()
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
bars = ax2.bar(month_names, monthly_avg.values, color="#3B82F6", edgecolor="white", linewidth=0.5)
ax2.set_title("Average Sales by Month", fontsize=11, fontweight="bold")
ax2.set_ylabel("Avg Sales ($)")
ax2.tick_params(axis="x", rotation=45)
for bar, val in zip(bars, monthly_avg.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8, f"${val:.0f}",
             ha="center", va="bottom", fontsize=7)
ax2.grid(axis="y", alpha=0.3)

# 2c. Yearly trend
ax3 = fig.add_subplot(gs[1, 1])
yearly_avg = df.groupby("year")["sales"].agg(["mean", "sum"])
ax3.bar(yearly_avg.index, yearly_avg["mean"], color="#10B981", edgecolor="white")
ax3.set_title("Average Weekly Sales by Year", fontsize=11, fontweight="bold")
ax3.set_ylabel("Avg Sales ($)")
ax3.set_xlabel("Year")
for i, (yr, row) in enumerate(yearly_avg.iterrows()):
    ax3.text(yr, row["mean"] + 5, f"${row['mean']:.0f}", ha="center", fontsize=9)
ax3.grid(axis="y", alpha=0.3)

# 2d. Promo vs non-promo
ax4 = fig.add_subplot(gs[2, 0])
promo_data    = df[df["promotion"] == 1]["sales"]
nonpromo_data = df[df["promotion"] == 0]["sales"]
ax4.hist(nonpromo_data, bins=35, alpha=0.7, color="#3B82F6", label=f"Regular (n={len(nonpromo_data)})")
ax4.hist(promo_data,    bins=15, alpha=0.7, color="#EF4444", label=f"Promo (n={len(promo_data)})")
ax4.set_title("Sales Distribution: Promo vs Regular", fontsize=11, fontweight="bold")
ax4.set_xlabel("Sales ($)")
ax4.set_ylabel("Frequency")
ax4.legend()
ax4.grid(alpha=0.3)

# 2e. Rolling mean
ax5 = fig.add_subplot(gs[2, 1])
rolling4  = df["sales"].rolling(4).mean()
rolling13 = df["sales"].rolling(13).mean()
ax5.plot(df.index, df["sales"],  color="#93C5FD", linewidth=0.7, alpha=0.6, label="Actual")
ax5.plot(df.index, rolling4,     color="#2563EB", linewidth=1.2, label="4-wk MA")
ax5.plot(df.index, rolling13,    color="#EF4444", linewidth=1.5, label="13-wk MA")
ax5.set_title("Rolling Moving Averages", fontsize=11, fontweight="bold")
ax5.set_ylabel("Sales ($)")
ax5.legend(fontsize=9)
ax5.grid(alpha=0.3)

plt.savefig(f"{OUT}/01_eda.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved → 01_eda.png")

# ═══════════════════════════════════════════════════════════
# 3.  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════
print("\n[3/7] Feature Engineering …")

df_feat = df.copy()

# Lag features
for lag in [1, 2, 4, 8, 13, 26, 52]:
    df_feat[f"lag_{lag}"] = df_feat["sales"].shift(lag)

# Rolling statistics
for w in [4, 8, 13]:
    df_feat[f"roll_mean_{w}"] = df_feat["sales"].shift(1).rolling(w).mean()
    df_feat[f"roll_std_{w}"]  = df_feat["sales"].shift(1).rolling(w).std()

# Cyclical encoding of month and week
df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12)
df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12)
df_feat["week_sin"]  = np.sin(2 * np.pi * df_feat["week"]  / 52)
df_feat["week_cos"]  = np.cos(2 * np.pi * df_feat["week"]  / 52)

# Time index
df_feat["t_index"] = np.arange(len(df_feat))

df_feat.dropna(inplace=True)

FEATURES = [c for c in df_feat.columns if c != "sales"]
TARGET   = "sales"

print(f"   Features created : {len(FEATURES)}")
print(f"   Usable rows      : {len(df_feat)}")

# ═══════════════════════════════════════════════════════════
# 4.  TRAIN / TEST SPLIT  (last 52 weeks = test)
# ═══════════════════════════════════════════════════════════
print("\n[4/7] Train / Test Split (last 52 weeks as test) …")

HORIZON = 52
train = df_feat.iloc[:-HORIZON]
test  = df_feat.iloc[-HORIZON:]

X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

print(f"   Train : {len(train)} weeks  ({train.index.min().date()} → {train.index.max().date()})")
print(f"   Test  : {len(test)}  weeks  ({test.index.min().date()} → {test.index.max().date()})")

# ═══════════════════════════════════════════════════════════
# 5.  MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════
print("\n[5/7] Training models …")

def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"   {name:<30} MAE={mae:7.1f}  RMSE={rmse:7.1f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {"Model": name, "MAE": round(mae,1), "RMSE": round(rmse,1),
            "R2": round(r2,4), "MAPE%": round(mape,2)}

results  = []
preds    = {}

# --- Baseline: 52-week moving average ---
baseline_pred = np.array([y_train.iloc[-52:].mean()] * HORIZON)
results.append(evaluate("Baseline (52-wk MA)", y_test, baseline_pred))
preds["Baseline"] = baseline_pred

# --- Linear Regression ---
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
results.append(evaluate("Linear Regression", y_test, lr_pred))
preds["Linear Regression"] = lr_pred

# --- Random Forest ---
rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=3,
                           random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
results.append(evaluate("Random Forest", y_test, rf_pred))
preds["Random Forest"] = rf_pred

# --- Gradient Boosting ---
gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                subsample=0.8, random_state=SEED)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
results.append(evaluate("Gradient Boosting", y_test, gb_pred))
preds["Gradient Boosting"] = gb_pred

results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUT}/model_results.csv", index=False)
print(f"\n   Results saved → model_results.csv")

# ═══════════════════════════════════════════════════════════
# 6.  VISUALISE FORECAST vs ACTUAL
# ═══════════════════════════════════════════════════════════
print("\n[6/7] Generating forecast plots …")

COLORS = {
    "Baseline":          "#94A3B8",
    "Linear Regression": "#F59E0B",
    "Random Forest":     "#10B981",
    "Gradient Boosting": "#8B5CF6",
}

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("Sales Forecasting — Forecast vs Actual (Test Period: Last 52 Weeks)",
             fontsize=15, fontweight="bold", y=1.01)

test_index = test.index

for ax, (model_name, pred) in zip(axes.flatten(), preds.items()):
    # Context: last 26 weeks of train
    ctx_idx  = train.index[-26:]
    ctx_vals = y_train.values[-26:]

    ax.plot(ctx_idx,   ctx_vals, color="#2563EB", linewidth=1.2, label="Historical")
    ax.plot(test_index, y_test.values, color="#1E293B", linewidth=1.5,
            label="Actual Sales", zorder=4)
    ax.plot(test_index, pred, color=COLORS[model_name], linewidth=1.8,
            linestyle="--", label=f"Forecast ({model_name})", zorder=5)

    # Confidence band (±1 std of residuals on train)
    train_resid = y_train.values - (rf.predict(X_train) if model_name in
                                     ["Random Forest","Gradient Boosting"] else
                                     lr.predict(X_train))
    std = np.std(train_resid)
    ax.fill_between(test_index, pred - std, pred + std,
                    alpha=0.15, color=COLORS[model_name], label="±1 std band")

    row_match = results_df[results_df["Model"].str.contains(model_name.split()[0])]
    row = row_match.iloc[0] if len(row_match) > 0 else results_df.iloc[0]
    ax.set_title(f"{model_name}\nMAE=${row['MAE']:.0f}  RMSE=${row['RMSE']:.0f}"
                 f"  R²={row['R2']:.3f}  MAPE={row['MAPE%']:.1f}%",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Sales ($)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(f"{OUT}/02_forecast_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved → 02_forecast_vs_actual.png")

# --- Feature importance (Random Forest) ---
feat_imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Sales Forecasting — Model Insights", fontsize=14, fontweight="bold")

ax = axes[0]
top_n = feat_imp.head(15)
colors_fi = ["#2563EB" if "lag" in f else "#10B981" if "roll" in f
             else "#F59E0B" if "month" in f or "week" in f or "quarter" in f
             else "#EF4444" for f in top_n.index]
bars = ax.barh(top_n.index[::-1], top_n.values[::-1], color=colors_fi[::-1], edgecolor="white")
ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=11, fontweight="bold")
ax.set_xlabel("Importance Score")
ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, top_n.values[::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8)

# Legend for colors
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#2563EB", label="Lag features"),
                   Patch(facecolor="#10B981", label="Rolling statistics"),
                   Patch(facecolor="#F59E0B", label="Calendar features"),
                   Patch(facecolor="#EF4444", label="Other")]
ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

# --- Model comparison bar chart ---
ax2 = axes[1]
metrics = ["MAE", "RMSE", "MAPE%"]
x = np.arange(len(results_df))
width = 0.25
palette = ["#3B82F6", "#EF4444", "#10B981"]

for i, (metric, color) in enumerate(zip(metrics, palette)):
    vals = results_df[metric].values
    # Normalise for visual comparison
    normed = vals / vals.max()
    bars2 = ax2.bar(x + i*width, vals, width, label=metric, color=color, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars2, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=7)

ax2.set_xticks(x + width)
ax2.set_xticklabels([r["Model"].replace(" ", "\n") for _, r in results_df.iterrows()], fontsize=9)
ax2.set_title("Model Comparison (MAE / RMSE / MAPE%)\nLower is better", fontsize=11, fontweight="bold")
ax2.set_ylabel("Error Value")
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/03_model_insights.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved → 03_model_insights.png")

# ═══════════════════════════════════════════════════════════
# 7.  FUTURE FORECAST (next 26 weeks)
# ═══════════════════════════════════════════════════════════
print("\n[7/7] Generating future 26-week forecast …")

# We use Gradient Boosting (usually most accurate)
# Build a simple recursive forecast using known calendar features
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(weeks=1),
                              periods=26, freq="W")

# Use last known sales values as seed for lag features
history_sales = list(df["sales"].values)

future_rows = []
for fd in future_dates:
    row = {
        "promotion":   1 if (fd.month == 11 and 22 <= fd.day <= 30) or
                            (fd.month == 12 and fd.day >= 15) or
                            (fd.month == 7  and fd.day <= 14) else 0,
        "month":       fd.month,
        "week":        fd.isocalendar()[1],
        "quarter":     fd.quarter,
        "year":        fd.year,
        "day_of_year": fd.day_of_year,
        "month_sin":   np.sin(2 * np.pi * fd.month / 12),
        "month_cos":   np.cos(2 * np.pi * fd.month / 12),
        "week_sin":    np.sin(2 * np.pi * fd.isocalendar()[1] / 52),
        "week_cos":    np.cos(2 * np.pi * fd.isocalendar()[1] / 52),
        "t_index":     len(df_feat) + len(future_rows),
        "lag_1":   history_sales[-1],
        "lag_2":   history_sales[-2],
        "lag_4":   history_sales[-4],
        "lag_8":   history_sales[-8],
        "lag_13":  history_sales[-13],
        "lag_26":  history_sales[-26],
        "lag_52":  history_sales[-52],
        "roll_mean_4":  np.mean(history_sales[-4:]),
        "roll_std_4":   np.std(history_sales[-4:]),
        "roll_mean_8":  np.mean(history_sales[-8:]),
        "roll_std_8":   np.std(history_sales[-8:]),
        "roll_mean_13": np.mean(history_sales[-13:]),
        "roll_std_13":  np.std(history_sales[-13:]),
    }
    pred_val = gb.predict(pd.DataFrame([row])[FEATURES])[0]
    history_sales.append(pred_val)
    row["predicted_sales"] = round(pred_val, 2)
    row["date"] = fd
    future_rows.append(row)

future_df = pd.DataFrame(future_rows).set_index("date")
future_df[["predicted_sales","promotion"]].to_csv(f"{OUT}/future_forecast_26weeks.csv")

# Plot
fig, ax = plt.subplots(figsize=(16, 6))
context_n = 52
ax.plot(df.index[-context_n:], df["sales"].values[-context_n:],
        color="#2563EB", linewidth=1.5, label="Historical Sales (last 52 wks)")
ax.plot(future_df.index, future_df["predicted_sales"],
        color="#8B5CF6", linewidth=2, linestyle="--", label="26-Week Forecast (GB)")
ax.scatter(future_df[future_df["promotion"] == 1].index,
           future_df[future_df["promotion"] == 1]["predicted_sales"],
           color="#EF4444", s=60, zorder=6, label="Promo Period", marker="*")

# Shaded forecast zone
ax.axvspan(future_df.index[0], future_df.index[-1], alpha=0.06, color="#8B5CF6")
ax.axvline(x=future_df.index[0], color="#8B5CF6", linestyle=":", linewidth=1.5, alpha=0.8)

# Confidence band
resid_std = np.std(y_test.values - gb_pred)
ax.fill_between(future_df.index,
                future_df["predicted_sales"] - resid_std,
                future_df["predicted_sales"] + resid_std,
                alpha=0.15, color="#8B5CF6", label="±1 std uncertainty")

ax.set_title("Sales Forecast — Next 26 Weeks (Gradient Boosting Model)", fontsize=13, fontweight="bold")
ax.set_ylabel("Sales ($)")
ax.set_xlabel("Date")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/04_future_forecast.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved → 04_future_forecast.png")
print("   Saved → future_forecast_26weeks.csv")

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)
print(results_df.to_string(index=False))
best = results_df.loc[results_df["MAPE%"].idxmin()]
print(f"\n  ✅  Best model : {best['Model']}")
print(f"      MAE  = ${best['MAE']:.1f}  |  RMSE = ${best['RMSE']:.1f}")
print(f"      R²   = {best['R2']:.4f}   |  MAPE = {best['MAPE%']:.2f}%")
print("\n  Output files saved to:", OUT)
print("  • 01_eda.png                  — Exploratory Data Analysis")
print("  • 02_forecast_vs_actual.png   — All models: forecast vs actual")
print("  • 03_model_insights.png       — Feature importance + model comparison")
print("  • 04_future_forecast.png      — 26-week future prediction")
print("  • model_results.csv           — Metrics table")
print("  • future_forecast_26weeks.csv — Future predictions")
print("=" * 60)
