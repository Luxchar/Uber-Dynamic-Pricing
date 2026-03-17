"""
Generate 1M synthetic rows that mirror the statistical distributions
of data/raw/dynamic_pricing.csv, then save the combined dataset.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_SYNTHETIC = 20_000

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "dynamic_pricing.csv"
OUT_CSV = PROJECT_ROOT / "data" / "raw" / "dynamic_pricing_synth.csv"

rng = np.random.default_rng(SEED)

# ── load real data to derive distributions ──────────────────────────────────
real = pd.read_csv(RAW_CSV)
n = N_SYNTHETIC

# ── categorical columns  (sample with replacement, preserving freq) ─────────


def sample_cat(col, size):
    counts = real[col].value_counts(normalize=True)
    return rng.choice(counts.index, size=size, p=counts.values)


location = sample_cat("Location_Category",      n)
loyalty = sample_cat("Customer_Loyalty_Status", n)
time_of_book = sample_cat("Time_of_Booking",         n)
vehicle = sample_cat("Vehicle_Type",            n)

# ── integer / float columns  (fit normal, clip to observed range) ────────────


def sample_int(col, size, low=None, high=None):
    mu, sigma = real[col].mean(), real[col].std()
    vals = rng.normal(mu, sigma, size).round().astype(int)
    lo = real[col].min() if low is None else low
    hi = real[col].max() if high is None else high
    return np.clip(vals, lo, hi)


def sample_float(col, size, decimals=2):
    mu, sigma = real[col].mean(), real[col].std()
    vals = rng.normal(mu, sigma, size)
    lo, hi = real[col].min(), real[col].max()
    return np.round(np.clip(vals, lo, hi), decimals)


riders = sample_int("Number_of_Riders",    n, low=1)
drivers = sample_int("Number_of_Drivers",   n, low=1)
past_rides = sample_int("Number_of_Past_Rides", n, low=0)
ratings = sample_float("Average_Ratings",   n, decimals=2)
duration = sample_int("Expected_Ride_Duration", n, low=1)

# ── derive Historical_Cost_of_Ride using a simple pricing formula ────────────
# fit a linear model on the real data so synthetic costs are realistic

cat_cols = ["Location_Category", "Customer_Loyalty_Status",
            "Time_of_Booking", "Vehicle_Type"]
num_cols = ["Number_of_Riders", "Number_of_Drivers", "Number_of_Past_Rides",
            "Average_Ratings", "Expected_Ride_Duration"]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ("num", "passthrough", num_cols),
])
model = Pipeline([("pre", pre), ("reg", LinearRegression())])
model.fit(real[cat_cols + num_cols], real["Historical_Cost_of_Ride"])

synth_features = pd.DataFrame({
    "Location_Category":      location,
    "Customer_Loyalty_Status": loyalty,
    "Time_of_Booking":        time_of_book,
    "Vehicle_Type":           vehicle,
    "Number_of_Riders":       riders,
    "Number_of_Drivers":      drivers,
    "Number_of_Past_Rides":   past_rides,
    "Average_Ratings":        ratings,
    "Expected_Ride_Duration": duration,
})

predicted_cost = model.predict(synth_features[cat_cols + num_cols])
# add realistic noise (std of real residuals)
residual_std = (real["Historical_Cost_of_Ride"] -
                model.predict(real[cat_cols + num_cols])).std()
cost = np.round(
    predicted_cost + rng.normal(0, residual_std, n),
    decimals=8
)
lo, hi = real["Historical_Cost_of_Ride"].min(), real["Historical_Cost_of_Ride"].max()
cost = np.clip(cost, lo, hi)

# ── assemble and save ────────────────────────────────────────────────────────
synth = pd.DataFrame({
    "Number_of_Riders":        riders,
    "Number_of_Drivers":       drivers,
    "Location_Category":       location,
    "Customer_Loyalty_Status": loyalty,
    "Number_of_Past_Rides":    past_rides,
    "Average_Ratings":         ratings,
    "Time_of_Booking":         time_of_book,
    "Vehicle_Type":            vehicle,
    "Expected_Ride_Duration":  duration,
    "Historical_Cost_of_Ride": cost,
})

combined = pd.concat([real, synth], ignore_index=True)
combined.to_csv(OUT_CSV, index=False)

print(f"Real rows      : {len(real):>10,}")
print(f"Synthetic rows : {len(synth):>10,}")
print(f"Combined rows  : {len(combined):>10,}")
print(f"Saved to       : {OUT_CSV}")
