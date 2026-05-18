"""Clean QQQ option chain snapshot for W6 calibration.

Input:  data/cleaned/qqq_options_snapshot.csv  (produced by refresh_options_data.py)
Output: data/cleaned/qqq_options_calib.csv

Cleaning rules (with rationale):
  1. T >= 0.05 (>= ~18 days)
       Ultra-short weekly options have huge IV inversion noise when prices
       are even slightly stale.
  2. OTM only: keep puts with K < S0 and calls with K > S0.
       Deep-ITM options are illiquid and their `lastPrice` is often months
       stale, producing absurd implied vols. This is also the academic
       standard for surface calibration -- the OTM side carries all the
       smile information anyway, via put-call parity.
  3. moneyness in [0.85, 1.15].
       Concentrate on the informative near-ATM region. Far-strike quotes
       are dominated by tail noise rather than vol structure.
  4. iv_market in [0.10, 0.60].
       Realistic range for QQQ. Anything outside is data error, not signal.
  5. Drop quotes where lastPrice differs from intrinsic + small premium by
       more than 5x  (sanity check for completely broken stale prices).
"""

from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd

INPUT_PATH = Path("data/cleaned/qqq_options_snapshot.csv")
OUTPUT_PATH = Path("data/cleaned/qqq_options_calib.csv")

# Filter thresholds -- can be tuned, but these are reasonable defaults.
T_MIN = 0.05            # at least ~18 calendar days
MONEYNESS_MIN = 0.85
MONEYNESS_MAX = 1.15
IV_MIN = 0.10
IV_MAX = 0.60


def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")
    print(f"S0 = {df['S0'].iloc[0]:.2f}, snapshot = {df['snapshot_date'].iloc[0]}")

    print("\n--- CLEANING ---")
    n0 = len(df)

    # Step 1: minimum time to expiry
    df = df[df["T"] >= T_MIN].copy()
    print(f"After T >= {T_MIN} (>= {int(T_MIN*365)} days):   {len(df)}  "
          f"(dropped {n0 - len(df)})")
    n1 = len(df)

    # Step 2: OTM only
    s0 = df["S0"].iloc[0]
    is_otm = ((df["type"] == "put") & (df["strike"] < s0)) | \
             ((df["type"] == "call") & (df["strike"] > s0))
    df = df[is_otm].copy()
    print(f"After OTM only (put K<S0, call K>S0):  {len(df)}  "
          f"(dropped {n1 - len(df)})")
    n2 = len(df)

    # Step 3: moneyness band
    df = df[(df["moneyness"] >= MONEYNESS_MIN) &
            (df["moneyness"] <= MONEYNESS_MAX)].copy()
    print(f"After moneyness in [{MONEYNESS_MIN}, {MONEYNESS_MAX}]:  {len(df)}  "
          f"(dropped {n2 - len(df)})")
    n3 = len(df)

    # Step 4: realistic IV range
    df = df[(df["iv_market"] >= IV_MIN) &
            (df["iv_market"] <= IV_MAX)].copy()
    print(f"After IV in [{IV_MIN}, {IV_MAX}]:           {len(df)}  "
          f"(dropped {n3 - len(df)})")
    n4 = len(df)

    # Step 5: sanity check on price vs intrinsic
    # OTM intrinsic is 0, so we just require price > 0.05 (no penny weirdness)
    df = df[df["price"] >= 0.05].copy()
    print(f"After price >= $0.05:               {len(df)}  "
          f"(dropped {n4 - len(df)})")

    # Final report
    print(f"\n=== FINAL: {len(df)} clean quotes ===\n")
    print("Rows per (expiry, type):")
    print(df.groupby(["expiry", "type"]).size().unstack(fill_value=0))

    print("\nIV statistics per expiry:")
    iv_stats = df.groupby("expiry")["iv_market"].describe()[
        ["count", "mean", "std", "min", "max"]
    ]
    print(iv_stats.round(3))

    print("\nMoneyness range per expiry:")
    m_range = df.groupby("expiry").agg(
        n=("moneyness", "size"),
        m_min=("moneyness", "min"),
        m_max=("moneyness", "max"),
    )
    print(m_range.round(3))

    # Save
    df = df.sort_values(["T", "strike", "type"]).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
