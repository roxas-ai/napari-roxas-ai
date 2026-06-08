import pytest
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)        # show all columns
pd.set_option("display.max_colwidth", None)      # don't truncate cell contents
pd.set_option("display.width", None)             # let pandas auto-detect terminal width
pd.set_option("display.expand_frame_repr", False)  # avoid wrapped multiple-line repr


class CellProcessor:
    def __init__(self, cells_table, config):
        self.cells_table = cells_table
        self.config = config

    def filter_cells(self):
        df = self.cells_table
        config = self.config

        if df is None or df.empty:
            return

        required = ["CWT_pith", "CWT_bark", "CWT_left", "CWT_right"]
        for col in required:
            if col not in df.columns:
                return

        # --- Settings ---
        ll_scaling = float(config.get("ll_scaling", 1.5))
        ul_scaling = float(config.get("ul_scaling", 3.0))
        opp_scaling = float(config.get("opp_scaling", 1.5))
        adj_scaling = float(config.get("adj_scaling", 2.5))
        px_per_um = float(config["pixels_per_um"])
        min_plausible = 1.0 / px_per_um

        # Pull numeric columns
        pi = pd.to_numeric(df["CWT_pith"], errors="coerce")
        ba = pd.to_numeric(df["CWT_bark"], errors="coerce")
        le = pd.to_numeric(df["CWT_left"], errors="coerce")
        ri = pd.to_numeric(df["CWT_right"], errors="coerce")

        # --- Step 1: combined quantiles ---
        tan_all = pd.concat([pi, ba], ignore_index=True).dropna()
        rad_all = pd.concat([le, ri], ignore_index=True).dropna()
        if tan_all.empty or rad_all.empty:
            return

        q1_tan, q3_tan = float(tan_all.quantile(0.25)), float(tan_all.quantile(0.75))
        q1_rad, q3_rad = float(rad_all.quantile(0.25)), float(rad_all.quantile(0.75))

        def _inside_iqr(x, q1, q3):
            return x.notna() & (x >= q1) & (x <= q3)

        confirmed_pi = _inside_iqr(pi, q1_tan, q3_tan)
        confirmed_ba = _inside_iqr(ba, q1_tan, q3_tan)
        confirmed_le = _inside_iqr(le, q1_rad, q3_rad)
        confirmed_ri = _inside_iqr(ri, q1_rad, q3_rad)

        cand_pi = pi.notna() & ~confirmed_pi
        cand_ba = ba.notna() & ~confirmed_ba
        cand_le = le.notna() & ~confirmed_le
        cand_ri = ri.notna() & ~confirmed_ri

        # Step 2: hard limits
        iqr_tan = max(q3_tan - q1_tan, 0.0)
        iqr_rad = max(q3_rad - q1_rad, 0.0)
        ll_tan, ul_tan = max(q1_tan - ll_scaling * iqr_tan, min_plausible), q3_tan + ul_scaling * iqr_tan
        ll_rad, ul_rad = max(q1_rad - ll_scaling * iqr_rad, min_plausible), q3_rad + ul_scaling * iqr_rad

        reason_pi = pd.Series([None]*len(pi), index=pi.index)
        reason_ba = pd.Series([None]*len(ba), index=ba.index)
        reason_le = pd.Series([None]*len(le), index=le.index)
        reason_ri = pd.Series([None]*len(ri), index=ri.index)

        def _apply_hard_limits(x, cand, ll, ul, reason_series):
            out = x.copy()
            mask = cand & x.notna() & ((x < ll) | (x > ul))
            out.loc[mask] = np.nan
            reason_series.loc[mask] = "hard limit"
            return out

        pi = _apply_hard_limits(pi, cand_pi, ll_tan, ul_tan, reason_pi)
        ba = _apply_hard_limits(ba, cand_ba, ll_tan, ul_tan, reason_ba)
        le = _apply_hard_limits(le, cand_le, ll_rad, ul_rad, reason_le)
        ri = _apply_hard_limits(ri, cand_ri, ll_rad, ul_rad, reason_ri)

        # Step 3: opposite-side filtering
        mask = cand_ba & ba.notna() & pi.notna() & (ba > opp_scaling * pi)
        ba.loc[mask] = np.nan
        reason_ba.loc[mask] = "opposite side"

        mask = cand_pi & pi.notna() & ba.notna() & (pi > opp_scaling * ba)
        pi.loc[mask] = np.nan
        reason_pi.loc[mask] = "opposite side"

        mask = cand_le & le.notna() & ri.notna() & (le > opp_scaling * ri)
        le.loc[mask] = np.nan
        reason_le.loc[mask] = "opposite side"

        mask = cand_ri & ri.notna() & le.notna() & (ri > opp_scaling * le)
        ri.loc[mask] = np.nan
        reason_ri.loc[mask] = "opposite side"

        # Step 4: adjacent-side filtering
        ave_lr = pd.Series(
            [np.nanmean([le.iloc[i], ri.iloc[i]]) if not (np.isnan(le.iloc[i]) and np.isnan(ri.iloc[i])) else np.nan
             for i in range(len(le))],
            index=le.index
        )
        ave_pb = pd.Series(
            [np.nanmean([ba.iloc[i], pi.iloc[i]]) if not (np.isnan(ba.iloc[i]) and np.isnan(pi.iloc[i])) else np.nan
             for i in range(len(ba))],
            index=ba.index
        )

        mask = cand_ba & ba.notna() & ave_lr.notna() & (ba > adj_scaling * ave_lr)
        ba.loc[mask] = np.nan
        reason_ba.loc[mask] = "adjacent side"

        mask = cand_pi & pi.notna() & ave_lr.notna() & (pi > adj_scaling * ave_lr)
        pi.loc[mask] = np.nan
        reason_pi.loc[mask] = "adjacent side"

        mask = cand_le & le.notna() & ave_pb.notna() & (le > adj_scaling * ave_pb)
        le.loc[mask] = np.nan
        reason_le.loc[mask] = "adjacent side"

        mask = cand_ri & ri.notna() & ave_pb.notna() & (ri > adj_scaling * ave_pb)
        ri.loc[mask] = np.nan
        reason_ri.loc[mask] = "adjacent side"

        # Write back filtered values + reasons
        df["CWT_pith"], df["CWT_bark"], df["CWT_left"], df["CWT_right"] = pi, ba, le, ri
        df["reason_pith"], df["reason_bark"], df["reason_left"], df["reason_right"] = reason_pi, reason_ba, reason_le, reason_ri


def test_filter_cells_with_reasons():
    # --- Generate 20-row example dataset ---
    np.random.seed(42)
    data = {
        "CWT_pith": np.random.uniform(1.0, 5.0, 20),
        "CWT_bark": np.random.uniform(1.0, 5.0, 20),
        "CWT_left": np.random.uniform(0.8, 4.5, 20),
        "CWT_right": np.random.uniform(0.8, 4.5, 20),
    }

    # Introduce extreme outliers
    data["CWT_pith"][5] = 20.0
    data["CWT_bark"][12] = 25.0
    data["CWT_left"][17] = 30.0
    data["CWT_right"][2] = 35.0

    df = pd.DataFrame(data)

    config = {
        "pixels_per_um": 1.0,
        "ll_scaling": 1.5,
        "ul_scaling": 3.0,
        "opp_scaling": 1.5,
        "adj_scaling": 2.5
    }

    processor = CellProcessor(df.copy(), config)

    print("\n--- BEFORE filtering ---")
    print(df)

    processor.filter_cells()
    out = processor.cells_table

    print("\n--- AFTER filtering ---")
    print(out)

    # Check that the extreme outliers were removed
    assert np.isnan(out["CWT_pith"].iloc[5])
    assert np.isnan(out["CWT_bark"].iloc[12])
    assert np.isnan(out["CWT_left"].iloc[17])
    assert np.isnan(out["CWT_right"].iloc[2])

    # Check that reason columns were set for the outliers
    assert out["reason_pith"].iloc[5] == "hard limit"
    assert out["reason_bark"].iloc[12] == "hard limit"
    assert out["reason_left"].iloc[17] == "hard limit"
    assert out["reason_right"].iloc[2] == "hard limit"
