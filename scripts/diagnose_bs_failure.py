"""Build BS IV surface, find best flat vol, and diagnose failures.

Usage:
    python scripts/diagnose_bs_failure.py --data data/spy_synthetic_options.csv
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.optimize import minimize_scalar

from volsurf.pricing import black_scholes as bs


def plot_surface_3d(calls_df, save_path):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    x = calls_df["moneyness"].values
    y = calls_df["T"].values
    z = calls_df["impliedVolatility"].values
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method="cubic")
    surf = ax.plot_surface(Xi, Yi, Zi, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.scatter(x, y, z, c="red", s=8, alpha=0.4)
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Maturity (years)")
    ax.set_zlabel("Implied Volatility")
    ax.set_title("BS Implied Volatility Surface", fontweight="bold")
    fig.colorbar(surf, shrink=0.5, aspect=10, label="IV")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_failure_diagnostic(df, flat_vol, save_path):
    fig = plt.figure(figsize=(16, 12))

    # (a) 3D surface
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    calls = df[df["option_type"] == "call"]
    x, y, z = calls["moneyness"].values, calls["T"].values, calls["impliedVolatility"].values
    xi = np.linspace(x.min(), x.max(), 80)
    yi = np.linspace(y.min(), y.max(), 40)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method="cubic")
    surf = ax1.plot_surface(Xi, Yi, Zi, cmap="viridis", edgecolor="none", alpha=0.85)
    ax1.set_title("(a) Implied Volatility Surface", fontweight="bold")
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    # (b) Skew slices
    ax2 = fig.add_subplot(2, 2, 2)
    T_vals = np.sort(df["T"].unique())
    selected = T_vals[:: max(1, len(T_vals) // 5)][:5]
    cmap = plt.get_cmap("viridis")
    for i, T in enumerate(selected):
        sl = df[(df["option_type"] == "call") & (np.abs(df["T"] - T) < 0.001)].sort_values("moneyness")
        if len(sl) < 3:
            continue
        ax2.plot(sl["moneyness"], sl["impliedVolatility"], marker="o", markersize=4,
                 label=f"T={T:.3f}", color=cmap(i / max(len(selected) - 1, 1)), linewidth=2)
    ax2.axvline(1.0, color="black", linestyle="--", alpha=0.5)
    ax2.axhline(flat_vol, color="red", linestyle="--", alpha=0.7, label=f"Flat vol = {flat_vol:.2%}")
    ax2.set_xlabel("Moneyness K/S")
    ax2.set_ylabel("Implied Volatility")
    ax2.set_title("(b) Skew / Smile by Maturity", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0.75, 1.2)

    # (c) Term structure
    ax3 = fig.add_subplot(2, 2, 3)
    for opt, color, label in [("call", "steelblue", "ATM Calls"), ("put", "darkorange", "ATM Puts")]:
        opt_df = df[df["option_type"] == opt]
        atm_data = []
        for T in T_vals:
            sl = opt_df[np.abs(opt_df["T"] - T) < 0.001]
            atm_sl = sl[np.abs(sl["moneyness"] - 1.0) < 0.02]
            if len(atm_sl) > 0:
                atm_data.append((T, atm_sl["impliedVolatility"].mean()))
        if atm_data:
            T_atm, iv_atm = zip(*atm_data)
            ax3.plot(T_atm, iv_atm, marker="s", color=color, label=label, linewidth=2)
    otm_data = []
    for T in T_vals:
        sl = df[(df["option_type"] == "put") & (np.abs(df["T"] - T) < 0.001)]
        otm_sl = sl[(sl["moneyness"] > 0.88) & (sl["moneyness"] < 0.92)]
        if len(otm_sl) > 0:
            otm_data.append((T, otm_sl["impliedVolatility"].mean()))
    if otm_data:
        T_otm, iv_otm = zip(*otm_data)
        ax3.plot(T_otm, iv_otm, marker="^", color="crimson", label="OTM Put (K/S~0.9)", linewidth=2)
    ax3.set_xlabel("Maturity T (years)")
    ax3.set_ylabel("Implied Volatility")
    ax3.set_title("(c) Term Structure", fontweight="bold")
    ax3.legend(fontsize=9)

    # (d) Deviation from flat vol
    ax4 = fig.add_subplot(2, 2, 4)
    deviations = df["impliedVolatility"] - flat_vol
    scatter = ax4.scatter(df["moneyness"], deviations, c=df["T"], cmap="coolwarm", s=15, alpha=0.6)
    ax4.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax4.axvline(1.0, color="black", linestyle=":", alpha=0.3)
    ax4.set_xlabel("Moneyness K/S")
    ax4.set_ylabel("IV Deviation from Flat Vol")
    ax4.set_title("(d) BS Failure: Smile & Skew Deviations", fontweight="bold")
    ax4.annotate("OTM puts: IV too high\n(crash insurance)", xy=(0.85, 0.04), fontsize=9, color="darkred")
    ax4.annotate("OTM calls: close to flat\n(no upside panic)", xy=(1.08, -0.005), fontsize=9, color="darkblue")
    fig.colorbar(scatter, ax=ax4, label="Maturity T")

    plt.suptitle("Black-Scholes: Where the Flat-Vol Assumption Fails", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/spy_synthetic_options.csv")
    parser.add_argument("--output-dir", type=str, default="output")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    spot = df["spot"].iloc[0]
    r = df["risk_free_rate"].iloc[0]
    q = df["dividend_yield"].iloc[0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find best single flat volatility
    def flat_vol_mse(vol):
        sse = 0.0
        for _, row in df.iterrows():
            model = bs.price(row["spot"], row["strike"], row["T"], r, vol, q, row["option_type"])
            sse += (model - row["midPrice"]) ** 2
        return sse / len(df)

    result = minimize_scalar(flat_vol_mse, bounds=(0.05, 0.50), method="bounded")
    flat_vol = result.x

    # Compute price errors
    df["flat_price"] = df.apply(
        lambda r: bs.price(r["spot"], r["strike"], r["T"], r["risk_free_rate"], flat_vol,
                        r["dividend_yield"], r["option_type"]), axis=1)
    df["price_error"] = df["flat_price"] - df["midPrice"]
    df["price_error_pct"] = df["price_error"] / df["midPrice"] * 100

    # Summary
    print("=" * 60)
    print("  BLACK-SCHOLES FLAT-VOL FAILURE ANALYSIS")
    print("=" * 60)
    print(f"  Spot:              {spot:.2f}")
    print(f"  Best flat vol:     {flat_vol:.4f} ({flat_vol:.2%})")
    print(f"  Overall MAPE:      {df['price_error_pct'].abs().mean():.2f}%")
    print(f"  Max abs error:     ${df['price_error'].abs().max():.4f}")
    print("-" * 60)
    print("  KEY FINDING: Flat vol systematically misprices:")
    print(f"    - Deep OTM puts:   {df[df['moneyness']<0.85]['price_error_pct'].abs().mean():.1f}% MAPE")
    print(f"    - OTM puts:        {df[(df['moneyness']>=0.85)&(df['moneyness']<0.95)]['price_error_pct'].abs().mean():.1f}% MAPE")
    print(f"    - ATM:             {df[(df['moneyness']>=0.98)&(df['moneyness']<=1.02)]['price_error_pct'].abs().mean():.1f}% MAPE")
    print("=" * 60)

    # Plots
    plot_surface_3d(df[df["option_type"] == "call"], output_dir / "bs_surface_3d.png")
    plot_failure_diagnostic(df, flat_vol, output_dir / "bs_failure_diagnostic.png")

    # Save processed data
    df.to_csv(output_dir / "bs_analysis.csv", index=False)
    print(f"Analysis saved to: {output_dir}")


if __name__ == "__main__":
    main()