"""Publication-quality Plotly visualisations — corrected for sparse data."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _build_surface_grid(
    df: pd.DataFrame,
    x_col: str = "moneyness",
    y_col: str = "T",
    z_col: str = "bs_iv",
    n_x: int = 80,
    n_y: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build interpolation grid. Uses LINEAR (not cubic) to avoid Runge oscillations."""
    x = df[x_col].values
    y = df[y_col].values
    z = df[z_col].values

    xi = np.linspace(x.min(), x.max(), n_x)
    yi = np.linspace(y.min(), y.max(), n_y)
    Xi, Yi = np.meshgrid(xi, yi)

    # LINEAR interpolation — stable with sparse data.
    # Use fill_value so there are no NaNs in the resulting surface grid.
    Zi = griddata((x, y), z, (Xi, Yi), method="linear", fill_value=0.001)

    # Clip values into a sensible IV range and floor any outside-hull points.
    Zi = np.clip(Zi, 0.001, 1.0)

    return Xi, Yi, Zi


def plot_surface_3d(
    df: pd.DataFrame,
    title: str = "Implied Volatility Surface",
    save_path: str | None = None,
    width: int = 1000,
    height: int = 750,
) -> go.Figure:
    """3D surface — corrected: linear interpolation, no negative values."""
    Xi, Yi, Zi = _build_surface_grid(df)

    fig = go.Figure(
        data=[
            go.Surface(
                x=Xi,
                y=Yi,
                z=Zi,
                colorscale="Viridis",
                cmin=0,  # hard floor at 0
                colorbar=dict(title="IV", x=0.9),
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
                ),
                hovertemplate="K/S: %{x:.3f}<br>T: %{y:.3f} yr<br>IV: %{z:.2%}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        scene=dict(
            xaxis_title="Moneyness (K / S)",
            yaxis_title="Maturity (years)",
            zaxis_title="Implied Volatility",
            zaxis=dict(range=[0, max(0.25, np.nanmax(Zi) * 1.1)]),  # never show negative z
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        width=width,
        height=height,
        margin=dict(l=0, r=0, b=0, t=50),
        template="plotly_white",
    )

    if save_path:
        try:
            fig.write_image(str(_ensure_dir(save_path)), scale=3, width=width, height=height)
            print(f"  Saved PNG: {save_path}")
        except Exception as e:
            print(f"  PNG export failed (Kaleido/WebGL): {e}")
            print(f"  Displaying interactively instead. Screenshot manually for slides.")
    return fig


def plot_surface_2d_contour(
    df: pd.DataFrame,
    title: str = "Implied Volatility Contours",
    save_path: str | None = None,
    width: int = 900,
    height: int = 700,
) -> go.Figure:
    """2D contour — corrected: linear interpolation, zmin=0."""
    Xi, Yi, Zi = _build_surface_grid(df, n_x=100, n_y=60)

    fig = go.Figure(
        data=[
            go.Contour(
                x=Xi[0, :],
                y=Yi[:, 0],
                z=Zi,
                colorscale="Viridis",
                zmin=0,  # hard floor
                colorbar=dict(title="IV", thickness=20),
                contours=dict(coloring="heatmap", showlabels=True, labelfont=dict(size=10)),
                hovertemplate="K/S: %{x:.3f}<br>T: %{y:.3f} yr<br>IV: %{z:.2%}<extra></extra>",
            ),
            go.Scatter(
                x=df["moneyness"],
                y=df["T"],
                mode="markers",
                marker=dict(size=5, color=df["bs_iv"], colorscale="Viridis", cmin=df["bs_iv"].min(), cmax=df["bs_iv"].max(), showscale=False),
                name="Market quotes",
                hovertemplate="K/S: %{x:.3f}<br>T: %{y:.3f}<br>IV: %{marker.color:.2%}<extra></extra>",
            ),
        ]
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_title="Moneyness (K / S)",
        yaxis_title="Maturity (years)",
        width=width,
        height=height,
        template="plotly_white",
        margin=dict(l=60, r=60, t=60, b=60),
    )

    if save_path:
        fig.write_image(str(_ensure_dir(save_path)), scale=3, width=width, height=height)
    return fig


def plot_skew_slices(
    df: pd.DataFrame,
    flat_vol: float | None = None,
    title: str = "Volatility Skew / Smile by Maturity",
    save_path: str | None = None,
    width: int = 1000,
    height: int = 700,
) -> go.Figure:
    """Skew slices — no interpolation, just raw data connected by lines."""
    T_vals = np.sort(df["T"].unique())
    selected = T_vals[:: max(1, len(T_vals) // 5)][:5]

    fig = go.Figure()
    cmap = ["#440154", "#31688e", "#35b779", "#fde725", "#f16913"]

    for i, T in enumerate(selected):
        sl = df[(df["option_type"] == "call") & (np.abs(df["T"] - T) < 0.001)].sort_values("moneyness")
        if len(sl) < 2:
            continue
        color = cmap[i % len(cmap)]
        fig.add_trace(
            go.Scatter(
                x=sl["moneyness"],
                y=sl["bs_iv"],
                mode="lines+markers",
                name=f"T = {T:.3f} yr",
                marker=dict(size=7, color=color),
                line=dict(width=2.5, color=color),
            )
        )

    fig.add_vline(x=1.0, line_dash="dash", line_color="black", annotation_text="ATM", annotation_position="top")

    if flat_vol is not None:
        fig.add_hline(
            y=flat_vol,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Flat vol = {flat_vol:.2%}",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_title="Moneyness (K / S)",
        yaxis_title="Implied Volatility",
        width=width,
        height=height,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=60, r=60, t=80, b=80),
    )

    if save_path:
        fig.write_image(str(_ensure_dir(save_path)), scale=3, width=width, height=height)
    return fig


def plot_term_structure(
    df: pd.DataFrame,
    title: str = "Term Structure of Implied Volatility",
    save_path: str | None = None,
    width: int = 1000,
    height: int = 700,
) -> go.Figure:
    """Term structure — raw data only, no interpolation."""
    T_vals = np.sort(df["T"].unique())
    fig = go.Figure()

    # ATM calls
    atm_data = []
    for T in T_vals:
        sl = df[(df["option_type"] == "call") & (np.abs(df["T"] - T) < 0.001)]
        atm_sl = sl[np.abs(sl["moneyness"] - 1.0) < 0.02]
        if len(atm_sl) > 0:
            atm_data.append((T, atm_sl["bs_iv"].mean()))
    if atm_data:
        T_atm, iv_atm = zip(*atm_data)
        fig.add_trace(
            go.Scatter(
                x=T_atm, y=iv_atm, mode="lines+markers", name="ATM Calls",
                marker=dict(size=10, symbol="square", color="#31688e"),
                line=dict(width=3, color="#31688e"),
            )
        )

    # ATM puts
    atm_data = []
    for T in T_vals:
        sl = df[(df["option_type"] == "put") & (np.abs(df["T"] - T) < 0.001)]
        atm_sl = sl[np.abs(sl["moneyness"] - 1.0) < 0.02]
        if len(atm_sl) > 0:
            atm_data.append((T, atm_sl["bs_iv"].mean()))
    if atm_data:
        T_atm, iv_atm = zip(*atm_data)
        fig.add_trace(
            go.Scatter(
                x=T_atm, y=iv_atm, mode="lines+markers", name="ATM Puts",
                marker=dict(size=10, symbol="square", color="#f16913"),
                line=dict(width=3, color="#f16913"),
            )
        )

    # OTM put ~0.95
    otm_data = []
    for T in T_vals:
        sl = df[(df["option_type"] == "put") & (np.abs(df["T"] - T) < 0.001)]
        otm_sl = sl[(sl["moneyness"] > 0.93) & (sl["moneyness"] < 0.97)]
        if len(otm_sl) > 0:
            otm_data.append((T, otm_sl["bs_iv"].mean()))
    if otm_data:
        T_otm, iv_otm = zip(*otm_data)
        fig.add_trace(
            go.Scatter(
                x=T_otm, y=iv_otm, mode="lines+markers", name="OTM Put (K/S ~ 0.95)",
                marker=dict(size=10, symbol="triangle-up", color="#d62728"),
                line=dict(width=3, color="#d62728"),
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_title="Maturity T (years)",
        yaxis_title="Implied Volatility",
        width=width,
        height=height,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=60, r=60, t=80, b=80),
    )

    if save_path:
        fig.write_image(str(_ensure_dir(save_path)), scale=3, width=width, height=height)
    return fig


def plot_flatvol_deviation(
    df: pd.DataFrame,
    flat_vol: float,
    title: str = "BS Failure: IV Deviation from Flat Vol",
    save_path: str | None = None,
    width: int = 1000,
    height: int = 700,
) -> go.Figure:
    """Scatter of raw deviations — no interpolation."""
    deviations = df["bs_iv"] - flat_vol

    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["moneyness"],
                y=deviations,
                mode="markers",
                marker=dict(
                    size=7,
                    color=df["T"],
                    colorscale="RdBu_r",
                    colorbar=dict(title="Maturity T", thickness=20),
                    showscale=True,
                ),
                hovertemplate="K/S: %{x:.3f}<br>Deviation: %{y:.2%}<br>T: %{marker.color:.3f}<extra></extra>",
            )
        ]
    )

    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.add_vline(x=1.0, line_dash="dot", line_color="black", annotation_text="ATM", annotation_position="top")

    fig.add_annotation(
        x=0.94, y=float(deviations.quantile(0.90)),
        text="OTM puts: IV too high",
        showarrow=False, font=dict(size=12, color="darkred"),
        bgcolor="white", bordercolor="darkred", borderwidth=1,
    )
    fig.add_annotation(
        x=1.06, y=float(deviations.quantile(0.10)),
        text="OTM calls: close to flat",
        showarrow=False, font=dict(size=12, color="darkblue"),
        bgcolor="white", bordercolor="darkblue", borderwidth=1, xanchor="right",
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_title="Moneyness (K / S)",
        yaxis_title="IV Deviation from Flat Vol",
        width=width,
        height=height,
        template="plotly_white",
        margin=dict(l=60, r=100, t=80, b=60),
    )

    if save_path:
        fig.write_image(str(_ensure_dir(save_path)), scale=3, width=width, height=height)
    return fig
