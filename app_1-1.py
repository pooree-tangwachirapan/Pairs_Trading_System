# app.py
# v1.1 (short-term trading): interval support + rolling beta/corr + residual z-score + strong guards

import re
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

TRADING_DAYS = 252

# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="Mini-Bloomberg: Quant Stats (Short-term)", layout="wide")
st.title("📊 Quantitative Analysis Tool (Short-term v1.1)")

# -----------------------------
# Helpers
# -----------------------------
def parse_tickers(text: str) -> list[str]:
    if not text:
        return []
    raw = re.split(r"[,\s]+", text.strip())
    tickers, seen = [], set()
    for t in raw:
        t = t.strip().upper()
        if not t:
            continue
        if not re.match(r"^[A-Z0-9\.\-\^=]{1,15}$", t):
            continue
        if t not in seen:
            tickers.append(t)
            seen.add(t)
    return tickers

def yfinance_safe_download(
    tickers: list[str],
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    interval: str,
    period: str | None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Intraday: prefer period (yfinance limits). Daily+: can use start/end.
    Returns raw yfinance df.
    """
    if not tickers:
        return pd.DataFrame()

    kwargs = dict(
        tickers=tickers,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
        threads=True,
        interval=interval,
    )

    # Intraday guard: start/end sometimes works but frequently fails; period is more reliable
    if period:
        kwargs["period"] = period
    else:
        kwargs["start"] = start
        kwargs["end"] = end

    df = yf.download(**kwargs)
    if df is None or df.empty:
        return pd.DataFrame()
    return df

def extract_close(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"].copy()
        else:
            close = df["Close"].copy()
            if isinstance(close, pd.Series):
                close = close.to_frame(name=tickers[0])
    except Exception:
        return pd.DataFrame()

    if isinstance(close, pd.Series):
        close = close.to_frame()
    close = close.sort_index()
    return close

def compute_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()
    prices = prices.sort_index()
    if method == "log":
        return np.log(prices).diff()
    return prices.pct_change()

def align_pairwise(returns: pd.DataFrame, y_col: str, x_col: str) -> pd.DataFrame:
    if returns is None or returns.empty:
        return pd.DataFrame()
    if x_col not in returns.columns or y_col not in returns.columns:
        return pd.DataFrame()
    return returns[[x_col, y_col]].dropna(how="any")

def rolling_beta_alpha_resid(x: pd.Series, y: pd.Series, window: int) -> pd.DataFrame:
    """
    Rolling beta = cov(y,x)/var(x)
    Rolling alpha = mean(y) - beta*mean(x)
    Residual = y - (alpha + beta*x)
    """
    df = pd.concat([x, y], axis=1).dropna()
    if df.empty:
        return pd.DataFrame()

    x_ = df.iloc[:, 0]
    y_ = df.iloc[:, 1]

    cov = y_.rolling(window).cov(x_)
    var = x_.rolling(window).var()
    beta = cov / var

    mx = x_.rolling(window).mean()
    my = y_.rolling(window).mean()
    alpha = my - beta * mx

    y_hat = alpha + beta * x_
    resid = y_ - y_hat

    out = pd.DataFrame(
        {"beta": beta, "alpha": alpha, "resid": resid},
        index=df.index
    )
    return out

def zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window).mean()
    sd = series.rolling(window).std(ddof=0)
    return (series - mu) / sd

def safe_stop(msg: str):
    st.error(msg)
    st.stop()

# -----------------------------
# Sidebar (Short-term)
# -----------------------------
st.sidebar.header("Data Settings (Short-term)")

interval = st.sidebar.selectbox(
    "Interval",
    ["1d", "1h", "30m", "15m", "5m"],
    index=1
)

# yfinance intraday limits (rough practical defaults)
# 5m/15m: use 5-30 days; 30m/1h: up to 60d; daily: use start/end
period_map = {
    "5m":  "30d",
    "15m": "60d",
    "30m": "60d",
    "1h":  "730d",   # yfinance often allows longer; still safer than start/end
    "1d":  None
}

use_period = period_map.get(interval)
if interval == "1d":
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2024-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
else:
    period = st.sidebar.selectbox(
        "History (period)",
        ["5d", "10d", "30d", "60d", "90d", "180d", "730d"],
        index=["5d","10d","30d","60d","90d","180d","730d"].index(use_period) if use_period in ["5d","10d","30d","60d","90d","180d","730d"] else 2
    )
    start_date = None
    end_date = None

st.sidebar.subheader("Return Settings")
return_method = st.sidebar.selectbox("Return Type", ["simple", "log"], index=0)

st.sidebar.subheader("Regression Pair")
target_stock = st.sidebar.text_input("Target", "AAPL").strip().upper()
benchmark_stock = st.sidebar.text_input("Benchmark", "SPY").strip().upper()

st.sidebar.subheader("Short-term Windows")
beta_window = st.sidebar.selectbox("Rolling Beta Window (bars)", [20, 30, 40, 60, 120], index=0)
z_window = st.sidebar.selectbox("Residual Z-score Window (bars)", [20, 30, 60, 120], index=0)
corr_window = st.sidebar.selectbox("Rolling Corr Window (bars)", [20, 30, 60, 120], index=2)

min_obs = st.sidebar.number_input("Min overlap bars", min_value=30, max_value=2000, value=120, step=10)

st.sidebar.subheader("Correlation Basket")
portfolio_text = st.sidebar.text_area(
    "Portfolio Symbols",
    "AAPL, MSFT, NVDA, AMD, TSLA, GOOGL, META, SPY"
)

tab1, tab2 = st.tabs(["⚡ Short-term Pair (Beta/Z)", "🔥 Correlation (Rolling + Matrix)"])

# -----------------------------
# TAB 1: Pair for short-term trading
# -----------------------------
with tab1:
    st.subheader(f"Pair Monitor: {target_stock} vs {benchmark_stock}  | interval={interval}")

    if st.button("Run / Refresh", type="primary"):
        tickers = parse_tickers(f"{target_stock},{benchmark_stock}")
        if len(tickers) < 2:
            safe_stop("ใส่ ticker ให้ครบ Target และ Benchmark")

        with st.spinner("Downloading price data..."):
            raw = yfinance_safe_download(
                tickers=tickers,
                start=pd.to_datetime(start_date) if start_date is not None else None,
                end=pd.to_datetime(end_date) if end_date is not None else None,
                interval=interval,
                period=period if interval != "1d" else None,
                auto_adjust=True
            )
        prices = extract_close(raw, tickers)
        if prices.empty:
            safe_stop("ดึงข้อมูลราคาไม่ได้ (เช็ค ticker / เปลี่ยน interval หรือ period)")

        rets = compute_returns(prices, method=return_method)
        pair = align_pairwise(rets, y_col=target_stock, x_col=benchmark_stock)

        if pair.empty or len(pair) < min_obs:
            safe_stop(f"overlap bars น้อยเกิน: {len(pair)} (ต้อง ≥ {min_obs}) — ลองเพิ่ม period/ช่วงเวลา หรือเปลี่ยน interval")

        x = pair[benchmark_stock]
        y = pair[target_stock]

        # Snapshot regression (latest overall, not rolling)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r2 = r_value**2

        # Rolling stats
        roll = rolling_beta_alpha_resid(x, y, window=int(beta_window))
        if roll.empty or roll.dropna().empty:
            safe_stop("ข้อมูลไม่พอทำ rolling beta/alpha/residual — ลองเพิ่ม period/ช่วงเวลา")

        roll = roll.dropna()
        roll["resid_z"] = zscore(roll["resid"], window=int(z_window))
        roll["roll_corr"] = x.loc[roll.index].rolling(int(corr_window)).corr(y.loc[roll.index])

        latest = roll.iloc[-1]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Snapshot Beta", f"{slope:.3f}")
        c2.metric("Snapshot R²", f"{r2:.3f}")
        c3.metric("Rolling Beta (latest)", f"{latest['beta']:.3f}")
        c4.metric("Rolling Corr (latest)", f"{latest['roll_corr']:.3f}" if pd.notna(latest["roll_corr"]) else "—")
        c5.metric("Residual Z (latest)", f"{latest['resid_z']:.2f}" if pd.notna(latest["resid_z"]) else "—")

        # Chart 1: Scatter + snapshot line (fast sanity)
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(x=x, y=y, mode="markers", name="returns"))
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = intercept + slope * x_line
        fig_sc.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="snapshot OLS"))
        fig_sc.update_layout(
            title="Scatter (returns) + Snapshot OLS line",
            xaxis_title=f"{benchmark_stock} returns",
            yaxis_title=f"{target_stock} returns",
            height=420
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # Chart 2: Rolling Beta
        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(x=roll.index, y=roll["beta"], mode="lines", name="rolling beta"))
        fig_b.update_layout(title=f"Rolling Beta (window={beta_window} bars)", xaxis_title="Time", yaxis_title="Beta", height=320)
        st.plotly_chart(fig_b, use_container_width=True)

        # Chart 3: Residual Z-score (signal-ish)
        z_th = st.slider("Z threshold (visual)", 0.5, 4.0, 2.0, 0.1)
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(x=roll.index, y=roll["resid_z"], mode="lines", name="residual z"))
        fig_z.add_hline(y=z_th)
        fig_z.add_hline(y=-z_th)
        fig_z.add_hline(y=0.0)
        fig_z.update_layout(
            title=f"Residual Z-score (window={z_window} bars) — mean reversion monitor",
            xaxis_title="Time",
            yaxis_title="Z",
            height=360
        )
        st.plotly_chart(fig_z, use_container_width=True)

        st.info(
            f"""
**แนวทางใช้สำหรับเทรดสั้น (ไม่ใช่คำแนะนำการลงทุน)**
- ดู **Rolling Corr**: ถ้าตกแรง/หลุดใกล้ 0 แปลว่า “คู่ไม่เดินด้วยกัน” ระวังเล่น mean-reversion
- ดู **Residual Z**: |Z| > {z_th:.1f} คือ “หลุดค่าเฉลี่ยหลาย sigma” (บางคนใช้เป็น trigger/เฝ้ารอ)
- ดู **Rolling Beta**: ถ้า beta เปลี่ยน regime → hedge ratio เปลี่ยน (อย่าใช้ค่าเก่า)
            """.strip()
        )

        # Export for audit
        out = pd.DataFrame({
            f"{benchmark_stock}_ret": x.loc[roll.index],
            f"{target_stock}_ret": y.loc[roll.index],
            "beta_roll": roll["beta"],
            "alpha_roll": roll["alpha"],
            "resid": roll["resid"],
            "resid_z": roll["resid_z"],
            "corr_roll": roll["roll_corr"],
        }, index=roll.index)

        st.download_button(
            "Download rolling pair stats (CSV)",
            data=out.to_csv(index=True).encode("utf-8"),
            file_name=f"pair_shortterm_{target_stock}_vs_{benchmark_stock}_{interval}.csv",
            mime="text/csv",
        )

# -----------------------------
# TAB 2: Correlation for short-term
# -----------------------------
with tab2:
    st.subheader(f"Correlation Tools | interval={interval}")

    if st.button("Generate Matrix", type="primary"):
        tickers = parse_tickers(portfolio_text)
        if len(tickers) < 2:
            safe_stop("ใส่ ticker อย่างน้อย 2 ตัว")

        with st.spinner("Downloading price data..."):
            raw = yfinance_safe_download(
                tickers=tickers,
                start=pd.to_datetime(start_date) if start_date is not None else None,
                end=pd.to_datetime(end_date) if end_date is not None else None,
                interval=interval,
                period=period if interval != "1d" else None,
                auto_adjust=True
            )
        prices = extract_close(raw, tickers)
        if prices.empty:
            safe_stop("ดึงข้อมูลราคาไม่ได้ (เช็ค ticker / เปลี่ยน interval หรือ period)")

        rets = compute_returns(prices, method=return_method)

        minp = st.slider("min_periods (matrix)", min_value=10, max_value=300, value=60, step=10)
        corr = rets.corr(min_periods=int(minp))

        if corr.isna().all().all():
            safe_stop("Correlation เป็น NaN หมด (overlap ไม่พอ) — เพิ่ม period/ช่วงเวลา หรือ ลด min_periods")

        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            zmin=-1, zmax=1,
            title=f"Correlation Matrix ({return_method} returns, min_periods={minp})"
        )
        fig_corr.update_layout(height=720)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Rolling correlation vs SPY (quick heat-of-the-moment view)
        st.markdown("### Rolling correlation vs a chosen anchor")
        anchor = st.selectbox("Anchor ticker", options=[t for t in tickers if t in rets.columns], index=min(0, len(tickers)-1))
        roll_w = st.selectbox("Rolling window (bars)", [20, 30, 60, 120], index=2)

        if anchor:
            roll_corr_df = pd.DataFrame(index=rets.index)
            for t in tickers:
                if t == anchor or t not in rets.columns:
                    continue
                roll_corr_df[t] = rets[anchor].rolling(int(roll_w)).corr(rets[t])

            roll_corr_df = roll_corr_df.dropna(how="all")
            if roll_corr_df.empty:
                st.warning("Rolling corr vs anchor ทำไม่ได้ (ข้อมูลไม่พอ)")
            else:
                fig_rc = go.Figure()
                for col in roll_corr_df.columns:
                    fig_rc.add_trace(go.Scatter(x=roll_corr_df.index, y=roll_corr_df[col], mode="lines", name=col))
                fig_rc.update_layout(
                    title=f"Rolling Corr vs {anchor} (window={roll_w} bars)",
                    xaxis_title="Time",
                    yaxis_title="Corr",
                    height=420
                )
                st.plotly_chart(fig_rc, use_container_width=True)

        counts = rets.notna().sum().sort_values(ascending=False).rename("non-NaN return obs")
        st.dataframe(counts.to_frame(), use_container_width=True)

        st.download_button(
            "Download correlation matrix (CSV)",
            data=corr.to_csv(index=True).encode("utf-8"),
            file_name=f"correlation_matrix_{interval}.csv",
            mime="text/csv",
        )

st.caption(
    "Short-term v1.1: เน้น rolling beta/corr + residual z-score เพื่อดูการหลุดความสัมพันธ์และ mean-reversion. "
    "ใช้ auto_adjust=True เพื่อลดปัญหา split/dividend. Intraday แนะนำใช้ period มากกว่า start/end."
)
