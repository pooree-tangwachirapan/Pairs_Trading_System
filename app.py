# app.py
# v1.0 (production-ish): Regression (Beta/Alpha) + Rolling + Correlation (pairwise) + Robust guards

import re
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="Mini-Bloomberg: Quant Stats (Practical)", layout="wide")
st.title("📊 Quantitative Analysis Tool (Practical v1.0)")

# -----------------------------
# Helpers
# -----------------------------
TRADING_DAYS = 252

def parse_tickers(text: str) -> list[str]:
    """
    Accepts comma/space/newline separated tickers.
    Returns unique, uppercased tickers (keeps order).
    """
    if not text:
        return []
    raw = re.split(r"[,\s]+", text.strip())
    tickers = []
    seen = set()
    for t in raw:
        t = t.strip().upper()
        if not t:
            continue
        # basic sanity filter (allow '.' for BRK.B style)
        if not re.match(r"^[A-Z0-9\.\-\^=]{1,15}$", t):
            continue
        if t not in seen:
            tickers.append(t)
            seen.add(t)
    return tickers

@st.cache_data(show_spinner=False)
def download_close_prices(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp, auto_adjust: bool = True) -> pd.DataFrame:
    """
    Downloads close prices (auto_adjust=True recommended: includes splits/dividends effect).
    Returns DataFrame with columns=tickers, index=DatetimeIndex.
    """
    if not tickers:
        return pd.DataFrame()

    # yfinance behavior:
    # - single ticker: columns = OHLCV
    # - multiple tickers: columns MultiIndex (field, ticker)
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
        threads=True,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Extract Close
    try:
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"].copy()
        else:
            close = df["Close"].copy()
            # if single ticker, close is Series
            if isinstance(close, pd.Series):
                close = close.to_frame(name=tickers[0])
    except Exception:
        return pd.DataFrame()

    # Ensure DataFrame & columns match requested tickers order where possible
    if isinstance(close, pd.Series):
        close = close.to_frame()
    # Sometimes yfinance returns columns with weird ordering/missing tickers
    close = close.sort_index()
    return close

def compute_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """
    method: 'simple' => pct_change
            'log'    => log returns
    Keeps NaNs (we do pairwise alignment later).
    """
    if prices is None or prices.empty:
        return pd.DataFrame()
    prices = prices.sort_index()
    if method == "log":
        rets = np.log(prices).diff()
    else:
        rets = prices.pct_change()
    return rets

def align_pairwise(returns: pd.DataFrame, y_col: str, x_col: str) -> pd.DataFrame:
    """
    Returns a 2-col DataFrame aligned and dropna pairwise.
    """
    if returns is None or returns.empty:
        return pd.DataFrame()
    cols = [c for c in [x_col, y_col] if c in returns.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    out = returns[[x_col, y_col]].dropna(how="any")
    return out

def annualize_alpha(alpha_daily: float) -> float:
    # geometric annualization
    return (1.0 + alpha_daily) ** TRADING_DAYS - 1.0

def rolling_beta_alpha(x: pd.Series, y: pd.Series, window: int) -> pd.DataFrame:
    """
    Rolling beta = cov(y,x)/var(x)
    Rolling alpha (daily) = mean(y) - beta*mean(x)
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

    out = pd.DataFrame({"beta": beta, "alpha_daily": alpha}, index=df.index)
    return out

def safe_metric(col, label, value, fmt=None, help_text=None):
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        col.metric(label, "—", help=help_text)
    else:
        col.metric(label, (fmt.format(value) if fmt else str(value)), help=help_text)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Data Settings")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

st.sidebar.subheader("Return Settings")
return_method = st.sidebar.selectbox("Return Type", ["simple", "log"], index=0)
min_obs = st.sidebar.number_input("Min observations (pairwise)", min_value=30, max_value=1000, value=120, step=10)

st.sidebar.subheader("1) Regression Settings")
target_stock = st.sidebar.text_input("Stock Symbol (Target)", "AAPL").strip().upper()
benchmark_stock = st.sidebar.text_input("Benchmark Symbol (Market)", "SPY").strip().upper()

st.sidebar.subheader("Rolling Settings")
rolling_window = st.sidebar.selectbox("Rolling Window (trading days)", [20, 60, 120, 252], index=1)

st.sidebar.subheader("2) Correlation Settings")
portfolio_text = st.sidebar.text_area(
    "Portfolio Symbols (comma / space separated)",
    "AAPL, MSFT, GOOGL, TSLA, AMD, NVDA, KO, PEP"
)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["📈 Linear Regression (Beta/Alpha)", "🔥 Correlation Matrix"])

# =====================================================
# TAB 1: Regression
# =====================================================
with tab1:
    st.subheader(f"Regression Analysis: {target_stock} vs {benchmark_stock}")

    run = st.button("Run Regression", type="primary")

    if run:
        tickers = parse_tickers(f"{target_stock},{benchmark_stock}")
        if len(tickers) < 2:
            st.error("กรุณาใส่ ticker ให้ครบทั้ง Target และ Benchmark")
            st.stop()

        with st.spinner("Downloading price data..."):
            prices = download_close_prices(tickers, pd.to_datetime(start_date), pd.to_datetime(end_date), auto_adjust=True)

        if prices.empty or prices.shape[0] < min_obs:
            st.error("ดึงข้อมูลราคาไม่ได้ หรือข้อมูลน้อยเกินไป (ลองเช็ค ticker / เปลี่ยนช่วงเวลา)")
            st.stop()

        rets = compute_returns(prices, method=return_method)

        pair = align_pairwise(rets, y_col=target_stock, x_col=benchmark_stock)
        if pair.empty or len(pair) < min_obs:
            st.error(
                f"ข้อมูล returns ที่ซ้อนกัน (pairwise) น้อยเกินไป: ได้ {len(pair)} แถว (ต้อง >= {min_obs})\n"
                "ลองขยายช่วงเวลา หรือเปลี่ยน ticker"
            )
            st.stop()

        x = pair[benchmark_stock]
        y = pair[target_stock]

        # Regression (scipy)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r2 = r_value ** 2

        # Practical interpretations
        alpha_daily = intercept
        alpha_annual = annualize_alpha(alpha_daily)
        beta = slope

        # Stats Box
        c1, c2, c3, c4 = st.columns(4)
        safe_metric(c1, "Beta (slope)", beta, "{:.4f}", "ความไวของหุ้นต่อ benchmark (ตัวคูณ)")
        safe_metric(c2, "Alpha (daily intercept)", alpha_daily, "{:.6f}", "ผลตอบแทนต่อวันเมื่อ benchmark = 0 (ไม่ใช่ CAPM alpha แบบหัก rf)")
        safe_metric(c3, "Alpha (annualized)", alpha_annual, "{:.2%}", "แปลงจาก daily alpha เป็นรายปีแบบทบต้น (ประมาณการ)")
        safe_metric(c4, "R²", r2, "{:.4f}", "สัดส่วนความแปรปรวนของหุ้นที่อธิบายได้ด้วย benchmark")

        # Scatter + manual regression line (to guarantee consistency)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Daily returns"))

        # Regression line: y = a + b x
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = intercept + slope * x_line
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Regression line"))

        fig.update_layout(
            title=f"{target_stock} vs {benchmark_stock} ({return_method} returns) — OLS",
            xaxis_title=f"{benchmark_stock} returns",
            yaxis_title=f"{target_stock} returns",
            height=520
        )
        st.plotly_chart(fig, use_container_width=True)

        # Interpretation box (unit-correct)
        st.info(
            f"""
**อ่านค่าแบบใช้งานจริง**
- **Beta = {beta:.2f}**: ถ้า {benchmark_stock} เปลี่ยน **+1% (0.01)** คาดว่า {target_stock} จะเปลี่ยนโดยเฉลี่ยประมาณ **{(beta*0.01)*100:.2f}%** (ยังไม่รวม alpha)
- **Alpha (daily) = {alpha_daily:.6f}**: ค่าคงที่รายวันของโมเดล (ไม่ได้หัก risk-free)  
- **R² = {r2:.2f}**: โมเดลอธิบายการแกว่งของ {target_stock} ได้ประมาณ **{r2*100:.1f}%** จาก {benchmark_stock}
- **p-value(beta) = {p_value:.4g}**, **std_err(beta) = {std_err:.4g}** (เป็น OLS แบบพื้นฐาน)
            """.strip()
        )

        # Rolling beta/alpha (stability check)
        st.markdown(f"### Rolling Beta/Alpha (window = {rolling_window} วันทำการ)")
        roll = rolling_beta_alpha(x, y, window=int(rolling_window))
        if roll.empty or roll.dropna().empty:
            st.warning("ทำ rolling ไม่ได้ (ข้อมูลไม่พอหลัง align) — ลองขยายช่วงเวลา")
        else:
            roll_plot = roll.copy()
            roll_plot["alpha_annualized"] = (1 + roll_plot["alpha_daily"]) ** TRADING_DAYS - 1

            fig_beta = go.Figure()
            fig_beta.add_trace(go.Scatter(x=roll_plot.index, y=roll_plot["beta"], mode="lines", name="Rolling Beta"))
            fig_beta.update_layout(title="Rolling Beta", xaxis_title="Date", yaxis_title="Beta", height=350)
            st.plotly_chart(fig_beta, use_container_width=True)

            fig_alpha = go.Figure()
            fig_alpha.add_trace(go.Scatter(x=roll_plot.index, y=roll_plot["alpha_annualized"], mode="lines", name="Rolling Alpha (annualized)"))
            fig_alpha.update_layout(title="Rolling Alpha (annualized)", xaxis_title="Date", yaxis_title="Alpha (annualized)", height=350)
            st.plotly_chart(fig_alpha, use_container_width=True)

        # Download aligned dataset for audit
        st.markdown("### Download")
        out = pair.copy()
        out.columns = [f"{benchmark_stock}_ret", f"{target_stock}_ret"]
        st.download_button(
            "Download aligned returns (CSV)",
            data=out.to_csv(index=True).encode("utf-8"),
            file_name=f"aligned_returns_{target_stock}_vs_{benchmark_stock}.csv",
            mime="text/csv",
        )

# =====================================================
# TAB 2: Correlation
# =====================================================
with tab2:
    st.subheader("Asset Correlation Heatmap (pairwise, min_periods guard)")

    gen = st.button("Generate Matrix", type="primary")

    if gen:
        tickers_list = parse_tickers(portfolio_text)
        if len(tickers_list) < 2:
            st.error("กรุณาใส่ ticker อย่างน้อย 2 ตัว")
            st.stop()

        with st.spinner("Downloading price data..."):
            prices = download_close_prices(tickers_list, pd.to_datetime(start_date), pd.to_datetime(end_date), auto_adjust=True)

        if prices.empty:
            st.error("ดึงข้อมูลราคาไม่ได้ (ลองเช็ค ticker / เปลี่ยนช่วงเวลา)")
            st.stop()

        rets = compute_returns(prices, method=return_method)

        # Pairwise correlation, but require enough overlap
        corr_min_periods = st.slider("min_periods (overlap required)", min_value=20, max_value=252, value=60, step=10)
        corr = rets.corr(min_periods=int(corr_min_periods))

        if corr.isna().all().all():
            st.error("Correlation ทั้งหมดเป็น NaN (ข้อมูลซ้อนกันไม่พอ) — ลองขยายช่วงเวลา หรือ ลด min_periods")
            st.stop()

        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            zmin=-1,
            zmax=1,
            title=f"Correlation Matrix ({return_method} returns, min_periods={corr_min_periods})",
        )
        fig_corr.update_layout(height=720)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("### Data Quality Check")
        # show counts of available returns per ticker (helps detect missing histories)
        counts = rets.notna().sum().sort_values(ascending=False).rename("non-NaN return obs")
        st.dataframe(counts.to_frame(), use_container_width=True)

        st.download_button(
            "Download correlation matrix (CSV)",
            data=corr.to_csv(index=True).encode("utf-8"),
            file_name="correlation_matrix.csv",
            mime="text/csv",
        )

# -----------------------------
# Footer notes
# -----------------------------
st.caption(
    "Notes: ใช้ auto_adjust=True (ใกล้เคียง Adj Close) เพื่อรวมผล dividend/split. "
    "Regression/Correlation ทำบน returns ไม่ใช่ราคา. Rolling ช่วยดูความสัมพันธ์เปลี่ยนตาม regime."
)
