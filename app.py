import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Mini-Bloomberg: Quant Stats", layout="wide")
st.title("📊 Quantitative Analysis Tool")

# 2. Sidebar: Input ข้อมูล
st.sidebar.header("Data Settings")
# เลือกช่วงเวลา
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Input สำหรับ Linear Regression
st.sidebar.subheader("1. Regression Settings")
target_stock = st.sidebar.text_input("Stock Symbol (Target)", "AAPL")
benchmark_stock = st.sidebar.text_input("Benchmark Symbol (Market)", "SPY")

# Input สำหรับ Correlation Matrix
st.sidebar.subheader("2. Correlation Settings")
portfolio_stocks = st.sidebar.text_area("Portfolio Symbols (comma separated)", "AAPL, MSFT, GOOGL, TSLA, AMD, NVDA, KO, PEP")

# 3. ฟังก์ชันดึงข้อมูล
@st.cache_data
def get_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Close']
    returns = data.pct_change().dropna()
    return returns

# --- Main App ---
tab1, tab2 = st.tabs(["📈 Linear Regression (Beta)", "🔥 Correlation Matrix"])

# === TAB 1: LINEAR REGRESSION ===
with tab1:
    st.subheader(f"Regression Analysis: {target_stock} vs {benchmark_stock}")
    
    if st.button("Run Regression"):
        try:
            # ดึงข้อมูล
            df_reg = get_data([target_stock, benchmark_stock], start_date, end_date)
            
            # เตรียมตัวแปร X (Market) และ Y (Stock)
            x = df_reg[benchmark_stock]
            y = df_reg[target_stock]
            
            # คำนวณ Regression ด้วย Scipy
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # แสดงค่าสถิติ (Stats Box)
            col1, col2, col3 = st.columns(3)
            col1.metric("Beta (Slope)", f"{slope:.4f}", help="ความผันผวนเทียบกับตลาด")
            col2.metric("Alpha (Intercept)", f"{intercept:.6f}", help="ผลตอบแทนส่วนเกิน")
            col3.metric("R-Squared", f"{r_value**2:.4f}", help="ความแม่นยำของโมเดล (ยิ่งใกล้ 1 ยิ่งแม่น)")
            
            # สร้างกราฟ Scatter Plot + Regression Line
            fig = px.scatter(df_reg, x=benchmark_stock, y=target_stock, 
                             trendline="ols", # สั่งให้ Plotly วาดเส้น Regression อัตโนมัติ
                             trendline_color_override="red",
                             title=f"Scatter Plot: {target_stock} vs {benchmark_stock}",
                             labels={benchmark_stock: f"Market Returns ({benchmark_stock})", target_stock: f"Stock Returns ({target_stock})"})
            
            st.plotly_chart(fig, use_container_width=True)
            
            # คำอธิบายผลลัพธ์
            st.info(f"""
            **การอ่านค่า:**
            - **Beta = {slope:.2f}:** ถ้า {benchmark_stock} เปลี่ยนแปลง 1%, {target_stock} จะเปลี่ยนแปลงประมาณ {slope:.2f}%
            - **R-Squared = {r_value**2:.2f}:** การเคลื่อนไหวของ {benchmark_stock} มีผลต่อ {target_stock} ประมาณ {(r_value**2)*100:.1f}%
            """)
            
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}. กรุณาตรวจสอบชื่อหุ้น")

# === TAB 2: CORRELATION MATRIX ===
with tab2:
    st.subheader("Asset Correlation Heatmap")
    
    if st.button("Generate Matrix"):
        try:
            # แปลง Input string เป็น list
            tickers_list = [x.strip() for x in portfolio_stocks.split(",")]
            
            # ดึงข้อมูล
            df_corr = get_data(tickers_list, start_date, end_date)
            
            # คำนวณ Correlation Matrix
            corr_matrix = df_corr.corr()
            
            # สร้างกราฟ Heatmap
            fig_corr = px.imshow(corr_matrix, 
                                 text_auto=".2f", # แสดงตัวเลขทศนิยม 2 ตำแหน่ง
                                 aspect="auto",
                                 color_continuous_scale="RdBu_r", # สีแดง-น้ำเงิน (แดง=Correlation สูง)
                                 zmin=-1, zmax=1,
                                 title="Portfolio Correlation Matrix")
            
            st.plotly_chart(fig_corr, use_container_width=True)
            
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}. กรุณาตรวจสอบชื่อหุ้น")