import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="AIR WATCH CHENNAI", page_icon="🌫️", layout="wide")
st.title("🌫️ AIR WATCH CHENNAI")
st.subheader("Air Quality Forecasting and Pollution Source Analysis")
st.warning("Demonstration dataset is SYNTHETIC. Source proxy columns are simulated and are not actual source-apportionment measurements.")
@st.cache_data
def demo():
    return pd.read_csv(Path(__file__).parent/"data"/"synthetic_chennai_air_quality.csv",parse_dates=["Date"])
uploaded=st.sidebar.file_uploader("Upload air-quality CSV",type=["csv"])
if uploaded:
    raw=pd.read_csv(uploaded)
    try: raw["Date"]=pd.to_datetime(raw["Date"])
    except: st.error("Date column must be parseable.");st.stop()
    data_label="Uploaded data"
else:
    raw=demo();data_label="Synthetic demo data"
required=["Date","PM2.5"]
if not set(required).issubset(raw.columns): st.error("CSV needs Date and PM2.5 columns.");st.stop()
raw=raw.dropna(subset=["Date","PM2.5"]).sort_values("Date")
raw["PM2.5"]=pd.to_numeric(raw["PM2.5"],errors="coerce")
raw=raw.dropna(subset=["PM2.5"])
for col in ["PM10","NO2","SO2","CO","O3"]:
    if col in raw: raw[col]=pd.to_numeric(raw[col],errors="coerce")
start,end=st.sidebar.date_input("Date range",value=(raw.Date.min().date(),raw.Date.max().date()))
if isinstance(start,tuple):start,end=start
data=raw[(raw.Date.dt.date>=start)&(raw.Date.dt.date<=end)].copy()
if data.empty:st.warning("No rows for selected dates.");st.stop()
st.caption(f"Data: {data_label} | {len(data)} rows | {data.Date.min().date()} – {data.Date.max().date()}")
st.metric("Average PM2.5",f"{data['PM2.5'].mean():.1f} µg/m³")
st.metric("Maximum PM2.5",f"{data['PM2.5'].max():.1f} µg/m³")
st.subheader("Module 1 — Air Quality Forecasting and Data Analysis")
if "PM10" in data:
    pollutant=st.selectbox("Pollutant for trend analysis", [c for c in ["PM2.5","PM10","NO2","SO2","CO","O3"] if c in data])
else: pollutant="PM2.5"
fig=px.line(data,x="Date",y=pollutant,title=f"{pollutant} trend")
st.plotly_chart(fig,use_container_width=True)
data["Month"]=data.Date.dt.month
monthly=data.groupby("Month")[pollutant].mean().reset_index()
st.plotly_chart(px.bar(monthly,x="Month",y=pollutant,title=f"Monthly average {pollutant}"),use_container_width=True)
st.subheader("Simple baseline forecasting (illustrative)")
# chronological split and 7-day moving average baseline forecast for held-out data
series=data[pollutant].dropna().reset_index(drop=True)
if len(series)>=30:
    split=int(len(series)*0.8)
    train,test=series.iloc[:split],series.iloc[split:]
    preds=[]
    history=list(train)
    for actual in test:
        preds.append(float(np.mean(history[-7:])) if len(history)>=7 else float(np.mean(history)))
        history.append(float(actual))
    y=np.array(test);p=np.array(preds)
    mae=float(np.mean(np.abs(y-p)));rmse=float(np.sqrt(np.mean((y-p)**2)))
    c1,c2=st.columns(2);c1.metric("Baseline MAE",f"{mae:.2f}");c2.metric("Baseline RMSE",f"{rmse:.2f}")
    st.caption("Chronological 80/20 split; 7-day rolling-average baseline. These metrics describe this selected data only.")
    forecast_df=pd.DataFrame({"Observed":y,"Baseline prediction":p})
    st.line_chart(forecast_df)
else: st.info("At least 30 records are recommended to evaluate the baseline.")
st.subheader("Module 2 — Pollution Source Analysis and Visualization")
proxy_cols=[c for c in ["Traffic_Proxy","Industry_Proxy","Dust_Proxy"] if c in data.columns]
if proxy_cols:
    proxy=data[proxy_cols].mean().reset_index();proxy.columns=["Proxy","Mean (synthetic units)"]
    st.plotly_chart(px.bar(proxy,x="Proxy",y="Mean (synthetic units)",title="Illustrative source proxy comparison"),use_container_width=True)
    st.caption("Proxy values are simulated demonstration indicators, not validated source contributions or causal estimates.")
if all(c in data for c in ["PM2.5","PM10","NO2","SO2","CO","O3"]):
    corr=data[["PM2.5","PM10","NO2","SO2","CO","O3"]].corr()
    st.plotly_chart(px.imshow(corr,text_auto=".2f",aspect="auto",title="Pollutant correlation heatmap"),use_container_width=True)
st.subheader("Data preview")
st.dataframe(data.tail(30),use_container_width=True)
st.download_button("Download selected data CSV",data.to_csv(index=False).encode(), "air_watch_selected_data.csv","text/csv")
st.info("This is a student analytics prototype, not a real-time air-quality alert service.")
