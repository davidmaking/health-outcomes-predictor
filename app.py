import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#streamlit run app.py

st.set_page_config(page_title="Walkability & Health", layout="wide")

from pathlib import Path

DATA_PATH = Path(__file__).parent / 'merged_data.csv'
FEATURES = ['walkability_mean', 'lapop1share', 'no_vehicle_ratio', 'MHI', 'black_pct', 'hispanic_pct']
TARGET = 'diabetes_pct'

REGION_MAP = {
    '09': 'Northeast', '23': 'Northeast', '25': 'Northeast', '33': 'Northeast',
    '44': 'Northeast', '50': 'Northeast', '34': 'Northeast', '36': 'Northeast', '42': 'Northeast',
    '17': 'Midwest', '18': 'Midwest', '26': 'Midwest', '39': 'Midwest',
    '55': 'Midwest', '19': 'Midwest', '20': 'Midwest', '27': 'Midwest',
    '29': 'Midwest', '31': 'Midwest', '38': 'Midwest', '46': 'Midwest',
    '10': 'South', '11': 'South', '12': 'South', '13': 'South',
    '24': 'South', '37': 'South', '45': 'South', '51': 'South',
    '54': 'South', '01': 'South', '21': 'South', '28': 'South',
    '47': 'South', '05': 'South', '22': 'South', '40': 'South', '48': 'South',
    '04': 'West', '08': 'West', '16': 'West', '30': 'West',
    '32': 'West', '35': 'West', '49': 'West', '56': 'West',
    '02': 'West', '06': 'West', '15': 'West', '41': 'West', '53': 'West',
}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, dtype={'GEOID_tract': str})
    df['GEOID_tract'] = df['GEOID_tract'].str.zfill(11)
    if 'state_fips' not in df.columns:
        df['state_fips'] = df['GEOID_tract'].str[:2]
    df['region'] = df['state_fips'].map(REGION_MAP).fillna('Other')
    df['income_q'] = pd.qcut(df['MHI'], 4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])
    return df



@st.cache_resource
def train_model():
    df = load_data()
    clean = df[FEATURES + [TARGET]].dropna()
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(clean[FEATURES], clean[TARGET])
    return model


df = load_data()
model = train_model()

st.title("Walkability, Food Access & Health Outcomes")
st.divider()

nat_median = df[TARGET].median()
col_left, col_right = st.columns([1, 1])

with col_left:
    walkability = st.slider(
        "Walkability score", 1.0, 20.0,
        float(df['walkability_mean'].median()), 0.5
    )
    no_vehicle = st.slider(
        "No-vehicle household ratio", 0.0, 0.80,
        float(df['no_vehicle_ratio'].median()), 0.01
    )
    food_access = st.slider(
        "Low food access share", 0.0, 0.80,
        float(df['lapop1share'].median()), 0.01,
    )

    st.markdown("**Controls**")
    mhi = st.slider(
        "Median household income ($)", 20000, 200000,
        int(df['MHI'].median()), 2500, format="$%d"
    )
    b_pct = st.slider(
        "Black population share", 0.0, 1.0,
        float(df['black_pct'].median()), 0.01
    )
    h_pct = st.slider(
        "Hispanic population share", 0.0, 1.0,
        float(df['hispanic_pct'].median()), 0.01
    )

pred = model.predict(pd.DataFrame([{
    'walkability_mean': walkability,
    'lapop1share':      food_access,
    'no_vehicle_ratio': no_vehicle,
    'MHI':              mhi,
    'black_pct':        b_pct,
    'hispanic_pct':     h_pct,
}]))[0]

pct_rank = float((df[TARGET].dropna() < pred).mean() * 100)

with col_right:
    st.markdown("### Predicted diabetes rate")
    st.metric(
        "", f"{pred:.1f}%",
        f"{pred - nat_median:+.1f}pp vs national median ({nat_median:.1f}%)",
        delta_color="inverse"
    )
    st.metric("National percentile", f"{pct_rank:.0f}th")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(df[TARGET].dropna(), bins=80, color='#d8d8d8', density=True, label='All US tracts')
    ax.axvline(pred, color='crimson', linewidth=2.5, label=f'Prediction: {pred:.1f}%')
    ax.axvline(
        nat_median, color='steelblue', linewidth=1.5, linestyle='--',
        label=f'National median: {nat_median:.1f}%'
    )
    ax.set_xlabel('Diabetes Rate (%)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
