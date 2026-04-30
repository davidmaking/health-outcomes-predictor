import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#streamlit run app.py

st.set_page_config(page_title="Walkability & Health", layout="wide")

DATA_PATH = '~/Desktop/projects/saas_df/data/merged_data.csv'
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


def geocode(address):
    try:
        r = requests.get(
            'https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress',
            params={
                'address': address,
                'benchmark': 'Public_AR_Current',
                'vintage': 'Census2020_Current',
                'layers': '10',
                'format': 'json'
            },
            timeout=10
        )
        matches = r.json()['result']['addressMatches']
        if matches:
            block = matches[0]['geographies']['Census Blocks'][0]
            return block['STATE'] + block['COUNTY'] + block['TRACT']
    except Exception:
        pass
    return None


def percentile_bar(ax, series, value, label, higher_is_better):
    pct = float((series.dropna() < value).mean() * 100)
    good = (higher_is_better and pct >= 50) or (not higher_is_better and pct <= 50)
    fill = '#4CAF50' if good else '#F44336'
    ax.barh([''], [pct], color=fill, height=0.5)
    ax.barh([''], [100 - pct], left=[pct], color='#eeeeee', height=0.5)
    ax.axvline(50, color='#aaaaaa', linewidth=0.8, linestyle='--')
    val_str = f"${value:,.0f}" if 'Income' in label else f"{value:.1f}"
    ax.set_title(f"{label}\n{val_str}  ·  {pct:.0f}th pct", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(['0', '50', '100'], fontsize=7)
    ax.tick_params(axis='y', length=0)
    ax.set_yticklabels([])


df = load_data()
model = train_model()

st.title("Walkability, Food Access & Health Outcomes")
st.divider()

tab1, tab2 = st.tabs(["Neighborhood Lookup", "What-If Explorer"])


with tab1:
    st.markdown("Enter any US address to pull up that census tract's walkability, food access, and health stats.")
    address = st.text_input(
        "Address",
        placeholder="e.g. 2400 Durant Avenue, Berkeley, CA 94720"
    )

    if address:
        with st.spinner("Looking up census tract..."):
            geoid = geocode(address)

        if geoid:
            geoid = str(geoid).zfill(11)
            match = df[df['GEOID_tract'] == geoid]

            if not match.empty:
                row = match.iloc[0]
                fd_val = row.get('LILATracts_1And10')
                fd = pd.notna(fd_val) and int(fd_val) == 1
                fd_label = 'Yes ⚠️' if fd else ('No ✓' if pd.notna(fd_val) else 'N/A')

                st.markdown(
                    f"**Tract:** `{geoid}` &nbsp;·&nbsp; "
                    f"**Region:** {row['region']} &nbsp;·&nbsp; "
                    f"**Income quartile:** {row['income_q']} &nbsp;·&nbsp; "
                    f"**Food desert:** {fd_label}"
                )
                st.divider()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "Walkability", f"{row['walkability_mean']:.1f} / 20",
                    f"{row['walkability_mean'] - df['walkability_mean'].median():+.1f} vs median"
                )
                c2.metric(
                    "Obesity", f"{row['obesity_pct']:.1f}%",
                    f"{row['obesity_pct'] - df['obesity_pct'].median():+.1f}pp vs median",
                    delta_color="inverse"
                )
                c3.metric(
                    "Diabetes", f"{row['diabetes_pct']:.1f}%",
                    f"{row['diabetes_pct'] - df['diabetes_pct'].median():+.1f}pp vs median",
                    delta_color="inverse"
                )
                c4.metric(
                    "Median income", f"${row['MHI']:,.0f}",
                    f"${row['MHI'] - df['MHI'].median():+,.0f} vs median"
                )

                st.subheader("National percentile")
                fig, axes = plt.subplots(1, 4, figsize=(14, 1.8))
                for ax, (col, label, hib) in zip(axes, [
                    ('walkability_mean', 'Walkability', True),
                    ('obesity_pct',      'Obesity %',   False),
                    ('diabetes_pct',     'Diabetes %',  False),
                    ('MHI',             'Median Income', True),
                ]):
                    percentile_bar(ax, df[col], row[col], label, hib)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            else:
                st.warning(f"Tract `{geoid}` was geocoded but not found in the dataset. Try a different address.")
        else:
            st.error("Address not recognized. Try including city and state — e.g. `123 Main St, Chicago IL`.")

with tab2:
    st.markdown("Adjust tract characteristics below to see the model's predicted diabetes rate update in real time.")

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
            "Low food access share", 0.0, 1.0,
            float(df['lapop1share'].median()), 0.01,
            help="Share of tract population with low grocery access within 1 mile"
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

        st.caption("Gradient Boosting · Test R² = 0.70 · MAE = 1.51 pp")
