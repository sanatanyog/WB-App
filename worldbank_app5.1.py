import wbgapi as wb
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import urllib.parse

# --- Simple Indicator Definitions ---
INDICATORS = {
    '1':  ('GDP',             'NY.GDP.MKTP.CD'),
    '2':  ('GDP per Capita',  'NY.GDP.PCAP.CD'),
    '3':  ('GDP (PPP)',       'NY.GDP.MKTP.PP.CD'),
    '4':  ('Per Capita (PPP)','NY.GDP.PCAP.PP.CD'),
    '5':  ('Inflation',       'FP.CPI.TOTL.ZG'),
    '6':  ('Pop. Growth',     'SP.POP.GROW'),
    '7':  ('Debt % of GDP',   'GC.DOD.TOTL.GD.ZS'),
    '8':  ('Poverty %',       'POVERTY_AUTO'),
    '9':  ('Unemployment %',  'SL.UEM.TOTL.ZS'),
    '10': ('Corruption',      'CC.EST')
}

# --- Helper Functions ---
@st.cache_data
def get_iso3_codes(countries):
    mapping = {
        c['value'].lower(): c['id']
        for c in wb.economy.list()
        if len(c['id']) == 3
    }
    return [mapping[name.lower()] for name in countries if name.lower() in mapping]

@st.cache_data
def detect_poverty_index_for_country(code):
    for name, ind in [
        ('Pov $2.15/day', 'SI.POV.DDAY'),
        ('Pov $4.20/day', 'SI.POV.LMIC'),
        ('Pov national',  'SI.POV.NAHC'),
        ('Gini Index',    'SI.POV.GINI')
    ]:
        df = wb.data.DataFrame(ind, [code]).transpose()
        df.index = df.index.map(lambda y: int(str(y).replace('YR','')))
        if not df.dropna(how='all').empty:
            return name, df.round(2).dropna(how='all')
    return None, None

@st.cache_data
def fetch_series(code, country_codes):
    df = wb.data.DataFrame(code, country_codes).transpose()
    df.index = df.index.map(lambda y: int(str(y).replace('YR','')))
    return df.round(2).dropna(how='all')

@st.cache_data
def fetch_wb_data(ind_list, country_codes, country_names):
    data = {}
    for label, code in ind_list:
        if code == 'POVERTY_AUTO':
            for cc, name in zip(country_codes, country_names):
                pname, pdf = detect_poverty_index_for_country(cc)
                if pdf is not None:
                    data[f"{pname} ({name})"] = pdf.rename(columns={cc: name})
        else:
            df = fetch_series(code, country_codes)
            if code == 'NY.GDP.MKTP.CD':
                df = df / 1e9  # GDP in billions
            df = df.rename(columns={cc: nm for cc, nm in zip(country_codes, country_names)})
            for nm in df.columns:
                data[f"{label} ({nm})"] = df[[nm]]
    return data

def make_search_link(country, year, indicator):
    query = f"{country} {year} {indicator} context"
    return f"https://www.google.com/search?q={urllib.parse.quote(query)}"

# --- Streamlit App ---
st.set_page_config(page_title="EconEasy", layout="wide")
st.title("EconEasy: Any Country, Any Story")

# 1️⃣ Country Selection
all_countries = [c['value'] for c in wb.economy.list() if len(c['id']) == 3]
selected_countries = st.multiselect("Select up to 5 countries:", all_countries, max_selections=5)
if not selected_countries:
    st.info("Please select at least one country.")
    st.stop()

country_codes = get_iso3_codes(selected_countries)

# 2️⃣ Indicator Selection
ind_keys = list(INDICATORS.keys())
selected
