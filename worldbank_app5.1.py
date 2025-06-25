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

@st.cache_data
def get_iso3_codes(countries):
    mapping = {c['value'].lower(): c['id'] 
               for c in wb.economy.list() if len(c['id']) == 3}
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
        if not df.dropna(how='all').empty:
            df.index = df.index.map(lambda y: int(str(y).replace('YR','')))
            return name, df.round(2).dropna(how='all')
    return None, None

@st.cache_data
def fetch_series(code, country_codes):
    df = wb.data.DataFrame(code, country_codes).transpose()
    df.index = df.index.map(lambda y: int(str(y).replace('YR','')))
    return df.round(2).dropna(how='all')

@st.cache_data
def fetch_wb_data(ind_codes, country_codes, country_names):
    data = {}
    for label, code in ind_codes:
        if code == 'POVERTY_AUTO':
            for cc, name in zip(country_codes, country_names):
                nm, df = detect_poverty_index_for_country(cc)
                if df is not None:
                    data[f"{nm} ({name})"] = df.rename(columns={cc: name})
        else:
            df = fetch_series(code, country_codes)
            # Scale GDP to billions
            if code == 'NY.GDP.MKTP.CD':
                df = df / 1e9
            # Rename columns with country names
            df = df.rename(columns={cc: name for cc, name in zip(country_codes, country_names)})
            for name_col in df.columns:
                data[f"{label} ({name_col})"] = df[[name_col]]
    return data

def make_search_link(country, year, ind_label):
    q = f"{country} {year} {ind_label} context"
    return f"https://www.google.com/search?q={urllib.parse.quote(q)}"

# --- Streamlit App ---
st.set_page_config(page_title="EconEasy", layout="wide")
st.title("EconEasy: Any Country, Any Story")

# 1️⃣ Countries
countries = [c['value'] for c in wb.economy.list() if len(c['id']) == 3]
selected = st.multiselect("Select up to 5 countries:", countries, max_selections=5)
if not selected:
    st.info("Choose at least one country.")
    st.stop()
codes = get_iso3_codes(selected)

# 2️⃣ Indicators
keys = list(INDICATORS.keys())
chosen = st.multiselect("Select indicators:", keys,
                        format_func=lambda k: INDICATORS[k][0])
if not chosen:
    st.warning("Choose at least one indicator.")
    st.stop()
ind_list = [INDICATORS[k] for k in chosen]

# Fetch data
data_dict = fetch_wb_data(ind_list, codes, selected)

# Combine into DataFrame
years = sorted({y for df in data_dict.values() for y in df.index})
df_out = pd.DataFrame({'Year': years})
for name, df in data_dict.items():
    key = name.replace(' ', '_').replace('%','').replace('(','').replace(')','')
    df_out = df_out.merge(df.rename(columns={df.columns[0]: key}),
                          left_on='Year', right_index=True, how='left')

# 3️⃣ Plot selection
cols = [c for c in df_out.columns if c!='Year']
to_plot = st.multiselect("Columns to plot:", cols, default=cols)
if to_plot:
    title = st.text_input("Chart title:", "Economic Indicators Over Time")
    # Decade filter
    if st.checkbox("Filter by decade"):
        decs = sorted({(y//10)*10 for y in df_out['Year']})
        sel_dec = st.selectbox("Decade:", [f"{d}s" for d in decs])
        d0 = int(sel_dec[:-1])
        df_plot = df_out[(df_out['Year']>=d0)&(df_out['Year']<d0+10)]
    else:
        df_plot = df_out

    # Drill-down
    drill = st.checkbox("Drill down to year")
    year_sel = None
    if drill:
        year_sel = st.selectbox("Year:", df_plot['Year'])

    # Separate axes
    abs_c, rate_c, idx_c = [], [], []
    for c in to_plot:
        if 'GDP' in c:
            abs_c.append(c)
        elif '%' in c:
            rate_c.append(c)
        else:
            idx_c.append(c)

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax2 = ax3 = None
    if rate_c and (abs_c or idx_c):
        ax2 = ax1.twinx()
    if idx_c and (abs_c or rate_c):
        ax3 = ax1.twinx(); ax3.spines['right'].set_position(('outward', 60))

    if drill and year_sel is not None:
        row = df_plot[df_plot['Year']==year_sel].iloc[0]
        for cols_list, ax, marker in [(abs_c, ax1, 'o'),
                                      (rate_c, ax2 or ax1, 's'),
                                      (idx_c,  ax3 or ax2 or ax1, '^')]:
            for c in cols_list:
                val = row[c]
                if pd.notna(val):
                    parts = c.rsplit('_',1)
                    lbl = f"{parts[0]} ({parts[-1]})" if len(parts)>1 else parts[0]
                    ax.scatter(year_sel, val, marker=marker, s=100, label=lbl)
        ax1.set_xticks([year_sel])
    else:
        for cols_list, ax, style in [(abs_c, ax1, '-'),
                                     (rate_c, ax2 or ax1, '--'),
                                     (idx_c,  ax3 or ax2 or ax1, ':')]:
            for c in cols_list:
                parts = c.rsplit('_',1)
                lbl = f"{parts[0]} ({parts[-1]})" if len(parts)>1 else parts[0]
                ax.plot(df_plot['Year'], df_plot[c], style, label=lbl)
        ax1.set_xlabel("Year")

    ax1.set_ylabel("Level")
    if ax2: ax2.set_ylabel("Rate (%)")
    if ax3: ax3.set_ylabel("Index")
    ax1.set_title(title)

    # Legend
    h, l = [], []
    for ax in [ax1, ax2, ax3]:
        if ax:
            hh, ll = ax.get_legend_handles_labels()
            h += hh; l += ll
    ax1.legend(h, l, bbox_to_anchor=(1.02,1), loc='upper_left')
    ax1.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

    # Context links
    if drill and year_sel is not None:
        st.markdown("#### Explore Context")
        for c in to_plot:
            parts = c.rsplit('_',1)
            ind = parts[0]
            cc = parts[-1]
            url = make_search_link(cc, year_sel, ind)
            st.markdown(f"[{cc} {year_sel} {ind} context]({url})")

    # Download
    st.markdown("### Download data")
    st.download_button("Download CSV",
                       df_plot.to_csv(index=False).encode(),
                       file_name="data.csv",
                       mime="text/csv")
else:
    st.warning("Select at least one series to visualize.")
