import wbgapi as wb
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import urllib.parse
from matplotlib.ticker import FuncFormatter

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
    return [
        c['id']
        for c in wb.economy.list()
        if len(c['id']) == 3 and c['value'] in countries
    ]

@st.cache_data
def detect_poverty_index_for_country(code):
    options = [
        ('Pov $2.15/day', 'SI.POV.DDAY'),
        ('Pov $4.20/day', 'SI.POV.LMIC'),
        ('Pov national',  'SI.POV.NAHC'),
        ('Gini Index',    'SI.POV.GINI')
    ]
    for name, ind in options:
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
                df = df / 1e9  # convert to billions
            df = df.rename(columns={cc: nm for cc, nm in zip(country_codes, country_names)})
            for nm in df.columns:
                data[f"{label} ({nm})"] = df[[nm]]
    return data

def choose_scale_and_unit(df, cols):
    if not cols:
        return 1, ''
    max_val = df[cols].abs().max().max()
    factor, unit = 1, ''
    if any('GDP' in c for c in cols):
        if max_val >= 1e3:
            factor, unit = 1e3, 'Trillion USD' if max_val >= 1e6 else 'Billion USD'
    if any('Per Capita' in c and 'PPP' in c for c in cols) and max_val >= 1e3:
        factor, unit = 1e3, 'Thousand'
    return factor, unit

def make_search_link(country, year, indicator):
    q = f"{country} {year} {indicator} context"
    return f"https://www.google.com/search?q={urllib.parse.quote(q)}"

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
keys = list(INDICATORS.keys())
selected_inds = st.multiselect("Select indicators:", keys, format_func=lambda k: INDICATORS[k][0])
if not selected_inds:
    st.warning("Please select at least one indicator.")
    st.stop()
ind_list = [INDICATORS[k] for k in selected_inds]

# Fetch and merge data
data_dict = fetch_wb_data(ind_list, country_codes, selected_countries)
years = sorted({y for df in data_dict.values() for y in df.index})
df_out = pd.DataFrame({'Year': years})
for name, df in data_dict.items():
    col = name.replace(' ', '_').replace('%','').replace('(','').replace(')','')
    df_out = df_out.merge(df.rename(columns={df.columns[0]: col}),
                          left_on='Year', right_index=True, how='left')

# 3️⃣ Plot Selection
available = [c for c in df_out.columns if c != 'Year']
plot_cols = st.multiselect("Columns to plot:", available, default=available)
if not plot_cols:
    st.warning("Select at least one series to plot.")
    st.stop()
title = st.text_input("Chart title:", "Economic Indicators Over Time")

# Plot size selector
size_opt = st.radio("Choose plot size:", ["Small (mobile)", "Medium", "Large (desktop)"], index=1)
figsize = (5,3) if size_opt=="Small (mobile)" else (12,7) if size_opt=="Large (desktop)" else (8,5)

# Decade filter
if st.checkbox("Filter by Decade"):
    decades = sorted({(y//10)*10 for y in df_out['Year']})
    sel_dec = st.selectbox("Decade:", [f"{d}s" for d in decades])
    d0 = int(sel_dec[:-1])
    df_plot = df_out[(df_out['Year']>=d0)&(df_out['Year']<d0+10)].copy()
else:
    df_plot = df_out.copy()

# Drill-down
sel_year = st.selectbox("Drill down to a specific year:", df_plot['Year']) if st.checkbox("Drill down to a specific year") else None

# Determine scaling
scale, y_unit = choose_scale_and_unit(df_plot, plot_cols)

# Separate axes
abs_cols, rate_cols, idx_cols = [], [], []
for c in plot_cols:
    if 'GDP' in c:
        abs_cols.append(c)
    elif '%' in c:
        rate_cols.append(c)
    else:
        idx_cols.append(c)

# Plotting
fig, ax1 = plt.subplots(figsize=figsize, layout='constrained')
ax2 = ax3 = None
if rate_cols and (abs_cols or idx_cols):
    ax2 = ax1.twinx()
if idx_cols and (abs_cols or rate_cols):
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward',60))

def plot_series(ax, cols, style, marker):
    for col in cols:
        series = df_plot[col] / scale
        if sel_year:
            val = series[df_plot['Year']==sel_year].iloc[0]
            ax.scatter(sel_year, val, marker=marker, s=80, label=col)
            ax.set_xticks([sel_year])
        else:
            ax.plot(df_plot['Year'], series, style, label=col)

plot_series(ax1, abs_cols, '-', 'o')
plot_series(ax2 or ax1, rate_cols, '--', 's')
plot_series(ax3 or ax2 or ax1, idx_cols, ':', '^')

# Axis labels
ax1.set_xlabel("Year")
ax1.set_ylabel(y_unit or "Percent (%)")
if ax2: ax2.set_ylabel("Percent (%)")
if ax3: ax3.set_ylabel("Index")
ax1.set_title(title)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
ax1.grid(True, linestyle='--', alpha=0.5)

# Shrink plot area for bottom legend
box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85])

# Bottom horizontal legend
handles, labels = [], []
for ax in (ax1, ax2, ax3):
    if ax:
        h, l = ax.get_legend_handles_labels()
        handles += h; labels += l
unique = dict(zip(labels, handles))
fig.legend(unique.values(), unique.keys(),
           loc='upper center', bbox_to_anchor=(0.5, 0.05),
           ncol=2, frameon=False, fontsize='small')

st.pyplot(fig)

# Context links
if sel_year:
    st.markdown("#### Explore Context")
    for col in plot_cols:
        ind, cc = col.rsplit('_',1)
        url = make_search_link(cc, sel_year, ind)
        st.markdown(f"[{cc} {sel_year} {ind} context]({url})")

# Download CSV
st.markdown("### Download Data")
st.download_button("Download CSV", df_plot.to_csv(index=False).encode(),
                   file_name="data.csv", mime="text/csv")
