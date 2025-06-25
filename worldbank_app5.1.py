import wbgapi as wb
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import urllib.parse

# --- Indicator Definitions ---
INDICATORS = {
    '1': ('GDP (current US$)', 'NY.GDP.MKTP.CD'),
    '2': ('GDP per capita (current US$)', 'NY.GDP.PCAP.CD'),
    '3': ('GDP, PPP (current international $)', 'NY.GDP.MKTP.PP.CD'),
    '4': ('GDP per capita, PPP (current international $)', 'NY.GDP.PCAP.PP.CD'),
    '5': ('Inflation, consumer prices (annual %)', 'FP.CPI.TOTL.ZG'),
    '6': ('Population growth (annual %)', 'SP.POP.GROW'),
    '7': ('Central government debt, total (% of GDP)', 'GC.DOD.TOTL.GD.ZS'),
    '8': ('Poverty Index (Auto-Detected)', 'POVERTY_AUTO'),
    '9': ('Unemployment Rate (%)', 'SL.UEM.TOTL.ZS'),
    '10': ('Control of Corruption (WGI)', 'CC.EST')
}

SHORT_NAMES = {
    'GDP': 'GDP (current US$)',
    'GDPpc': 'GDP per capita (current US$)',
    'GDP_PPP': 'GDP, PPP (current international $)',
    'GDPpc_PPP': 'GDP per capita, PPP (current international $)',
    'Inflation': 'Inflation, consumer prices (annual %)',
    'PopGrowth': 'Population growth (annual %)',
    'Debt': 'Central government debt, total (% of GDP)',
    'Unemployment Rate (%)': 'Unemployment Rate (%)',
    'Control of Corruption (WGI)': 'Control of Corruption (WGI)'
}

POVERTY_INDICATORS = [
    ('Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)', 'SI.POV.DDAY'),
    ('Poverty headcount ratio at $4.20 a day (2021 PPP) (% of population)', 'SI.POV.LMIC'),
    ('Poverty headcount ratio at national poverty lines (% of population)', 'SI.POV.NAHC'),
    ('Gini index', 'SI.POV.GINI')
]

def get_full_indicator_name(abbr):
    return SHORT_NAMES.get(abbr, abbr)

@st.cache_data
def get_iso3_codes(user_countries):
    all_names = {c['value'].lower(): c['id'] for c in wb.economy.list() if len(c['id']) == 3}
    return [all_names[name.lower()] for name in user_countries if name.lower() in all_names]

@st.cache_data
def detect_poverty_index_for_country(country_code):
    for ind_name, ind_code in POVERTY_INDICATORS:
        df = wb.data.DataFrame(ind_code, [country_code]).transpose()
        if not df.dropna(how='all').empty:
            return (ind_name, ind_code)
    return (None, None)

@st.cache_data
def fetch_unemployment_data(country_codes):
    df = wb.data.DataFrame('SL.UEM.TOTL.ZS', country_codes).transpose()
    df.index = df.index.map(lambda x: int(str(x).replace('YR','')) if str(x).startswith('YR') else int(x))
    return df.round(2).dropna(how='all')

@st.cache_data
def fetch_corruption_data(country_codes):
    df = wb.data.DataFrame('CC.EST', country_codes).transpose()
    df.index = df.index.map(lambda x: int(str(x).replace('YR','')) if str(x).startswith('YR') else int(x))
    return df.round(3).dropna(how='all')

@st.cache_data
def fetch_wb_data(indicator_codes, country_codes, country_names):
    data = {}
    for ind_name, ind_code in indicator_codes:
        # Auto-detect poverty
        if ind_code == 'POVERTY_AUTO':
            for code in country_codes:
                if code == 'IND':
                    # India-specific MPI
                    mpi_df = pd.DataFrame({'IND': [29.17, 14.96]}, index=[2016, 2021])
                    data["India MPI (NITI Aayog)"] = mpi_df
                    # World Bank $2.15 and $4.20 thresholds
                    for label, code2 in [('2.15$/day', 'SI.POV.DDAY'), ('4.20$/day', 'SI.POV.LMIC')]:
                        df2 = wb.data.DataFrame(code2, [code]).transpose()
                        df2.index = df2.index.map(lambda x: int(str(x).replace('YR','')) if str(x).startswith('YR') else int(x))
                        data[f"{label} Poverty (World Bank)"] = df2.round(2).dropna(how='all')
                else:
                    pname, pcode = detect_poverty_index_for_country(code)
                    if pcode:
                        dfp = wb.data.DataFrame(pcode, [code]).transpose()
                        dfp.index = dfp.index.map(lambda x: int(str(x).replace('YR','')) if str(x).startswith('YR') else int(x))
                        data[f"{pname} ({code})"] = dfp.round(2).dropna(how='all')

        # Unemployment
        elif ind_code == 'SL.UEM.TOTL.ZS':
            df = fetch_unemployment_data(country_codes)
            for code, name in zip(country_codes, country_names):
                if code in df.columns:
                    col = df[[code]].rename(columns={code: name})
                    data[f"Unemployment Rate (%) - {name}"] = col

        # Corruption
        elif ind_code == 'CC.EST':
            df = fetch_corruption_data(country_codes)
            for code, name in zip(country_codes, country_names):
                if code in df.columns:
                    col = df[[code]].rename(columns={code: name})
                    data[f"Control of Corruption (WGI) - {name}"] = col

        # All other indicators
        else:
            df = wb.data.DataFrame(ind_code, country_codes).transpose()
            df.index = df.index.map(lambda x: int(str(x).replace('YR','')) if str(x).startswith('YR') else int(x))
            # Scale large GDP series
            if ind_code in ['NY.GDP.MKTP.CD', 'NY.GDP.MKTP.PP.CD']:
                df = df / 1e9
            df = df.round(2).dropna(how='all')
            data[ind_name] = df

    return data

def make_google_search_link(country, year, indicator_abbr):
    full_indicator = get_full_indicator_name(indicator_abbr)
    query = f"{country} {year} {full_indicator} economic context"
    return f"https://www.google.com/search?q={urllib.parse.quote(query)}"

# --- Streamlit UI ---
st.set_page_config(page_title="EconEasy: World Bank CXO Dashboard", layout="wide")
st.title("EconEasy: World Bank Data for Executives")

# 1️⃣ Country Selection
st.markdown("#### Select up to 5 countries")
all_countries = [c['value'] for c in wb.economy.list() if len(c['id']) == 3]
selected_countries = st.multiselect("Countries:", all_countries, max_selections=5)

if not selected_countries:
    st.info("Please select at least one country to proceed.")
    st.stop()

country_codes = get_iso3_codes(selected_countries)

# 2️⃣ Indicator Selection
st.markdown("#### Choose Indicators")
indicator_keys = list(INDICATORS.keys())
selected_inds = st.multiselect(
    "Indicators",
    indicator_keys,
    format_func=lambda k: INDICATORS[k][0]
)

if not selected_inds:
    st.warning("Select at least one indicator.")
    st.stop()

indicator_codes = [INDICATORS[k] for k in selected_inds]
data_dict = fetch_wb_data(indicator_codes, country_codes, selected_countries)

# Build combined DataFrame
all_years = sorted({yr for df in data_dict.values() for yr in df.index})
output = pd.DataFrame({'YEAR': all_years})
for name, df in data_dict.items():
    col_key = name.replace(' ', '_').replace('%','').replace('(','').replace(')','').replace('-','_')
    output = output.merge(df.rename(columns={df.columns[0]: col_key}),
                          left_on='YEAR', right_index=True, how='left')

# 3️⃣ Columns to Plot
st.markdown("#### Select Data Series to Plot")
available_cols = [c for c in output.columns if c != 'YEAR']
plot_cols = st.multiselect("Plot Columns:", available_cols, default=available_cols)

if plot_cols:
    title = st.text_input("Chart Title", "Economic Indicators Over Time")
    # 4️⃣ Decade Filter
    decades = sorted({(y // 10)*10 for y in output['YEAR']})
    if st.checkbox("Filter by Decade"):
        decade = st.selectbox("Decade:", [f"{d}s" for d in decades])
        d0 = int(decade[:-1])
        output = output[(output['YEAR'] >= d0) & (output['YEAR'] < d0+10)]
    # 5️⃣ Drill-Down
    drill = st.checkbox("Drill Down to Specific Year")
    sel_year = None
    if drill:
        sel_year = st.selectbox("Year:", output['YEAR'].tolist())

    # Separate axes
    abs_cols, usd_cols, perc_cols = [], [], []
    for col in plot_cols:
        if "GDP" in col and "per_capita" not in col:
            abs_cols.append(col)
        elif "per_capita" in col or "Unemployment" in col:
            usd_cols.append(col)
        else:
            perc_cols.append(col)

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax3 = None
    if usd_cols and (abs_cols or perc_cols):
        ax2 = ax1.twinx()
    if perc_cols and (abs_cols or usd_cols):
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))

    if drill and sel_year is not None:
        row = output[output['YEAR'] == sel_year].iloc[0]
        for cols, ax, marker in [
            (abs_cols, ax1, 'o'),
            (usd_cols, ax2 or ax1, 's'),
            (perc_cols, ax3 or ax2 or ax1, '^')
        ]:
            for c in cols:
                v = row[c]
                if pd.notna(v):
                    ax.scatter(sel_year, v, s=100, marker=marker, label=f"{c}: {v}")
        ax1.set_xticks([sel_year])
    else:
        for cols, ax, style in [
            (abs_cols, ax1, '-'),
            (usd_cols, ax2 or ax1, '--'),
            (perc_cols, ax3 or ax2 or ax1, ':')
        ]:
            for c in cols:
                ax.plot(output['YEAR'], output[c], style, label=c)
        ax1.set_xlabel("Year")

    ax1.set_ylabel("Absolute / Level")
    if ax2: ax2.set_ylabel("Per Capita / Rate")
    if ax3: ax3.set_ylabel("Index / Percentage")
    plt.title(title)

    # Legend
    handles, labels = [], []
    for ax in [ax1, ax2, ax3]:
        if ax:
            h, l = ax.get_legend_handles_labels()
            handles += h; labels += l
    ax1.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # Google links on drill-down
    if drill and sel_year is not None:
        st.markdown("#### Explore Context")
        for c in plot_cols:
            country = c.split('_')[0]
            ind = c[len(country)+1:]
            url = make_google_search_link(country, sel_year, ind)
            st.markdown(f"[{country} {sel_year} {get_full_indicator_name(ind)}]({url})")

    # Download
    st.markdown("### Download Data")
    st.download_button(
        "Download CSV",
        data=output.to_csv(index=False).encode(),
        file_name="wb_data.csv",
        mime="text/csv"
    )

else:
    st.warning("Select at least one series to visualize.")
