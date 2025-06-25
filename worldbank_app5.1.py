import wbgapi as wb
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import urllib.parse

# --- Indicator Definitions (Human-friendly) ---
INDICATORS = {
    '1':  ('Total Economy Size (GDP, US$)',        'NY.GDP.MKTP.CD'),
    '2':  ('Average Income per Person (US$)',       'NY.GDP.PCAP.CD'),
    '3':  ('Economy Size Adjusted for Cost (PPP)',  'NY.GDP.MKTP.PP.CD'),
    '4':  ('Average Income per Person (PPP, $)',    'NY.GDP.PCAP.PP.CD'),
    '5':  ('Inflation Rate (%)',                    'FP.CPI.TOTL.ZG'),
    '6':  ('Population Growth Rate (%)',            'SP.POP.GROW'),
    '7':  ('Government Debt % of GDP',              'GC.DOD.TOTL.GD.ZS'),
    '8':  ('Poverty Rate (%)',                      'POVERTY_AUTO'),
    '9':  ('Unemployment Rate (%)',                 'SL.UEM.TOTL.ZS'),
    '10': ('Corruption Index Score (–2.5 to +2.5)', 'CC.EST')
}

SHORT_NAMES = {
    'GDP':    'Total Economy Size (GDP, US$)',
    'GDPpc':  'Average Income per Person (US$)',
    'GDP_PPP':'Economy Size Adjusted for Cost (PPP)',
    'GDPpc_PPP':'Average Income per Person (PPP, $)',
    'Inflation':'Inflation Rate (%)',
    'PopGrowth':'Population Growth Rate (%)',
    'Debt':   'Government Debt % of GDP',
    'Unemployment Rate (%)':'Unemployment Rate (%)',
    'Corruption Index Score (–2.5 to +2.5)':'Corruption Index Score'
}

POVERTY_INDICATORS = [
    ('Poverty at $2.15/day (%)', 'SI.POV.DDAY'),
    ('Poverty at $4.20/day (%)', 'SI.POV.LMIC'),
    ('Poverty at national lines (%)','SI.POV.NAHC'),
    ('Gini Index',                 'SI.POV.GINI')
]

def get_full_indicator_name(abbr):
    return SHORT_NAMES.get(abbr, abbr)

@st.cache_data
def get_iso3_codes(user_countries):
    all_names = {c['value'].lower(): c['id'] for c in wb.economy.list() if len(c['id']) == 3}
    return [all_names[name.lower()] for name in user_countries if name.lower() in all_names]

@st.cache_data
def detect_poverty_index_for_country(country_code):
    for name, code in POVERTY_INDICATORS:
        df = wb.data.DataFrame(code, [country_code]).transpose()
        if not df.dropna(how='all').empty:
            return (name, code)
    return (None, None)

@st.cache_data
def fetch_indicator_data(code, country_codes):
    # Fetch and clean any indicator series
    df = wb.data.DataFrame(code, country_codes).transpose()
    df.index = df.index.map(lambda y: int(str(y).replace('YR','')) if str(y).startswith('YR') else int(y))
    return df.round(2).dropna(how='all')

@st.cache_data
def fetch_wb_data(indicator_codes, country_codes, country_names):
    data = {}
    for ind_name, ind_code in indicator_codes:
        # Poverty auto-detect
        if ind_code == 'POVERTY_AUTO':
            for cc in country_codes:
                if cc == 'IND':
                    # India-specific MPI
                    mpi = pd.DataFrame({'IND': [29.17, 14.96]}, index=[2016, 2021])
                    data["India MPI (NITI Aayog)"] = mpi
                    for label, code2 in [('Pov $2.15/day', 'SI.POV.DDAY'), ('Pov $4.20/day', 'SI.POV.LMIC')]:
                        df2 = fetch_indicator_data(code2, [cc])
                        data[f"{label} (World Bank)"] = df2
                else:
                    pname, pcode = detect_poverty_index_for_country(cc)
                    if pcode:
                        data[f"{pname} ({cc})"] = fetch_indicator_data(pcode, [cc])

        # Unemployment
        elif ind_code == 'SL.UEM.TOTL.ZS':
            df = fetch_indicator_data(ind_code, country_codes)
            for cc, name in zip(country_codes, country_names):
                if cc in df.columns:
                    col = df[[cc]].rename(columns={cc: name})
                    data[f"Unemployment Rate - {name}"] = col

        # Corruption
        elif ind_code == 'CC.EST':
            df = fetch_indicator_data(ind_code, country_codes)
            for cc, name in zip(country_codes, country_names):
                if cc in df.columns:
                    col = df[[cc]].rename(columns={cc: name})
                    data[f"Corruption Score - {name}"] = col

        # Other indicators
        else:
            df = fetch_indicator_data(ind_code, country_codes)
            if ind_code in ['NY.GDP.MKTP.CD', 'NY.GDP.MKTP.PP.CD']:
                df = df / 1e9  # scale GDP to billions
            data[ind_name] = df

    return data

def make_google_search_link(country, year, indicator_abbr):
    full = get_full_indicator_name(indicator_abbr)
    q = f"{country} {year} {full} economic context"
    return f"https://www.google.com/search?q={urllib.parse.quote(q)}"

# --- Streamlit UI ---
st.set_page_config(page_title="EconEasy: World Bank CXO Dashboard", layout="wide")
st.title("EconEasy: Built for the Curious, Not Just the Suits")

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
keys = list(INDICATORS.keys())
selected_inds = st.multiselect("Indicators:", keys, format_func=lambda k: INDICATORS[k][0])
if not selected_inds:
    st.warning("Select at least one indicator.")
    st.stop()
indicator_codes = [INDICATORS[k] for k in selected_inds]

# Fetch data
data_dict = fetch_wb_data(indicator_codes, country_codes, selected_countries)

# Build combined DataFrame
all_years = sorted({yr for df in data_dict.values() for yr in df.index})
output = pd.DataFrame({'Year': all_years})
for name, df in data_dict.items():
    col_key = name.replace(' ', '_').replace('%','').replace('(','').replace(')','').replace('-','_')
    output = output.merge(df.rename(columns={df.columns[0]: col_key}),
                          left_on='Year', right_index=True, how='left')

# 3️⃣ Plot Selection
st.markdown("#### Select Data Series to Plot")
available_cols = [c for c in output.columns if c != 'Year']
plot_cols = st.multiselect("Plot Columns:", available_cols, default=available_cols)

if plot_cols:
    title = st.text_input("Chart Title", "Economic Indicators Over Time")

    # 4️⃣ Decade Filter
    decades = sorted({(y // 10)*10 for y in output['Year']})
    if st.checkbox("Filter by Decade"):
        decade = st.selectbox("Decade:", [f"{d}s" for d in decades])
        d0 = int(decade[:-1])
        output = output[(output['Year'] >= d0) & (output['Year'] < d0+10)]

    # 5️⃣ Drill-Down
    if st.checkbox("🔍 Drill Down to Specific Year"):
        sel_year = st.selectbox("Year:", output['Year'].tolist())
    else:
        sel_year = None

    # Separate axes
    abs_cols, rate_cols, idx_cols = [], [], []
    for col in plot_cols:
        if "GDP" in col:
            abs_cols.append(col)
        elif "Inflation" in col or "Unemployment" in col:
            rate_cols.append(col)
        else:
            idx_cols.append(col)

    fig, ax1 = plt.subplots(figsize=(12,7))
    ax2 = ax3 = None
    if rate_cols and (abs_cols or idx_cols):
        ax2 = ax1.twinx()
    if idx_cols and (abs_cols or rate_cols):
        ax3 = ax1.twinx(); ax3.spines['right'].set_position(('outward', 60))

    if sel_year is not None:
        row = output[output['Year'] == sel_year].iloc[0]
        for cols, ax, marker in [
            (abs_cols, ax1, 'o'),
            (rate_cols, ax2 or ax1, 's'),
            (idx_cols,  ax3 or ax2 or ax1, '^')
        ]:
            for c in cols:
                v = row[c]
                if pd.notna(v):
                    # Build legend label with country code
                    parts = c.rsplit('_', 1)
                    label = f"{parts[0]} ({parts[-1]})" if len(parts)>1 else parts[0]
                    ax.scatter(sel_year, v, marker=marker, s=100, label=label)
        ax1.set_xticks([sel_year])
    else:
        for cols, ax, style in [
            (abs_cols, ax1, '-'),
            (rate_cols, ax2 or ax1, '--'),
            (idx_cols,  ax3 or ax2 or ax1, ':')
        ]:
            for c in cols:
                parts = c.rsplit('_', 1)
                label = f"{parts[0]} ({parts[-1]})" if len(parts)>1 else parts[0]
                ax.plot(output['Year'], output[c], style, label=label)
        ax1.set_xlabel("Year")

    ax1.set_ylabel("Absolute / Level")
    if ax2: ax2.set_ylabel("Rate (%)")
    if ax3: ax3.set_ylabel("Index Score")
    ax1.set_title(title)

    # Legend
    handles, labels = [], []
    for ax in [ax1, ax2, ax3]:
        if ax:
            h, l = ax.get_legend_handles_labels()
            handles += h; labels += l
    ax1.legend(handles, labels, bbox_to_anchor=(1.02,1), loc='upper left')
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # Context links if drilling down
    if sel_year is not None:
        st.markdown("#### Explore Context")
        for c in plot_cols:
            country = c.split('_')[-1]
            ind = c.rsplit('_',1)[0]
            url = make_google_search_link(country, sel_year, ind)
            st.markdown(f"[{country} {sel_year} {get_full_indicator_name(ind)}]({url})")

    # Download data
    st.markdown("### Download Data")
    st.download_button(
        "Download CSV",
        data=output.to_csv(index=False).encode(),
        file_name="worldbank_data.csv",
        mime="text/csv"
    )
else:
    st.warning("Select at least one series to visualize.")
