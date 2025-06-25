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
    '9': ('Unemployment Rate (%)', 'SL.UEM.TOTL.ZS')
}

SHORT_NAMES = {
    'GDP': 'GDP (current US$)',
    'GDPpc': 'GDP per capita (current US$)',
    'GDP_PPP': 'GDP, PPP (current international $)',
    'GDPpc_PPP': 'GDP per capita, PPP (current international $)',
    'Inflation': 'Inflation, consumer prices (annual %)',
    'PopGrowth': 'Population growth (annual %)',
    'Debt': 'Central government debt, total (% of GDP)',
    'Unemployment Rate (%)': 'Unemployment Rate (%)'
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
    codes = []
    for name in user_countries:
        code = all_names.get(name.lower())
        if code:
            codes.append(code)
    return codes

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
def fetch_wb_data(indicator_codes, country_codes):
    data = {}
    for ind_name, ind_code in indicator_codes:
        if ind_code == 'POVERTY_AUTO':
            for code in country_codes:
                if code == 'IND':
                    mpi_df = pd.DataFrame({'IND': [29.17, 14.96]}, index=[2016, 2021])
                    data["India MPI (NITI Aayog) (IND)"] = mpi_df
                    wb_215 = wb.data.DataFrame('SI.POV.DDAY', [code]).transpose()
                    wb_215.index = wb_215.index.map(lambda x: int(str(x).replace('YR','')) if str(x).startswith('YR') else int(x))
                    data["$2.15/day Poverty (World Bank) (IND)"] = wb_215.round(2).dropna(how='all')
                    wb_420 = wb.data.DataFrame('SI.POV.LMIC', [code]).transpose()
                    wb_420.index = wb_420.index.map(lambda x: int(str(x).replace('YR','')) if str(x).startswith('YR') else int(x))
                    data["$4.20/day Poverty (World Bank) (IND)"] = wb_420.round(2).dropna(how='all')
                else:
                    poverty_name, poverty_code = detect_poverty_index_for_country(code)
                    if poverty_code:
                        df = wb.data.DataFrame(poverty_code, [code]).transpose()
                        df.index = df.index.map(lambda x: int(str(x).replace('YR','')) if str(x).startswith('YR') else int(x))
                        data[f"{poverty_name} ({code})"] = df.round(2).dropna(how='all')
        elif ind_code == 'SL.UEM.TOTL.ZS':
            df = fetch_unemployment_data(country_codes)
            for code, country in zip(country_codes, selected_countries):
                if code in df.columns:
                    data[f"{country} Unemployment Rate (%)"] = df[[code]].rename(columns={code: code})
        else:
            df = wb.data.DataFrame(ind_code, country_codes).transpose()
            df.index = df.index.map(lambda x: int(str(x).replace('YR','')) if str(x).startswith('YR') else int(x))
            if ind_code in ['NY.GDP.MKTP.CD', 'NY.GDP.MKTP.PP.CD']:
                df = df / 1e9
            data[ind_name] = df.round(2).dropna(how='all')
    return data

def make_google_search_link(country, year, indicator_abbr):
    full_indicator = get_full_indicator_name(indicator_abbr)
    query = f"{country} {year} {full_indicator} economic context"
    return f"https://www.google.com/search?q={urllib.parse.quote(query)}"

# --- Streamlit UI ---
st.set_page_config(page_title="EconEasy: World Bank CXO Dashboard", layout="wide")
st.title("EconEasy: World Bank Data for Executives")

# Step 1: Country Selection
st.markdown("#### 1️⃣ Select up to 5 countries to compare")
st.info("Pick up to 5 countries you want to compare. Start typing a country's name to search quickly.")
all_countries = [c['value'] for c in wb.economy.list() if len(c['id']) == 3]
selected_countries = st.multiselect("Select up to 5 countries:", all_countries, max_selections=5)

if selected_countries:
    country_codes = get_iso3_codes(selected_countries)

    # Step 2: Indicator Selection
    st.markdown("#### 2️⃣ Choose indicators to compare")
    indicator_keys = list(INDICATORS.keys())
    selected_inds = st.multiselect(
        "Indicators",
        indicator_keys,
        format_func=lambda k: INDICATORS[k][0]
    )

    if selected_inds:
        indicator_codes = [INDICATORS[k] for k in selected_inds]
        data_dict = fetch_wb_data(indicator_codes, country_codes)

        # Build unified DataFrame
        all_years = sorted({y for df in data_dict.values() for y in df.index})
        output = pd.DataFrame({'YEAR': all_years})
        for name, df in data_dict.items():
            col_key = name.split(' (')[0].replace(' ', '_').replace('%','').replace('(','').replace(')','')
            output = output.merge(
                df.rename(columns={df.columns[0]: col_key}),
                left_on='YEAR', right_index=True, how='left'
            )

        # Step 3: Chart Columns Selection
        st.markdown("#### 3️⃣ Choose which data to visualize")
        available_cols = [c for c in output.columns if c != "YEAR"]
        plot_cols = st.multiselect("Choose columns to plot:", available_cols, default=available_cols)

        if plot_cols:
            chart_title = st.text_input("Chart title", "Indicator Comparison Over Time")

            # Step 4: Decade Filter
            st.markdown("#### 4️⃣ (Optional) Filter by Decade")
            decades = sorted({(y // 10) * 10 for y in output['YEAR']})
            enable_decade = st.checkbox("Enable Decade Filter")
            output_filtered = output.copy()
            if enable_decade:
                decade = st.selectbox("Select Decade", [""] + [f"{d}s" for d in decades])
                if decade:
                    d = int(decade[:-1])
                    output_filtered = output[(output['YEAR']>=d)&(output['YEAR']<d+10)]

            # Step 5: Year Drill-Down
            st.markdown("#### 5️⃣ (Optional) Drill Down to a Specific Year")
            drill_down = st.checkbox("🔍 Drill down to a specific year")
            selected_year = None
            if drill_down:
                selected_year = st.selectbox("Select Year", output_filtered['YEAR'].tolist())

            # Assign axes
            abs_cols, usd_cols, perc_cols = [], [], []
            for col in plot_cols:
                if "GDP" in col and "per_capita" not in col:
                    abs_cols.append(col)
                elif "per_capita" in col:
                    usd_cols.append(col)
                else:
                    perc_cols.append(col)

            # Plot
            fig, ax1 = plt.subplots(figsize=(12, 7))
            ax2 = ax3 = None
            if usd_cols and (abs_cols or perc_cols):
                ax2 = ax1.twinx()
            if perc_cols and (abs_cols or usd_cols):
                ax3 = ax1.twinx()
                ax3.spines['right'].set_position(('outward', 60))

            # Time series vs scatter
            if drill_down and selected_year is not None:
                row = output_filtered[output_filtered['YEAR']==selected_year].iloc[0]
                for col_list, axis, marker in [(abs_cols, ax1, 'o'), (usd_cols, ax2 or ax1, 's'), (perc_cols, ax3 or ax2 or ax1, '^')]:
                    for col in col_list:
                        val = row[col]
                        if pd.notna(val):
                            axis.scatter(selected_year, val, marker=marker, s=100, label=f"{col}: {val}")
                ax1.set_xlabel("Year")
                ax1.set_xticks([selected_year])
            else:
                for col_list, axis, fmt in [(abs_cols, ax1, '-'), (usd_cols, ax2 or ax1, '--'), (perc_cols, ax3 or ax2 or ax1, ':')]:
                    for col in col_list:
                        axis.plot(output_filtered['YEAR'], output_filtered[col], fmt, label=col)

                ax1.set_xlabel("Year")

            # Labels and legends
            ax1.set_ylabel("Absolute Value / GDP")
            if ax2: ax2.set_ylabel("Per Capita")
            if ax3: ax3.set_ylabel("Percentage")
            ax1.set_title(chart_title)
            lines, labels = [], []
            for ax in [ax1, ax2, ax3]:
                if ax:
                    l, lb = ax.get_legend_handles_labels()
                    lines += l; labels += lb
            ax1.legend(lines, labels, bbox_to_anchor=(1.02, 1), loc='upper left')
            ax1.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)

            # Google search links (for drill-down)
            if drill_down and selected_year is not None:
                st.markdown("#### Discover Key Events Behind the Data")
                for col in plot_cols:
                    country = col.split('_')[0]
                    ind = col[len(country)+1:]
                    url = make_google_search_link(country, selected_year, ind)
                    st.markdown(f"[Explore {country} {selected_year} {get_full_indicator_name(ind)} context]({url})")

            # Download data
            st.markdown("### 💾 Download Data")
            st.download_button(
                label="Download CSV",
                data=output_filtered.to_csv(index=False).encode('utf-8'),
                file_name='worldbank_data.csv',
                mime='text/csv'
            )
        else:
            st.warning("Please select at least one data series to plot")
    else:
        st.warning("Select at least one indicator to proceed")
else:
    st.info("Select up to 5 countries to get started")
