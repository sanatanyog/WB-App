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
    '8': ('Poverty Index (Auto-Detected)', 'POVERTY_AUTO')
}

SHORT_NAMES = {
    'GDP': 'GDP (current US$)',
    'GDPpc': 'GDP per capita (current US$)',
    'GDP_PPP': 'GDP, PPP (current international $)',
    'GDPpc_PPP': 'GDP per capita, PPP (current international $)',
    'Inflation': 'Inflation, consumer prices (annual %)',
    'PopGrowth': 'Population growth (annual %)',
    'Debt': 'Central government debt, total (% of GDP)'
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
def fetch_wb_data(indicator_codes, country_codes):
    data = {}
    for ind_name, ind_code in indicator_codes:
        if ind_code == 'POVERTY_AUTO':
            for code in country_codes:
                if code == 'IND':
                    # 1. NITI Aayog MPI (percentages)
                    mpi_df = pd.DataFrame({'IND': [29.17, 14.96]}, index=[2016, 2021])
                    data["India MPI (NITI Aayog) (IND)"] = mpi_df

                    # 2. World Bank $2.15/day (percentages)
                    wb_215 = wb.data.DataFrame('SI.POV.DDAY', [code]).transpose()
                    wb_215.index = wb_215.index.map(lambda x: int(str(x).replace('YR', '')) if str(x).startswith('YR') else int(x))
                    wb_215 = wb_215.round(2).dropna(how='all')
                    data["$2.15/day Poverty (World Bank) (IND)"] = wb_215

                    # 3. World Bank $4.20/day (percentages)
                    wb_420 = wb.data.DataFrame('SI.POV.LMIC', [code]).transpose()
                    wb_420.index = wb_420.index.map(lambda x: int(str(x).replace('YR', '')) if str(x).startswith('YR') else int(x))
                    wb_420 = wb_420.round(2).dropna(how='all')
                    data["$4.20/day Poverty (World Bank) (IND)"] = wb_420
                else:
                    poverty_name, poverty_code = detect_poverty_index_for_country(code)
                    if poverty_code:
                        df = wb.data.DataFrame(poverty_code, [code]).transpose()
                        df.index = df.index.map(lambda x: int(str(x).replace('YR', '')) if str(x).startswith('YR') else int(x))
                        df = df.round(2).dropna(how='all')
                        data[f"{poverty_name} ({code})"] = df
        else:
            df = wb.data.DataFrame(ind_code, country_codes).transpose()
            df.index = df.index.map(lambda x: int(str(x).replace('YR', '')) if str(x).startswith('YR') else int(x))
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

# Step 1: Country Selection
st.markdown("#### 1️⃣ Select up to 5 countries to compare")
st.info("Pick up to 5 countries you want to compare. Start typing a country's name to search quickly.")
all_countries = [c['value'] for c in wb.economy.list() if len(c['id']) == 3]
selected_countries = st.multiselect(
    "Select up to 5 countries:",
    all_countries,
    max_selections=5,
    help="Pick up to 5 countries to compare. Start typing to search for a country."
)

if selected_countries:
    country_codes = get_iso3_codes(selected_countries)

    # Step 2: Indicator Selection
    st.markdown("#### 2️⃣ Choose indicators to compare")
    st.info("Choose one or more economic indicators to compare (e.g., GDP, Inflation, Population Growth).")
    indicator_keys = list(INDICATORS.keys())
    selected_inds = st.multiselect(
        "Indicators",
        indicator_keys,
        format_func=lambda k: INDICATORS[k][0],
        help="Select economic indicators like GDP, Inflation, etc. You can pick multiple indicators."
    )

    if selected_inds:
        indicator_codes = [INDICATORS[k] for k in selected_inds]
        data_dict = fetch_wb_data(indicator_codes, country_codes)

        all_years = set()
        for df in data_dict.values():
            all_years.update(df.index)
        years_sorted = sorted(all_years)
        output = pd.DataFrame({'YEAR': years_sorted})

        for ind_name, df in data_dict.items():
            # --- India: handle all three poverty indicators ---
            if 'India MPI' in ind_name and 'IND' in df.columns:
                output = output.merge(df[['IND']].rename(columns={'IND': 'India_MPI'}), left_on='YEAR', right_index=True, how='left')
            elif '$2.15/day Poverty' in ind_name and 'IND' in df.columns:
                output = output.merge(df[['IND']].rename(columns={'IND': 'India_Poverty_2.15'}), left_on='YEAR', right_index=True, how='left')
            elif '$4.20/day Poverty' in ind_name and 'IND' in df.columns:
                output = output.merge(df[['IND']].rename(columns={'IND': 'India_Poverty_4.20'}), left_on='YEAR', right_index=True, how='left')
            else:
                for code, country in zip(country_codes, selected_countries):
                    if 'Poverty' in ind_name:
                        if code in df.columns:
                            output = output.merge(df[[code]].rename(columns={code: f"{country}_Poverty"}), left_on='YEAR', right_index=True, how='left')
                    else:
                        short_ind = [k for k, v in SHORT_NAMES.items() if v == ind_name]
                        if short_ind:
                            short_ind = short_ind[0]
                        else:
                            short_ind = ind_name
                        colname = f"{country}_{short_ind}"
                        if code in df.columns:
                            output = output.merge(df[[code]].rename(columns={code: colname}), left_on='YEAR', right_index=True, how='left')

        # Step 3: Chart Columns Selection (AUTO-POPULATED)
        st.markdown("#### 3️⃣ Choose which data to visualize")
        st.info("All relevant country-indicator combinations are automatically selected. You may deselect any if you wish.")
        available_cols = [col for col in output.columns if col != "YEAR"]
        plot_cols = st.multiselect(
            "Choose columns to plot:",
            available_cols,
            default=available_cols,  # Auto-select all by default
            help="Select which country-indicator combinations to display on the chart."
        )

        if plot_cols:
            chart_title = st.text_input("Chart title", "Indicator Comparison Over Time")

            # --- Step 4: Decade Filter ---
            st.markdown("#### 4️⃣ (Optional) Filter by Decade")
            st.info(
                "You can focus on a specific decade (like the 1990s or 2000s) to see only the data from that period. "
                "This helps you compare economic trends within a particular decade."
            )
            enable_decade = st.checkbox(
                "Enable Decade Filter (optional)",
                help="Check this to filter data by decade (e.g., 1990s, 2000s)."
            )
            decades = sorted(set([(y // 10) * 10 for y in output['YEAR']]))
            decade_labels = [""] + [f"{d}s" for d in decades]
            output_filtered = output.copy()

            if enable_decade:
                selected_decade_label = st.selectbox(
                    "Select Decade (optional)",
                    decade_labels,
                    index=0,
                    help="Pick a decade to see only data from that period."
                )
                if selected_decade_label:
                    selected_decade = int(selected_decade_label[:-1])
                    output_filtered = output[(output['YEAR'] >= selected_decade) & (output['YEAR'] < selected_decade + 10)]
                else:
                    output_filtered = output.copy()
            else:
                output_filtered = output.copy()

            # --- Step 5: Year Drill-Down ---
            st.markdown("#### 5️⃣ (Optional) Drill Down to a Specific Year")
            st.info(
                "Want to focus on a particular year? Enable this option to select a single year and see detailed data for that year. "
                "This is useful if you want to understand what happened in a specific year."
            )
            drill_down = st.checkbox(
                "🔍 Drill down to a specific year (optional)",
                key="year_drill",
                help="Check this to select a specific year for detailed analysis."
            )
            selected_year = None
            if drill_down:
                available_years = output_filtered['YEAR'].tolist()
                if available_years:
                    selected_year = st.selectbox(
                        "Select Year",
                        available_years,
                        help="Pick the year you want to focus on."
                    )
                else:
                    st.warning("No data available for selected decade/year")

            # --- Assign columns to axis types ---
            abs_cols = []
            usd_cols = []
            perc_cols = []
            for col in plot_cols:
                indicator = '_'.join(col.split('_')[1:])
                if indicator in ['GDP', 'GDP_PPP']:
                    abs_cols.append(col)
                elif indicator in ['GDPpc', 'GDPpc_PPP']:
                    usd_cols.append(col)
                elif indicator in ['Inflation', 'PopGrowth', 'Debt']:
                    perc_cols.append(col)
                elif 'Poverty' in col or 'MPI' in col:
                    perc_cols.append(col)

            # --- Color and Style Logic ---
            countries = list({col.split('_')[0] for col in plot_cols})
            indicators = list({'_'.join(col.split('_')[1:]) for col in plot_cols})
            country_colors = plt.cm.tab10(range(len(countries)))
            color_map = {country: country_colors[i] for i, country in enumerate(countries)}
            line_styles = ['-', '--', '-.', ':']
            style_map = {ind: line_styles[i % len(line_styles)] for i, ind in enumerate(indicators)}

            fig, ax1 = plt.subplots(figsize=(12, 7))
            ax2 = ax3 = None

            # --- Plotting ---
            if drill_down and selected_year is not None:
                year_data = output_filtered[output_filtered['YEAR'] == selected_year]
                if usd_cols and (abs_cols or perc_cols):
                    ax2 = ax1.twinx()
                if perc_cols and (abs_cols or usd_cols):
                    ax3 = ax1.twinx()
                    ax3.spines['right'].set_position(('outward', 60))
                for col in abs_cols:
                    if year_data[col].notna().sum() > 0:
                        country = col.split('_')[0]
                        indicator = '_'.join(col.split('_')[1:])
                        color = color_map[country]
                        linestyle = style_map[indicator]
                        value = year_data[col].values[0]
                        ax1.scatter(selected_year, value, color=color, s=120, label=f"{country} - {get_full_indicator_name(indicator)}: ${value}B")
                        ax1.text(selected_year, value, f"${value}B", fontsize=10, ha='left', va='bottom', color=color)
                for col in usd_cols:
                    if year_data[col].notna().sum() > 0:
                        country = col.split('_')[0]
                        indicator = '_'.join(col.split('_')[1:])
                        color = color_map[country]
                        linestyle = style_map[indicator]
                        value = year_data[col].values[0]
                        if ax2:
                            ax2.scatter(selected_year, value, color=color, s=120, marker='s', label=f"{country} - {get_full_indicator_name(indicator)}: ${value}")
                            ax2.text(selected_year, value, f"${value}", fontsize=10, ha='left', va='bottom', color=color)
                        else:
                            ax1.scatter(selected_year, value, color=color, s=120, marker='s', label=f"{country} - {get_full_indicator_name(indicator)}: ${value}")
                            ax1.text(selected_year, value, f"${value}", fontsize=10, ha='left', va='bottom', color=color)
                for col in perc_cols:
                    if year_data[col].notna().sum() > 0:
                        country = col.split('_')[0]
                        indicator = '_'.join(col.split('_')[1:])
                        # --- India: special color for MPI and poverty lines ---
                        if col == 'India_MPI':
                            color = "#FF9933"
                            linestyle = ':'
                        elif col == 'India_Poverty_2.15':
                            color = "#003399"
                            linestyle = '--'
                        elif col == 'India_Poverty_4.20':
                            color = "#228B22"
                            linestyle = '-.'
                        else:
                            color = color_map.get(country, 'black')
                            linestyle = style_map.get(indicator, '-')
                        value = year_data[col].values[0]
                        if ax3:
                            ax3.scatter(selected_year, value, color=color, s=120, marker='^', label=f"{country} - {indicator}: {value}%")
                            ax3.text(selected_year, value, f"{value}%", fontsize=10, ha='left', va='bottom', color=color)
                        elif ax2:
                            ax2.scatter(selected_year, value, color=color, s=120, marker='^', label=f"{country} - {indicator}: {value}%")
                            ax2.text(selected_year, value, f"{value}%", fontsize=10, ha='left', va='bottom', color=color)
                        else:
                            ax1.scatter(selected_year, value, color=color, s=120, marker='^', label=f"{country} - {indicator}: {value}%")
                            ax1.text(selected_year, value, f"{value}%", fontsize=10, ha='left', va='bottom', color=color)
                ax1.set_xlabel("Year")
                ax1.set_xticks([selected_year])
                if ax2 and ax3:
                    ax1.set_ylabel("GDP/PPP (Billions USD)")
                    ax2.set_ylabel("Per Capita (USD)")
                    ax3.set_ylabel("Percentage Indicators (%)")
                elif ax2:
                    ax1.set_ylabel("GDP/PPP (Billions USD)")
                    ax2.set_ylabel("Per Capita (USD) / Percentage (%)")
                elif ax3:
                    ax1.set_ylabel("GDP/PPP (Billions USD) / Per Capita (USD)")
                    ax3.set_ylabel("Percentage Indicators (%)")
                else:
                    ax1.set_ylabel("Value")
                ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                if ax2: ax2.legend(bbox_to_anchor=(1.02, 0.7), loc='upper left')
                if ax3: ax3.legend(bbox_to_anchor=(1.02, 0.3), loc='upper left')
            else:
                if usd_cols and (abs_cols or perc_cols):
                    ax2 = ax1.twinx()
                if perc_cols and (abs_cols or usd_cols):
                    ax3 = ax1.twinx()
                    ax3.spines['right'].set_position(('outward', 60))
                for col in abs_cols:
                    if output_filtered[col].notna().sum() > 0:
                        country = col.split('_')[0]
                        indicator = '_'.join(col.split('_')[1:])
                        color = color_map[country]
                        linestyle = style_map[indicator]
                        ax1.plot(output_filtered['YEAR'], output_filtered[col], label=f"{country} - {get_full_indicator_name(indicator)} (Billions USD)", color=color, linestyle=linestyle)
                for col in usd_cols:
                    if output_filtered[col].notna().sum() > 0:
                        country = col.split('_')[0]
                        indicator = '_'.join(col.split('_')[1:])
                        color = color_map[country]
                        linestyle = style_map[indicator]
                        if ax2:
                            ax2.plot(output_filtered['YEAR'], output_filtered[col], label=f"{country} - {get_full_indicator_name(indicator)} (USD)", color=color, linestyle=linestyle)
                        else:
                            ax1.plot(output_filtered['YEAR'], output_filtered[col], label=f"{country} - {get_full_indicator_name(indicator)} (USD)", color=color, linestyle=linestyle)
                for col in perc_cols:
                    if output_filtered[col].notna().sum() > 0:
                        country = col.split('_')[0]
                        indicator = '_'.join(col.split('_')[1:])
                        # --- India: special color for MPI and poverty lines ---
                        if col == 'India_MPI':
                            color = "#FF9933"
                            linestyle = ':'
                        elif col == 'India_Poverty_2.15':
                            color = "#003399"
                            linestyle = '--'
                        elif col == 'India_Poverty_4.20':
                            color = "#228B22"
                            linestyle = '-.'
                        else:
                            color = color_map.get(country, 'black')
                            linestyle = style_map.get(indicator, '-')
                        if ax3:
                            ax3.plot(output_filtered['YEAR'], output_filtered[col], label=f"{country} - {indicator} (%)", color=color, linestyle=linestyle)
                        elif ax2:
                            ax2.plot(output_filtered['YEAR'], output_filtered[col], label=f"{country} - {indicator} (%)", color=color, linestyle=linestyle)
                        else:
                            ax1.plot(output_filtered['YEAR'], output_filtered[col], label=f"{country} - {indicator} (%)", color=color, linestyle=linestyle)
                ax1.set_xlabel("Year")
                if ax2 and ax3:
                    ax1.set_ylabel("GDP/PPP (Billions USD)")
                    ax2.set_ylabel("Per Capita (USD)")
                    ax3.set_ylabel("Percentage Indicators (%)")
                elif ax2:
                    ax1.set_ylabel("GDP/PPP (Billions USD)")
                    ax2.set_ylabel("Per Capita (USD) / Percentage (%)")
                elif ax3:
                    ax1.set_ylabel("GDP/PPP (Billions USD) / Per Capita (USD)")
                    ax3.set_ylabel("Percentage Indicators (%)")
                else:
                    ax1.set_ylabel("Value")
                ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                if ax2: ax2.legend(bbox_to_anchor=(1.02, 0.7), loc='upper left')
                if ax3: ax3.legend(bbox_to_anchor=(1.02, 0.3), loc='upper left')
            ax1.set_title(chart_title)
            ax1.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)

            # --- Google Search Links (shown below chart) ---
            if drill_down and selected_year is not None:
                st.markdown("### 🔗 Google Search: Context for Selected Year")
                st.info(
                    "Click the links below to search Google for news, reports, or events that might explain the economic data for your selected country and year. "
                    "This helps you understand the real-world context behind the numbers."
                )
                for col in plot_cols:
                    if col in output_filtered.columns:
                        country = col.split('_')[0]
                        indicator_abbr = '_'.join(col.split('_')[1:])
                        search_url = make_google_search_link(country, selected_year, indicator_abbr)
                        st.link_button(f"Explore {country} {selected_year} {get_full_indicator_name(indicator_abbr)} context", search_url)

            # --- Data Download ---
            st.markdown("### 💾 Download Data")
            st.download_button(
                label="Download CSV",
                data=output_filtered.to_csv(index=False).encode('utf-8'),
                file_name='worldbank_data.csv',
                mime='text/csv'
            )
        else:
            st.warning("Please select at least one indicator")
else:
    st.info("Select up to 5 countries to get started")
