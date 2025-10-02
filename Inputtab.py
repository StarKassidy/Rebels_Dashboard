# ===== Import libraries =====
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ===== Load & preprocess data =====
recdata = pd.read_excel('KPI Team.xlsx', sheet_name='Blad1', header=1)
recdata = recdata.drop(index=[37, 38, 39, 40], errors='ignore')
recdata['Begin datum'] = pd.to_datetime(recdata['Begin datum'], format='%d/%m/%Y', errors='coerce')
recdata['Eind datum'] = pd.to_datetime(recdata['Eind datum'], format='%d/%m/%Y', errors='coerce')
recdata['Week'] = recdata['Begin datum'].dt.isocalendar().week
recdata = recdata.fillna(0)
recdata = recdata.replace('holiday', 0)

recdata['Name'] = recdata['Name'].astype(str)
for col in ['Cold call', 'Qualification', 'Introductions', 'InMails']:
    recdata[col] = recdata[col].astype(int)
recdata['Response rate'] = recdata['Response rate'].astype(str).apply(
    lambda x: float(x.replace('%', '')) / 100 if '%' in x else float(x)
)

recdata['Time to hire'] = (recdata['Eind datum'] - recdata['Begin datum']).dt.days
recdata['Responses (accepted or declined)'] = ((recdata['InMails'] + recdata['Cold call']) * recdata['Response rate']).round().astype(int)
recdata['Introductions to first contact ratio (%)'] = recdata.apply(
    lambda row: round(row['Introductions'] / (row['InMails'] + row['Cold call']) * 100, 2)
    if (row['InMails'] + row['Cold call']) > 0 else 0, axis=1
)
recdata['Candidate employment'] = recdata['Introductions'].apply(lambda x: 1 if x > 3 else 0)
recdata = recdata[recdata["Name"].str.lower() != "eindtotaal"]

# ===== Week filter (dropdown menu) =====
week_options = sorted(recdata['Week'].unique().tolist())
selected_week = st.selectbox("Selecteer week", week_options)

# ===== Filter data per geselecteerde week =====
filtered_data = recdata[recdata['Week'] == int(selected_week)]
week_label = f"Week {selected_week}"

# ===== Bereken KPI's per week =====
avg_inmails = filtered_data["InMails"].mean()
avg_coldcalls = filtered_data["Cold call"].mean()
avg_qualification = filtered_data["Qualification"].mean()
avg_response = filtered_data["Response rate"].mean()

targets = {
    "InMails": 100,
    "Cold call": 120,
    "Response rate": 0.2,
    "Qualification": 15
}

chart_title_prefix = "Voortgang Gemiddelde"

# ===== Function to create donut chart =====
def plot_donut(kpi_name, value, target, title, color="#636EFA"):
    remaining = max(target - value, 0)
    values = [min(value, target), remaining]
    percent_achieved = min(value / target * 100, 100) if target > 0 else 0

    fig = px.pie(
        names=[kpi_name, "Nog te behalen"],
        values=values,
        hole=0.5,
        color_discrete_sequence=[color, "#E5ECF6"],
        height=300
    )
    fig.update_traces(textinfo='none', sort=False)
    fig.add_annotation(
        text=f"{percent_achieved:.0f}%",
        x=0.5, y=0.5,
        font_size=28,
        showarrow=False
    )
    fig.update_layout(title_text=title, margin=dict(t=40, b=0, l=0, r=0))
    return fig

# ===== Function to create response rate progress bar =====
def plot_response_rate_bar(avg_value, target, title):
    color = "#00CC96" if avg_value >= target else "#EF553B"
    fig = go.Figure(go.Bar(
        x=[avg_value*100],
        y=[""],
        orientation='h',
        marker_color=color,
        width=0.6
    ))
    fig.add_shape(
        type="line",
        x0=target*100, x1=target*100,
        y0=-0.5, y1=0.5,
        line=dict(color="red", width=4, dash="dash")
    )
    fig.update_layout(
        xaxis=dict(range=[0,100], title="Percentage (%)"),
        yaxis=dict(showticklabels=False),
        showlegend=False,
        height=150,
        margin=dict(l=20, r=20, t=60, b=20),
        title=dict(text=title, x=0.5, xanchor='center', yanchor='top')
    )
    return fig

# ===== Function to create colored bar charts per recruiter =====
def colored_bar_chart(data, x_col, y_col, title, target, is_percentage=False):
    colors = ["#00CC96" if v >= target else "#EF553B" for v in data[y_col]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data[x_col],
        y=data[y_col],
        marker_color=colors,
        text=[f"{v:.0%}" if is_percentage else str(v) for v in data[y_col]],
        textposition='auto'
    ))
    fig.add_shape(
        type="line",
        x0=-0.5, x1=len(data[x_col])-0.5,
        y0=target, y1=target,
        line=dict(color="red", width=3, dash="dash")
    )
    y_max = max(data[y_col].max(), target)
    fig.update_yaxes(range=[0, y_max*1.2], tickformat=".0%" if is_percentage else None)
    fig.update_layout(title=title, height=300, showlegend=False)
    return fig

# ===== Streamlit layout =====
st.set_page_config(page_title="Recruitment KPI Dashboard", layout="wide")
st.title("📊 Recruitment KPI Dashboard")

tab1, tab2 = st.tabs(["Input KPI's", "Output KPI's"])

with tab1:
    st.header("Input KPI's")

    # --- Donut charts per week ---
    col1, col2, col3 = st.columns(3)
    with col1:
        fig_avg_inmails = plot_donut("InMails", avg_inmails, targets["InMails"],
                                     f"{chart_title_prefix} InMails ({week_label})", color="#636EFA")
        st.plotly_chart(fig_avg_inmails, use_container_width=True)
    with col2:
        fig_avg_coldcalls = plot_donut("Cold Calls", avg_coldcalls, targets["Cold call"],
                                       f"{chart_title_prefix} Cold Calls ({week_label})", color="#EF553B")
        st.plotly_chart(fig_avg_coldcalls, use_container_width=True)
    with col3:
        fig_avg_qualification = plot_donut("Qualification", avg_qualification, targets["Qualification"],
                                           f"{chart_title_prefix} Kwalificatiecalls ({week_label})", color="#AB63FA")
        st.plotly_chart(fig_avg_qualification, use_container_width=True)

    # --- Response Rate ---
    fig_avg_response_bar = plot_response_rate_bar(avg_response, targets['Response rate'],
                                                  f"Gemiddelde Response Rate ({week_label})")
    st.plotly_chart(fig_avg_response_bar, use_container_width=True)

    # --- Per recruiter breakdown met targetlijnen en kleur ---
    st.subheader(f"Per Recruiter Breakdown ({week_label})")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fig_inmails = colored_bar_chart(filtered_data, "Name", "InMails", "InMails per Recruiter", targets["InMails"])
        st.plotly_chart(fig_inmails, use_container_width=True)

    with col2:
        fig_coldcalls = colored_bar_chart(filtered_data, "Name", "Cold call", "Cold Calls per Recruiter", targets["Cold call"])
        st.plotly_chart(fig_coldcalls, use_container_width=True)

    with col3:
        fig_response = colored_bar_chart(filtered_data, "Name", "Response rate", "Response Rate per Recruiter", targets["Response rate"], is_percentage=True)
        st.plotly_chart(fig_response, use_container_width=True)

    with col4:
        fig_qualification = colored_bar_chart(filtered_data, "Name", "Qualification", "Qualification per Recruiter", targets["Qualification"])
        st.plotly_chart(fig_qualification, use_container_width=True)

with tab2:
    st.header("Output KPI's")

    # =========================
    # MAAND-BASED KPI'S
    # =========================
    st.subheader("Maandelijkse Doelstelling")

    recdata["Maand"] = recdata["Begin datum"].dt.to_period("M")
    maand_labels = recdata["Maand"].dt.strftime("%B %Y").unique()
    maand_map = dict(zip(maand_labels, recdata["Maand"].unique()))

    geselecteerde_maand_label = st.selectbox(
        "Selecteer maand", sorted(maand_labels), key="maand_selectie"
    )
    geselecteerde_maand = maand_map[geselecteerde_maand_label]

    maand_data = recdata[recdata["Maand"] == geselecteerde_maand]
    hires_per_recruiter = maand_data.groupby("Name")["Candidate employment"].sum().reset_index()

    goal = 1
    recruiters_totaal = hires_per_recruiter["Name"].nunique()
    recruiters_gehaald = (hires_per_recruiter["Candidate employment"] >= goal).sum()
    percentage_gehaald = (recruiters_gehaald / recruiters_totaal * 100) if recruiters_totaal > 0 else 0

    # KPI kaart
    st.metric(
        f"Recruiters met ≥{goal} hire ({geselecteerde_maand_label})",
        f"{recruiters_gehaald}/{recruiters_totaal} ({percentage_gehaald:.0f}%)"
    )

    # Bar chart per recruiter
    kleuren = ["#00CC96" if v >= goal else "#EF553B" for v in hires_per_recruiter["Candidate employment"]]
    fig_goal = go.Figure()
    fig_goal.add_trace(go.Bar(
        x=hires_per_recruiter["Name"],
        y=hires_per_recruiter["Candidate employment"],
        marker_color=kleuren,
        text=hires_per_recruiter["Candidate employment"],
        textposition="auto"
    ))
    fig_goal.add_shape(
        type="line",
        x0=-0.5, x1=len(hires_per_recruiter["Name"]) - 0.5,
        y0=goal, y1=goal,
        line=dict(color="red", width=3, dash="dash")
    )
    fig_goal.update_yaxes(title="Aantal aannames", range=[0, max(hires_per_recruiter["Candidate employment"].max(), goal) + 1])
    fig_goal.update_layout(
        title=f"Aannames per Recruiter in {geselecteerde_maand_label} (doel = {goal})",
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig_goal, use_container_width=True)

    st.markdown("---")  # duidelijke scheiding

    # =========================
    # WEEK-BASED KPI'S
    # =========================
    st.subheader(f"Weekanalyse: {week_label}")

    # KPI metrics
    totaal_introductions = filtered_data["Introductions"].sum()
    totaal_employment = filtered_data["Candidate employment"].sum()
    gem_tijd_tot_aannemen = filtered_data["Time to hire"].replace(0, pd.NA).mean()
    conversie_percentage = (totaal_employment / totaal_introductions * 100) if totaal_introductions > 0 else 0

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_col1.metric("Totale Introductions", totaal_introductions)
    kpi_col2.metric("Aangenomen kandidaten", totaal_employment)
    kpi_col3.metric("Conversie %", f"{conversie_percentage:.1f}%")
    kpi_col4.metric("Gem. Time to Hire (dagen)", f"{gem_tijd_tot_aannemen:.1f}" if pd.notna(gem_tijd_tot_aannemen) else "N/B")

    # Introductions vs Employment
    st.subheader("Introductions vs. Aangenomen Kandidaten per Recruiter")
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Bar(
        x=filtered_data["Name"],
        y=filtered_data["Introductions"],
        name="Introductions",
        marker_color="#636EFA"
    ))
    trend_fig.add_trace(go.Bar(
        x=filtered_data["Name"],
        y=filtered_data["Candidate employment"],
        name="Aangenomen kandidaten",
        marker_color="#00CC96"
    ))
    trend_fig.update_layout(
        barmode='group',
        title="Weekanalyse: Introductions en Aangenomen Kandidaten",
        height=400
    )
    st.plotly_chart(trend_fig, use_container_width=True)

    # Funnel chart
    st.subheader("Wervingsfunnel (van activiteiten tot aanname)")
    funnel_fig = go.Figure(go.Funnel(
        y=["InMails + Cold Calls", "Responses", "Introductions", "Aangenomen kandidaten"],
        x=[
            (filtered_data["InMails"].sum() + filtered_data["Cold call"].sum()),
            filtered_data["Responses (accepted or declined)"].sum(),
            totaal_introductions,
            totaal_employment
        ],
        textinfo="value+percent previous"
    ))
    funnel_fig.update_layout(title="Funnel: van contact tot aanname")
    st.plotly_chart(funnel_fig, use_container_width=True)

