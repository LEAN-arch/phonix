# main.py
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import json
from datetime import datetime
import logging
import warnings
from pathlib import Path

# Import from our refactored modules
from core import DataManager, PredictiveAnalyticsEngine, EnvFactors
from utils import load_config, ReportGenerator

# --- System Setup ---
st.set_page_config(page_title="RedShield AI: Phoenix v3.2.8", layout="wide", initial_sidebar_state="expanded")
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Setup logging
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(name)s - %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler("logs/redshield_phoenix.log")])
logger = logging.getLogger(__name__)

# --- Main Dashboard Class ---
class Dashboard:
    """Handles the rendering of the Streamlit user interface."""
    def __init__(self, dm: DataManager, engine: PredictiveAnalyticsEngine):
        self.dm = dm
        self.engine = engine
        self.config = dm.config

        if 'historical_data' not in st.session_state:
            st.session_state['historical_data'] = []
            
        # --- ENHANCEMENT: Initialize session_state with the new, detailed EnvFactors ---
        if 'env_factors' not in st.session_state:
            avg_pop_density = self.dm.zones_gdf['population'].mean() if not self.dm.zones_gdf.empty else 50000
            st.session_state['env_factors'] = EnvFactors(
                # Original factors
                is_holiday=False, weather="Clear", traffic_level=1.0, major_event=False, 
                population_density=avg_pop_density, air_quality_index=50.0, heatwave_alert=False,
                # New strategic factors with default values
                day_type='Weekday', 
                time_of_day='Midday', 
                public_event_type='None',
                hospital_divert_status=0.0, 
                police_activity='Normal', 
                school_in_session=True
            )

    def render(self):
        """Main rendering loop for the Streamlit application."""
        st.title("RedShield AI: Phoenix v3.2.8")
        st.markdown("**Real-time Emergency Response Optimization Platform**")

        self._render_sidebar()

        env_factors = st.session_state['env_factors']
        historical_data = st.session_state['historical_data']

        with st.spinner("Analyzing data and generating predictions..."):
            current_incidents = self.dm.get_current_incidents(env_factors)
            kpi_df = self.engine.generate_kpis(historical_data, env_factors, current_incidents)
            forecast_df = self.engine.generate_forecast(historical_data, env_factors, kpi_df)
            allocations = self.engine.generate_allocation_recommendations(kpi_df, forecast_df)
        
        st.session_state['kpi_df'] = kpi_df
        st.session_state['forecast_df'] = forecast_df
        st.session_state['allocations'] = allocations

        col1, col2 = st.columns([3, 2])
        with col1:
            st.header("Risk Heatmap & Allocations")
            self._render_map(kpi_df, allocations)
        with col2:
            st.header("Ambulance Allocation")
            alloc_df = pd.DataFrame(list(allocations.items()), columns=['Zone', 'Recommended Units']).set_index('Zone')
            st.dataframe(alloc_df, use_container_width=True)
            
            st.header("Key Risk Indicators")
            # --- ENHANCEMENT: Display the new, more advanced Integrated_Risk_Score ---
            if not kpi_df.empty and all(c in kpi_df.columns for c in ['Zone', 'Integrated_Risk_Score', 'Violence Clustering Score', 'Medical Surge Score', 'Spatial Spillover Risk']):
                display_kpis = kpi_df[['Zone', 'Integrated_Risk_Score', 'Violence Clustering Score', 'Medical Surge Score', 'Spatial Spillover Risk']].set_index('Zone')
                st.dataframe(display_kpis.style.format("{:.3f}").background_gradient(cmap='Reds', subset=['Integrated_Risk_Score']), use_container_width=True)

        st.header("Risk Forecast")
        if not forecast_df.empty:
            forecast_pivot = forecast_df.pivot(index='Zone', columns='Horizon (Hours)', values='Combined Risk')
            st.dataframe(forecast_pivot.style.format("{:.3f}").background_gradient(cmap='YlOrRd', axis=1), use_container_width=True)

        self._render_explanations(kpi_df)

    def _render_sidebar(self):
        st.sidebar.header("Environmental Factors")
        env = st.session_state['env_factors']
        
        is_holiday = st.sidebar.checkbox("Is Holiday", value=env.is_holiday, key="is_holiday_sb")
        weather = st.sidebar.selectbox("Weather", ["Clear", "Rain", "Fog"], index=["Clear", "Rain", "Fog"].index(env.weather), key="weather_sb")
        traffic = st.sidebar.slider("General Traffic Level", 0.5, 3.0, env.traffic_level, 0.1, key="traffic_sb", help="A general multiplier for traffic congestion.")
        aqi = st.sidebar.slider("Air Quality Index (AQI)", 0.0, 500.0, env.air_quality_index, 5.0, key="aqi_sb")
        heatwave = st.sidebar.checkbox("Heatwave Alert", value=env.heatwave_alert, key="heatwave_sb")
        
        # --- ENHANCEMENT: New section for detailed strategic factors ---
        st.sidebar.subheader("Detailed Contextual Factors")
        
        day_type = st.sidebar.selectbox("Day Type", ['Weekday', 'Friday', 'Weekend'], index=['Weekday', 'Friday', 'Weekend'].index(env.day_type))
        time_of_day = st.sidebar.selectbox("Time of Day", ['Morning Rush', 'Midday', 'Evening Rush', 'Night'], index=['Morning Rush', 'Midday', 'Evening Rush', 'Night'].index(env.time_of_day))
        
        public_event_type = st.sidebar.selectbox("Public Event Type", ['None', 'Sporting Event', 'Concert/Festival', 'Public Protest'], index=['None', 'Sporting Event', 'Concert/Festival', 'Public Protest'].index(env.public_event_type))
        major_event = public_event_type != 'None' # Keep the old 'major_event' boolean in sync

        hospital_divert_status = st.sidebar.slider("Hospital Divert Status (%)", 0, 100, int(env.hospital_divert_status * 100), 5, help="Percentage of hospitals on diversion, indicating system strain.") / 100.0

        police_activity = st.sidebar.selectbox("Police Activity Level", ['Low', 'Normal', 'High'], index=['Low', 'Normal', 'High'].index(env.police_activity))
        
        school_in_session = st.sidebar.checkbox("School In Session", value=env.school_in_session)

        # --- ENHANCEMENT: Create new EnvFactors object with all variables (old and new) ---
        new_env = EnvFactors(
            is_holiday=is_holiday, weather=weather, traffic_level=traffic, major_event=major_event,
            population_density=env.population_density, air_quality_index=aqi, heatwave_alert=heatwave,
            day_type=day_type, time_of_day=time_of_day, public_event_type=public_event_type,
            hospital_divert_status=hospital_divert_status, police_activity=police_activity, school_in_session=school_in_session
        )

        if new_env != st.session_state['env_factors']:
            st.session_state['env_factors'] = new_env
            st.rerun()

        st.sidebar.header("Data Management")
        uploaded_file = st.sidebar.file_uploader("Upload Historical Incidents (JSON)", type=["json"])
        if uploaded_file:
            try:
                st.session_state['historical_data'] = json.load(uploaded_file)
                st.sidebar.success(f"{len(st.session_state['historical_data'])} historical records loaded.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error loading data: {e}")

        st.sidebar.header("Report Generation")
        if st.sidebar.button("Generate & Download PDF Report", use_container_width=True):
            kpi_df = st.session_state.get('kpi_df', pd.DataFrame())
            forecast_df = st.session_state.get('forecast_df', pd.DataFrame())
            allocations = st.session_state.get('allocations', {})
            with st.spinner("Generating Report..."):
                pdf_buffer = ReportGenerator.generate_pdf_report(kpi_df, forecast_df, allocations, st.session_state.env_factors)
            if pdf_buffer.getvalue():
                st.sidebar.download_button(label="Download PDF Report", data=pdf_buffer, file_name=f"RedShield_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
            else:
                st.sidebar.error("Report generation failed."); st.sidebar.info("Check logs for details.")
    
    def _render_map(self, kpi_df: pd.DataFrame, allocations: dict):
        if self.dm.zones_gdf.empty or kpi_df.empty:
            st.warning("No zone data available for map.")
            return
        try:
            map_gdf = self.dm.zones_gdf.join(kpi_df.set_index('Zone'))
            map_gdf.reset_index(inplace=True)
            center = map_gdf.unary_union.centroid
            m = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="cartodbpositron")
            
            # --- ENHANCEMENT: Use the new, more advanced Integrated_Risk_Score for the map ---
            risk_col = 'Integrated_Risk_Score' if 'Integrated_Risk_Score' in map_gdf.columns else 'Ensemble Risk Score'
            
            choropleth = folium.Choropleth(
                geo_data=map_gdf.to_json(), data=map_gdf,
                columns=['name', risk_col], key_on='feature.properties.name',
                fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.2, legend_name='Integrated Risk Score'
            ).add_to(m)
            tooltip_html = "<b>Zone:</b> {name}<br><b>Risk Score:</b> {risk:.3f}"
            map_gdf['tooltip'] = map_gdf.apply(lambda row: tooltip_html.format(name=row['name'], risk=row[risk_col]), axis=1)
            folium.GeoJson(map_gdf, style_function=lambda x: {'color': 'black', 'weight': 1, 'fillOpacity': 0},
                           tooltip=folium.features.GeoJsonTooltip(fields=['tooltip'], labels=False)).add_to(m)
            st_folium(m, use_container_width=True, height=550)
        except Exception as e:
            logger.error(f"Failed to render map: {e}", exc_info=True)
            st.error(f"Error rendering map: {e}")

    def _render_explanations(self, kpi_df: pd.DataFrame):
        st.header("Methodology & KPI Explanations")

        with st.expander("View Detailed KPI Breakdown"):
            st.markdown("This table shows the raw values for the key analytical indicators explained below.")
            
            # --- ENHANCEMENT: Expand the list of detailed KPIs to include the new advanced metrics ---
            detailed_kpi_cols = [
                'Zone', 'Incident Probability', 'Expected Incident Volume', 'Risk Entropy', 
                'Anomaly Score', 'Spatial Spillover Risk', 'Resource Adequacy Index', 
                'Chaos Sensitivity Score', 'Bayesian Confidence Score', 'Information Value Index',
                'STGP_Risk', 'HMM_State_Risk', 'GNN_Structural_Risk', 'Game_Theory_Tension',
                'Integrated_Risk_Score'
            ]
            
            # Filter to only columns that actually exist in the dataframe to avoid errors
            available_cols = [col for col in detailed_kpi_cols if col in kpi_df.columns]
            
            if not kpi_df.empty and len(available_cols) > 1:
                st.dataframe(kpi_df[available_cols].set_index('Zone').style.format("{:.3f}"))
            else:
                st.warning("Could not display detailed KPIs. One or more required columns are missing.")

        with st.expander("View Predictive Methodologies"):
            st.subheader("High-Level Goal & Strategy")
            st.markdown("""
            The fundamental goal of RedShield AI is to move emergency response from a purely *reactive* model ("wait for a call, then dispatch") to a *proactive, predictive* one ("anticipate where calls will occur, and pre-position resources").

            To achieve this, the system employs an **Ensemble Modeling** strategy. Instead of relying on a single predictive algorithm, it uses multiple, diverse methodologies, each looking at the problem from a different angle. The final prediction is a "wisdom of the crowds" consensus from these individual models. This makes the system more robust, as the weakness of one model is offset by the strengths of others.
            """)

            st.subheader("Predictive Methodologies & Their Math")
            st.markdown("""
            **1. Hawkes Process (Self-Excitation Model)**
            - *What It Is:* A model for "self-exciting" events, famous in seismology for modeling aftershocks. It's used here to model incident clustering.
            - *Mathematical Concept:* The rate of events `λ(t)` is the sum of a background rate `μ` and the influence of all past events, which decays over time.
            > `λ(t) = μ + Σ [κ * exp(-β(t-tᵢ))]`
            - *Significance:* Perfect for modeling how one incident (e.g., a gang shooting) can trigger "aftershock" incidents (retaliation) in the same area. A high `Trauma Clustering Score` indicates a high probability of a follow-up incident.

            **2. SIR Model (Epidemiological Model)**
            - *What It Is:* A classic model for the spread of disease through a population (Susceptible, Infected, Recovered).
            - *Mathematical Concept:* The rate of new "infections" is proportional to the contact between Susceptible and Infected groups: `β * S * I`.
            - *Significance:* We model non-viral phenomena that "spread" due to shared environmental conditions. For example, a heatwave "infects" the vulnerable population with heatstroke. A high `Medical Surge Score` warns of a potential surge in calls related to environmental factors.

            **3. Graph Theory & The Laplacian Matrix**
            - *What It Is:* A model of the city's road network to quantify how risk "spills over" from one zone to its neighbors.
            - *Mathematical Concept:* The Graph Laplacian `L` is a matrix representing connectivity. Multiplying it by the risk vector shows how risk diffuses.
            > `Spillover = α * L * RiskVector`
            - *Significance:* An incident doesn't happen in a vacuum. A highway closure in one zone increases accident risk in adjacent zones. A high `Spatial Spillover Risk` is an early warning of imported risk.

            **4. Bayesian Network (Probabilistic "What-If" Engine)**
            - *What It Is:* A graph representing probabilistic relationships between variables (e.g., Weather, Holiday -> Incident Rate).
            - *Mathematical Concept:* It uses Bayes' Theorem to calculate the probability of an outcome given a set of evidence. `P(IncidentRate | Holiday=True, Weather=Rain)`.
            - *Significance:* Formalizes the intuition of experienced dispatchers (e.g., a holiday plus a heatwave is high risk), providing a robust baseline risk level based on the overall environment.

            **5. Chaos Theory (Lyapunov Exponent)**
            - *What It Is:* A measure of the "unpredictability" or "fragility" of a system.
            - *Mathematical Concept:* It measures how quickly the system's state changes in response to small perturbations. A positive exponent indicates chaos.
            - *Significance:* It measures the **volatility** of the 911 call system. A high `Chaos Sensitivity Score` warns that the system is unstable and a small event could trigger a disproportionately large cascade of incidents.
            """)

            st.subheader("Key Performance Indicator (KPI) Definitions")
            st.markdown("""
            - **Incident Probability `P(Et|Ft)`:** The calculated likelihood (0-1) of at least one incident occurring in a zone, based on the Bayesian network's assessment of environmental factors and historical priors.
            - **Expected Incident Volume:** A Poisson-based estimate of the *number* of incidents to expect in a zone during a given time window, derived from the base incident rate.
            - **Risk Entropy:** Measures the *uncertainty* of the spatial risk distribution. High entropy means risk is spread evenly and unpredictably across all zones. Low entropy means risk is concentrated in a few clear hotspots.
            - **Anomaly Score:** Based on the Kullback-Leibler divergence, this score is high when the *pattern* of current incidents (e.g., more trauma in a typically quiet zone) significantly deviates from the historical baseline.
            - **Spatial Spillover Risk:** The risk a zone "imports" from its neighbors. A quiet zone next to a chaotic one will have a high spillover risk.
            - **Resource Adequacy Index:** A system-wide score indicating if currently available units are sufficient to handle the total expected incident volume (`Available Units / Needed Units`). A value below 1.0 indicates a resource deficit.
            - **Chaos Sensitivity Score:** A measure of system volatility. A high score suggests the system is in a fragile state where small events could trigger large, unpredictable consequences.
            - **Bayesian Confidence Score:** A measure (0-1) of how certain the Bayesian network is about its `Incident Probability` prediction, derived from the variance of the posterior distribution.
            - **Information Value Index:** A proxy measure of how "actionable" the current set of predictions is. A high value (high standard deviation in risk scores) means there are clear, differentiable hotspots, making resource allocation decisions easier and more impactful.
            - **(Advanced) STGP Risk:** A proxy for a Spatiotemporal Gaussian Process, this score is high for zones that are geographically close to recent, high-severity incidents.
            - **(Advanced) HMM State Risk:** A proxy for a Hidden Markov Model, this score identifies zones in a latent high-risk 'state' based on a combination of volatility and resource strain.
            - **(Advanced) GNN Structural Risk:** A proxy for a Graph Neural Network, this score reflects a zone's intrinsic vulnerability based on its structural importance (centrality) within the city's road network.
            - **(Advanced) Game Theory Tension:** Models the competition for resources. A zone's tension is high when its expected incident demand is a large fraction of the total system-wide demand.
            - **(Ultimate) Integrated Risk Score:** The final, most sophisticated risk metric, combining the original ensemble score with the four advanced analytical KPIs for a comprehensive assessment.
            """)

def main():
    """Main function to run the application."""
    try:
        config = load_config()
        data_manager = DataManager(config)
        engine = PredictiveAnalyticsEngine(data_manager, config)
        dashboard = Dashboard(data_manager, engine)
        dashboard.render()
    except Exception as e:
        logger.error(f"A fatal error occurred in the application: {e}", exc_info=True)
        st.error(f"A fatal application error occurred: {e}. Please check logs and configuration file.")

if __name__ == "__main__":
    main()
