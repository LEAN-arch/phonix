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
st.set_page_config(page_title="RedShield AI: Phoenix v3.2.2", layout="wide", initial_sidebar_state="expanded")
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
        if 'env_factors' not in st.session_state:
            avg_pop_density = self.dm.zones_gdf['population'].mean() if not self.dm.zones_gdf.empty else 50000
            st.session_state['env_factors'] = EnvFactors(is_holiday=False, weather="Clear", traffic_level=1.0, major_event=False, population_density=avg_pop_density, air_quality_index=50.0, heatwave_alert=False)

    def render(self):
        """Main rendering loop for the Streamlit application."""
        st.title("RedShield AI: Phoenix v3.2.2")
        st.markdown("**Real-time Emergency Response Optimization Platform**")

        self._render_sidebar()

        env_factors = st.session_state['env_factors']
        historical_data = st.session_state['historical_data']

        with st.spinner("Analyzing data and generating predictions..."):
            current_incidents = self.dm.get_current_incidents(env_factors)
            kpi_df = self.engine.generate_kpis(historical_data, env_factors, current_incidents)
            forecast_df = self.engine.generate_forecast(historical_data, env_factors, kpi_df)
            allocations = self.engine.generate_allocation_recommendations(kpi_df, forecast_df)
        
        col1, col2 = st.columns([3, 2])
        with col1:
            st.header("Risk Heatmap & Allocations")
            self._render_map(kpi_df, allocations)
        with col2:
            st.header("Ambulance Allocation")
            alloc_df = pd.DataFrame(list(allocations.items()), columns=['Zone', 'Recommended Units']).set_index('Zone')
            st.dataframe(alloc_df, use_container_width=True)
            
            st.header("Key Risk Indicators")
            display_kpis = kpi_df[['Zone', 'Ensemble Risk Score', 'Violence Clustering Score', 'Medical Surge Score', 'Spatial Spillover Risk']].set_index('Zone')
            st.dataframe(display_kpis.style.format("{:.2f}").background_gradient(cmap='Reds', subset=['Ensemble Risk Score']), use_container_width=True)

        st.header("Risk Forecast")
        if not forecast_df.empty:
            forecast_pivot = forecast_df.pivot(index='Zone', columns='Horizon (Hours)', values='Combined Risk')
            st.dataframe(forecast_pivot.style.format("{:.2f}").background_gradient(cmap='YlOrRd', axis=1), use_container_width=True)

    def _render_sidebar(self):
        st.sidebar.header("Environmental Factors")
        env = st.session_state['env_factors']
        is_holiday = st.sidebar.checkbox("Is Holiday", value=env.is_holiday)
        weather = st.sidebar.selectbox("Weather", ["Clear", "Rain", "Fog"], index=["Clear", "Rain", "Fog"].index(env.weather))
        traffic = st.sidebar.slider("Traffic Level", 0.5, 3.0, env.traffic_level, 0.1)
        event = st.sidebar.checkbox("Major Event", value=env.major_event)
        aqi = st.sidebar.slider("Air Quality Index (AQI)", 0.0, 500.0, env.air_quality_index, 5.0)
        heatwave = st.sidebar.checkbox("Heatwave Alert", value=env.heatwave_alert)

        new_env = EnvFactors(is_holiday, weather, traffic, event, env.population_density, aqi, heatwave)
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
            with st.spinner("Generating Report..."):
                kpi_df = self.engine.generate_kpis(st.session_state.historical_data, st.session_state.env_factors, self.dm.get_current_incidents(st.session_state.env_factors))
                forecast_df = self.engine.generate_forecast(st.session_state.historical_data, st.session_state.env_factors, kpi_df)
                allocations = self.engine.generate_allocation_recommendations(kpi_df, forecast_df)
                pdf_buffer = ReportGenerator.generate_pdf_report(kpi_df, forecast_df, allocations, st.session_state.env_factors)
            if pdf_buffer.getvalue():
                st.sidebar.download_button(label="Download PDF", data=pdf_buffer, file_name=f"RedShield_Report_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf")
    
    def _render_map(self, kpi_df: pd.DataFrame, allocations: dict):
        if self.dm.zones_gdf.empty or kpi_df.empty:
            st.warning("No zone data available for map.")
            return

        try:
            map_gdf = self.dm.zones_gdf.join(kpi_df.set_index('Zone'))
            # --- BUG FIX STARTS HERE ---
            # Reset the index so 'name' becomes a column that Folium can find.
            map_gdf.reset_index(inplace=True)
            # --- BUG FIX ENDS HERE ---

            center = map_gdf.unary_union.centroid
            m = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="cartodbpositron")

            choropleth = folium.Choropleth(
                geo_data=map_gdf.to_json(),
                data=map_gdf,
                # --- BUG FIX STARTS HERE ---
                # Use the correct column 'name' for the key and 'Ensemble Risk Score' for the value.
                columns=['name', 'Ensemble Risk Score'],
                # --- BUG FIX ENDS HERE ---
                key_on='feature.properties.name', # The key in the GeoJSON is now under properties
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name='Ensemble Risk Score'
            ).add_to(m)

            # Use GeoJsonTooltip for a cleaner implementation
            folium.GeoJsonTooltip(
                fields=["name", "Ensemble Risk Score"],
                aliases=["Zone:", "Risk Score:"],
                style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
            ).add_to(choropleth.geojson)
            
            st_folium(m, use_container_width=True, height=550)
        except Exception as e:
            logger.error(f"Failed to render map: {e}", exc_info=True)
            st.error(f"Error rendering map: {e}")

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
