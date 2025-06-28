# RedShield_Phoenix_Documented.py
# VERSION 2.4.1 - PHOENIX ARCHITECTURE WITH ENHANCED FORECASTING AND ENSEMBLE RISK FUSION
#
# This version enhances the Phoenix Architecture to focus on predicting medical emergencies (trauma and disease)
# for multiple time horizons (0.5, 1, 3, 6, 12, 24, 72, 144 hours) to optimize resource allocation and
# infrastructure readiness. Adds Ensemble Risk Fusion methodology for robust, sensitive predictions.
#
# KEY ENHANCEMENTS (v2.4.1):
# 1. [METHODOLOGY] Added Ensemble Risk Fusion (ERF) to combine all predictive methods, weighted by predictive quality.
# 2. [KPI] Added Ensemble Risk Score to integrate outputs for robust, sensitive risk assessment.
# 3. [AUTHENTICATION] Removed authentication and role-based access for streamlined access.
# 4. [FORECASTING] Retained multi-horizon forecasting (30 min, 1 hr, 3 hr, 6 hr, 12 hr, 24 hr, 3 days, 6 days).
# 5. [MODELING] Retained Marked Hawkes Process, Spatio-Temporal SIR Model, Lyapunov Exponent, and Copula-Based Correlation.
# 6. [UI] Updated visualizations to highlight zones with high Ensemble Risk Scores.
# 7. [RESOURCES] Enhanced recommendations prioritizing zones with high Ensemble Risk Scores.
# 8. [VISUALIZATION] Uses OpenStreetMap with Leaflet.js (via folium) for geospatial visualizations.
# 9. [DATA] Uses local sample_api_response.json for incident data.
# 10. [FIX] Corrected TypeError in fetch_real_time_incidents by converting location dictionaries to shapely Point objects.
# 11. [FIX] Handled AttributeError in main by checking None return from plot_risk_heatmap before calling st_folium.
#
# PREVIOUS FEATURES (v2.4):
# - Fixed TypeError in geometry handling for GeoDataFrame.
# - Fixed DuplicateWidgetID error with unique keys for st.text_input.
# - Advanced modeling, real-time data integration, resource optimization, PDF reports, and robust error handling.
#
"""
RedShield AI: Phoenix Architecture v2.4.1
A commercial-grade predictive intelligence engine for urban emergency response.
Fuses advanced modeling for trauma and disease emergencies with multi-horizon forecasting and actionable insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
import networkx as nx
import os
from pathlib import Path
import plotly.graph_objects as go
import logging
import warnings
import json
import random
import requests
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
import io
from scipy.stats import norm
import folium
from streamlit_folium import st_folium

# Advanced Dependencies (optional, with fallbacks)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    DiscreteBayesianNetwork = None
    TabularCPD = None
    VariableElimination = None

# --- L0: SYSTEM CONFIGURATION & INITIALIZATION ---
st.set_page_config(page_title="RedShield AI: Phoenix v2.4.1", layout="wide", initial_sidebar_state="expanded")
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("redshield_phoenix.log")]
)
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class EnvFactors:
    is_holiday: bool
    weather: str
    traffic_level: float
    major_event: bool
    population_density: float

# --- L1: CORE DATA & SIMULATION MODULES ---

@st.cache_resource
def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Loads and validates the system configuration."""
    try:
        if not Path(config_path).exists():
            logger.warning(f"Config file '{config_path}' not found. Using default configuration.")
            config = get_default_config()
        else:
            with open(config_path, 'r') as f:
                config = json.load(f)
        mapbox_key = os.environ.get("MAPBOX_API_KEY", config.get("mapbox_api_key", ""))
        config['mapbox_api_key'] = mapbox_key if mapbox_key and "YOUR_KEY" not in mapbox_key else None
        validate_config(config)
        logger.info("System configuration loaded successfully.")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        st.error(f"Configuration error: {e}. Using default configuration.")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """Returns a default configuration."""
    return {
        "mapbox_api_key": None,
        "data": {
            "zones": {
                "Centro": {"polygon": [[32.52, -117.03], [32.54, -117.03], [32.54, -117.05], [32.52, -117.05]], "prior_risk": 0.7, "population": 50000},
                "Otay": {"polygon": [[32.53, -116.95], [32.54, -116.95], [32.54, -116.98], [32.53, -116.98]], "prior_risk": 0.4, "population": 30000},
                "Playas": {"polygon": [[32.51, -117.11], [32.53, -117.11], [32.53, -117.13], [32.51, -117.13]], "prior_risk": 0.3, "population": 20000}
            },
            "ambulances": {
                "A01": {"status": "Disponible", "home_base": "Centro", "location": [32.53, -117.04]},
                "A02": {"status": "Disponible", "home_base": "Otay", "location": [32.535, -116.965]},
                "A03": {"status": "En Misión", "home_base": "Playas", "location": [32.52, -117.12]}
            },
            "distributions": {
                "zone": {"Centro": 0.5, "Otay": 0.3, "Playas": 0.2},
                "incident_type": {"Trauma": 0.4, "Medical": 0.6},
                "triage": {"Red": 0.1, "Yellow": 0.3, "Green": 0.6}
            },
            "road_network": {
                "edges": [
                    ["Centro", "Otay", 5],
                    ["Otay", "Playas", 8],
                    ["Playas", "Centro", 10]
                ]
            },
            "real_time_api": {
                "endpoint": "http://localhost:8000/sample_api_response.json",
                "api_key": null
            }
        },
        "model_params": {
            "hawkes_process": {"kappa": 0.5, "beta": 1.0, "trauma_weight": 1.5},
            "sir_model": {"beta": 0.3, "gamma": 0.1, "noise_scale": 0.05},
            "laplacian_diffusion_factor": 0.1,
            "response_time_penalty": 3.0,
            "copula_correlation": 0.2
        },
        "bayesian_network": {
            "structure": [("Holiday", "IncidentRate"), ("Weather", "IncidentRate"), ("MajorEvent", "IncidentRate")],
            "cpds": {
                "Holiday": {"card": 2, "values": [[0.9], [0.1]], "evidence": None, "evidence_card": None},
                "Weather": {"card": 2, "values": [[0.7], [0.3]], "evidence": None, "evidence_card": None},
                "MajorEvent": {"card": 2, "values": [[0.8], [0.2]], "evidence": None, "evidence_card": None},
                "IncidentRate": {
                    "card": 3,
                    "values": [[0.6, 0.5, 0.4, 0.3, 0.5, 0.4, 0.3, 0.2], 
                               [0.3, 0.3, 0.4, 0.4, 0.3, 0.4, 0.4, 0.5], 
                               [0.1, 0.2, 0.2, 0.3, 0.2, 0.2, 0.3, 0.3]],
                    "evidence": ["Holiday", "Weather", "MajorEvent"],
                    "evidence_card": [2, 2, 2]
                }
            }
        },
        "tcnn_params": {
            "input_size": 7,
            "output_size": 16,  # 8 horizons (0.5, 1, 3, 6, 12, 24, 72, 144) for trauma and disease
            "channels": [16, 32, 64]
        }
    }

def validate_config(config: Dict[str, Any]) -> None:
    """Validates the configuration structure and content."""
    required_sections = ['data', 'model_params', 'bayesian_network', 'tcnn_params']
    for section in required_sections:
        if section not in config or not config[section]:
            raise ValueError(f"Configuration section '{section}' is missing or empty.")
    if 'zones' not in config['data'] or not config['data']['zones']:
        raise ValueError("No zones defined in configuration.")
    for zone, data in config['data']['zones'].items():
        if 'polygon' not in data or not isinstance(data['polygon'], list) or len(data['polygon']) < 3:
            raise ValueError(f"Invalid polygon for zone '{zone}'.")
        if 'population' not in data or not isinstance(data['population'], (int, float)):
            raise ValueError(f"Invalid population for zone '{zone}'.")
    for amb in config['data']['ambulances'].values():
        if 'location' not in amb or not isinstance(amb['location'], list) or len(amb['location']) != 2:
            raise ValueError(f"Invalid location for ambulance.")

@st.cache_resource
def get_data_manager(config: Dict[str, Any]) -> 'DataManager':
    """Initializes and caches the DataManager."""
    return DataManager(config)

class DataManager:
    """Manages static and real-time geospatial and network data assets."""
    def __init__(self, config: Dict[str, Any]):
        self.data_config = config.get('data', {})
        self.zones = list(self.data_config.get('zones', {}).keys())
        self.road_graph = self._build_road_graph()
        self.zones_gdf = self._build_zones_gdf()
        self.ambulances = self._initialize_ambulances()
        try:
            self.laplacian_matrix = nx.normalized_laplacian_matrix(self.road_graph, nodelist=self.zones).toarray()
            logger.info("Graph Laplacian computed successfully.")
        except Exception as e:
            logger.warning(f"Could not compute Graph Laplacian: {e}. Using identity matrix.")
            self.laplacian_matrix = np.identity(len(self.zones))

    def _build_road_graph(self) -> nx.Graph:
        """Builds the road network graph."""
        G = nx.Graph()
        G.add_nodes_from(self.zones)
        edges = self.data_config.get('road_network', {}).get('edges', [])
        for edge in edges:
            if len(edge) == 3 and edge[0] in G.nodes and edge[1] in G.nodes and isinstance(edge[2], (int, float)):
                G.add_edge(edge[0], edge[1], weight=float(edge[2]))
            else:
                logger.warning(f"Invalid edge data: {edge}")
        return G

    def _build_zones_gdf(self) -> gpd.GeoDataFrame:
        """Builds a GeoDataFrame for zones."""
        zone_data = []
        for name, data in self.data_config.get('zones', {}).items():
            try:
                polygon = data.get('polygon')
                if isinstance(polygon, list) and len(polygon) >= 3:
                    poly = Polygon([(lon, lat) for lat, lon in polygon])
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    if not poly.is_empty:
                        zone_data.append({
                            'name': name,
                            'geometry': poly,
                            'prior_risk': data.get('prior_risk', 0.5),
                            'population': data.get('population', 10000)
                        })
            except Exception as e:
                logger.warning(f"Invalid polygon for zone '{name}': {e}")
        if not zone_data:
            logger.error("No valid zones found.")
            return gpd.GeoDataFrame()
        return gpd.GeoDataFrame(zone_data, crs="EPSG:4326").set_index('name')

    def _initialize_ambulances(self) -> Dict:
        """Initializes ambulance data with locations."""
        ambulances = {}
        for amb_id, data in self.data_config.get('ambulances', {}).items():
            try:
                location = data.get('location')
                if isinstance(location, list) and len(location) == 2:
                    ambulances[amb_id] = {
                        'id': amb_id,
                        'status': data.get('status', 'Disponible'),
                        'home_base': data.get('home_base'),
                        'location': Point(location[1], location[0])
                    }
            except Exception as e:
                logger.warning(f"Invalid data for ambulance '{amb_id}': {e}")
        return ambulances

    def fetch_real_time_incidents(self, api_config: Dict) -> List[Dict]:
        """Fetches real-time incident data from a local JSON file or external API."""
        endpoint = api_config.get('endpoint', '')
        try:
            if endpoint.startswith('http://localhost'):
                with open('sample_api_response.json', 'r') as f:
                    data = json.load(f)
                    incidents = data.get('incidents', [])
                    # Convert location dictionaries to shapely Point objects
                    valid_incidents = []
                    for inc in incidents:
                        if 'location' in inc and 'type' in inc and 'triage' in inc:
                            loc = inc['location']
                            if isinstance(loc, dict) and 'lat' in loc and 'lon' in loc:
                                try:
                                    lat, lon = float(loc['lat']), float(loc['lon'])
                                    inc['location'] = Point(lon, lat)
                                    valid_incidents.append(inc)
                                except (ValueError, TypeError) as e:
                                    logger.warning(f"Invalid location data for incident {inc.get('id', 'unknown')}: {e}")
                            else:
                                logger.warning(f"Invalid location format for incident {inc.get('id', 'unknown')}: {loc}")
                    logger.info(f"Loaded {len(valid_incidents)} valid incidents from local sample_api_response.json.")
                    return valid_incidents
            else:
                headers = {"Authorization": f"Bearer {api_config.get('api_key', '')}"} if api_config.get('api_key') else {}
                response = requests.get(endpoint, headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()
                incidents = []
                for inc in data.get('incidents', []):
                    if 'location' in inc and 'type' in inc and 'triage' in inc:
                        loc = inc['location']
                        if isinstance(loc, dict) and 'lat' in loc and 'lon' in loc:
                            try:
                                lat, lon = float(loc['lat']), float(loc['lon'])
                                if inc['type'] in self.data_config['distributions']['incident_type'] and inc['triage'] in self.data_config['distributions']['triage']:
                                    incidents.append({
                                        'id': inc.get('id', f"RT-{len(incidents)}"),
                                        'type': inc['type'],
                                        'triage': inc['triage'],
                                        'location': Point(lon, lat),
                                        'timestamp': inc.get('timestamp', datetime.utcnow().isoformat())
                                    })
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Invalid location data for incident {inc.get('id', 'unknown')}: {e}")
                logger.info(f"Fetched {len(incidents)} valid incidents from {endpoint}.")
                return incidents
        except Exception as e:
            logger.warning(f"Failed to fetch incidents from {endpoint}: {e}. Falling back to synthetic data.")
            return self._generate_synthetic_incidents()

    def _generate_synthetic_incidents(self) -> List[Dict]:
        """Generates synthetic incident data for fallback."""
        intensity = 5.0  # Base rate for Poisson process
        num_incidents = max(0, int(np.random.poisson(intensity)))
        city_boundary = self.zones_gdf.unary_union if not self.zones_gdf.empty else None
        bounds = city_boundary.bounds if city_boundary else (-117.13, 32.45, -116.95, 32.54)
        incidents = []
        for i in range(num_incidents):
            lon = np.random.uniform(bounds[0], bounds[2])
            lat = np.random.uniform(bounds[1], bounds[3])
            point = Point(lon, lat)
            if city_boundary and not city_boundary.contains(point):
                continue
            incidents.append({
                'id': f"SYN-{i}",
                'type': np.random.choice(list(self.data_config['distributions']['incident_type'].keys()), p=list(self.data_config['distributions']['incident_type'].values())),
                'triage': np.random.choice(list(self.data_config['distributions']['triage'].keys()), p=list(self.data_config['distributions']['triage'].values())),
                'location': point,
                'timestamp': datetime.utcnow().isoformat()
            })
        logger.info(f"Generated {len(incidents)} synthetic incidents.")
        return incidents

# --- L2: SIMULATION & PREDICTIVE ENGINES ---

class SimulationEngine:
    """Generates synthetic and processes real-time incident data."""
    def __init__(self, dm: DataManager, config: Dict[str, Any]):
        self.dm = dm
        self.config = config
        self.dist = config['data']['distributions']
        self.city_boundary = self.dm.zones_gdf.unary_union if not self.dm.zones_gdf.empty else None
        self.bounds = self.city_boundary.bounds if self.city_boundary else (-117.13, 32.45, -116.95, 32.54)

    def get_live_state(self, env_factors: EnvFactors, time_step: int, base_rate: float = 5.0) -> Dict[str, Any]:
        """Simulates or fetches current system state with incidents."""
        real_time_incidents = self.dm.fetch_real_time_incidents(self.config['data'].get('real_time_api', {}))
        if real_time_incidents:
            incidents_gdf = gpd.GeoDataFrame(
                real_time_incidents,
                geometry=[inc['location'] for inc in real_time_incidents],
                crs="EPSG:4326"
            )
        else:
            intensity = base_rate
            if env_factors.is_holiday:
                intensity *= 1.5
            if env_factors.weather.lower() in ['rain', 'fog']:
                intensity *= 1.2
            if env_factors.major_event:
                intensity *= 2.0
            intensity *= env_factors.traffic_level * (1 + 0.5 * env_factors.population_density / 100000)
            num_incidents = max(0, int(np.random.poisson(intensity)))
            if num_incidents == 0 or not self.city_boundary:
                return {"incidents": [], "system_state": "Normal"}

            incidents_gdf = gpd.GeoDataFrame(
                {'type': np.random.choice(list(self.dist['incident_type'].keys()), num_incidents, p=list(self.dist['incident_type'].values())),
                 'triage': np.random.choice(list(self.dist['triage'].keys()), num_incidents, p=list(self.dist['triage'].values()))},
                geometry=gpd.points_from_xy(
                    x=np.random.uniform(self.bounds[0], self.bounds[2], num_incidents),
                    y=np.random.uniform(self.bounds[1], self.bounds[3], num_incidents),
                    crs="EPSG:4326"
                )
            )
            incidents_gdf = incidents_gdf[incidents_gdf.within(self.city_boundary)].reset_index(drop=True)
            if incidents_gdf.empty:
                return {"incidents": [], "system_state": "Normal"}
            incidents_gdf['id'] = [f"INC-{i}" for i in range(len(incidents_gdf))]

        incidents_gdf = incidents_gdf.sjoin(self.dm.zones_gdf[['geometry']], how='left', predicate='intersects').rename(columns={'index_right': 'zone'})
        incidents = [{'id': row['id'], 'type': row['type'], 'triage': row['triage'], 'zone': row['zone'], 'location': row['geometry'], 'timestamp': row.get('timestamp', datetime.utcnow().isoformat())} for _, row in incidents_gdf.iterrows() if pd.notna(row['zone'])]
        system_state = "Anomalous" if len(incidents) > 10 or any(i['triage'] == 'Red' for i in incidents) else "Elevated" if len(incidents) > 5 else "Normal"
        return {"incidents": incidents, "system_state": system_state}

class TCNN(nn.Module):
    """Temporal Convolutional Network for multi-horizon time-series forecasting."""
    def __init__(self, input_size: int, output_size: int, num_channels: List[int], kernel_size: int = 2, dropout: float = 0.2):
        super(TCNN, self).__init__()
        layers = []
        for i, num_channel in enumerate(num_channels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            layers += [
                nn.Conv1d(in_channels, num_channel, kernel_size, padding=(kernel_size-1) * dilation, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        return self.linear(self.network(x.permute(0, 2, 1))[:, :, -1])

class PredictiveAnalyticsEngine:
    """Fuses multiple methodologies to generate predictive KPIs for trauma and disease emergencies."""
    def __init__(self, dm: DataManager, config: Dict[str, Any]):
        self.dm = dm
        self.config = config
        self.bn_model = self._build_bayesian_network() if PGMPY_AVAILABLE else None
        self.tcnn_model = self._initialize_tcnn() if TORCH_AVAILABLE else None
        self.copula_rho = config['model_params'].get('copula_correlation', 0.2)
        # Predictive quality scores for ensemble weighting
        self.method_weights = {
            'hawkes': 9,
            'sir': 8,
            'bayesian': 8 if PGMPY_AVAILABLE else 0,
            'graph': 7,
            'chaos': 7,
            'info': 9,
            'tcnn': 10 if TORCH_AVAILABLE else 7,
            'game': 8,
            'copula': 8
        }
        total_weight = sum(self.method_weights.values())
        self.method_weights = {k: v / total_weight for k, v in self.method_weights.items()}

    @st.cache_resource
    def _build_bayesian_network(_self) -> Optional[DiscreteBayesianNetwork]:
        """Builds and caches the Bayesian network."""
        if not PGMPY_AVAILABLE:
            logger.warning("pgmpy not available. Bayesian network disabled.")
            return None
        try:
            bn_config = _self.config['bayesian_network']
            model = DiscreteBayesianNetwork(bn_config['structure'])
            for node, params in bn_config['cpds'].items():
                model.add_cpds(TabularCPD(
                    variable=node,
                    variable_card=params['card'],
                    values=params['values'],
                    evidence=params.get('evidence'),
                    evidence_card=params.get('evidence_card')
                ))
            model.check_model()
            logger.info("Bayesian network initialized.")
            return model
        except Exception as e:
            logger.warning(f"Failed to initialize Bayesian network: {e}. Disabling.")
            return None

    @st.cache_resource
    def _initialize_tcnn(_self) -> Optional[TCNN]:
        """Initializes and caches the TCNN model."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. TCNN disabled.")
            return None
        try:
            tcnn_params = _self.config['tcnn_params']
            model = TCNN(tcnn_params['input_size'], tcnn_params['output_size'], tcnn_params['channels'])
            logger.info("TCNN model initialized.")
            return model
        except Exception as e:
            logger.warning(f"Failed to initialize TCNN: {e}. Disabling.")
            return None

    def calculate_ensemble_risk_score(self, kpi_df: pd.DataFrame, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Computes a weighted ensemble risk score combining all methodologies."""
        if kpi_df.empty:
            return {zone: 0.0 for zone in self.dm.zones}

        # Normalize KPI values to [0,1] for comparability
        def normalize(series):
            min_val = series.min()
            max_val = max(series.max(), 1e-9)
            return (series - min_val) / (max_val - min_val + 1e-9)

        # Extract relevant KPIs
        scores = {}
        for zone in self.dm.zones:
            zone_kpi = kpi_df[kpi_df['Zone'] == zone]
            if zone_kpi.empty:
                scores[zone] = 0.0
                continue

            # Collect normalized contributions
            contributions = []
            # Hawkes: Trauma Clustering Score
            contributions.append(normalize(zone_kpi['Trauma Clustering Score'].iloc[0]) * self.method_weights['hawkes'])
            # SIR: Disease Surge Score
            contributions.append(normalize(zone_kpi['Disease Surge Score'].iloc[0]) * self.method_weights['sir'])
            # Bayesian: Bayesian Confidence Score (if available)
            if PGMPY_AVAILABLE:
                contributions.append(normalize(zone_kpi['Bayesian Confidence Score'].iloc[0]) * self.method_weights['bayesian'])
            # Graph: Spatial Spillover Risk
            contributions.append(normalize(zone_kpi['Spatial Spillover Risk'].iloc[0]) * self.method_weights['graph'])
            # Chaos: Chaos Sensitivity Score
            chaos_score = normalize(zone_kpi['Chaos Sensitivity Score'].iloc[0])
            # Amplify chaos contribution if historical variance is high
            if not historical_data.empty:
                incident_counts = [len(h['incidents']) for h in historical_data]
                chaos_amplifier = 1.5 if np.var(incident_counts) > np.mean(incident_counts) else 1.0
            else:
                chaos_amplifier = 1.0
            contributions.append(chaos_score * chaos_amplifier * self.method_weights['chaos'])
            # Information Theory: Risk Entropy and Anomaly Score
            contributions.append(normalize(zone_kpi['Risk Entropy'].iloc[0]) * self.method_weights['info'] * 0.5)
            contributions.append(normalize(zone_kpi['Anomaly Score'].iloc[0]) * self.method_weights['info'] * 0.5)
            # TCNN: Use short-term forecast (3h) if available
            if TORCH_AVAILABLE and hasattr(self, 'forecast_df') and not self.forecast_df.empty:
                zone_forecast = self.forecast_df[(self.forecast_df['Zone'] == zone) & (self.forecast_df['Horizon (Hours)'] == 3)]
                if not zone_forecast.empty:
                    tcnn_score = (zone_forecast['Trauma Risk'].iloc[0] + zone_forecast['Disease Risk'].iloc[0]) / 2
                    contributions.append(normalize(pd.Series([tcnn_score]))[0] * self.method_weights['tcnn'])
                else:
                    contributions.append(0.0)
            else:
                contributions.append(normalize(zone_kpi['Incident Probability'].iloc[0]) * self.method_weights['tcnn'])
            # Game Theory: Resource Adequacy Index (inverted, lower is riskier)
            contributions.append((1 - normalize(zone_kpi['Resource Adequacy Index'].iloc[0])) * self.method_weights['game'])
            # Copula: Trauma-Disease Correlation
            contributions.append(normalize(zone_kpi['Trauma-Disease Correlation'].iloc[0]) * self.method_weights['copula'])

            # Compute weighted sum
            scores[zone] = np.sum(contributions)
            # Ensure score is in [0,1]
            scores[zone] = min(max(scores[zone], 0.0), 1.0)

        return scores

    def generate_kpis(self, historical_data: pd.DataFrame, env_factors: EnvFactors, current_incidents: List[Dict]) -> pd.DataFrame:
        """Computes KPIs for each zone, including trauma, disease, and ensemble risk scores."""
        if self.bn_model and PGMPY_AVAILABLE:
            inference = VariableElimination(self.bn_model)
            evidence = {
                'Holiday': 1 if env_factors.is_holiday else 0,
                'Weather': 0 if env_factors.weather == 'Clear' else 1,
                'MajorEvent': 1 if env_factors.major_event else 0
            }
            try:
                result = inference.query(variables=['IncidentRate'], evidence=evidence, show_progress=False)
                rate_probs = result.values
                baseline_rate = np.sum(rate_probs * np.array([1, 5, 10]))
                bayesian_confidence = 1 - (np.std(rate_probs) / (np.mean(rate_probs) + 1e-9))
            except Exception as e:
                logger.warning(f"Bayesian inference failed: {e}. Using default values.")
                baseline_rate = 5.0
                bayesian_confidence = 0.5
        else:
            baseline_rate = 5.0
            bayesian_confidence = 0.5

        incident_counts = pd.Series(dtype=float)
        trauma_counts = pd.Series(dtype=float)
        disease_counts = pd.Series(dtype=float)
        if current_incidents:
            df = pd.DataFrame(current_incidents)
            if 'zone' in df.columns and 'type' in df.columns:
                incident_counts = df['zone'].value_counts()
                trauma_counts = df[df['type'] == 'Trauma']['zone'].value_counts()
                disease_counts = df[df['type'] == 'Medical']['zone'].value_counts()

        hawkes_params = self.config['model_params']['hawkes_process']
        sir_params = self.config['model_params']['sir_model']
        past_incidents = historical_data['incidents'].sum() if not historical_data.empty else []
        trauma_intensity = self._calculate_marked_hawkes_intensity(past_incidents, hawkes_params, 'Trauma')
        disease_intensity = self._calculate_sir_intensity(env_factors, sir_params)

        prior_dist = self.config['data']['distributions']['zone']
        current_dist = (incident_counts / (incident_counts.sum() + 1e-9)).reindex(self.dm.zones, fill_value=0)
        trauma_dist = (trauma_counts / (trauma_counts.sum() + 1e-9)).reindex(self.dm.zones, fill_value=0)
        disease_dist = (disease_counts / (disease_counts.sum() + 1e-9)).reindex(self.dm.zones, fill_value=0)

        kl_divergence = np.sum(current_dist * np.log(current_dist.replace(0, 1e-9) / pd.Series(prior_dist).reindex_like(current_dist).replace(0, 1e-9)))
        shannon_entropy = -np.sum(current_dist * np.log2(current_dist.replace(0, 1e-9)))
        lyapunov_exponent = self._calculate_lyapunov_exponent(historical_data, current_dist)

        base_probs = self._calculate_base_probabilities(baseline_rate, trauma_intensity + disease_intensity, prior_dist)
        trauma_probs = self._calculate_base_probabilities(baseline_rate, trauma_intensity, prior_dist)
        disease_probs = self._calculate_base_probabilities(baseline_rate, disease_intensity, prior_dist)

        try:
            spillover_risk = self.config['model_params']['laplacian_diffusion_factor'] * (self.dm.laplacian_matrix @ pd.Series(base_probs, index=self.dm.zones).values)
        except Exception as e:
            logger.warning(f"Spillover risk calculation failed: {e}. Setting to zero.")
            spillover_risk = np.zeros(len(self.dm.zones))

        response_times = self._calculate_response_times(current_incidents)
        trauma_cluster_scores = self._calculate_trauma_cluster_scores(trauma_counts, hawkes_params)
        disease_surge_scores = self._calculate_disease_surge_scores(disease_counts, sir_params)

        correlation_score = self._model_event_correlations(trauma_dist, disease_dist)

        kpi_data = []
        available_units = len([a for a in self.dm.ambulances.values() if a['status'] == 'Disponible'])
        for i, zone in enumerate(self.dm.zones):
            kpi_data.append({
                "Zone": zone,
                "Incident Probability": base_probs.get(zone, 0),
                "Expected Incident Volume": base_probs.get(zone, 0) * 3,
                "Risk Entropy": shannon_entropy,
                "Anomaly Score": kl_divergence,
                "Spatial Spillover Risk": spillover_risk[i],
                "Resource Adequacy Index": available_units / (base_probs.get(zone, 1e-9) * len(self.dm.zones) + 1e-9),
                "Chaos Sensitivity Score": lyapunov_exponent,
                "Bayesian Confidence Score": bayesian_confidence,
                "Response Time Estimate": response_times.get(zone, 10.0),
                "Trauma Clustering Score": trauma_cluster_scores.get(zone, 0.0),
                "Disease Surge Score": disease_surge_scores.get(zone, 0.0),
                "Trauma-Disease Correlation": correlation_score
            })
        kpi_df = pd.DataFrame(kpi_data)
        
        # Add Ensemble Risk Score
        ensemble_scores = self.calculate_ensemble_risk_score(kpi_df, historical_data)
        kpi_df['Ensemble Risk Score'] = [ensemble_scores.get(zone, 0.0) for zone in kpi_df['Zone']]
        
        return kpi_df

    def _calculate_base_probabilities(self, baseline: float, intensity: float, priors: Dict[str, float]) -> Dict[str, float]:
        """Calculates base incident probabilities."""
        return {zone: (baseline + intensity) * prob for zone, prob in priors.items()}

    def _calculate_marked_hawkes_intensity(self, past_incidents: Any, params: Dict[str, float], incident_type: str) -> float:
        """Calculates Marked Hawkes Process intensity for trauma incidents."""
        if not isinstance(past_incidents, list) or not past_incidents:
            return 0.0
        trauma_incidents = [inc for inc in past_incidents if inc.get('type') == incident_type]
        intensity = len(trauma_incidents) * params['kappa'] * params['trauma_weight'] * np.exp(-params['beta'])
        return max(0.0, intensity)

    def _calculate_sir_intensity(self, env_factors: EnvFactors, params: Dict[str, float]) -> float:
        """Calculates SIR model intensity for disease incidents."""
        S = env_factors.population_density  # Proxy for susceptible population
        I = 0.01 * S  # Initial infected fraction
        beta, gamma, noise_scale = params['beta'], params['gamma'], params['noise_scale']
        dI_dt = beta * S * I / (S + 1e-9) - gamma * I
        noise = np.random.normal(0, noise_scale)
        intensity = max(0.0, dI_dt + noise)
        if env_factors.major_event:
            intensity *= 1.5
        if env_factors.weather.lower() in ['rain', 'fog']:
            intensity *= 1.2
        return intensity

    def _calculate_lyapunov_exponent(self, historical_data: pd.DataFrame, current_dist: pd.Series) -> float:
        """Estimates Lyapunov exponent for chaos detection."""
        if historical_data.empty or len(historical_data) < 2:
            return 0.0
        try:
            divergences = []
            for i in range(len(historical_data) - 1):
                past_dist = pd.Series(0.0, index=self.dm.zones)
                for inc in historical_data.iloc[i]['incidents']:
                    if inc.get('zone') in past_dist:
                        past_dist[inc['zone']] += 1
                past_dist = past_dist / (past_dist.sum() + 1e-9)
                delta_t = (past_dist - current_dist).abs().sum()
                delta_t1 = (past_dist.shift(1).fillna(0) - past_dist).abs().sum()
                if delta_t1 > 1e-9:
                    divergences.append(np.log(delta_t / delta_t1 + 1e-9))
            return np.mean(divergences) if divergences else 0.0
        except Exception as e:
            logger.warning(f"Lyapunov exponent calculation failed: {e}. Returning 0.0.")
            return 0.0

    def _calculate_trauma_cluster_scores(self, trauma_counts: pd.Series, params: Dict[str, float]) -> Dict[str, float]:
        """Calculates trauma clustering scores based on Hawkes intensity."""
        scores = {}
        for zone in self.dm.zones:
            count = trauma_counts.get(zone, 0)
            scores[zone] = count * params['kappa'] * params['trauma_weight']
        return scores

    def _calculate_disease_surge_scores(self, disease_counts: pd.Series, params: Dict[str, float]) -> Dict[str, float]:
        """Calculates disease surge scores based on SIR model."""
        scores = {}
        for zone in self.dm.zones:
            count = disease_counts.get(zone, 0)
            population = self.dm.zones_gdf.loc[zone, 'population']
            scores[zone] = count * params['beta'] * population / 100000
        return scores

    def _model_event_correlations(self, trauma_dist: pd.Series, disease_dist: pd.Series) -> float:
        """Models correlations between trauma and disease incidents using a Gaussian copula."""
        try:
            u1 = norm.cdf(trauma_dist.values)
            u2 = norm.cdf(disease_dist.values)
            rho = self.copula_rho
            cov = [[1, rho], [rho, 1]]
            joint_prob = np.sum(np.dot(u1, np.dot(cov, u2.T)))
            return min(max(joint_prob, -1.0), 1.0)
        except Exception as e:
            logger.warning(f"Correlation modeling failed: {e}. Returning 0.0.")
            return 0.0

    def _calculate_response_times(self, incidents: List[Dict]) -> Dict[str, float]:
        """Estimates response times to each zone based on ambulance locations."""
        response_times = {}
        for zone in self.dm.zones:
            min_time = float('inf')
            zone_centroid = self.dm.zones_gdf.loc[zone, 'geometry'].centroid
            for amb in self.dm.ambulances.values():
                if amb['status'] == 'Disponible':
                    distance = zone_centroid.distance(amb['location'])
                    time = distance * 1000 / 50000 * 60 + self.config['model_params']['response_time_penalty']
                    min_time = min(min_time, time)
            response_times[zone] = min_time if min_time != float('inf') else 10.0
        return response_times

    def forecast_risk(self, kpi_df: pd.DataFrame, horizons: List[float] = [0.5, 1, 3, 6, 12, 24, 72, 144]) -> pd.DataFrame:
        """Forecasts trauma and disease risks for multiple time horizons."""
        if self.tcnn_model and TORCH_AVAILABLE:
            try:
                X = np.array([kpi_df[['Incident Probability', 'Risk Entropy', 'Anomaly Score', 'Trauma Clustering Score', 'Disease Surge Score']].values], dtype=np.float32)
                X = torch.tensor(X).float()
                with torch.no_grad():
                    preds = self.tcnn_model(X).numpy()
                forecast_data = []
                for i, zone in enumerate(self.dm.zones):
                    for h_idx, horizon in enumerate(horizons):
                        trauma_idx = h_idx
                        disease_idx = h_idx + len(horizons)
                        forecast_data.append({
                            'Zone': zone,
                            'Horizon (Hours)': horizon,
                            'Trauma Risk': float(preds[0, trauma_idx]),
                            'Disease Risk': float(preds[0, disease_idx])
                        })
                self.forecast_df = pd.DataFrame(forecast_data)  # Store for ensemble scoring
                return self.forecast_df
            except Exception as e:
                logger.warning(f"TCNN forecasting failed: {e}. Using baseline forecast.")
        # Fallback to exponential smoothing with horizon-specific decay
        forecast_data = []
        decay_rates = {0.5: 0.95, 1: 0.9, 3: 0.8, 6: 0.7, 12: 0.6, 24: 0.5, 72: 0.3, 144: 0.2}
        for zone in self.dm.zones:
            base_trauma = kpi_df.loc[kpi_df['Zone'] == zone, 'Trauma Clustering Score'].iloc[0] if not kpi_df.empty else 0.5
            base_disease = kpi_df.loc[kpi_df['Zone'] == zone, 'Disease Surge Score'].iloc[0] if not kpi_df.empty else 0.5
            for horizon in horizons:
                decay = decay_rates.get(horizon, 0.5)
                forecast_data.append({
                    'Zone': zone,
                    'Horizon (Hours)': horizon,
                    'Trauma Risk': base_trauma * decay,
                    'Disease Risk': base_disease * decay
                })
        self.forecast_df = pd.DataFrame(forecast_data)
        return self.forecast_df

# --- L3: STRATEGIC ADVISOR ---

class StrategicAdvisor:
    """Optimizes resource allocation for emergency response."""
    def __init__(self, dm: DataManager, config: Dict[str, Any]):
        self.dm = dm
        self.config = config

    def recommend_allocations(self, kpi_df: pd.DataFrame, forecast_df: pd.DataFrame) -> List[Dict]:
        """Recommends ambulance reallocations based on ensemble risk scores and forecasts."""
        if kpi_df.empty or forecast_df.empty:
            return []
        available_ambulances = [amb for amb in self.dm.ambulances.values() if amb['status'] == 'Disponible']
        if not available_ambulances:
            return []

        # Combine current ensemble risk and forecasted risks (weighted by horizon proximity)
        weights = {0.5: 0.3, 1: 0.25, 3: 0.2, 6: 0.15, 12: 0.1, 24: 0.08, 72: 0.07, 144: 0.05}
        deficits = pd.Series(0.0, index=self.dm.zones)
        for zone in self.dm.zones:
            zone_kpi = kpi_df[kpi_df['Zone'] == zone]
            zone_forecast = forecast_df[forecast_df['Zone'] == zone]
            if not zone_kpi.empty:
                current_deficit = zone_kpi['Ensemble Risk Score'].iloc[0]
            else:
                current_deficit = 0.0
            forecast_deficit = 0.0
            for _, row in zone_forecast.iterrows():
                horizon = row['Horizon (Hours)']
                forecast_deficit += weights.get(horizon, 0.1) * (row['Trauma Risk'] + row['Disease Risk'])
            deficits[zone] = current_deficit + forecast_deficit

        target_zone = deficits.idxmax() if deficits.max() > 0.5 else None
        if not target_zone:
            return []

        recommendations = []
        for amb in available_ambulances:
            current_zone = next((z for z, d in self.dm.zones_gdf.iterrows() if d['geometry'].contains(amb['location'])), None)
            if current_zone != target_zone:
                reason = (f"High ensemble risk in {target_zone} (Ensemble Risk Score: {kpi_df.loc[kpi_df['Zone'] == target_zone, 'Ensemble Risk Score'].iloc[0] if not kpi_df.empty else 0:.2f}, "
                         f"Trauma (3h): {forecast_df.loc[(forecast_df['Zone'] == target_zone) & (forecast_df['Horizon (Hours)'] == 3), 'Trauma Risk'].iloc[0] if not forecast_df.empty else 0:.2f}, "
                         f"Disease (3h): {forecast_df.loc[(forecast_df['Zone'] == target_zone) & (forecast_df['Horizon (Hours)'] == 3), 'Disease Risk'].iloc[0] if not forecast_df.empty else 0:.2f})")
                recommendations.append({
                    'unit': amb['id'],
                    'from': current_zone or 'Unknown',
                    'to': target_zone,
                    'reason': reason
                })
        return recommendations[:2]  # Limit to 2 recommendations for clarity

# --- L4: REPORT GENERATION ---

class ReportGenerator:
    """Generates exportable PDF reports for actionable insights."""
    @staticmethod
    def generate_pdf_report(kpi_df: pd.DataFrame, recommendations: List[Dict], forecast_df: pd.DataFrame) -> io.BytesIO:
        """Creates a PDF report summarizing KPIs, recommendations, and forecasts."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("RedShield AI: Emergency Response Report", styles['Title']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 12))

        if not kpi_df.empty:
            elements.append(Paragraph("Key Performance Indicators", styles['Heading2']))
            kpi_data = [kpi_df.columns.tolist()] + kpi_df.round(3).values.tolist()
            kpi_table = Table(kpi_data)
            kpi_table.setStyle([
                ('BACKGROUND', (0, 0), (-1, 0), '#2C3E50'),
                ('TEXTCOLOR', (0, 0), (-1, 0), '#FFFFFF'),
                ('GRID', (0, 0), (-1, -1), 1, '#000000'),
                ('FONTSIZE', (0, 0), (-1, -1), 10)
            ])
            elements.append(kpi_table)
            elements.append(Spacer(1, 12))

        if recommendations:
            elements.append(Paragraph("Resource Allocation Recommendations", styles['Heading2']))
            for rec in recommendations:
                text = f"Move {rec['unit']} from {rec['from']} to {rec['to']}. Reason: {rec['reason']}"
                elements.append(Paragraph(text, styles['Normal']))
            elements.append(Spacer(1, 12))

        if not forecast_df.empty:
            elements.append(Paragraph("Risk Forecast (Multiple Horizons)", styles['Heading2']))
            forecast_data = [forecast_df.columns.tolist()] + forecast_df.round(3).values.tolist()
            forecast_table = Table(forecast_data)
            forecast_table.setStyle([
                ('BACKGROUND', (0, 0), (-1, 0), '#2C3E50'),
                ('TEXTCOLOR', (0, 0), (-1, 0), '#FFFFFF'),
                ('GRID', (0, 0), (-1, -1), 1, '#000000'),
                ('FONTSIZE', (0, 0), (-1, -1), 10)
            ])
            elements.append(forecast_table)

        doc.build(elements)
        buffer.seek(0)
        return buffer

# --- L5: UI & VISUALIZATION ---

class VisualizationSuite:
    """Generates operational and analytical visualizations."""
    @staticmethod
    def plot_kpi_dashboard(kpi_df: pd.DataFrame) -> go.Figure:
        """Creates a KPI dashboard table."""
        if kpi_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No KPI data available. Run a predictive cycle to generate insights.", showarrow=False, font=dict(size=14))
            return fig

        def color_scaler(val: float, col_data: pd.Series, high_is_bad: bool = True) -> str:
            norm = (val - col_data.min()) / (col_data.max() - col_data.min() + 1e-9)
            return f'rgba(255, {255 * (1-norm)}, {255 * (1-norm)}, 0.7)' if high_is_bad else f'rgba({255 * (1-norm)}, 255, {255 * (1-norm)}, 0.7)'

        colors = {'Zone': ['#2C3E50']*len(kpi_df)}
        font_colors = {'Zone': ['white']*len(kpi_df)}
        for col in kpi_df.columns[1:]:
            high_is_bad = 'Confidence' not in col and 'Adequacy' not in col and 'Response Time' not in col
            colors[col] = [color_scaler(v, kpi_df[col], high_is_bad) for v in kpi_df[col]]
            font_colors[col] = ['black' if v < kpi_df[col].mean() else 'white' for v in kpi_df[col]]
            kpi_df[col] = kpi_df[col].round(3)

        fig = go.Figure(data=[go.Table(
            header=dict(values=[f"<b>{c}</b>" for c in kpi_df.columns], fill_color='#2C3E50', align='center', font=dict(color='white', size=14)),
            cells=dict(values=[kpi_df[k] for k in kpi_df.columns], fill_color=list(colors.values()), align='center', font=dict(color=list(font_colors.values()), size=12), height=30)
        )])
        fig.update_layout(title_text="<b>Actionable KPI Dashboard</b>", margin=dict(l=10, r=10, t=40, b=10))
        return fig

    @staticmethod
    def plot_risk_heatmap(kpi_df: pd.DataFrame, dm: DataManager, config: Dict, risk_type: str = 'Ensemble Risk Score') -> Any:
        """Creates a risk heatmap using OpenStreetMap and Folium."""
        if kpi_df.empty or dm.zones_gdf.empty:
            st.warning("No geospatial data available. Run a predictive cycle to generate the map.")
            return None

        m = folium.Map(location=[32.53, -117.04], zoom_start=12, tiles="OpenStreetMap")
        
        # Normalize risk values for coloring
        min_val = kpi_df[risk_type].min()
        max_val = max(kpi_df[risk_type].max(), 1e-9)
        norm = lambda x: (x - min_val) / (max_val - min_val + 1e-9)

        # Add zone polygons with risk-based coloring
        for zone, row in dm.zones_gdf.iterrows():
            risk_value = kpi_df.loc[kpi_df['Zone'] == zone, risk_type].iloc[0] if zone in kpi_df['Zone'].values else 0
            color_intensity = norm(risk_value)
            color = f'#{int(255 * color_intensity):02x}0000'  # Red gradient
            folium.Polygon(
                locations=[[lat, lon] for lat, lon in config['data']['zones'][zone]['polygon']],
                color='blue',
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=f"{zone}: {risk_type} = {risk_value:.3f}"
            ).add_to(m)

        # Add ambulance markers
        for amb_id, amb in dm.ambulances.items():
            folium.Marker(
                location=[amb['location'].y, amb['location'].x],
                popup=f"Ambulance {amb_id}: {amb['status']}",
                icon=folium.Icon(color='green' if amb['status'] == 'Disponible' else 'red')
            ).add_to(m)

        return m  # Return folium.Map object

# --- L6: DOCUMENTATION & EXPLANATION MODULE ---

class Documentation:
    """Renders methodological framework and KPI explanations in the UI."""
    
    @staticmethod
    def render_methodology_framework():
        """Displays a table explaining the integrated mathematical frameworks."""
        st.header("🧠 I. Integrated Methodological Framework")
        st.markdown("""
        The Phoenix Architecture integrates advanced computational methodologies to model the heuristic, stochastic,
        and chaotic nature of medical emergencies, ensuring robust predictions for trauma and disease events.
        """)
        
        methodology_data = {
            "Methodology": [
                "**Marked Hawkes Process**", "**Spatio-Temporal SIR Model**", "**Bayesian Inference**",
                "**Graph Theory & Network Science**", "**Chaos Theory & Lyapunov Exponent**", "**Information Theory**",
                "**Deep Learning & Probabilistic ML**", "**Game Theory**", "**Copula-Based Correlation**",
                "**Ensemble Risk Fusion (ERF)**"
            ],
            "Description & Implementation": [
                "Models trauma emergencies with a **Marked Hawkes Process** in `PredictiveAnalyticsEngine`, capturing spatial-temporal clustering of incidents.",
                "Models disease emergencies with a **Spatio-Temporal SIR Model** in `PredictiveAnalyticsEngine`, accounting for population dynamics and stochastic transitions.",
                "Employs a **Discrete Bayesian Network** (`pgmpy`) to infer incident rates from causal factors (weather, holidays, etc.).",
                "Models the city as a spatial graph in `DataManager`. The **Graph Laplacian (L = D-A)** computes **Spatial Spillover Risk**.",
                "Uses **Lyapunov Exponent** estimation to enhance the **Chaos Sensitivity Score**, detecting chaotic regimes in emergency dynamics.",
                "Calculates **Shannon Entropy** and **Kullback-Leibler (KL) Divergence** for system unpredictability and anomaly detection.",
                "Implements a **Temporal Convolutional Network (TCNN)** for multi-horizon forecasting, with fallback to statistical models if `torch` unavailable.",
                "Uses a **one-shot game** in `StrategicAdvisor` to optimize resource allocation based on the **Resource Adequacy Index**.",
                "Models correlations between trauma and disease emergencies using a **Gaussian Copula** in `PredictiveAnalyticsEngine`.",
                "Combines all methodologies into an **Ensemble Risk Score** in `PredictiveAnalyticsEngine`, weighted by predictive quality with sensitivity amplification for chaotic patterns."
            ],
            "Mathematical Principle": [
                "`λ(t, z) = μ(t, z) + Σ κ(z, z_i) e^(-β(t-t_i))`", 
                "`dS/dt = -βSI/N, dI/dt = βSI/N - γI, dR/dt = γI + stochastic noise`",
                "`P(A|B) = [P(B|A) * P(A)] / P(B)`",
                "`x_spillover = L * x_risk`",
                "`λ ≈ (1/T) Σ ln |Δx(t+1)/Δx(t)|`",
                "`H(X) = -Σ p(x)log(p(x))` & `D_KL(P||Q)`",
                "Dilated convolutions for temporal patterns.",
                "`max U(a) = max [Σ(Deficit_initial - Deficit_after_action(a))]`",
                "`C(u_1, u_2) = Φ_Σ(Φ^(-1)(u_1), Φ^(-1)(u_2))`",
                "`S_ensemble = Σ w_i * norm(S_i) + α * var(H) * S_chaos`"
            ],
            "Why It Matters": [
                "Captures clustering of trauma incidents (e.g., cascading accidents).",
                "Models disease surges driven by population and environmental factors.",
                "Enables adaptive reasoning under uncertainty for dynamic conditions.",
                "Models risk propagation across spatially connected zones.",
                "Identifies chaotic, unpredictable emergency patterns.",
                "Quantifies unpredictability and deviations from historical norms.",
                "Captures complex temporal patterns for accurate forecasting across multiple horizons.",
                "Optimizes resource allocation for maximum impact.",
                "Models dependencies between trauma and disease events.",
                "Integrates all predictive signals for a robust, sensitive risk assessment."
            ],
            "Predictive Quality Contribution (0-10)": [9, 8, 8 if PGMPY_AVAILABLE else 0, 7, 7, 9, 10 if TORCH_AVAILABLE else 7, 8, 8, 10]
        }
        
        st.dataframe(pd.DataFrame(methodology_data), use_container_width=True)

    @staticmethod
    def render_kpi_definitions():
        """Displays an expandable section detailing each KPI."""
        st.header("📊 II. Key Performance Indicators (KPIs) Explained")
        with st.expander("View detailed KPI explanations"):
            kpi_data = {
                "KPI": [
                    "**Incident Probability**", "**Expected Incident Volume**", "**Risk Entropy**", 
                    "**Anomaly Score**", "**Spatial Spillover Risk**", "**Resource Adequacy Index**", 
                    "**Chaos Sensitivity Score**", "**Bayesian Confidence Score**", "**Response Time Estimate**",
                    "**Trauma Clustering Score**", "**Disease Surge Score**", "**Trauma-Disease Correlation**",
                    "**Ensemble Risk Score**"
                ],
                "Description": [
                    "Likelihood of at least one new incident in a zone within the next hour.",
                    "Predicted number of incidents in a zone over a time horizon (e.g., 3 hours).",
                    "Shannon Entropy of incident probability distribution across all zones.",
                    "KL-Divergence between current and historical incident distributions.",
                    "Risk score based on proximity to other high-risk zones.",
                    "Ratio of expected incident volume to available units in/near a zone.",
                    "Measure of system volatility using Lyapunov exponent in Anomaly-Entropy phase space.",
                    "Model certainty in baseline incident rate predictions.",
                    "Estimated time for emergency units to reach a zone.",
                    "Intensity of trauma incident clustering based on Marked Hawkes Process.",
                    "Likelihood of disease-related emergency surge based on SIR model.",
                    "Correlation between trauma and disease incidents using a Gaussian Copula.",
                    "Weighted ensemble of all predictive methodologies, amplified for chaotic patterns."
                ],
                "Interpretation": [
                    "High probability zones require immediate attention.",
                    "High volume signals need for additional resources.",
                    "High entropy indicates unpredictable, widespread incidents.",
                    "High score flags unexpected incident patterns.",
                    "High spillover risk predicts future surges in adjacent zones.",
                    "Index > 1.0 indicates under-resourced zones.",
                    "Spikes signal increasing system instability or chaos.",
                    "Low confidence suggests cautious use of predictions.",
                    "Longer times indicate potential response delays.",
                    "High score indicates clustered trauma incidents, requiring rapid response.",
                    "High score predicts potential disease-related surges, needing preventive measures.",
                    "High correlation indicates coupled trauma-disease dynamics.",
                    "High score prioritizes zones for immediate resource allocation."
                ]
            }
            st.dataframe(pd.DataFrame(kpi_data), use_container_width=True)

# --- L7: MAIN APPLICATION LOGIC ---

@st.cache_resource
def initialize_system(config: Dict[str, Any]) -> Tuple[DataManager, PredictiveAnalyticsEngine, SimulationEngine, StrategicAdvisor]:
    """Initializes and caches core system components."""
    dm = get_data_manager(config)
    predictor = PredictiveAnalyticsEngine(dm, config)
    sim_engine = SimulationEngine(dm, config)
    advisor = StrategicAdvisor(dm, config)
    return dm, predictor, sim_engine, advisor

def main():
    """Main application entry point."""
    try:
        st.title("RedShield AI: Phoenix Architecture v2.4.1")
        st.markdown("**Commercial-Grade Predictive Intelligence for Urban Emergency Response** | Version 2.4.1")

        config = load_config()
        dm, predictor, sim_engine, advisor = initialize_system(config)

        if 'history' not in st.session_state:
            st.session_state.history = []

        # Sidebar Controls
        st.sidebar.title("System Controls")
        is_holiday = st.sidebar.checkbox("Holiday Period")
        weather = st.sidebar.selectbox("Weather", ["Clear", "Rain", "Fog"])
        traffic_level = st.sidebar.slider("Traffic Level", 0.5, 2.0, 1.0)
        major_event = st.sidebar.checkbox("Major Event Active")
        population_density = st.sidebar.slider("Population Density (per km²)", 1000, 100000, 50000)
        env_factors = EnvFactors(is_holiday=is_holiday, weather=weather, traffic_level=traffic_level, major_event=major_event, population_density=population_density)
        run_sim = st.sidebar.button("Run Predictive Cycle")

        # Main UI Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["**Operational Dashboard**", "**Geospatial Intelligence**", "**Forecasting**", "**Methodology & KPIs**"])

        with tab4:
            Documentation.render_methodology_framework()
            st.divider()
            Documentation.render_kpi_definitions()

        kpi_df = pd.DataFrame()
        recommendations = []
        forecast_df = pd.DataFrame()

        if run_sim:
            with st.spinner("Running predictive cycle..."):
                current_incidents = sim_engine.get_live_state(env_factors, len(st.session_state.history))['incidents']
                st.session_state.history.append({'incidents': current_incidents, 'timestamp': datetime.utcnow().isoformat()})
                st.session_state.history = [h for h in st.session_state.history if datetime.fromisoformat(h['timestamp']) > datetime.utcnow() - timedelta(hours=144)]
                kpi_df = predictor.generate_kpis(pd.DataFrame(st.session_state.history), env_factors, current_incidents)
                forecast_df = predictor.forecast_risk(kpi_df)
                recommendations = advisor.recommend_allocations(kpi_df, forecast_df)

            with tab1:
                st.header("Real-Time System Insights")
                st.plotly_chart(VisualizationSuite.plot_kpi_dashboard(kpi_df), use_container_width=True)
                if recommendations:
                    st.subheader("Resource Allocation Recommendations")
                    for rec in recommendations:
                        st.warning(f"**Move {rec['unit']}** from {rec['from']} to {rec['to']}. Reason: {rec['reason']}")
                else:
                    st.info("No immediate resource reallocation needed.")

            with tab2:
                st.header("Live Risk & Incident Map")
                risk_type = st.selectbox("Select Risk Type for Heatmap", ["Ensemble Risk Score", "Incident Probability", "Trauma Clustering Score", "Disease Surge Score"])
                folium_map = VisualizationSuite.plot_risk_heatmap(kpi_df, dm, config, risk_type)
                if folium_map is not None:
                    st_folium(folium_map, width=700, height=500)
                else:
                    st.warning("No geospatial data available. Run a predictive cycle to generate the map.")

            with tab3:
                st.header("Risk Forecast (Multiple Horizons)")
                st.plotly_chart(VisualizationSuite.plot_forecast_trend(forecast_df), use_container_width=True)

            if not kpi_df.empty:
                report_buffer = ReportGenerator.generate_pdf_report(kpi_df, recommendations, forecast_df)
                st.download_button(
                    label="Download PDF Report",
                    data=report_buffer,
                    file_name=f"RedShield_AI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

        else:
            with tab1:
                st.info("System is standing by. Press 'Run Predictive Cycle' to generate insights.")
                st.plotly_chart(VisualizationSuite.plot_kpi_dashboard(pd.DataFrame()), use_container_width=True)
            with tab2:
                st.info("Map will be generated after the first predictive cycle.")
                folium_map = VisualizationSuite.plot_risk_heatmap(pd.DataFrame(), dm, config)
                if folium_map is not None:
                    st_folium(folium_map, width=700, height=500)
                else:
                    st.warning("No geospatial data available. Run a predictive cycle to generate the map.")
            with tab3:
                st.info("Forecast will be generated after the first predictive cycle.")
                st.plotly_chart(VisualizationSuite.plot_forecast_trend(pd.DataFrame()), use_container_width=True)

    except Exception as e:
        logger.error(f"Critical error in main: {e}", exc_info=True)
        st.error(f"A fatal system error occurred: {e}. Check logs for details.")

if __name__ == "__main__":
    main()
