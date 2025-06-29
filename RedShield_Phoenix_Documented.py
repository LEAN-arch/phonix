import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import networkx as nx
import os
from pathlib import Path
import plotly.graph_objects as go
import logging
import warnings
import json
import requests
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
import io
from scipy.stats import norm
import folium
from streamlit_folium import st_folium

# --- Project-specific imports ---
from models import TCNN, TORCH_AVAILABLE

# --- Optional dependency check for pgmpy ---
try:
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    class DiscreteBayesianNetwork: pass
    class TabularCPD: pass
    class VariableElimination: pass

# --- L0: SYSTEM CONFIGURATION & INITIALIZATION ---
st.set_page_config(page_title="RedShield AI: Phoenix v3.1.0", layout="wide", initial_sidebar_state="expanded")
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Setup logging
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/redshield_phoenix.log")
    ]
)
logger = logging.getLogger(__name__)

# --- Constants ---
POPULATION_DENSITY_NORMALIZATION = 100000.0

@dataclass(frozen=True)
class EnvFactors:
    is_holiday: bool
    weather: str
    traffic_level: float
    major_event: bool
    population_density: float
    air_quality_index: float
    heatwave_alert: bool

@dataclass(frozen=True)
class ZoneAttributes:
    name: str
    geometry: Polygon
    prior_risk: float
    population: float
    crime_rate_modifier: float

@st.cache_resource
def load_config(config_path: str = "config.json") -> Dict[str, any]:
    try:
        if not Path(config_path).exists():
            logger.warning(f"Config file '{config_path}' not found. Generating and using default configuration.")
            config = get_default_config()
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
        else:
            with open(config_path, 'r') as f:
                config = json.load(f)

        mapbox_key = os.environ.get("MAPBOX_API_KEY", config.get("mapbox_api_key", ""))
        config['mapbox_api_key'] = mapbox_key if mapbox_key and "YOUR_KEY" not in mapbox_key else None

        validate_config(config)
        logger.info("System configuration loaded and validated successfully.")
        return config
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to load or validate config: {e}. Falling back to default configuration.", exc_info=True)
        st.warning(f"Configuration error: {e}. Using default configuration.")
        return get_default_config()

def get_default_config() -> Dict[str, any]:
    return {
        "mapbox_api_key": None,
        "forecast_horizons_hours": [0.5, 1, 3, 6, 12, 24, 72, 144],
        "data": {
            "zones": {
                "Centro": {
                    "polygon": [[32.52, -117.03], [32.54, -117.03], [32.54, -117.05], [32.52, -117.05]],
                    "prior_risk": 0.7,
                    "population": 50000,
                    "crime_rate_modifier": 1.2
                },
                "Otay": {
                    "polygon": [[32.53, -116.95], [32.54, -116.95], [32.54, -116.98], [32.53, -116.98]],
                    "prior_risk": 0.4,
                    "population": 30000,
                    "crime_rate_modifier": 0.8
                },
                "Playas": {
                    "polygon": [[32.51, -117.11], [32.53, -117.11], [32.53, -117.13], [32.51, -117.13]],
                    "prior_risk": 0.3,
                    "population": 20000,
                    "crime_rate_modifier": 1.0
                }
            },
            "ambulances": {
                "A01": {"status": "Disponible", "home_base": "Centro", "location": [32.53, -117.04]},
                "A02": {"status": "Disponible", "home_base": "Otay", "location": [32.535, -116.965]},
                "A03": {"status": "En Misión", "home_base": "Playas", "location": [32.52, -117.12]}
            },
            "distributions": {
                "zone": {"Centro": 0.5, "Otay": 0.3, "Playas": 0.2},
                "incident_type": {
                    "Trauma-Violence": 0.2,
                    "Trauma-Accident": 0.2,
                    "Medical-Chronic": 0.4,
                    "Medical-Acute": 0.2
                },
                "triage": {"Red": 0.1, "Yellow": 0.3, "Green": 0.6}
            },
            "road_network": {"edges": [["Centro", "Otay", 5], ["Otay", "Playas", 8], ["Playas", "Centro", 10]]},
            "real_time_api": {"endpoint": "sample_api_response.json", "api_key": None}
        },
        "model_params": {
            "hawkes_process": {
                "kappa": 0.5,
                "beta": 1.0,
                "trauma_weight": 1.5,
                "violence_weight": 1.8,
                "aqi_multiplier": 1.5
            },
            "sir_model": {"beta": 0.3, "gamma": 0.1, "noise_scale": 0.05},
            "laplacian_diffusion_factor": 0.1,
            "response_time_penalty": 3.0,
            "copula_correlation": 0.2,
            "ensemble_weights": {
                "hawkes": 9,
                "sir": 8,
                "bayesian": 8,
                "graph": 7,
                "chaos": 7,
                "info": 9,
                "tcnn": 10,
                "game": 8,
                "copula": 8,
                "violence": 9,
                "accident": 8,
                "medical": 8
            },
            "chaos_amplifier": 1.5,
            "fallback_forecast_decay_rates": {
                "0.5": 0.95, "1": 0.9, "3": 0.8, "6": 0.7, "12": 0.6,
                "24": 0.5, "72": 0.3, "144": 0.2
            },
            "allocation_forecast_weights": {
                "0.5": 0.3, "1": 0.25, "3": 0.2, "6": 0.15, "12": 0.1,
                "24": 0.08, "72": 0.07, "144": 0.05
            }
        },
        "bayesian_network": {
            "structure": [
                ("Holiday", "IncidentRate"),
                ("Weather", "IncidentRate"),
                ("MajorEvent", "IncidentRate"),
                ("AirQuality", "IncidentRate"),
                ("Heatwave", "IncidentRate")
            ],
            "cpds": {
                "Holiday": {"card": 2, "values": [[0.9], [0.1]], "evidence": None, "evidence_card": None},
                "Weather": {"card": 2, "values": [[0.7], [0.3]], "evidence": None, "evidence_card": None},
                "MajorEvent": {"card": 2, "values": [[0.8], [0.2]], "evidence": None, "evidence_card": None},
                "AirQuality": {"card": 2, "values": [[0.8], [0.2]], "evidence": None, "evidence_card": None},
                "Heatwave": {"card": 2, "values": [[0.9], [0.1]], "evidence": None, "evidence_card": None},
                "IncidentRate": {
                    "card": 3,
                    "values": [
                        [0.6, 0.5, 0.4, 0.3, 0.5, 0.4, 0.3, 0.2] * 4,
                        [0.3, 0.3, 0.4, 0.4, 0.3, 0.4, 0.4, 0.5] * 4,
                        [0.1, 0.2, 0.2, 0.3, 0.2, 0.2, 0.3, 0.3] * 4
                    ],
                    "evidence": ["Holiday", "Weather", "MajorEvent", "AirQuality", "Heatwave"],
                    "evidence_card": [2, 2, 2, 2, 2]
                }
            }
        },
        "tcnn_params": {
            "input_size": 8,
            "output_size": 24,
            "channels": [16, 32, 64],
            "kernel_size": 2,
            "dropout": 0.2
        }
    }

def validate_config(config: Dict[str, any]) -> None:
    required_sections = ['data', 'model_params', 'bayesian_network', 'tcnn_params']
    for section in required_sections:
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Configuration section '{section}' is missing or invalid.")
    
    zones = config.get('data', {}).get('zones', {})
    if not zones:
        raise ValueError("No zones defined in configuration.")
        
    for zone, data in zones.items():
        if 'polygon' not in data or not isinstance(data['polygon'], list) or len(data['polygon']) < 3:
            raise ValueError(f"Invalid polygon for zone '{zone}'.")
        if 'population' not in data or not isinstance(data['population'], (int, float)) or data['population'] <= 0:
            raise ValueError(f"Invalid population for zone '{zone}'.")
        if 'crime_rate_modifier' not in data or not isinstance(data['crime_rate_modifier'], (int, float)):
            logger.warning(f"Invalid or missing crime_rate_modifier for zone '{zone}'. Setting to default (1.0).")
            data['crime_rate_modifier'] = 1.0
        elif data['crime_rate_modifier'] <= 0:
            logger.warning(f"Non-positive crime_rate_modifier ({data['crime_rate_modifier']}) for zone '{zone}'. Setting to default (1.0).")
            data['crime_rate_modifier'] = 1.0
            
    for amb_id, amb_data in config.get('data', {}).get('ambulances', {}).items():
        if 'location' not in amb_data or not isinstance(amb_data['location'], list) or len(amb_data['location']) != 2:
            raise ValueError(f"Invalid location for ambulance '{amb_id}'.")
        if 'home_base' not in amb_data or amb_data['home_base'] not in zones:
            raise ValueError(f"Ambulance '{amb_id}' has an invalid home_base '{amb_data.get('home_base')}'.")

@st.cache_resource
def get_data_manager(config: Dict[str, any]) -> 'DataManager':
    return DataManager(config)

class DataManager:
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.data_config = config['data']
        self.zones = list(self.data_config['zones'].keys())
        self.zones_gdf = self._build_zones_gdf()
        self.road_graph = self._build_road_graph()
        self.ambulances = self._initialize_ambulances()
        try:
            self.laplacian_matrix = nx.normalized_laplacian_matrix(self.road_graph, nodelist=self.zones).toarray()
            logger.info("Graph Laplacian computed successfully.")
        except Exception as e:
            logger.warning(f"Could not compute Graph Laplacian: {e}. Using identity matrix as fallback.")
            self.laplacian_matrix = np.identity(len(self.zones))

    def _build_road_graph(self) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(self.zones)
        edges = self.data_config.get('road_network', {}).get('edges', [])
        for u, v, weight in edges:
            if u in G.nodes and v in G.nodes and isinstance(weight, (int, float)):
                G.add_edge(u, v, weight=float(weight))
            else:
                logger.warning(f"Skipping invalid edge data: {[u, v, weight]}")
        return G

    def _build_zones_gdf(self) -> gpd.GeoDataFrame:
        zone_data = []
        for name, data in self.data_config['zones'].items():
            try:
                poly = Polygon([(lon, lat) for lat, lon in data['polygon']])
                if not poly.is_valid:
                    poly = poly.buffer(0)
                zone_data.append({
                    'name': name,
                    'geometry': poly,
                    'prior_risk': data.get('prior_risk', 0.5),
                    'population': data.get('population', 10000),
                    'crime_rate_modifier': data.get('crime_rate_modifier', 1.0)
                })
            except Exception as e:
                logger.error(f"Failed to create polygon for zone '{name}': {e}", exc_info=True)
        if not zone_data:
            st.error("Fatal: No valid zones could be loaded from configuration.")
            return gpd.GeoDataFrame()
        return gpd.GeoDataFrame(zone_data, crs="EPSG:4326").set_index('name')

    def _initialize_ambulances(self) -> Dict[str, Dict]:
        return {
            amb_id: {
                'id': amb_id,
                'status': data.get('status', 'Disponible'),
                'home_base': data.get('home_base'),
                'location': Point(data['location'][1], data['location'][0])
            }
            for amb_id, data in self.data_config['ambulances'].items()
        }

    def get_current_incidents(self, env_factors: EnvFactors) -> List[Dict]:
        api_config = self.data_config.get('real_time_api', {})
        endpoint = api_config.get('endpoint', '')
        try:
            if endpoint.startswith('http'):
                headers = {"Authorization": f"Bearer {api_config.get('api_key', '')}"} if api_config.get('api_key') else {}
                response = requests.get(endpoint, headers=headers, timeout=10)
                response.raise_for_status()
                raw_incidents = response.json().get('incidents', [])
                logger.info(f"Fetched {len(raw_incidents)} incidents from API: {endpoint}")
            else:
                with open(endpoint, 'r') as f:
                    raw_incidents = json.load(f).get('incidents', [])
                logger.info(f"Loaded {len(raw_incidents)} incidents from local file: {endpoint}")
            valid_incidents = self._validate_and_process_incidents(raw_incidents)
            return valid_incidents if valid_incidents else self._generate_synthetic_incidents(env_factors)
        except (requests.RequestException, FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to get real-time incidents from '{endpoint}': {e}. Falling back to synthetic data.")
            return self._generate_synthetic_incidents(env_factors)

    def _validate_and_process_incidents(self, incidents: List[Dict]) -> List[Dict]:
        valid_incidents = []
        for inc in incidents:
            if not all(k in inc for k in ['id', 'type', 'triage', 'location']):
                logger.warning(f"Skipping incident {inc.get('id', 'N/A')}: Missing required fields.")
                continue
            loc = inc['location']
            if not isinstance(loc, dict) or 'lat' not in loc or 'lon' not in loc:
                logger.warning(f"Skipping incident {inc.get('id')}: Invalid location format.")
                continue
            try:
                lat, lon = float(loc['lat']), float(loc['lon'])
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    raise ValueError("Coordinates out of bounds.")
                inc['location'] = Point(lon, lat)
                if inc['type'] not in self.data_config['distributions']['incident_type']:
                    inc['type'] = 'Medical-Chronic'
                valid_incidents.append(inc)
            except (ValueError, TypeError):
                logger.warning(f"Skipping incident {inc['id']}: Invalid location data.")
        return valid_incidents

    def _generate_synthetic_incidents(self, env_factors: EnvFactors) -> List[Dict]:
        intensity = 5.0
        if env_factors.is_holiday: intensity *= 1.5
        if env_factors.weather.lower() in ['rain', 'fog']: intensity *= 1.2
        if env_factors.major_event: intensity *= 2.0
        intensity *= env_factors.traffic_level * (1 + 0.5 * env_factors.population_density / POPULATION_DENSITY_NORMALIZATION)
        if env_factors.air_quality_index > 100: intensity *= (1 + env_factors.air_quality_index / 500.0)
        if env_factors.heatwave_alert: intensity *= 1.3
        num_incidents = max(0, int(np.random.poisson(intensity)))
        if num_incidents == 0 or self.zones_gdf.empty: return []

        city_boundary = self.zones_gdf.unary_union
        bounds = city_boundary.bounds
        incidents = [{
            'id': f"SYN-{i}",
            'type': np.random.choice(
                list(self.data_config['distributions']['incident_type'].keys()),
                p=list(self.data_config['distributions']['incident_type'].values())
            ),
            'triage': np.random.choice(
                list(self.data_config['distributions']['triage'].keys()),
                p=list(self.data_config['distributions']['triage'].values())
            ),
            'location': Point(np.random.uniform(bounds[0], bounds[2]), np.random.uniform(bounds[1], bounds[3])),
            'timestamp': datetime.utcnow().isoformat()
        } for i in range(num_incidents)]
        logger.info(f"Generated {len(incidents)} synthetic incidents.")
        return incidents

    def _generate_sample_history_file(self) -> io.BytesIO:
        sample_history = [
            {
                'incidents': [
                    {
                        'id': f"SAMPLE-{i}-{j}",
                        'type': np.random.choice(
                            list(self.data_config['distributions']['incident_type'].keys()),
                            p=list(self.data_config['distributions']['incident_type'].values())
                        ),
                        'triage': np.random.choice(
                            list(self.data_config['distributions']['triage'].keys()),
                            p=list(self.data_config['distributions']['triage'].values())
                        ),
                        'zone': zone,
                        'location': Point(np.random.uniform(bounds[0], bounds[2]), np.random.uniform(bounds[1], bounds[3])),
                        'timestamp': (datetime.utcnow() - timedelta(hours=i*24)).isoformat()
                    } for j in range(np.random.randint(1, 5)) for zone in self.zones
                ],
                'timestamp': (datetime.utcnow() - timedelta(hours=i*24)).isoformat()
            } for i in range(3)
        ]
        buffer = io.BytesIO()
        buffer.write(json.dumps(sample_history, indent=2).encode('utf-8'))
        buffer.seek(0)
        return buffer
class PredictiveAnalyticsEngine:
    def __init__(self, dm: DataManager, config: Dict[str, any]):
        self.dm = dm
        self.config = config
        self.bn_model = self._build_bayesian_network()
        self.tcnn_model = self._initialize_tcnn()
        
        weights_config = self.config['model_params']['ensemble_weights']
        self.method_weights = {
            'hawkes': weights_config.get('hawkes', 0),
            'sir': weights_config.get('sir', 0),
            'bayesian': weights_config.get('bayesian', 0) if PGMPY_AVAILABLE else 0,
            'graph': weights_config.get('graph', 0),
            'chaos': weights_config.get('chaos', 0),
            'info': weights_config.get('info', 0),
            'tcnn': weights_config.get('tcnn', 0) if TORCH_AVAILABLE else weights_config.get('tcnn_fallback', 7),
            'game': weights_config.get('game', 0),
            'copula': weights_config.get('copula', 0),
            'violence': weights_config.get('violence', 0),
            'accident': weights_config.get('accident', 0),
            'medical': weights_config.get('medical', 0)
        }
        total_weight = sum(self.method_weights.values())
        self.method_weights = {k: v / total_weight for k, v in self.method_weights.items()} if total_weight > 0 else {}

    @st.cache_resource
    def _build_bayesian_network(_self) -> Optional[DiscreteBayesianNetwork]:
        if not PGMPY_AVAILABLE: return None
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
        if not TORCH_AVAILABLE: return None
        try:
            model = TCNN(**_self.config['tcnn_params'])
            logger.info("TCNN model initialized.")
            return model
        except Exception as e:
            logger.warning(f"Failed to initialize TCNN: {e}. Disabling.")
            return None

    def calculate_ensemble_risk_score(self, kpi_df: pd.DataFrame, historical_data: List[Dict]) -> Dict[str, float]:
        if kpi_df.empty or not self.method_weights: return {zone: 0.0 for zone in self.dm.zones}
        def normalize(series):
            min_val, max_val = series.min(), max(series.max(), 1e-9)
            return (series - min_val) / (max_val - min_val + 1e-9)
        
        scores = {}
        for zone in self.dm.zones:
            zone_kpi = kpi_df[kpi_df['Zone'] == zone]
            if zone_kpi.empty:
                scores[zone] = 0.0
                continue
            
            chaos_score = normalize(zone_kpi['Chaos Sensitivity Score'].iloc[0])
            chaos_amplifier = self.config['model_params'].get('chaos_amplifier', 1.5) if np.var([len(h['incidents']) for h in historical_data if h]) > np.mean([len(h['incidents']) for h in historical_data if h]) else 1.0
            
            contributions = [
                normalize(zone_kpi['Trauma Clustering Score'].iloc[0]) * self.method_weights['hawkes'],
                normalize(zone_kpi['Disease Surge Score'].iloc[0]) * self.method_weights['sir'],
                normalize(zone_kpi['Bayesian Confidence Score'].iloc[0]) * self.method_weights.get('bayesian', 0),
                normalize(zone_kpi['Spatial Spillover Risk'].iloc[0]) * self.method_weights['graph'],
                chaos_score * chaos_amplifier * self.method_weights['chaos'],
                (normalize(zone_kpi['Risk Entropy'].iloc[0]) * 0.5 + normalize(zone_kpi['Anomaly Score'].iloc[0]) * 0.5) * self.method_weights['info'],
                (1 - normalize(zone_kpi['Resource Adequacy Index'].iloc[0])) * self.method_weights['game'],
                normalize(zone_kpi['Trauma-Disease Correlation'].iloc[0]) * self.method_weights['copula'],
                normalize(zone_kpi['Violence Clustering Score'].iloc[0]) * self.method_weights['violence'],
                normalize(zone_kpi['Accident Clustering Score'].iloc[0]) * self.method_weights['accident'],
                normalize(zone_kpi['Medical Surge Score'].iloc[0]) * self.method_weights['medical']
            ]
            if TORCH_AVAILABLE and hasattr(self, 'forecast_df') and not self.forecast_df.empty:
                zone_forecast = self.forecast_df[(self.forecast_df['Zone'] == zone) & (self.forecast_df['Horizon (Hours)'] == 3)]
                if not zone_forecast.empty:
                    tcnn_score = (
                        zone_forecast['Violence Risk'].iloc[0] +
                        zone_forecast['Accident Risk'].iloc[0] +
                        zone_forecast['Medical Risk'].iloc[0]
                    ) / 3.0
                    contributions.append(normalize(pd.Series([tcnn_score]))[0] * self.method_weights['tcnn'])
            else:
                contributions.append(normalize(zone_kpi['Incident Probability'].iloc[0]) * self.method_weights.get('tcnn', 0))

            scores[zone] = min(max(np.sum(contributions), 0.0), 1.0)
        return scores

    def generate_kpis(self, historical_data: List[Dict], env_factors: EnvFactors, current_incidents: List[Dict]) -> pd.DataFrame:
        kpi_data = [{
            'Zone': zone,
            'Incident Probability': 0.0,
            'Expected Incident Volume': 0.0,
            'Risk Entropy': 0.0,
            'Anomaly Score': 0.0,
            'Spatial Spillover Risk': 0.0,
            'Resource Adequacy Index': 0.0,
            'Chaos Sensitivity Score': 0.0,
            'Bayesian Confidence Score': 0.0,
            'Response Time Estimate': 10.0,
            'Trauma Clustering Score': 0.0,
            'Disease Surge Score': 0.0,
            'Trauma-Disease Correlation': 0.0,
            'Violence Clustering Score': 0.0,
            'Accident Clustering Score': 0.0,
            'Medical Surge Score': 0.0,
            'Ensemble Risk Score': 0.0
        } for zone in self.dm.zones]

        if not historical_data and not current_incidents:
            logger.warning("No historical or current incident data provided. Returning empty KPI DataFrame.")
            return pd.DataFrame(kpi_data)

        # Combine historical and current incidents
        all_incidents = []
        for record in historical_data + [{'timestamp': pd.Timestamp.now(), 'incidents': current_incidents}]:
            for incident in record.get('incidents', []):
                incident_copy = incident.copy()
                incident_copy['timestamp'] = record['timestamp']
                all_incidents.append(incident_copy)

        if not all_incidents:
            logger.warning("No valid incidents found in data. Returning empty KPI DataFrame.")
            return pd.DataFrame(kpi_data)

        df = pd.DataFrame(all_incidents)
        
        # Map locations to zones
        def get_zone(point):
            for zone, row in self.dm.zones_gdf.iterrows():
                if row['geometry'].contains(point):
                    return zone
            return None

        if 'location' in df.columns:
            df['zone'] = df['location'].apply(get_zone)
        else:
            possible_zone_columns = ['zone', 'Zone', 'zone_id', 'ZoneID']
            zone_column = None
            for col in possible_zone_columns:
                if col in df.columns:
                    zone_column = col
                    break
            if zone_column:
                df['zone'] = df[zone_column]
            else:
                logger.error("No zone or location data found in incidents. Expected 'location' or one of: %s", possible_zone_columns)
                return pd.DataFrame(kpi_data)

        # Filter out incidents with no valid zone
        df = df[df['zone'].isin(self.dm.zones)]
        if df.empty:
            logger.warning("No incidents with valid zones after mapping. Returning empty KPI DataFrame.")
            return pd.DataFrame(kpi_data)

        if self.bn_model:
            try:
                inference = VariableElimination(self.bn_model)
                evidence = {
                    'Holiday': 1 if env_factors.is_holiday else 0,
                    'Weather': 1 if env_factors.weather != 'Clear' else 0,
                    'MajorEvent': 1 if env_factors.major_event else 0,
                    'AirQuality': 1 if env_factors.air_quality_index > 100 else 0,
                    'Heatwave': 1 if env_factors.heatwave_alert else 0
                }
                result = inference.query(variables=['IncidentRate'], evidence=evidence, show_progress=False)
                rate_probs = result.values
                baseline_rate = np.sum(rate_probs * np.array([1, 5, 10]))
                bayesian_confidence = 1 - (np.std(rate_probs) / (np.mean(rate_probs) + 1e-9))
            except Exception as e:
                logger.warning(f"Bayesian inference failed: {e}. Using defaults.")
                baseline_rate, bayesian_confidence = 5.0, 0.5
        else:
            baseline_rate, bayesian_confidence = 5.0, 0.5

        incident_counts = df['zone'].value_counts()
        violence_counts = df[df['type'] == 'Trauma-Violence']['zone'].value_counts()
        accident_counts = df[df['type'] == 'Trauma-Accident']['zone'].value_counts()
        medical_counts = df[df['type'].isin(['Medical-Chronic', 'Medical-Acute'])]['zone'].value_counts()
        trauma_counts = df[df['type'].isin(['Trauma-Violence', 'Trauma-Accident'])]['zone'].value_counts()
        disease_counts = df[df['type'].isin(['Medical-Chronic', 'Medical-Acute'])]['zone'].value_counts()
        
        past_incidents = sum([h.get('incidents', []) for h in historical_data if isinstance(h, dict)], [])
        violence_intensity = self._calculate_violence_intensity(past_incidents, env_factors, self.config['model_params']['hawkes_process'])
        accident_intensity = self._calculate_accident_intensity(past_incidents, env_factors, self.config['model_params']['hawkes_process'])
        medical_intensity = self._calculate_medical_intensity(env_factors, self.config['model_params']['sir_model'])
        trauma_intensity = violence_intensity + accident_intensity
        disease_intensity = medical_intensity
        
        # Suppress NumPy warnings for division and invalid operations
        with np.errstate(divide='ignore', invalid='ignore'):
            current_dist = (incident_counts / (incident_counts.sum() + 1e-9)).reindex(self.dm.zones, fill_value=0)
            if current_dist.sum() == 0 or not np.isfinite(current_dist).all():
                kl_divergence = 0.0
                shannon_entropy = 0.0
            else:
                prior_dist = self.config['data']['distributions']['zone']
                kl_divergence = np.sum(current_dist * np.log(current_dist.replace(0, 1e-9) / pd.Series(prior_dist).reindex_like(current_dist).replace(0, 1e-9)))
                shannon_entropy = -np.sum(current_dist * np.log2(current_dist.replace(0, 1e-9)))
                if not np.isfinite(kl_divergence): kl_divergence = 0.0
                if not np.isfinite(shannon_entropy): shannon_entropy = 0.0

        lyapunov_exponent = self._calculate_lyapunov_exponent(historical_data, current_dist)
        
        base_probs = self._calculate_base_probabilities(baseline_rate, trauma_intensity + disease_intensity, prior_dist)
        spillover_risk = self.config['model_params']['laplacian_diffusion_factor'] * (self.dm.laplacian_matrix @ pd.Series(base_probs, index=self.dm.zones).values)
        response_times = self._calculate_response_times(current_incidents)
        trauma_cluster_scores = self._calculate_trauma_cluster_scores(trauma_counts, self.config['model_params']['hawkes_process'])
        disease_surge_scores = self._calculate_disease_surge_scores(disease_counts, self.config['model_params']['sir_model'])
        violence_cluster_scores = self._calculate_violence_cluster_scores(violence_counts, self.config['model_params']['hawkes_process'])
        accident_cluster_scores = self._calculate_accident_cluster_scores(accident_counts, self.config['model_params']['hawkes_process'])
        medical_surge_scores = self._calculate_medical_surge_scores(medical_counts, self.config['model_params']['sir_model'])
        
        trauma_dist = (trauma_counts / (trauma_counts.sum() + 1e-9)).reindex(self.dm.zones, fill_value=0)
        disease_dist = (disease_counts / (disease_counts.sum() + 1e-9)).reindex(self.dm.zones, fill_value=0)
        correlation_score = self._model_event_correlations(trauma_dist, disease_dist)

        available_units = sum(1 for a in self.dm.ambulances.values() if a['status'] == 'Disponible')
        kpi_data = [{
            'Zone': zone,
            'Incident Probability': base_probs.get(zone, 0),
            'Expected Incident Volume': base_probs.get(zone, 0) * 3,
            'Risk Entropy': shannon_entropy,
            'Anomaly Score': kl_divergence,
            'Spatial Spillover Risk': spillover_risk[i],
            'Resource Adequacy Index': available_units / (base_probs.get(zone, 1e-9) * len(self.dm.zones) + 1e-9),
            'Chaos Sensitivity Score': lyapunov_exponent,
            'Bayesian Confidence Score': bayesian_confidence,
            'Response Time Estimate': response_times.get(zone, 10.0),
            'Trauma Clustering Score': trauma_cluster_scores.get(zone, 0.0),
            'Disease Surge Score': disease_surge_scores.get(zone, 0.0),
            'Trauma-Disease Correlation': correlation_score,
            'Violence Clustering Score': violence_cluster_scores.get(zone, 0.0),
            'Accident Clustering Score': accident_cluster_scores.get(zone, 0.0),
            'Medical Surge Score': medical_surge_scores.get(zone, 0.0)
        } for i, zone in enumerate(self.dm.zones)]
        
        kpi_df = pd.DataFrame(kpi_data)
        ensemble_scores = self.calculate_ensemble_risk_score(kpi_df, historical_data)
        kpi_df['Ensemble Risk Score'] = kpi_df['Zone'].map(ensemble_scores)
        return kpi_df

    def _calculate_base_probabilities(self, baseline: float, intensity: float, priors: Dict[str, float]) -> Dict[str, float]:
        return {
            zone: (baseline + intensity) * prob * self.dm.zones_gdf.loc[zone, 'crime_rate_modifier']
            for zone, prob in priors.items()
        }

    def _calculate_violence_intensity(self, past_incidents: List[Dict], env_factors: EnvFactors, params: Dict[str, float]) -> float:
        violence_incidents = [inc for inc in past_incidents if inc.get('type') == 'Trauma-Violence']
        intensity = len(violence_incidents) * params.get('kappa', 0.5) * params.get('violence_weight', 1.8) * np.exp(-params.get('beta', 1.0))
        if env_factors.air_quality_index > 100:
            intensity *= params.get('aqi_multiplier', 1.5) * (env_factors.air_quality_index / 500.0)
        if env_factors.heatwave_alert:
            intensity *= 1.3
        return max(0.0, intensity)

    def _calculate_accident_intensity(self, past_incidents: List[Dict], env_factors: EnvFactors, params: Dict[str, float]) -> float:
        accident_incidents = [inc for inc in past_incidents if inc.get('type') == 'Trauma-Accident']
        intensity = len(accident_incidents) * params.get('kappa', 0.5) * params.get('trauma_weight', 1.5) * np.exp(-params.get('beta', 1.0))
        if env_factors.traffic_level > 1.0:
            intensity *= env_factors.traffic_level
        if env_factors.weather.lower() in ['rain', 'fog']:
            intensity *= 1.2
        return max(0.0, intensity)

    def _calculate_medical_intensity(self, env_factors: EnvFactors, params: Dict[str, float]) -> float:
        S, I = env_factors.population_density, 0.01 * env_factors.population_density
        beta, gamma, noise_scale = params['beta'], params['gamma'], params['noise_scale']
        intensity = max(0.0, beta * S * I / (S + 1e-9) - gamma * I + np.random.normal(0, noise_scale))
        if env_factors.major_event:
            intensity *= 1.5
        if env_factors.weather.lower() in ['rain', 'fog']:
            intensity *= 1.2
        if env_factors.air_quality_index > 100:
            intensity *= params.get('aqi_multiplier', 1.5) * (env_factors.air_quality_index / 500.0)
        if env_factors.heatwave_alert:
            intensity *= 1.3
        return intensity

    def _calculate_lyapunov_exponent(self, historical_data: List[Dict], current_dist: pd.Series) -> float:
        if len(historical_data) < 2: return 0.0
        try:
            incident_counts_history = [pd.Series({inc['zone']: 1 for inc in h.get('incidents', []) if 'zone' in inc}).sum() for h in historical_data]
            series = pd.Series(incident_counts_history)
            if len(series) < 2 or series.std() == 0: return 0.0
            tau = 1
            m = 2
            N = len(series) - (m - 1) * tau
            y = np.array([series[i:i + (m - 1) * tau + 1:tau] for i in range(N)])
            distances = [np.linalg.norm(y - y[i], axis=1) for i in range(N)]
            divergence = [np.log(d[i+1] / (d[i] + 1e-9)) for i, d in enumerate(distances[:-1]) if d[i] > 0]
            return np.mean(divergence) if divergence and np.isfinite(divergence).all() else 0.0
        except Exception: return 0.0

    def _calculate_trauma_cluster_scores(self, trauma_counts: pd.Series, params: Dict[str, float]) -> Dict[str, float]:
        return {
            zone: trauma_counts.get(zone, 0) * params['kappa'] * params['trauma_weight'] * self.dm.zones_gdf.loc[zone, 'crime_rate_modifier']
            for zone in self.dm.zones
        }

    def _calculate_disease_surge_scores(self, disease_counts: pd.Series, params: Dict[str, float]) -> Dict[str, float]:
        return {
            zone: disease_counts.get(zone, 0) * params['beta'] * self.dm.zones_gdf.loc[zone, 'population'] / POPULATION_DENSITY_NORMALIZATION
            for zone in self.dm.zones
        }

    def _calculate_violence_cluster_scores(self, violence_counts: pd.Series, params: Dict[str, float]) -> Dict[str, float]:
        return {
            zone: violence_counts.get(zone, 0) * params['kappa'] * params['violence_weight'] * self.dm.zones_gdf.loc[zone, 'crime_rate_modifier']
            for zone in self.dm.zones
        }

    def _calculate_accident_cluster_scores(self, accident_counts: pd.Series, params: Dict[str, float]) -> Dict[str, float]:
        return {
            zone: accident_counts.get(zone, 0) * params['kappa'] * params['trauma_weight'] * self.dm.zones_gdf.loc[zone, 'crime_rate_modifier']
            for zone in self.dm.zones
        }

    def _calculate_medical_surge_scores(self, medical_counts: pd.Series, params: Dict[str, float]) -> Dict[str, float]:
        return {
            zone: medical_counts.get(zone, 0) * params['beta'] * self.dm.zones_gdf.loc[zone, 'population'] / POPULATION_DENSITY_NORMALIZATION
            for zone in self.dm.zones
        }

    def _model_event_correlations(self, trauma_dist: pd.Series, disease_dist: pd.Series) -> float:
        try:
            trauma_dist, disease_dist = trauma_dist.reindex(self.dm.zones, fill_value=0.0), disease_dist.reindex(self.dm.zones, fill_value=0.0)
            if trauma_dist.sum() == 0 or disease_dist.sum() == 0 or trauma_dist.std() == 0 or disease_dist.std() == 0:
                return 0.0
            u1, u2 = norm.cdf(trauma_dist.values), norm.cdf(disease_dist.values)
            rho = self.config['model_params'].get('copula_correlation', 0.2)
            if not np.all(np.isfinite(u1)) or not np.all(np.isfinite(u2)): return 0.0
            return np.corrcoef(u1, u2)[0, 1] if len(u1) > 1 else 0.0
        except Exception: return 0.0

    def _calculate_response_times(self, incidents: List[Dict]) -> Dict[str, float]:
        return {
            zone: min(
                [
                    self.dm.zones_gdf.loc[zone, 'geometry'].centroid.distance(amb['location']) * 1000 / 50000 * 60 +
                    self.config['model_params']['response_time_penalty']
                    for amb in self.dm.ambulances.values() if amb['status'] == 'Disponible'
                ] or [10.0]
            )
            for zone in self.dm.zones
        }

    def forecast_risk(self, kpi_df: pd.DataFrame) -> pd.DataFrame:
        horizons = self.config.get('forecast_horizons_hours', [0.5, 1, 3, 6, 12, 24, 72, 144])
        if self.tcnn_model and not kpi_df.empty:
            try:
                features = [
                    'Incident Probability',
                    'Risk Entropy',
                    'Anomaly Score',
                    'Trauma Clustering Score',
                    'Disease Surge Score',
                    'Violence Clustering Score',
                    'Accident Clustering Score',
                    'Medical Surge Score'
                ]
                X = kpi_df[features].values.astype(np.float32).reshape(1, len(self.dm.zones), -1)
                
                import torch
                with torch.no_grad():
                    preds = self.tcnn_model(torch.from_numpy(X)).numpy().flatten()
                
                forecast_data = []
                for zone_idx, zone in enumerate(self.dm.zones):
                    for h_idx, horizon in enumerate(horizons):
                        violence_idx = h_idx
                        accident_idx = h_idx + len(horizons)
                        medical_idx = h_idx + 2 * len(horizons)
                        forecast_data.append({
                            'Zone': zone,
                            'Horizon (Hours)': horizon,
                            'Violence Risk': float(preds[violence_idx]),
                            'Accident Risk': float(preds[accident_idx]),
                            'Medical Risk': float(preds[medical_idx]),
                            'Trauma Risk': float(preds[violence_idx]) + float(preds[accident_idx]),
                            'Disease Risk': float(preds[medical_idx])
                        })
                self.forecast_df = pd.DataFrame(forecast_data)
                return self.forecast_df
            except Exception as e:
                logger.warning(f"TCNN forecasting failed: {e}. Using baseline forecast.")
        
        decay_rates = self.config['model_params']['fallback_forecast_decay_rates']
        forecast_data = []
        for zone in self.dm.zones:
            zone_kpi = kpi_df[kpi_df['Zone'] == zone]
            base_violence = zone_kpi['Violence Clustering Score'].iloc[0] if not zone_kpi.empty else 0
            base_accident = zone_kpi['Accident Clustering Score'].iloc[0] if not zone_kpi.empty else 0
            base_medical = zone_kpi['Medical Surge Score'].iloc[0] if not zone_kpi.empty else 0
            base_trauma = zone_kpi['Trauma Clustering Score'].iloc[0] if not zone_kpi.empty else 0
            base_disease = zone_kpi['Disease Surge Score'].iloc[0] if not zone_kpi.empty else 0
            for horizon in horizons:
                decay = decay_rates.get(str(horizon), 0.5)
                forecast_data.append({
                    'Zone': zone,
                    'Horizon (Hours)': horizon,
                    'Violence Risk': base_violence * decay,
                    'Accident Risk': base_accident * decay,
                    'Medical Risk': base_medical * decay,
                    'Trauma Risk': base_trauma * decay,
                    'Disease Risk': base_disease * decay
                })
        self.forecast_df = pd.DataFrame(forecast_data)
        return self.forecast_df

class StrategicAdvisor:
    def __init__(self, dm: DataManager, config: Dict[str, any]):
        self.dm = dm
        self.config = config

    def recommend_allocations(self, kpi_df: pd.DataFrame, forecast_df: pd.DataFrame) -> List[Dict]:
        if kpi_df.empty or forecast_df.empty: return []
        available_ambulances = [amb for amb in self.dm.ambulances.values() if amb['status'] == 'Disponible']
        if not available_ambulances: return []

        weights = {float(k): v for k, v in self.config['model_params']['allocation_forecast_weights'].items()}
        deficits = pd.Series(0.0, index=self.dm.zones)
        for zone in self.dm.zones:
            current_deficit = kpi_df.loc[kpi_df['Zone'] == zone, 'Ensemble Risk Score'].iloc[0]
            forecast_deficit = sum(
                weights.get(row['Horizon (Hours)'], 0) * (
                    row['Violence Risk'] + row['Accident Risk'] + row['Medical Risk']
                ) for _, row in forecast_df[forecast_df['Zone'] == zone].iterrows()
            )
            deficits[zone] = current_deficit + forecast_deficit
        
        recommendations = []
        sorted_deficits = deficits.sort_values(ascending=False)
        for target_zone in sorted_deficits.index:
            if not available_ambulances: break
            if sorted_deficits[target_zone] > 0.5:
                zone_centroid = self.dm.zones_gdf.loc[target_zone, 'geometry'].centroid
                closest_amb = min(
                    available_ambulances,
                    key=lambda amb: zone_centroid.distance(amb['location'])
                )
                current_zone = next(
                    (z for z, d in self.dm.zones_gdf.iterrows() if d['geometry'].contains(closest_amb['location'])),
                    "Unknown"
                )
                
                if current_zone != target_zone:
                    reason = f"High integrated risk in {target_zone} (Ensemble Score: {kpi_df.loc[kpi_df['Zone'] == target_zone, 'Ensemble Risk Score'].iloc[0]:.2f}, 3hr Forecast: {(forecast_df.loc[(forecast_df['Zone'] == target_zone) & (forecast_df['Horizon (Hours)'] == 3), ['Violence Risk', 'Accident Risk', 'Medical Risk']].sum(axis=1).iloc[0] if not forecast_df.empty else 0):.2f})"
                    recommendations.append({
                        'unit': closest_amb['id'],
                        'from': current_zone,
                        'to': target_zone,
                        'reason': reason
                    })
                    available_ambulances.remove(closest_amb)

        return recommendations[:2]
class ReportGenerator:
    @staticmethod
    def generate_pdf_report(kpi_df: pd.DataFrame, recommendations: List[Dict], forecast_df: pd.DataFrame) -> io.BytesIO:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = [
            Paragraph("RedShield AI: Emergency Response Report", styles['Title']),
            Spacer(1, 12),
            Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']),
            Spacer(1, 12)
        ]
        if not kpi_df.empty:
            elements.append(Paragraph("Key Performance Indicators", styles['Heading2']))
            kpi_data = [kpi_df.columns.tolist()] + kpi_df.round(3).values.tolist()
            kpi_table = Table(kpi_data)
            kpi_table.setStyle([
                ('BACKGROUND', (0, 0), (-1, 0), '#2C3E50'),
                ('TEXTCOLOR', (0, 0), (-1, 0), '#FFFFFF'),
                ('GRID', (0, 0), (-1, -1), 1, '#000000'),
                ('FONTSIZE', (0, 0), (-1, -1), 8)
            ])
            elements.append(kpi_table)
            elements.append(Spacer(1, 12))
        if recommendations:
            elements.append(Paragraph("Resource Allocation Recommendations", styles['Heading2']))
            for rec in recommendations:
                elements.append(Paragraph(
                    f"Move {rec['unit']} from {rec['from']} to {rec['to']}. Reason: {rec['reason']}",
                    styles['Normal']
                ))
            elements.append(Spacer(1, 12))
        if not forecast_df.empty:
            elements.append(Paragraph("Risk Forecast (Multiple Horizons)", styles['Heading2']))
            forecast_data = [forecast_df.columns.tolist()] + forecast_df.round(3).values.tolist()
            forecast_table = Table(forecast_data)
            glimpse_table.setStyle([
                ('BACKGROUND', (0, 0), (-1, 0), '#2C3E50'),
                ('TEXTCOLOR', (0, 0), (-1, 0), '#FFFFFF'),
                ('GRID', (0, 0), (-1, -1), 1, '#000000'),
                ('FONTSIZE', (0, 0), (-1, -1), 8)
            ])
            elements.append(forecast_table)
        doc.build(elements)
        buffer.seek(0)
        return buffer

class VisualizationSuite:
    @staticmethod
    def plot_kpi_dashboard(kpi_df: pd.DataFrame) -> go.Figure:
        if kpi_df.empty:
            return go.Figure().add_annotation(
                text="No KPI data available. Run a predictive cycle.",
                showarrow=False,
                font=dict(size=16, color="#2C3E50")
            )
        def color_scaler(val: float, col_data: pd.Series, high_is_bad: bool = True) -> str:
            norm_val = (val - col_data.min()) / (col_data.max() - col_data.min() + 1e-9)
            r, g = (int(255 * norm_val), int(255 * (1 - norm_val))) if high_is_bad else (int(255 * (1 - norm_val)), int(255 * norm_val))
            return f'rgba({r}, {g}, 0, 0.7)'
        
        display_df = kpi_df.round(3)
        colors = {'Zone': [['#2C3E50'] * len(display_df)]}
        font_colors = {'Zone': [['white'] * len(display_df)]}
        for col in display_df.columns[1:]:
            high_is_bad = 'Confidence' not in col and 'Adequacy' not in col
            colors[col] = [[color_scaler(v, display_df[col], high_is_bad) for v in display_df[col]]]
            font_colors[col] = [['black'] * len(display_df)]
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[f"<b>{c}</b>" for c in display_df.columns],
                fill_color='#2C3E50',
                align='center',
                font=dict(color='white', size=14),
                line_color='white',
                height=40
            ),
            cells=dict(
                values=[display_df[k] for k in display_df.columns],
                fill_color=np.concatenate(list(colors.values()), axis=0).T,
                align='center',
                font=dict(color=np.concatenate(list(font_colors.values()), axis=0).T, size=12),
                height=30,
                line_color='white'
            )
        )])
        fig.update_layout(
            title_text="<b>Real-Time System Insights: KPI Dashboard</b>",
            title_font=dict(size=20, color='#2C3E50'),
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    @staticmethod
    def plot_radar_chart(kpi_df: pd.DataFrame) -> go.Figure:
        if kpi_df.empty:
            return go.Figure().add_annotation(
                text="No KPI data available for radar chart.",
                showarrow=False,
                font=dict(size=16, color="#2C3E50")
            )
        
        metrics = [
            'Ensemble Risk Score',
            'Violence Clustering Score',
            'Accident Clustering Score',
            'Medical Surge Score',
            'Response Time Estimate'
        ]
        fig = go.Figure()
        
        for zone in kpi_df['Zone']:
            values = kpi_df[kpi_df['Zone'] == zone][metrics].values.flatten()
            values = (values - kpi_df[metrics].min().values) / (kpi_df[metrics].max().values - kpi_df[metrics].min().values + 1e-9)
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=zone,
                line=dict(width=2),
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
                angularaxis=dict(showline=True, linewidth=2, gridcolor="rgba(0,0,0,0.2)")
            ),
            showlegend=True,
            title_text="<b>Zone Risk Profile Comparison</b>",
            title_font=dict(size=20, color='#2C3E50'),
            margin=dict(l=50, r=50, t=80, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        return fig

    @staticmethod
    def plot_risk_breakdown(kpi_df: pd.DataFrame) -> go.Figure:
        if kpi_df.empty:
            return go.Figure().add_annotation(
                text="No KPI data available for risk breakdown.",
                showarrow=False,
                font=dict(size=16, color="#2C3E50")
            )
        
        fig = go.Figure(data=[
            go.Bar(
                name='Violence Risk',
                x=kpi_df['Zone'],
                y=kpi_df['Violence Clustering Score'],
                marker_color='#E74C3C'
            ),
            go.Bar(
                name='Accident Risk',
                x=kpi_df['Zone'],
                y=kpi_df['Accident Clustering Score'],
                marker_color='#F1C40F'
            ),
            go.Bar(
                name='Medical Risk',
                x=kpi_df['Zone'],
                y=kpi_df['Medical Surge Score'],
                marker_color='#3498DB'
            )
        ])
        
        fig.update_layout(
            barmode='stack',
            title_text="<b>Risk Breakdown by Incident Type</b>",
            title_font=dict(size=20, color='#2C3E50'),
            xaxis_title="Zone",
            yaxis_title="Risk Score",
            xaxis=dict(tickfont=dict(size=12)),
            yaxis=dict(tickfont=dict(size=12)),
            margin=dict(l=50, r=50, t=80, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        return fig

    @staticmethod
    def plot_risk_heatmap(kpi_df: pd.DataFrame, dm: DataManager, config: Dict, risk_type: str) -> Optional[folium.Map]:
        if dm.zones_gdf.empty:
            logger.warning("Cannot plot heatmap, zones_gdf is empty.")
            return None
        map_center = [32.53, -117.04]
        if config.get('mapbox_api_key'):
            tiles = f"https://api.mapbox.com/styles/v1/mapbox/light-v10/tiles/{{z}}/{{x}}/{{y}}?access_token={config['mapbox_api_key']}"
            attr = "Mapbox"
        else:
            tiles = "OpenStreetMap"
            attr = "OpenStreetMap"
        
        m = folium.Map(location=map_center, zoom_start=12, tiles=tiles, attr=attr)
        
        if not kpi_df.empty and risk_type in kpi_df.columns:
            merged_gdf = dm.zones_gdf.join(kpi_df.set_index('Zone'))
            folium.Choropleth(
                geo_data=merged_gdf,
                name='Risk Choropleth',
                data=merged_gdf,
                columns=[merged_gdf.index, risk_type],
                key_on='feature.id',
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name=f'{risk_type} by Zone',
                highlight=True
            ).add_to(m)
        
        for zone, row in dm.zones_gdf.iterrows():
            popup_text = f"<b>Zone: {zone}</b><br>"
            if not kpi_df.empty and zone in kpi_df['Zone'].values:
                kpi_row = kpi_df.loc[kpi_df['Zone'] == zone].iloc[0]
                popup_text += (
                    f"Ensemble Risk: {kpi_row['Ensemble Risk Score']:.3f}<br>"
                    f"Violence Risk: {kpi_row['Violence Clustering Score']:.3f}<br>"
                    f"Accident Risk: {kpi_row['Accident Clustering Score']:.3f}<br>"
                    f"Medical Risk: {kpi_row['Medical Surge Score']:.3f}<br>"
                    f"Response Time: {kpi_row['Response Time Estimate']:.1f} min"
                )
            folium.Marker(
                location=[row.geometry.centroid.y, row.geometry.centroid.x],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(icon='info-sign', color='blue')
            ).add_to(m)

        for amb_id, amb in dm.ambulances.items():
            folium.Marker(
                location=[amb['location'].y, amb['location'].x],
                popup=f"Ambulance {amb_id}: {amb['status']}",
                icon=folium.Icon(
                    color='green' if amb['status'] == 'Disponible' else 'red',
                    icon='plus-sign'
                )
            ).add_to(m)
        
        return m

    @staticmethod
    def plot_gauge_chart(kpi_df: pd.DataFrame) -> go.Figure:
        if kpi_df.empty:
            return go.Figure().add_annotation(
                text="No KPI data available for gauge chart.",
                showarrow=False,
                font=dict(size=16, color="#2C3E50")
            )
        
        fig = go.Figure()
        for i, zone in enumerate(kpi_df['Zone']):
            risk_score = kpi_df.loc[kpi_df['Zone'] == zone, 'Ensemble Risk Score'].iloc[0]
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={'text': f"<b>{zone}</b>", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "black"},
                    'bar': {'color': "#2C3E50"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "#2ECC71"},
                        {'range': [0.3, 0.7], 'color': "#F1C40F"},
                        {'range': [0.7, 1], 'color': "#E74C3C"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.5
                    }
                },
                domain={
                    'row': i // 2,
                    'column': i % 2
                }
            ))
        
        fig.update_layout(
            grid={'rows': (len(kpi_df) + 1) // 2, 'columns': 2, 'pattern': "independent"},
            title_text="<b>Ensemble Risk Scores by Zone</b>",
            title_font=dict(size=20, color='#2C3E50'),
            margin=dict(l=50, r=50, t=80, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

def main():
    try:
        config = load_config()
    except Exception as e:
        st.error(f"Failed to load configuration: {e}. Using default configuration.")
        config = get_default_config()

    dm = get_data_manager(config)
    analytics = PredictiveAnalyticsEngine(dm, config)
    advisor = StrategicAdvisor(dm, config)
    viz = VisualizationSuite()
    report_gen = ReportGenerator()

    if 'run_count' not in st.session_state:
        st.session_state.run_count = 0
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = []
    if 'env_factors' not in st.session_state:
        st.session_state.env_factors = EnvFactors(
            is_holiday=False,
            weather="Clear",
            traffic_level=1.0,
            major_event=False,
            population_density=10000.0,
            air_quality_index=50.0,
            heatwave_alert=False
        )

    st.title("RedShield AI: Phoenix v3.1.0")
    st.markdown("**Commercial-grade predictive intelligence for urban emergency response**")

    with st.sidebar:
        st.header("Control Panel")
        with st.expander("Environmental Factors", expanded=True):
            holiday = st.checkbox("Is Holiday", value=st.session_state.env_factors.is_holiday)
            weather = st.selectbox(
                "Weather",
                options=["Clear", "Rain", "Fog"],
                index=["Clear", "Rain", "Fog"].index(st.session_state.env_factors.weather)
            )
            traffic = st.number_input(
                "Traffic Level (0-2)", min_value=0.0, max_value=2.0, value=st.session_state.env_factors.traffic_level, step=0.1
            )
            major_event = st.checkbox("Major Event", value=st.session_state.env_factors.major_event)
            pop_density = st.number_input(
                "Population Density", min_value=1000.0, max_value=100000.0, value=float(st.session_state.env_factors.population_density), step=1000.0
            )
            air_quality = st.number_input(
                "Air Quality Index (0-500)", min_value=0.0, max_value=500.0, value=float(st.session_state.env_factors.air_quality_index), step=10.0
            )
            heatwave = st.checkbox("Heatwave Alert", value=st.session_state.env_factors.heatwave_alert)
            
            st.session_state.env_factors = EnvFactors(
                is_holiday=holiday,
                weather=weather,
                traffic_level=traffic,
                major_event=major_event,
                population_density=pop_density,
                air_quality_index=air_quality,
                heatwave_alert=heatwave
            )

        uploaded_file = st.file_uploader("Upload Historical Incident Data (JSON)", type="json")
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                st.session_state.historical_data = data if isinstance(data, list) else []
                st.success("Historical data uploaded successfully.")
            except Exception as e:
                st.error(f"Failed to load file: {e}")

        if st.button("Download Sample History"):
            buffer = dm._generate_sample_history_file()
            st.download_button(
                label="Download Sample History JSON",
                data=buffer,
                file_name="sample_history.json",
                mime="application/json"
            )

    st.header("Real-Time System Insights")
    current_incidents = dm.get_current_incidents(st.session_state.env_factors)
    kpi_df = analytics.generate_kpis(st.session_state.historical_data, st.session_state.env_factors, current_incidents)
    forecast_df = analytics.forecast_risk(kpi_df)
    recommendations = advisor.recommend_allocations(kpi_df, forecast_df)

    st.subheader("Actionable KPI Dashboard")
    st.plotly_chart(viz.plot_kpi_dashboard(kpi_df), use_container_width=True)

    st.subheader("Zone Risk Profile Comparison")
    st.plotly_chart(viz.plot_radar_chart(kpi_df), use_container_width=True)

    st.subheader("Risk Breakdown by Incident Type")
    st.plotly_chart(viz.plot_risk_breakdown(kpi_df), use_container_width=True)

    st.subheader("Ensemble Risk Scores")
    st.plotly_chart(viz.plot_gauge_chart(kpi_df), use_container_width=True)

    st.subheader("Risk Heatmap")
    heatmap = viz.plot_risk_heatmap(kpi_df, dm, config, 'Ensemble Risk Score')
    if heatmap:
        st_folium(heatmap, width=700, height=500
