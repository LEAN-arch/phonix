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

# --- TCNN Model Definition ---
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        class Module: pass
        class Conv1d: pass
        class Dropout: pass
        class Linear: pass
        class ReLU: pass
        class AdaptiveAvgPool1d: pass
        class Flatten: pass

class TCNN(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, input_size: int, output_size: int, channels: List[int], kernel_size: int, dropout: float):
        if not TORCH_AVAILABLE:
            super().__init__()
            return
        super(TCNN, self).__init__()
        layers = []
        in_channels = input_size
        for out_channels in channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding='same'),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        layers.extend([
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, output_size)
        ])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if not TORCH_AVAILABLE:
            return torch.zeros(1, 24) if 'torch' in globals() else np.zeros((1, 24))
        return self.model(x)

# --- Optional dependency check for pgmpy ---
try:
    from pgmpy.models import BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    class BayesianNetwork: pass
    class TabularCPD: pass
    class VariableElimination: pass

# --- L0: SYSTEM CONFIGURATION & INITIALIZATION ---
st.set_page_config(page_title="RedShield AI: Phoenix v3.2.0", layout="wide", initial_sidebar_state="expanded")
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Setup logging
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/redshield_phoenix.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# --- Constants ---
POPULATION_DENSITY_NORMALIZATION = 100000.0
DEFAULT_HORIZONS = [0.5, 1, 3, 6, 12, 24, 72, 144]

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
        config = get_default_config()
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config.update(json.load(f))
        else:
            logger.info(f"Config file '{config_path}' not found. Generating default configuration.")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)

        mapbox_key = os.environ.get("MAPBOX_API_KEY", config.get("mapbox_api_key", ""))
        config['mapbox_api_key'] = mapbox_key if mapbox_key and "YOUR_KEY" not in mapbox_key else None

        logger.debug(f"Loaded config: {json.dumps(config, indent=2)}")
        validate_config(config)
        logger.info("System configuration loaded and validated successfully.")
        return config
    except (json.JSONDecodeError, ValueError, OSError) as e:
        logger.error(f"Failed to load or validate config: {e}. Using default configuration.", exc_info=True)
        st.warning(f"Configuration error: {e}. Using default configuration.")
        return get_default_config()

def get_default_config() -> Dict[str, any]:
    return {
        "mapbox_api_key": None,
        "forecast_horizons_hours": DEFAULT_HORIZONS,
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
                "hawkes": 9, "sir": 8, "bayesian": 8, "graph": 7, "chaos": 7,
                "info": 9, "tcnn": 10, "tcnn_fallback": 7, "game": 8, "copula": 8,
                "violence": 9, "accident": 8, "medical": 8
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
            "input_size": 9,
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
        if not all(isinstance(coord, list) and len(coord) == 2 for coord in data['polygon']):
            raise ValueError(f"Invalid coordinate format in polygon for zone '{zone}'.")
        if 'population' not in data or not isinstance(data['population'], (int, float)) or data['population'] <= 0:
            raise ValueError(f"Invalid population for zone '{zone}'.")
        if 'crime_rate_modifier' not in data or not isinstance(data['crime_rate_modifier'], (int, float)):
            logger.warning(f"Invalid or missing crime_rate_modifier for zone '{zone}' (value: {data.get('crime_rate_modifier')}). Setting to default (1.0).")
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
            self.laplacian_matrix = np.eye(len(self.zones))

    def _build_road_graph(self) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(self.zones)
        edges = self.data_config.get('road_network', {}).get('edges', [])
        for u, v, weight in edges:
            if u in G.nodes and v in G.nodes and isinstance(weight, (int, float)) and weight > 0:
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
                if poly.is_empty:
                    raise ValueError("Resulting polygon is empty after validation.")
                zone_data.append({
                    'name': name,
                    'geometry': poly,
                    'prior_risk': min(max(data.get('prior_risk', 0.5), 0.0), 1.0),
                    'population': max(data.get('population', 10000), 1),
                    'crime_rate_modifier': max(data.get('crime_rate_modifier', 1.0), 0.1)
                })
            except Exception as e:
                logger.error(f"Failed to create polygon for zone '{name}': {e}", exc_info=True)
        if not zone_data:
            st.error("Fatal: No valid zones could be loaded from configuration.")
            return gpd.GeoDataFrame()
        gdf = gpd.GeoDataFrame(zone_data, crs="EPSG:4326").set_index('name')
        return gdf

    def _initialize_ambulances(self) -> Dict[str, Dict]:
        return {
            amb_id: {
                'id': amb_id,
                'status': data.get('status', 'Disponible'),
                'home_base': data.get('home_base'),
                'location': Point(float(data['location'][1]), float(data['location'][0]))
            }
            for amb_id, data in self.data_config['ambulances'].items()
        }

    def get_current_incidents(self, env_factors: EnvFactors) -> List[Dict]:
        api_config = self.data_config.get('real_time_api', {})
        endpoint = api_config.get('endpoint', '')
        try:
            if endpoint.startswith(('http://', 'https://')):
                headers = {"Authorization": f"Bearer {api_config.get('api_key', '')}"} if api_config.get('api_key') else {}
                response = requests.get(endpoint, headers=headers, timeout=10)
                response.raise_for_status()
                raw_incidents = response.json().get('incidents', [])
                logger.info(f"Fetched {len(raw_incidents)} incidents from API: {endpoint}")
            else:
                with open(endpoint, 'r', encoding='utf-8') as f:
                    raw_incidents = json.load(f).get('incidents', [])
                logger.info(f"Loaded {len(raw_incidents)} incidents from local file: {endpoint}")
            valid_incidents = self._validate_and_process_incidents(raw_incidents)
            return valid_incidents or self._generate_synthetic_incidents(env_factors)
        except (requests.RequestException, FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to get real-time incidents from '{endpoint}': {e}. Falling back to synthetic data.", exc_info=True)
            return self._generate_synthetic_incidents(env_factors)

    def _validate_and_process_incidents(self, incidents: List[Dict]) -> List[Dict]:
        valid_incidents = []
        incident_types = self.data_config['distributions']['incident_type'].keys()
        for inc in incidents:
            if not all(k in inc for k in ['id', 'type', 'triage', 'location']):
                logger.warning(f"Skipping incident {inc.get('id', 'N/A')}: Missing required fields.")
                continue
            loc = inc['location']
            if not isinstance(loc, dict) or 'lat' not in loc or 'lon' not in loc:
                logger.warning(f"Skipping incident {inc.get('id', 'N/A')}: Invalid location format.")
                continue
            try:
                lat, lon = float(loc['lat']), float(loc['lon'])
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    raise ValueError("Coordinates out of bounds.")
                inc['location'] = Point(lon, lat)
                inc['type'] = inc['type'] if inc['type'] in incident_types else 'Medical-Chronic'
                valid_incidents.append(inc)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping incident {inc.get('id', 'N/A')}: Invalid location data: {e}")
        return valid_incidents

    def _generate_synthetic_incidents(self, env_factors: EnvFactors) -> List[Dict]:
        if self.zones_gdf.empty:
            logger.warning("No valid zones available for synthetic incident generation.")
            return []
        
        intensity = 5.0
        if env_factors.is_holiday:
            intensity *= 1.5
        if env_factors.weather.lower() in ['rain', 'fog']:
            intensity *= 1.2
        if env_factors.major_event:
            intensity *= 2.0
        intensity *= env_factors.traffic_level * (1 + 0.5 * env_factors.population_density / POPULATION_DENSITY_NORMALIZATION)
        if env_factors.air_quality_index > 100:
            intensity *= (1 + env_factors.air_quality_index / 500.0)
        if env_factors.heatwave_alert:
            intensity *= 1.3
        
        num_incidents = max(0, int(np.random.poisson(intensity)))
        if num_incidents == 0:
            return []
        
        city_boundary = self.zones_gdf.unary_union
        bounds = city_boundary.bounds
        incidents = []
        for i in range(num_incidents):
            for _ in range(10):  # Retry up to 10 times to find a valid point
                point = Point(np.random.uniform(bounds[0], bounds[2]), np.random.uniform(bounds[1], bounds[3]))
                if city_boundary.contains(point):
                    incidents.append({
                        'id': f"SYN-{i}",
                        'type': np.random.choice(
                            list(self.data_config['distributions']['incident_type'].keys()),
                            p=list(self.data_config['distributions']['incident_type'].values())
                        ),
                        'triage': np.random.choice(
                            list(self.data_config['distributions']['triage'].keys()),
                            p=list(self.data_config['distributions']['triage'].values())
                        ),
                        'location': point,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    break
        logger.info(f"Generated {len(incidents)} synthetic incidents.")
        return incidents

    def generate_sample_history_file(self) -> io.BytesIO:
        if self.zones_gdf.empty:
            logger.warning("No valid zones available for sample history generation.")
            return io.BytesIO()
        
        city_boundary = self.zones_gdf.unary_union
        bounds = city_boundary.bounds
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
        self.forecast_df = pd.DataFrame()
        
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
    def _build_bayesian_network(_self) -> Optional[BayesianNetwork]:
        if not PGMPY_AVAILABLE:
            logger.info("pgmpy not available. Bayesian network disabled.")
            return None
        try:
            bn_config = _self.config['bayesian_network']
            nodes = set(bn_config['cpds'].keys())
            for edge in bn_config['structure']:
                nodes.add(edge[0])
                nodes.add(edge[1])
            model = BayesianNetwork()
            model.add_nodes_from(nodes)
            model.add_edges_from(bn_config['structure'])
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
            logger.warning(f"Failed to initialize Bayesian network: {e}. Disabling.", exc_info=True)
            return None

    @st.cache_resource
    def _initialize_tcnn(_self) -> Optional[TCNN]:
        if not TORCH_AVAILABLE:
            logger.info("PyTorch not available. TCNN disabled.")
            return None
        try:
            model = TCNN(**_self.config['tcnn_params'])
            model.eval()
            logger.info("TCNN model initialized.")
            return model
        except Exception as e:
            logger.warning(f"Failed to initialize TCNN: {e}. Disabling.", exc_info=True)
            return None

    def calculate_ensemble_risk_score(self, kpi_df: pd.DataFrame, historical_data: List[Dict]) -> Dict[str, float]:
        if kpi_df.empty or not self.method_weights:
            return {zone: 0.0 for zone in self.dm.zones}
        
        def normalize(series: pd.Series) -> pd.Series:
            min_val, max_val = series.min(), max(series.max(), 1e-9)
            return (series - min_val) / (max_val - min_val + 1e-9)
        
        scores = {}
        historical_counts = [len(h.get('incidents', [])) for h in historical_data if h]
        chaos_amplifier = self.config['model_params'].get('chaos_amplifier', 1.5) if historical_counts and np.var(historical_counts) > np.mean(historical_counts) else 1.0
        
        for zone in self.dm.zones:
            zone_kpi = kpi_df[kpi_df['Zone'] == zone]
            if zone_kpi.empty:
                scores[zone] = 0.0
                continue
            
            contributions = []
            for metric, weight_key in [
                ('Trauma Clustering Score', 'hawkes'),
                ('Disease Surge Score', 'sir'),
                ('Bayesian Confidence Score', 'bayesian'),
                ('Spatial Spillover Risk', 'graph'),
                ('Chaos Sensitivity Score', 'chaos'),
                ('Risk Entropy', 'info'),
                ('Anomaly Score', 'info'),
                ('Resource Adequacy Index', 'game'),
                ('Trauma-Disease Correlation', 'copula'),
                ('Violence Clustering Score', 'violence'),
                ('Accident Clustering Score', 'accident'),
                ('Medical Surge Score', 'medical')
            ]:
                value = zone_kpi[metric].iloc[0] if metric in zone_kpi.columns else 0.0
                weight = self.method_weights.get(weight_key, 0)
                if metric == 'Resource Adequacy Index':
                    value = 1 - value
                if metric in ['Risk Entropy', 'Anomaly Score']:
                    value *= 0.5
                if metric == 'Chaos Sensitivity Score':
                    value *= chaos_amplifier
                contributions.append(normalize(pd.Series([value]))[0] * weight)
            
            if TORCH_AVAILABLE and not self.forecast_df.empty:
                zone_forecast = self.forecast_df[(self.forecast_df['Zone'] == zone) & (self.forecast_df['Horizon (Hours)'] == 3)]
                if not zone_forecast.empty:
                    tcnn_score = (
                        zone_forecast['Violence Risk'].iloc[0] +
                        zone_forecast['Accident Risk'].iloc[0] +
                        zone_forecast['Medical Risk'].iloc[0]
                    ) / 3.0
                    contributions.append(normalize(pd.Series([tcnn_score]))[0] * self.method_weights.get('tcnn', 0))
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
            logger.warning("No historical or current incident data provided. Returning default KPI DataFrame.")
            return pd.DataFrame(kpi_data)

        all_incidents = []
        for record in historical_data + [{'timestamp': pd.Timestamp.now().isoformat(), 'incidents': current_incidents}]:
            if not isinstance(record, dict) or 'incidents' not in record:
                continue
            for incident in record.get('incidents', []):
                if not isinstance(incident, dict):
                    continue
                incident_copy = incident.copy()
                incident_copy['timestamp'] = record['timestamp']
                all_incidents.append(incident_copy)

        if not all_incidents:
            logger.warning("No valid incidents found in data. Returning default KPI DataFrame.")
            return pd.DataFrame(kpi_data)

        df = pd.DataFrame(all_incidents)
        
        def get_zone(point):
            if not isinstance(point, Point):
                return None
            for zone, row in self.dm.zones_gdf.iterrows():
                if row['geometry'].contains(point):
                    return zone
            return None

        if 'location' in df.columns:
            df['zone'] = df['location'].apply(get_zone)
        else:
            possible_zone_columns = ['zone', 'Zone', 'zone_id', 'ZoneID']
            zone_column = next((col for col in possible_zone_columns if col in df.columns), None)
            if zone_column:
                df['zone'] = df[zone_column]
            else:
                logger.error("No zone or location data found in incidents. Expected 'location' or one of: %s", possible_zone_columns)
                return pd.DataFrame(kpi_data)

        df = df[df['zone'].isin(self.dm.zones)]
        if df.empty:
            logger.warning("No incidents with valid zones after mapping. Returning default KPI DataFrame.")
            return pd.DataFrame(kpi_data)

        if self.bn_model and PGMPY_AVAILABLE:
            try:
                inference = VariableElimination(self.bn_model)
                evidence = {
                    'Holiday': 1 if env_factors.is_holiday else 0,
                    'Weather': 1 if env_factors.weather.lower() != 'clear' else 0,
                    'MajorEvent': 1 if env_factors.major_event else 0,
                    'AirQuality': 1 if env_factors.air_quality_index > 100 else 0,
                    'Heatwave': 1 if env_factors.heatwave_alert else 0
                }
                result = inference.query(variables=['IncidentRate'], evidence=evidence, show_progress=False)
                rate_probs = result.values
                baseline_rate = np.sum(rate_probs * np.array([1, 5, 10]))
                bayesian_confidence = 1 - (np.std(rate_probs) / (np.mean(rate_probs) + 1e-9))
            except Exception as e:
                logger.warning(f"Bayesian inference failed: {e}. Using defaults.", exc_info=True)
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
        
        with np.errstate(divide='ignore', invalid='ignore'):
            current_dist = (incident_counts / (incident_counts.sum() + 1e-9)).reindex(self.dm.zones, fill_value=0)
            prior_dist = pd.Series(self.config['data']['distributions']['zone']).reindex(self.dm.zones, fill_value=1e-9)
            kl_divergence = np.sum(current_dist * np.log(current_dist.replace(0, 1e-9) / prior_dist))
            shannon_entropy = -np.sum(current_dist * np.log2(current_dist.replace(0, 1e-9)))
            kl_divergence = 0.0 if not np.isfinite(kl_divergence) else kl_divergence
            shannon_entropy = 0.0 if not np.isfinite(shannon_entropy) else shannon_entropy

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

    def _calculate_base_probabilities(self, baseline: float, intensity: float, priors: pd.Series) -> Dict[str, float]:
        return {
            zone: min(max((baseline + intensity) * prob * self.dm.zones_gdf.loc[zone, 'crime_rate_modifier'], 0.0), 1.0)
            for zone, prob in priors.items()
        }

    def _calculate_violence_intensity(self, past_incidents: List[Dict], env_factors: EnvFactors, params: Dict[str, float]) -> float:
        violence_incidents = [inc for inc in past_incidents if isinstance(inc, dict) and inc.get('type') == 'Trauma-Violence']
        intensity = len(violence_incidents) * params.get('kappa', 0.5) * params.get('violence_weight', 1.8) * np.exp(-params.get('beta', 1.0))
        if env_factors.air_quality_index > 100:
            intensity *= params.get('aqi_multiplier', 1.5) * (env_factors.air_quality_index / 500.0)
        if env_factors.heatwave_alert:
            intensity *= 1.3
        return max(0.0, intensity)

    def _calculate_accident_intensity(self, past_incidents: List[Dict], env_factors: EnvFactors, params: Dict[str, float]) -> float:
        accident_incidents = [inc for inc in past_incidents if isinstance(inc, dict) and inc.get('type') == 'Trauma-Accident']
        intensity = len(accident_incidents) * params.get('kappa', 0.5) * params.get('trauma_weight', 1.5) * np.exp(-params.get('beta', 1.0))
        if env_factors.traffic_level > 1.0:
            intensity *= env_factors.traffic_level
        if env_factors.weather.lower() in ['rain', 'fog']:
            intensity *= 1.2
        return max(0.0, intensity)

    def _calculate_medical_intensity(self, env_factors: EnvFactors, params: Dict[str, float]) -> float:
        S = env_factors.population_density
        I = 0.01 * env_factors.population_density
        beta, gamma, noise_scale = params.get('beta', 0.3), params.get('gamma', 0.1), params.get('noise_scale', 0.05)
        intensity = beta * S * I / (S + 1e-9) - gamma * I + np.random.normal(0, noise_scale)
        if env_factors.major_event:
            intensity *= 1.5
        if env_factors.weather.lower() in ['rain', 'fog']:
            intensity *= 1.2
        if env_factors.air_quality_index > 100:
            intensity *= params.get('aqi_multiplier', 1.5) * (env_factors.air_quality_index / 500.0)
        if env_factors.heatwave_alert:
            intensity *= 1.3
        return max(0.0, intensity)

    def _calculate_lyapunov_exponent(self, historical_data: List[Dict], current_dist: pd.Series) -> float:
        if len(historical_data) < 2:
            return 0.0
        try:
            incident_counts_history = []
            for h in historical_data:
                if not isinstance(h, dict) or 'incidents' not in h:
                    continue
                counts = pd.Series({inc.get('zone', ''): 1 for inc in h.get('incidents', []) if isinstance(inc, dict) and inc.get('zone') in self.dm.zones}).sum()
                incident_counts_history.append(counts)
            
            series = pd.Series(incident_counts_history)
            if len(series) < 2 or series.std() == 0:
                return 0.0
            tau = 1
            m = 2
            N = len(series) - (m - 1) * tau
            if N <= 0:
                return 0.0
            y = np.array([series[i:i + (m - 1) * tau + 1:tau] for i in range(N)])
            distances = [np.linalg.norm(y - y[i], axis=1) for i in range(N)]
            divergence = [np.log(d[i+1] / (d[i] + 1e-9)) for i, d in enumerate(distances[:-1]) if d[i] > 0 and i + 1 < len(d)]
            return np.mean(divergence) if divergence and np.isfinite(divergence).all() else 0.0
        except Exception as e:
            logger.warning(f"Lyapunov exponent calculation failed: {e}. Returning 0.0.", exc_info=True)
            return 0.0

    def _calculate_trauma_cluster_scores(self, trauma_counts: pd.Series, params: Dict[str, float]) -> Dict[str, float]:
        return {
            zone: trauma_counts.get(zone, 0) * params.get('kappa', 0.5) * params.get('trauma_weight', 1.5) * self.dm.zones_gdf.loc[zone, 'crime_rate_modifier']
            for zone in self.dm.zones
        }

    def _calculate_disease_surge_scores(self, disease_counts: pd.Series, params: Dict[str, float]) -> Dict[str, float]:
        return {
            zone: disease_counts.get(zone, 0) * params.get('beta', 0.3) * self.dm.zones_gdf.loc[zone, 'population'] / POPULATION_DENSITY_NORMALIZATION
            for zone in self.dm.zones
        }

    def _calculate_violence_cluster_scores(self, violence_counts: pd.Series, params: Dict[str, float]) -> Dict[str, float]:
        return {
            zone: violence_counts.get(zone, 0) * params.get('kappa', 0.5) * params.get('violence_weight', 1.8) * self.dm.zones_gdf.loc[zone, 'crime_rate_modifier']
            for zone in self.dm.zones
        }

    def _calculate_accident_cluster_scores(self, accident_counts: pd.Series, params: Dict[str, float]) -> Dict[str, float]:
        return {
            zone: accident_counts.get(zone, 0) * params.get('kappa', 0.5) * params.get('trauma_weight', 1.5) * self.dm.zones_gdf.loc[zone, 'crime_rate_modifier']
            for zone in self.dm.zones
        }

    def _calculate_medical_surge_scores(self, medical_counts: pd.Series, params: Dict[str, float]) -> Dict[str, float]:
        return {
            zone: medical_counts.get(zone, 0) * params.get('beta', 0.3) * self.dm.zones_gdf.loc[zone, 'population'] / POPULATION_DENSITY_NORMALIZATION
            for zone in self.dm.zones
        }

    def _model_event_correlations(self, trauma_dist: pd.Series, disease_dist: pd.Series) -> float:
        try:
            trauma_dist = trauma_dist.reindex(self.dm.zones, fill_value=0.0)
            disease_dist = disease_dist.reindex(self.dm.zones, fill_value=0.0)
            if trauma_dist.sum() == 0 or disease_dist.sum() == 0 or trauma_dist.std() == 0 or disease_dist.std() == 0:
                return 0.0
            u1, u2 = norm.cdf(trauma_dist.values), norm.cdf(disease_dist.values)
            if not np.all(np.isfinite(u1)) or not np.all(np.isfinite(u2)):
                return 0.0
            return np.corrcoef(u1, u2)[0, 1] if len(u1) > 1 else 0.0
        except Exception as e:
            logger.warning(f"Event correlation calculation failed: {e}. Returning 0.0.", exc_info=True)
            return 0.0

    def _calculate_response_times(self, incidents: List[Dict]) -> Dict[str, float]:
        available_ambulances = [amb for amb in self.dm.ambulances.values() if amb['status'] == 'Disponible']
        response_times = {}
        for zone in self.dm.zones:
            if not available_ambulances:
                response_times[zone] = 10.0
                continue
            zone_centroid = self.dm.zones_gdf.loc[zone, 'geometry'].centroid
            distances = [
                zone_centroid.distance(amb['location']) * 1000 / 50000 * 60 + self.config['model_params']['response_time_penalty']
                for amb in available_ambulances
            ]
            response_times[zone] = min(distances) if distances else 10.0
        return response_times

    def forecast_risk(self, kpi_df: pd.DataFrame) -> pd.DataFrame:
        horizons = self.config.get('forecast_horizons_hours', DEFAULT_HORIZONS)
        if kpi_df.empty:
            logger.warning("Empty KPI DataFrame provided for forecasting. Returning empty forecast.")
            forecast_data = [{
                'Zone': zone,
                'Horizon (Hours)': horizon,
                'Violence Risk': 0.0,
                'Accident Risk': 0.0,
                'Medical Risk': 0.0,
                'Trauma Risk': 0.0,
                'Disease Risk': 0.0
            } for zone in self.dm.zones for horizon in horizons]
            self.forecast_df = pd.DataFrame(forecast_data)
            return self.forecast_df

        if self.tcnn_model and TORCH_AVAILABLE:
            try:
                features = [
                    'Incident Probability', 'Risk Entropy', 'Anomaly Score',
                    'Trauma Clustering Score', 'Disease Surge Score',
                    'Violence Clustering Score', 'Accident Clustering Score',
                    'Medical Surge Score'
                ]
                feature_df = kpi_df[features].fillna(0).astype(np.float32)
                if feature_df.empty or len(self.dm.zones) == 0:
                    raise ValueError("Invalid input data for TCNN.")
                X = feature_df.values.T.reshape(1, len(features), len(self.dm.zones))
                with torch.no_grad():
                    preds = self.tcnn_model(torch.from_numpy(X)).numpy().flatten()
                
                forecast_data = []
                for zone_idx, zone in enumerate(self.dm.zones):
                    for h_idx, horizon in enumerate(horizons):
                        violence_idx = h_idx
                        accident_idx = h_idx + len(horizons)
                        medical_idx = h_idx + 2 * len(horizons)
                        violence_risk = max(float(preds[violence_idx]), 0.0) if violence_idx < len(preds) else 0.0
                        accident_risk = max(float(preds[accident_idx]), 0.0) if accident_idx < len(preds) else 0.0
                        medical_risk = max(float(preds[medical_idx]), 0.0) if medical_idx < len(preds) else 0.0
                        forecast_data.append({
                            'Zone': zone,
                            'Horizon (Hours)': horizon,
                            'Violence Risk': violence_risk,
                            'Accident Risk': accident_risk,
                            'Medical Risk': medical_risk,
                            'Trauma Risk': violence_risk + accident_risk,
                            'Disease Risk': medical_risk
                        })
                self.forecast_df = pd.DataFrame(forecast_data)
                return self.forecast_df
            except Exception as e:
                logger.warning(f"TCNN forecasting failed: {e}. Using baseline forecast.", exc_info=True)
        
        decay_rates = self.config['model_params']['fallback_forecast_decay_rates']
        forecast_data = []
        for zone in self.dm.zones:
            zone_kpi = kpi_df[kpi_df['Zone'] == zone]
            base_violence = zone_kpi['Violence Clustering Score'].iloc[0] if not zone_kpi.empty else 0.0
            base_accident = zone_kpi['Accident Clustering Score'].iloc[0] if not zone_kpi.empty else 0.0
            base_medical = zone_kpi['Medical Surge Score'].iloc[0] if not zone_kpi.empty else 0.0
            base_trauma = zone_kpi['Trauma Clustering Score'].iloc[0] if not zone_kpi.empty else 0.0
            base_disease = zone_kpi['Disease Surge Score'].iloc[0] if not zone_kpi.empty else 0.0
            for horizon in horizons:
                decay = decay_rates.get(str(horizon), 0.5)
                forecast_data.append({
                    'Zone': zone,
                    'Horizon (Hours)': horizon,
                    'Violence Risk': max(base_violence * decay, 0.0),
                    'Accident Risk': max(base_accident * decay, 0.0),
                    'Medical Risk': max(base_medical * decay, 0.0),
                    'Trauma Risk': max(base_trauma * decay, 0.0),
                    'Disease Risk': max(base_disease * decay, 0.0)
                })
        self.forecast_df = pd.DataFrame(forecast_data)
        return self.forecast_df

class StrategicAdvisor:
    def __init__(self, dm: DataManager, config: Dict[str, any]):
        self.dm = dm
        self.config = config

    def recommend_allocations(self, kpi_df: pd.DataFrame, forecast_df: pd.DataFrame) -> List[Dict]:
        if kpi_df.empty or forecast_df.empty or not self.dm.ambulances:
            logger.warning("Insufficient data for allocation recommendations.")
            return []
        
        available_ambulances = [amb for amb in self.dm.ambulances.values() if amb['status'] == 'Disponible']
        if not available_ambulances:
            logger.info("No available ambulances for reallocation.")
            return []

        weights = {float(k): v for k, v in self.config['model_params']['allocation_forecast_weights'].items()}
        deficits = pd.Series(0.0, index=self.dm.zones)
        for zone in self.dm.zones:
            zone_kpi = kpi_df[kpi_df['Zone'] == zone]
            if zone_kpi.empty:
                continue
            current_deficit = zone_kpi['Ensemble Risk Score'].iloc[0]
            zone_forecast = forecast_df[forecast_df['Zone'] == zone]
            forecast_deficit = sum(
                weights.get(row['Horizon (Hours)'], 0) * (
                    row['Violence Risk'] + row['Accident Risk'] + row['Medical Risk']
                ) for _, row in zone_forecast.iterrows()
            )
            deficits[zone] = current_deficit + forecast_deficit
        
        recommendations = []
        sorted_deficits = deficits.sort_values(ascending=False)
        for target_zone in sorted_deficits.index:
            if not available_ambulances:
                break
            if sorted_deficits[target_zone] <= 0.5:
                continue
            zone_centroid = self.dm.zones_gdf.loc[target_zone, 'geometry'].centroid
            closest_amb = min(
                available_ambulances,
                key=lambda amb: zone_centroid.distance(amb['location']) if isinstance(amb['location'], Point) else float('inf')
            )
            current_zone = next(
                (z for z, d in self.dm.zones_gdf.iterrows() if d['geometry'].contains(closest_amb['location'])),
                "Unknown"
            )
            
            if current_zone != target_zone and current_zone != "Unknown":
                zone_forecast = forecast_df[(forecast_df['Zone'] == target_zone) & (forecast_df['Horizon (Hours)'] == 3)]
                forecast_sum = zone_forecast[['Violence Risk', 'Accident Risk', 'Medical Risk']].sum(axis=1).iloc[0] if not zone_forecast.empty else 0.0
                reason = f"High integrated risk in {target_zone} (Ensemble Score: {kpi_df.loc[kpi_df['Zone'] == target_zone, 'Ensemble Risk Score'].iloc[0]:.2f}, 3hr Forecast: {forecast_sum:.2f})"
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
        try:
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
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), '#ECF0F1'),
                    ('TEXTCOLOR', (0, 1), (-1, -1), '#000000'),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, '#000000')
                ])
                elements.append(kpi_table)
                elements.append(Spacer(1, 12))

            if recommendations:
                elements.append(Paragraph("Resource Allocation Recommendations", styles['Heading2']))
                recommendation_data = [['Unit', 'From', 'To', 'Reason']] + [
                    [rec['unit'], rec['from'], rec['to'], rec['reason']] for rec in recommendations
                ]
                rec_table = Table(recommendation_data)
                rec_table.setStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), '#2C3E50'),
                    ('TEXTCOLOR', (0, 0), (-1, 0), '#FFFFFF'),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), '#ECF0F1'),
                    ('TEXTCOLOR', (0, 1), (-1, -1), '#000000'),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, '#000000')
                ])
                elements.append(rec_table)
                elements.append(Spacer(1, 12))

            if not forecast_df.empty:
                elements.append(Paragraph("Risk Forecast", styles['Heading2']))
                forecast_data = [forecast_df.columns.tolist()] + forecast_df.round(3).values.tolist()
                forecast_table = Table(forecast_data)
                forecast_table.setStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), '#2C3E50'),
                    ('TEXTCOLOR', (0, 0), (-1, 0), '#FFFFFF'),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), '#ECF0F1'),
                    ('TEXTCOLOR', (0, 1), (-1, -1), '#000000'),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, '#000000')
                ])
                elements.append(forecast_table)
                elements.append(Spacer(1, 12))

            doc.build(elements)
            buffer.seek(0)
            return buffer
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}", exc_info=True)
            return io.BytesIO()

class VisualizationSuite:
    @staticmethod
    def plot_risk_heatmap(dm: DataManager, kpi_df: pd.DataFrame) -> folium.Map:
        if dm.zones_gdf.empty or kpi_df.empty:
            logger.warning("Empty GeoDataFrame or KPI DataFrame provided for heatmap.")
            return folium.Map(location=[32.53, -117.03], zoom_start=12)
        
        try:
            center = dm.zones_gdf.unary_union.centroid
            heatmap = folium.Map(location=[center.y, center.x], zoom_start=12)
            colors = {0.0: '#00FF00', 0.5: '#FFFF00', 1.0: '#FF0000'}
            
            for idx, row in dm.zones_gdf.iterrows():
                risk_score = kpi_df[kpi_df['Zone'] == idx]['Ensemble Risk Score'].iloc[0] if idx in kpi_df['Zone'].values else 0.0
                risk_score = min(max(risk_score, 0.0), 1.0)
                if risk_score < 0.5:
                    color = colors[0.0]
                elif risk_score < 0.75:
                    color = colors[0.5]
                else:
                    color = colors[1.0]
                folium.GeoJson(
                    row['geometry'],
                    style_function=lambda x, color=color: {
                        'fillColor': color,
                        'color': '#000000',
                        'weight': 2,
                        'fillOpacity': 0.5
                    },
                    tooltip=idx
                ).add_to(heatmap)
            
            for amb in dm.ambulances.values():
                if not isinstance(amb['location'], Point):
                    continue
                folium.Marker(
                    location=[amb['location'].y, amb['location'].x],
                    popup=f"{amb['id']} ({amb['status']})",
                    icon=folium.Icon(
                        color='blue' if amb['status'] == 'Disponible' else 'red',
                        icon='ambulance',
                        prefix='fa'
                    )
                ).add_to(heatmap)
            
            return heatmap
        except Exception as e:
            logger.error(f"Failed to generate heatmap: {e}", exc_info=True)
            return folium.Map(location=[32.53, -117.03], zoom_start=12)

def main():
    config = load_config()
    dm = get_data_manager(config)
    pae = PredictiveAnalyticsEngine(dm, config)
    advisor = StrategicAdvisor(dm, config)
    
    st.sidebar.header("Environmental Factors")
    is_holiday = st.sidebar.checkbox("Is Holiday?", value=False)
    weather = st.sidebar.selectbox("Weather", ["Clear", "Rain", "Fog"], index=0)
    traffic_level = st.sidebar.slider("Traffic Level", 0.0, 2.0, 1.0)
    major_event = st.sidebar.checkbox("Major Event?", value=False)
    population_density = st.sidebar.slider("Population Density (per sq km)", 1000, 100000, 50000)
    air_quality_index = st.sidebar.slider("Air Quality Index", 0, 500, 50)
    heatwave_alert = st.sidebar.checkbox("Heatwave Alert?", value=False)
    
    env_factors = EnvFactors(
        is_holiday=is_holiday,
        weather=weather,
        traffic_level=traffic_level,
        major_event=major_event,
        population_density=population_density,
        air_quality_index=air_quality_index,
        heatwave_alert=heatwave_alert
    )
    
    current_incidents = dm.get_current_incidents(env_factors)
    historical_data = []
    uploaded_file = st.sidebar.file_uploader("Upload Historical Data (JSON)", type=['json'])
    if uploaded_file:
        try:
            historical_data = json.load(uploaded_file)
        except Exception as e:
            st.error(f"Failed to load historical data: {e}")
    else:
        historical_data = json.load(dm.generate_sample_history_file())
    
    kpi_df = pae.generate_kpis(historical_data, env_factors, current_incidents)
    forecast_df = pae.forecast_risk(kpi_df)
    recommendations = advisor.recommend_allocations(kpi_df, forecast_df)
    
    st.header("Key Performance Indicators")
    if not kpi_df.empty:
        st.dataframe(kpi_df.round(3))
    else:
        st.warning("No KPI data available.")
    
    st.header("Risk Forecast")
    if not forecast_df.empty:
        st.dataframe(forecast_df.round(3))
    else:
        st.warning("No forecast data available.")
    
    st.header("Resource Allocation Recommendations")
    if recommendations:
        st.table(recommendations)
    else:
        st.info("No reallocation recommendations at this time.")
    
    st.header("Risk Heatmap")
    heatmap = VisualizationSuite.plot_risk_heatmap(dm, kpi_df)
    st_folium(heatmap, width=700, height=500, key="heatmap")
    
    st.header("Generate Report")
    if st.button("Download PDF Report"):
        pdf_buffer = ReportGenerator.generate_pdf_report(kpi_df, recommendations, forecast_df)
        if pdf_buffer.getvalue():
            st.download_button(
                label="Download Report",
                data=pdf_buffer,
                file_name=f"RedShield_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
        else:
            st.error("Failed to generate PDF report.")

if __name__ == "__main__":
    main()
