# core.py
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import networkx as nx
import logging
import warnings
import json
import requests
from datetime import datetime, timedelta
from scipy.stats import norm
import hashlib
import streamlit as st
import io

# --- Optional Dependency Handling ---
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn: Module = object
    logging.info("PyTorch not found. TCNN model will be disabled.")

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
    logging.info("pgmpy not found. Bayesian network will be disabled.")

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

POPULATION_DENSITY_NORMALIZATION = 100000.0

# --- L1: DATA STRUCTURES ---
# --- ENHANCEMENT: The EnvFactors dataclass is expanded to include new strategic factors ---
@dataclass(frozen=True)
class EnvFactors:
    # Existing factors
    is_holiday: bool
    weather: str
    traffic_level: float
    major_event: bool # This will be derived from public_event_type for backward compatibility
    population_density: float
    air_quality_index: float
    heatwave_alert: bool
    # New strategic factors added based on SME input
    day_type: str
    time_of_day: str
    public_event_type: str
    hospital_divert_status: float
    police_activity: str
    school_in_session: bool


# --- L2: DEEP LEARNING MODEL ---
class TCNN(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, input_size: int, output_size: int, channels: List[int], kernel_size: int, dropout: float):
        if not TORCH_AVAILABLE:
            self.model = None
            self.output_size = output_size
            return
        super().__init__()
        layers = []
        in_channels = input_size
        for out_channels in channels:
            layers.extend([nn.Conv1d(in_channels, out_channels, kernel_size, padding='same'), nn.ReLU(), nn.Dropout(dropout)])
            in_channels = out_channels
        layers.extend([nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(in_channels, output_size)])
        self.model = nn.Sequential(*layers)
        self.output_size = output_size

    def forward(self, x):
        if not TORCH_AVAILABLE or self.model is None:
            return torch.zeros(x.shape[0], self.output_size)
        return self.model(x)

# --- L3: CORE LOGIC CLASSES ---
class DataManager:
    """Manages all data loading, validation, and preparation."""
    def __init__(self, config: Dict[str, Any]):
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
            logger.warning(f"Could not compute Graph Laplacian: {e}. Using identity matrix fallback.")
            self.laplacian_matrix = np.eye(len(self.zones))

    def _build_road_graph(self) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(self.zones)
        edges = self.data_config.get('road_network', {}).get('edges', [])
        for u, v, weight in edges:
            if u in G.nodes and v in G.nodes and isinstance(weight, (int, float)) and weight > 0:
                G.add_edge(u, v, weight=float(weight))
        return G

    def _build_zones_gdf(self) -> gpd.GeoDataFrame:
        zone_data = []
        for name, data in self.data_config['zones'].items():
            poly = Polygon([(lon, lat) for lat, lon in data['polygon']])
            poly = poly.buffer(0) if not poly.is_valid else poly
            if poly.is_empty:
                raise ValueError(f"Polygon for zone '{name}' is invalid or empty.")
            zone_data.append({'name': name, 'geometry': poly, **data})
        if not zone_data:
            raise RuntimeError("Fatal: No valid zones could be loaded from configuration.")
        return gpd.GeoDataFrame(zone_data, crs="EPSG:4326").set_index('name')

    def _initialize_ambulances(self) -> Dict[str, Any]:
        return { amb_id: {'id': amb_id, 'status': data.get('status', 'Disponible'), 'home_base': data.get('home_base'), 'location': Point(float(data['location'][1]), float(data['location'][0]))} for amb_id, data in self.data_config['ambulances'].items() }

    def get_current_incidents(self, env_factors: EnvFactors) -> List[Dict[str, Any]]:
        api_config = self.data_config.get('real_time_api', {})
        endpoint = api_config.get('endpoint', '')
        try:
            if endpoint.startswith(('http://', 'https://')):
                headers = {"Authorization": f"Bearer {api_config.get('api_key')}"} if api_config.get('api_key') else {}
                response = requests.get(endpoint, headers=headers, timeout=10)
                response.raise_for_status()
                incidents = response.json().get('incidents', [])
            else:
                with open(endpoint, 'r', encoding='utf-8') as f:
                    incidents = json.load(f).get('incidents', [])
            valid_incidents = self._validate_incidents(incidents)
            return valid_incidents if valid_incidents else self._generate_synthetic_incidents(env_factors)
        except Exception as e:
            logger.warning(f"Failed to get real-time incidents from '{endpoint}': {e}. Falling back to synthetic data.")
            return self._generate_synthetic_incidents(env_factors)
    
    def _validate_incidents(self, incidents: List[Dict]) -> List[Dict]:
        valid_incidents = []
        for inc in incidents:
            loc = inc.get('location')
            if all(k in inc for k in ['id', 'type', 'triage']) and isinstance(loc, dict) and 'lat' in loc and 'lon' in loc:
                try:
                    inc['location']['lat'] = float(loc['lat'])
                    inc['location']['lon'] = float(loc['lon'])
                    valid_incidents.append(inc)
                except (ValueError, TypeError):
                    logger.warning(f"Skipping incident {inc.get('id', 'N/A')} due to invalid location data.")
        return valid_incidents

    def _generate_synthetic_incidents(self, env_factors: EnvFactors) -> List[Dict[str, Any]]:
        intensity = 5.0 * (1.5 if env_factors.is_holiday else 1.0) * (1.2 if env_factors.weather in ['Rain', 'Fog'] else 1.0) * (2.0 if env_factors.major_event else 1.0)
        num_incidents = int(np.random.poisson(intensity))
        if num_incidents == 0: return []
        city_boundary = self.zones_gdf.unary_union
        bounds = city_boundary.bounds
        incidents = []
        for i in range(num_incidents):
            lon, lat = np.random.uniform(bounds[0], bounds[2]), np.random.uniform(bounds[1], bounds[3])
            if city_boundary.contains(Point(lon, lat)):
                incidents.append({'id': f"SYN-{i}", 'type': np.random.choice(list(self.data_config['distributions']['incident_type'].keys())), 'triage': 'Red', 'location': {'lat': lat, 'lon': lon}, 'timestamp': datetime.utcnow().isoformat()})
        logger.info(f"Generated {len(incidents)} synthetic incidents.")
        return incidents
    
    def generate_sample_history_file(self) -> io.BytesIO:
        # ENHANCEMENT: Update this call to use the new EnvFactors structure with default values
        default_env = EnvFactors(
            is_holiday=False, weather="Clear", traffic_level=1.0, major_event=False,
            population_density=50000, air_quality_index=50.0, heatwave_alert=False,
            day_type='Weekday', time_of_day='Midday', public_event_type='None',
            hospital_divert_status=0.0, police_activity='Normal', school_in_session=True
        )
        buffer = io.BytesIO()
        buffer.write(json.dumps([{'incidents': self._generate_synthetic_incidents(default_env), 'timestamp': (datetime.utcnow() - timedelta(hours=i*24)).isoformat()} for i in range(3)], indent=2).encode('utf-8'))
        buffer.seek(0)
        return buffer

class PredictiveAnalyticsEngine:
    """Encapsulates all predictive models and risk calculation logic."""
    def __init__(self, dm: DataManager, config: Dict[str, Any]):
        self.dm = dm
        self.config = config
        self.model_params = config['model_params']
        self.bn_model = self._build_bayesian_network() if PGMPY_AVAILABLE else None
        self.tcnn_model = TCNN(**config['tcnn_params']) if TORCH_AVAILABLE else None
        self.forecast_df = pd.DataFrame()
        
        weights_config = self.model_params['ensemble_weights']
        self.method_weights = {k: v for k, v in weights_config.items()}
        if not PGMPY_AVAILABLE: self.method_weights['bayesian'] = 0
        if not TORCH_AVAILABLE: self.method_weights['tcnn'] = self.method_weights.get('tcnn_fallback', 0)
        total_weight = sum(self.method_weights.values())
        if total_weight > 0:
            self.method_weights = {k: v / total_weight for k, v in self.method_weights.items()}

    @st.cache_resource(hash_funcs={dict: lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()})
    def _build_bayesian_network(_self) -> Optional[BayesianNetwork]:
        if not PGMPY_AVAILABLE: return None
        try:
            bn_config = _self.config['bayesian_network']
            model = BayesianNetwork(bn_config['structure'])
            for node, params in bn_config['cpds'].items():
                model.add_cpds(TabularCPD(variable=node, variable_card=params['card'], values=params['values'], evidence=params.get('evidence'), evidence_card=params.get('evidence_card')))
            model.check_model()
            logger.info("Bayesian network initialized.")
            return model
        except Exception as e:
            logger.warning(f"Failed to initialize Bayesian network: {e}. Disabling.", exc_info=True)
            return None

    @st.cache_data
    def generate_kpis(_self, historical_data: List[Dict], env_factors: EnvFactors, current_incidents: List[Dict]) -> pd.DataFrame:
        # --- EXPANSION: Add new advanced KPI columns ---
        kpi_cols = [
            'Incident Probability', 'Expected Incident Volume', 'Risk Entropy', 'Anomaly Score', 
            'Spatial Spillover Risk', 'Resource Adequacy Index', 'Chaos Sensitivity Score', 
            'Bayesian Confidence Score', 'Information Value Index', 'Response Time Estimate', 
            'Trauma Clustering Score', 'Disease Surge Score', 'Trauma-Disease Correlation', 
            'Violence Clustering Score', 'Accident Clustering Score', 'Medical Surge Score', 
            'Ensemble Risk Score',
            # New Advanced KPIs
            'STGP_Risk', 'HMM_State_Risk', 'GNN_Structural_Risk', 'Game_Theory_Tension',
            'Integrated_Risk_Score'
        ]
        kpi_df = pd.DataFrame(0, index=_self.dm.zones, columns=kpi_cols, dtype=float)

        all_incidents = [inc for h in historical_data for inc in h.get('incidents', [])] + current_incidents
        if not all_incidents:
            return kpi_df.reset_index().rename(columns={'index': 'Zone'})

        incident_df = pd.DataFrame(all_incidents)
        locations = [Point(loc['lon'], loc['lat']) for loc in incident_df['location']]
        incident_gdf = gpd.GeoDataFrame(incident_df, geometry=locations, crs="EPSG:4326")
        
        incidents_with_zones_raw = gpd.sjoin(incident_gdf, _self.dm.zones_gdf, how="inner", predicate="within")
        
        cols_to_keep = list(incident_gdf.columns) + ['name']
        incidents_with_zones = incidents_with_zones_raw[cols_to_keep].copy()
        incidents_with_zones.rename(columns={'name': 'Zone'}, inplace=True)
        incidents_with_zones.drop_duplicates(subset=['id'], keep='first', inplace=True)

        if incidents_with_zones.empty: return kpi_df.reset_index().rename(columns={'index': 'Zone'})

        incident_counts = incidents_with_zones['Zone'].value_counts().reindex(_self.dm.zones, fill_value=0)
        violence_counts = incidents_with_zones[incidents_with_zones['type'] == 'Trauma-Violence']['Zone'].value_counts().reindex(_self.dm.zones, fill_value=0)
        accident_counts = incidents_with_zones[incidents_with_zones['type'] == 'Trauma-Accident']['Zone'].value_counts().reindex(_self.dm.zones, fill_value=0)
        medical_counts = incidents_with_zones[incidents_with_zones['type'].isin(['Medical-Chronic', 'Medical-Acute'])]['Zone'].value_counts().reindex(_self.dm.zones, fill_value=0)
        
        # --- EXPANSION: Create dynamic multipliers from new strategic factors ---
        day_time_multiplier = {'Weekday': 1.0, 'Friday': 1.2, 'Weekend': 1.3}.get(env_factors.day_type, 1.0)
        day_time_multiplier *= {'Morning Rush': 1.1, 'Midday': 0.9, 'Evening Rush': 1.2, 'Night': 1.4}.get(env_factors.time_of_day, 1.0)
        
        event_multiplier = 1.0
        violence_event_mod = 1.0
        medical_event_mod = 1.0
        if env_factors.public_event_type != 'None':
            event_multiplier = {'Sporting Event': 1.6, 'Concert/Festival': 1.8, 'Public Protest': 2.0}.get(env_factors.public_event_type, 1.0)
            violence_event_mod = {'Sporting Event': 1.8, 'Public Protest': 2.5}.get(env_factors.public_event_type, 1.0)
            medical_event_mod = {'Concert/Festival': 2.0}.get(env_factors.public_event_type, 1.0)

        effective_traffic = env_factors.traffic_level * (1.0 if env_factors.school_in_session else 0.8)
        police_activity_mod = {'Low': 1.1, 'Normal': 1.0, 'High': 0.85}.get(env_factors.police_activity, 1.0)
        system_strain_penalty = 1.0 + (env_factors.hospital_divert_status * 2.0) # 0% divert -> 1x, 100% divert -> 3x penalty
        
        if _self.bn_model:
            try:
                evidence = {'Holiday': 1 if env_factors.is_holiday else 0, 'Weather': 1 if env_factors.weather != 'Clear' else 0, 'MajorEvent': 1 if env_factors.major_event else 0, 'AirQuality': 1 if env_factors.air_quality_index > 100 else 0, 'Heatwave': 1 if env_factors.heatwave_alert else 0}
                result = inference.query(variables=['IncidentRate'], evidence=evidence, show_progress=False)
                rate_probs = result.values
                baseline_rate = np.sum(rate_probs * np.array([1, 5, 10]))
                kpi_df['Bayesian Confidence Score'] = 1 - (np.std(rate_probs) / (np.mean(rate_probs) + 1e-9))
            except Exception as e:
                logger.warning(f"Bayesian inference failed: {e}. Using defaults.")
                baseline_rate, kpi_df['Bayesian Confidence Score'] = 5.0, 0.5
        else:
            baseline_rate, kpi_df['Bayesian Confidence Score'] = 5.0, 0.5

        # --- EXPANSION: Apply new multipliers to the baseline rate ---
        baseline_rate *= day_time_multiplier * event_multiplier
        
        current_dist = incident_counts / (incident_counts.sum() + 1e-9)
        prior_dist = pd.Series(_self.config['data']['distributions']['zone']).reindex(_self.dm.zones, fill_value=1e-9)
        with np.errstate(divide='ignore', invalid='ignore'):
            kl_divergence = np.nansum(current_dist * np.log(current_dist / prior_dist))
            shannon_entropy = -np.nansum(current_dist * np.log2(current_dist))
        kpi_df['Anomaly Score'] = kl_divergence
        kpi_df['Risk Entropy'] = shannon_entropy

        kpi_df['Chaos Sensitivity Score'] = _self._calculate_lyapunov_exponent(historical_data)
        base_probs = (baseline_rate * prior_dist * _self.dm.zones_gdf['crime_rate_modifier']).clip(0, 1)
        kpi_df['Spatial Spillover Risk'] = _self.model_params['laplacian_diffusion_factor'] * (_self.dm.laplacian_matrix @ base_probs.values)

        hawkes_params = _self.model_params['hawkes_process']
        sir_params = _self.model_params['sir_model']
        
        # --- EXPANSION: Apply multipliers to sub-model scores ---
        kpi_df['Violence Clustering Score'] = (violence_counts * hawkes_params['kappa'] * hawkes_params['violence_weight'] * violence_event_mod * police_activity_mod).clip(0, 1)
        kpi_df['Accident Clustering Score'] = (accident_counts * hawkes_params['kappa'] * hawkes_params['trauma_weight'] * effective_traffic).clip(0, 1)
        kpi_df['Medical Surge Score'] = (_self.dm.zones_gdf['population'].apply(lambda s: sir_params['beta'] * medical_counts.get(s, 0) / (s + 1e-9) - sir_params['gamma']) * medical_event_mod).clip(0, 1)
        
        kpi_df['Trauma Clustering Score'] = (kpi_df['Violence Clustering Score'] + kpi_df['Accident Clustering Score']) / 2
        kpi_df['Disease Surge Score'] = kpi_df['Medical Surge Score']
        
        kpi_df['Incident Probability'] = base_probs
        kpi_df['Expected Incident Volume'] = (base_probs * 10 * effective_traffic).round()
        
        available_units = sum(1 for a in _self.dm.ambulances.values() if a['status'] == 'Disponible')
        needed_units = kpi_df['Expected Incident Volume'].sum()
        
        # --- EXPANSION: Apply system strain penalty to resource and response KPIs ---
        kpi_df['Resource Adequacy Index'] = (available_units / (needed_units * system_strain_penalty + 1e-9)).clip(0, 1)
        kpi_df['Response Time Estimate'] = (10.0 * system_strain_penalty) * (1 + _self.model_params['response_time_penalty'] * (1-kpi_df['Resource Adequacy Index']))
        
        kpi_df['Ensemble Risk Score'] = _self.calculate_ensemble_risk_score(kpi_df, historical_data)
        kpi_df['Information Value Index'] = kpi_df['Ensemble Risk Score'].std()

        # --- EXPANSION: ADVANCED ANALYTICS LAYER ---
        # Call the new orchestrator method to calculate and add advanced KPIs
        advanced_kpis = _self._calculate_advanced_kpis(kpi_df, incidents_with_zones)
        for key, value in advanced_kpis.items():
            kpi_df[key] = value

        # Combine base ensemble and advanced KPIs into a final, superior score
        kpi_df['Integrated_Risk_Score'] = (
            0.6 * kpi_df['Ensemble Risk Score'] +
            0.1 * kpi_df['STGP_Risk'] +
            0.1 * kpi_df['HMM_State_Risk'] +
            0.1 * kpi_df['GNN_Structural_Risk'] +
            0.1 * kpi_df['Game_Theory_Tension']
        ).clip(0, 1)
        # --- END OF ADVANCED ANALYTICS LAYER EXPANSION ---

        return kpi_df.fillna(0).reset_index().rename(columns={'index': 'Zone'})
        
    def _calculate_lyapunov_exponent(_self, historical_data: List[Dict]) -> float:
        if len(historical_data) < 2: return 0.0
        try:
            series = pd.Series([len(h.get('incidents', [])) for h in historical_data])
            if len(series) < 10 or series.std() == 0: return 0.0
            return np.log(series.diff().abs().mean() + 1)
        except Exception: return 0.0

    def calculate_ensemble_risk_score(_self, kpi_df: pd.DataFrame, historical_data: List[Dict]) -> pd.Series:
        if kpi_df.empty or not _self.method_weights:
            return pd.Series(0.0, index=kpi_df.index)
        
        normalized_scores_df = pd.DataFrame(index=kpi_df.index)
        
        def normalize(series: pd.Series) -> pd.Series:
            min_val, max_val = series.min(), series.max()
            if max_val > min_val:
                return (series - min_val) / (max_val - min_val)
            return pd.Series(0.0, index=series.index)
            
        chaos_amp = _self.model_params.get('chaos_amplifier', 1.5) if historical_data and np.var([len(h.get('incidents',[])) for h in historical_data]) > np.mean([len(h.get('incidents',[])) for h in historical_data]) else 1.0

        component_map = { 'hawkes': 'Trauma Clustering Score', 'sir': 'Disease Surge Score', 'bayesian': 'Bayesian Confidence Score', 'graph': 'Spatial Spillover Risk', 'chaos': 'Chaos Sensitivity Score', 'info': 'Risk Entropy', 'game': 'Resource Adequacy Index', 'violence': 'Violence Clustering Score', 'accident': 'Accident Clustering Score', 'medical': 'Medical Surge Score'}
        
        for weight_key, metric in component_map.items():
            if metric in kpi_df.columns and _self.method_weights.get(weight_key, 0) > 0:
                col = kpi_df[metric].copy()
                if metric == 'Resource Adequacy Index': col = 1 - col
                if metric == 'Chaos Sensitivity Score': col *= chaos_amp
                normalized_scores_df[weight_key] = normalize(col)

        if _self.method_weights.get('tcnn', 0) > 0 and not _self.forecast_df.empty:
            tcnn_risk = _self.forecast_df[_self.forecast_df['Horizon (Hours)'] == 3].set_index('Zone')[['Violence Risk', 'Accident Risk', 'Medical Risk']].mean(axis=1)
            normalized_scores_df['tcnn'] = normalize(tcnn_risk.reindex(_self.dm.zones, fill_value=0))

        weights = pd.Series(_self.method_weights)
        aligned_scores, aligned_weights = normalized_scores_df.align(weights, axis=1, fill_value=0)
        final_scores = aligned_scores.dot(aligned_weights)
        
        return final_scores.clip(0, 1)

    # --- EXPANSION: ADDITION OF ADVANCED ANALYTICS LAYER METHODS ---
    
    def _calculate_advanced_kpis(self, kpi_df: pd.DataFrame, incidents_with_zones: gpd.GeoDataFrame) -> Dict[str, pd.Series]:
        """Orchestrator for calculating KPIs from advanced models."""
        advanced_kpis = {
            "STGP_Risk": self._calculate_stgp_risk(incidents_with_zones),
            "HMM_State_Risk": self._calculate_hmm_risk(kpi_df),
            "GNN_Structural_Risk": self._calculate_gnn_risk(),
            "Game_Theory_Tension": self._calculate_game_theory_tension(kpi_df)
        }
        return advanced_kpis

    def _calculate_stgp_risk(self, incidents_with_zones: gpd.GeoDataFrame) -> pd.Series:
        """
        Proxy for a Spatiotemporal Gaussian Process (ST-GP).
        This simplified version calculates risk based on proximity to recent high-severity incidents.
        A real ST-GP would provide a full covariance-based prediction with uncertainty.
        """
        stgp_risk = pd.Series(0.0, index=self.dm.zones)
        if incidents_with_zones.empty:
            return stgp_risk
        
        hotspots = incidents_with_zones[incidents_with_zones['triage'] == 'Red']
        if hotspots.empty:
            return stgp_risk

        zone_centroids = self.dm.zones_gdf.geometry.centroid
        for zone_name, centroid in zone_centroids.items():
            distances = hotspots.geometry.distance(centroid)
            length_scale = 0.05
            risk_contribution = np.exp(-0.5 * (distances / length_scale)**2)
            stgp_risk[zone_name] = risk_contribution.sum()
        
        max_risk = stgp_risk.max()
        return (stgp_risk / (max_risk + 1e-9)).clip(0, 1)

    def _calculate_hmm_risk(self, kpi_df: pd.DataFrame) -> pd.Series:
        """
        Proxy for a Hidden Markov Model (HMM).
        A real HMM would infer a hidden state (e.g., 'Calm', 'Agitated', 'Critical').
        This proxy uses thresholding on existing KPIs to simulate state transitions.
        """
        is_volatile = kpi_df['Chaos Sensitivity Score'] > 0.5
        is_strained = kpi_df['Resource Adequacy Index'] < 0.5
        
        hmm_state = pd.Series(0, index=self.dm.zones)
        hmm_state[is_volatile] = 1
        hmm_state[is_strained] = 1
        hmm_state[is_volatile & is_strained] = 2
        
        return (hmm_state / 2.0).clip(0, 1)

    def _calculate_gnn_risk(self) -> pd.Series:
        """
        Proxy for a Graph Neural Network (GNN).
        This proxy uses pre-computed graph centrality as a stand-in for structural importance,
        representing a zone's intrinsic vulnerability due to its position in the network.
        """
        if not hasattr(self, '_centrality'):
            # Calculate and cache centrality as it's static
            self._centrality = pd.Series(nx.betweenness_centrality(self.dm.road_graph), name="centrality")
        
        gnn_risk = self._centrality.copy()
        max_risk = gnn_risk.max()
        return (gnn_risk / (max_risk + 1e-9)).clip(0, 1)

    def _calculate_game_theory_tension(self, kpi_df: pd.DataFrame) -> pd.Series:
        """
        Proxy for a Game Theory model.
        This models the "tension" for resources between competing high-risk zones.
        """
        expected_incidents = kpi_df['Expected Incident Volume']
        total_expected = expected_incidents.sum()
        
        if total_expected == 0:
            return pd.Series(0.0, index=kpi_df.index)
            
        tension = expected_incidents / total_expected
        return tension.clip(0, 1)

    # --- END OF ADVANCED ANALYTICS LAYER ADDITION ---

    def generate_forecast(self, historical_data: List[Dict], env_factors: EnvFactors, kpi_df: pd.DataFrame) -> pd.DataFrame:
        if kpi_df.empty: return pd.DataFrame()
        forecast_data = []
        decay_rates = self.model_params['fallback_forecast_decay_rates']
        for _, row in kpi_df.iterrows():
            for horizon in self.config['forecast_horizons_hours']:
                decay = decay_rates.get(str(horizon), 0.5)
                # --- EXPANSION: Use the most advanced score for forecasting ---
                combined_risk_score = row.get('Integrated_Risk_Score', row['Ensemble Risk Score'])
                forecast_data.append({
                    'Zone': row['Zone'], 'Horizon (Hours)': horizon,
                    'Violence Risk': row['Violence Clustering Score'] * decay,
                    'Accident Risk': row['Accident Clustering Score'] * decay,
                    'Medical Risk': row['Medical Surge Score'] * decay,
                    'Combined Risk': combined_risk_score * decay
                })
        
        forecast_df = pd.DataFrame(forecast_data)
        if forecast_df.empty:
            self.forecast_df = forecast_df
            return self.forecast_df

        risk_cols_to_clip = ['Violence Risk', 'Accident Risk', 'Medical Risk', 'Combined Risk']
        forecast_df[risk_cols_to_clip] = forecast_df[risk_cols_to_clip].clip(0, 1)
        
        self.forecast_df = forecast_df
        return self.forecast_df

    def generate_allocation_recommendations(self, kpi_df: pd.DataFrame, forecast_df: pd.DataFrame) -> Dict[str, int]:
        if kpi_df.empty:
            return {zone: 0 for zone in self.dm.zones}
        available_units = sum(1 for a in self.dm.ambulances.values() if a['status'] == 'Disponible')
        if available_units == 0: return {zone: 0 for zone in self.dm.zones}
        
        # --- EXPANSION: Base allocation on the most sophisticated risk score available ---
        risk_scores = kpi_df.set_index('Zone').get('Integrated_Risk_Score', kpi_df.set_index('Zone')['Ensemble Risk Score'])
        
        total_risk = risk_scores.sum()
        if total_risk == 0:
            allocations = {zone: available_units // len(self.dm.zones) for zone in self.dm.zones}
            allocations[self.dm.zones[0]] += available_units % len(self.dm.zones)
            return allocations
            
        allocations = (available_units * risk_scores / total_risk).round().astype(int).to_dict()
        allocated_units = sum(allocations.values())
        diff = available_units - allocated_units
        if diff != 0:
            risk_order = risk_scores.sort_values(ascending=(diff < 0)).index
            for i in range(abs(diff)):
                allocations[risk_order[i % len(risk_order)]] += np.sign(diff)
        return allocations
