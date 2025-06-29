import pandas as pd
import logging

logger = logging.getLogger(__name__)

def generate_kpis(self, historical_data: list, env_factors: 'EnvFactors', current_incidents: list) -> pd.DataFrame:
    """Generate KPIs for each zone based on historical data and environmental factors."""
    kpi_data = {
        'Zone': [],
        'Ensemble Risk Score': [],
        'Violence Clustering Score': [],
        'Accident Clustering Score': [],
        'Medical Surge Score': [],
        'Response Time Estimate': [],
        'Resource Adequacy Score': [],
        'Prediction Confidence': []
    }

    if not historical_data and not current_incidents:
        logger.warning("No historical or current incident data provided. Returning empty KPI DataFrame.")
        return pd.DataFrame(kpi_data)

    # Combine historical and current incidents
    all_incidents = historical_data + [{'timestamp': pd.Timestamp.now(), 'incidents': current_incidents}]
    
    # Convert to DataFrame
    rows = []
    for record in all_incidents:
        for incident in record.get('incidents', []):
            incident['timestamp'] = record['timestamp']
            rows.append(incident)
    
    if not rows:
        logger.warning("No incidents found in data. Returning empty KPI DataFrame.")
        return pd.DataFrame(kpi_data)
    
    df = pd.DataFrame(rows)
    
    # Check for zone column and standardize
    possible_zone_columns = ['zone', 'Zone', 'zone_id', 'ZoneID']
    zone_column = None
    for col in possible_zone_columns:
        if col in df.columns:
            zone_column = col
            break
    
    if zone_column is None:
        logger.error("No zone column found in incident data. Expected one of: %s", possible_zone_columns)
        return pd.DataFrame(kpi_data)
    
    if zone_column != 'zone':
        df = df.rename(columns={zone_column: 'zone'})
    
    # Proceed with KPI calculations
    incident_counts = df['zone'].value_counts()
    
    for zone in self.data_manager.zones_gdf.index:
        kpi_data['Zone'].append(zone)
        
        # Incident-based KPIs
        zone_incidents = df[df['zone'] == zone]
        incident_count = incident_counts.get(zone, 0)
        
        # Calculate violence, accident, and medical scores
        violence_score = len(zone_incidents[zone_incidents.get('type', '') == 'Violence'])
        accident_score = len(zone_incidents[zone_incidents.get('type', '') == 'Accident'])
        medical_score = len(zone_incidents[zone_incidents.get('type', '') == 'Medical'])
        
        # Normalize scores
        max_incidents = max(incident_counts.max() if not incident_counts.empty else 1, 1)
        violence_score = violence_score / max_incidents
        accident_score = accident_score / max_incidents
        medical_score = medical_score / max_incidents
        
        # Ensemble risk score
        weights = self.config.get('model_weights', {'violence': 0.4, 'accident': 0.3, 'medical': 0.3})
        ensemble_risk = (
            weights['violence'] * violence_score +
            weights['accident'] * accident_score +
            weights['medical'] * medical_score
        )
        
        # Adjust for environmental factors
        env_impact = (
            1.2 if env_factors.is_holiday else 1.0 +
            1.3 if env_factors.weather in ['Rain', 'Fog'] else 1.0 +
            env_factors.traffic_level * 0.5 +
            1.5 if env_factors.major_event else 0.0 +
            env_factors.population_density / 100000 +
            env_factors.air_quality_index / 500 +
            1.2 if env_factors.heatwave_alert else 0.0
        )
        ensemble_risk *= env_impact
        ensemble_risk = min(1.0, max(0.0, ensemble_risk))  # Clamp to [0, 1]
        
        # Response time estimate
        response_time = 5.0  # Base time in minutes
        available_units = sum(1 for amb in self.data_manager.ambulances.values() if amb['status'] == 'Disponible')
        response_time += incident_count * 0.5
        response_time *= (1 + env_factors.traffic_level)
        response_time = min(response_time, 30.0)
        
        # Resource adequacy
        resource_adequacy = available_units / (incident_count + 1)
        resource_adequacy = min(1.0, max(0.0, resource_adequacy))
        
        # Prediction confidence
        confidence = 0.95 if len(historical_data) > 10 else 0.85
        confidence *= (1 - env_factors.traffic_level * 0.1)
        
        kpi_data['Ensemble Risk Score'].append(ensemble_risk)
        kpi_data['Violence Clustering Score'].append(violence_score)
        kpi_data['Accident Clustering Score'].append(accident_score)
        kpi_data['Medical Surge Score'].append(medical_score)
        kpi_data['Response Time Estimate'].append(response_time)
        kpi_data['Resource Adequacy Score'].append(resource_adequacy)
        kpi_data['Prediction Confidence'].append(confidence)
    
    return pd.DataFrame(kpi_data)
