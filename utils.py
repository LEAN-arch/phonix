# utils.py
import json
import os
from pathlib import Path
import logging
import io
from datetime import datetime
from typing import Dict, Any

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Assuming core.py is in the same directory for type hinting
from core import EnvFactors

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS = [0.5, 1, 3, 6, 12, 24, 72, 144]

def get_default_config() -> Dict[str, Any]:
    """Returns the hardcoded default configuration dictionary."""
    return {
        "mapbox_api_key": None,
        "forecast_horizons_hours": DEFAULT_HORIZONS,
        "data": {
            "zones": {
                "Centro": {
                    "polygon": [[32.52, -117.03], [32.54, -117.03], [32.54, -117.05], [32.52, -117.05]],
                    "prior_risk": 0.7, "population": 50000, "crime_rate_modifier": 1.2
                },
                "Otay": {
                    "polygon": [[32.53, -116.95], [32.54, -116.95], [32.54, -116.98], [32.53, -116.98]],
                    "prior_risk": 0.4, "population": 30000, "crime_rate_modifier": 0.8
                },
                "Playas": {
                    "polygon": [[32.51, -117.11], [32.53, -117.11], [32.53, -117.13], [32.51, -117.13]],
                    "prior_risk": 0.3, "population": 20000, "crime_rate_modifier": 1.0
                }
            },
            "ambulances": {
                "A01": {"status": "Disponible", "home_base": "Centro", "location": [32.53, -117.04]},
                "A02": {"status": "Disponible", "home_base": "Otay", "location": [32.535, -116.965]},
                "A03": {"status": "En Misión", "home_base": "Playas", "location": [32.52, -117.12]}
            },
            "distributions": {
                "zone": {"Centro": 0.5, "Otay": 0.3, "Playas": 0.2},
                "incident_type": {"Trauma-Violence": 0.2, "Trauma-Accident": 0.2, "Medical-Chronic": 0.4, "Medical-Acute": 0.2},
                "triage": {"Red": 0.1, "Yellow": 0.3, "Green": 0.6}
            },
            "road_network": {"edges": [["Centro", "Otay", 5], ["Otay", "Playas", 8], ["Playas", "Centro", 10]]},
            "real_time_api": {"endpoint": "sample_api_response.json", "api_key": None}
        },
        "model_params": {
            "hawkes_process": {"kappa": 0.5, "beta": 1.0, "trauma_weight": 1.5, "violence_weight": 1.8, "aqi_multiplier": 1.5},
            "sir_model": {"beta": 0.3, "gamma": 0.1, "noise_scale": 0.05},
            "laplacian_diffusion_factor": 0.1,
            "response_time_penalty": 3.0,
            "copula_correlation": 0.2,
            "ensemble_weights": { "hawkes": 9, "sir": 8, "bayesian": 8, "graph": 7, "chaos": 7, "info": 9, "tcnn": 10, "tcnn_fallback": 7, "game": 8, "copula": 8, "violence": 9, "accident": 8, "medical": 8 },
            "chaos_amplifier": 1.5,
            "fallback_forecast_decay_rates": {"0.5": 4, "1": 0.9, "3": 0.8, "6": 0.7, "12": 0.6, "24": 0.5, "72": 0.3, "144": 0.2},
            "allocation_forecast_weights": {"0.5": 0.3, "1": 0.25, "3": 0.2, "6": 0.15, "12": 0.1, "24": 0.08, "72": 0.07, "144": 0.05}
        },
        "bayesian_network": {
            "structure": [["Holiday", "IncidentRate"], ["Weather", "IncidentRate"], ["MajorEvent", "IncidentRate"], ["AirQuality", "IncidentRate"], ["Heatwave", "IncidentRate"]],
            "cpds": {
                "Holiday": {"card": 2, "values": [[0.9], [0.1]], "evidence": None, "evidence_card": None},
                "Weather": {"card": 2, "values": [[0.7], [0.3]], "evidence": None, "evidence_card": None},
                "MajorEvent": {"card": 2, "values": [[0.8], [0.2]], "evidence": None, "evidence_card": None},
                "AirQuality": {"card": 2, "values": [[0.8], [0.2]], "evidence": None, "evidence_card": None},
                "Heatwave": {"card": 2, "values": [[0.9], [0.1]], "evidence": None, "evidence_card": None},
                "IncidentRate": { "card": 3, "values": [[0.6, 0.5, 0.4, 0.3, 0.5, 0.4, 0.3, 0.2] * 4, [0.3, 0.3, 0.4, 0.4, 0.3, 0.4, 0.4, 0.5] * 4, [0.1, 0.2, 0.2, 0.3, 0.2, 0.2, 0.3, 0.3] * 4], "evidence": ["Holiday", "Weather", "MajorEvent", "AirQuality", "Heatwave"], "evidence_card": [2, 2, 2, 2, 2]}
            }
        },
        "tcnn_params": {"input_size": 8, "output_size": 24, "channels": [16, 32, 64], "kernel_size": 2, "dropout": 0.2}
    }

def validate_config(config: Dict[str, Any]) -> bool:
    """Validates the configuration, returning True if modifications were made."""
    modified = False
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
        if not isinstance(data.get('crime_rate_modifier'), (int, float)) or data.get('crime_rate_modifier', 0) <= 0:
            logger.warning(f"Invalid crime_rate_modifier for zone '{zone}'. Setting to 1.0.")
            data['crime_rate_modifier'] = 1.0
            modified = True
            
    for amb_id, amb_data in config.get('data', {}).get('ambulances', {}).items():
        if not isinstance(amb_data.get('location'), list) or len(amb_data['location']) != 2:
            raise ValueError(f"Invalid location for ambulance '{amb_id}'.")
        if amb_data.get('home_base') not in zones:
            raise ValueError(f"Ambulance '{amb_id}' has an invalid home_base.")

    return modified

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Loads, validates, and returns the system configuration.
    It will update the config file only if validation makes changes.
    """
    try:
        config = get_default_config()
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                # Deep merge could be used here, but for this structure, update is fine.
                config.update(user_config)
        
        mapbox_key = os.environ.get("MAPBOX_API_KEY", config.get("mapbox_api_key"))
        if mapbox_key and "YOUR_KEY" not in mapbox_key:
            config['mapbox_api_key'] = mapbox_key
        else:
            config['mapbox_api_key'] = None

        if validate_config(config):
            logger.info("Configuration was modified during validation. Saving changes.")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)

        logger.info("System configuration loaded and validated successfully.")
        return config
    except Exception as e:
        logger.error(f"Failed to load or validate config: {e}. Using default configuration.", exc_info=True)
        config = get_default_config()
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        return config

class ReportGenerator:
    """Handles the generation of PDF reports."""
    @staticmethod
    def generate_pdf_report(kpi_df: pd.DataFrame, forecast_df: pd.DataFrame, allocations: Dict[str, int], env_factors: EnvFactors) -> io.BytesIO:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, title="RedShield AI: Phoenix Incident Report")
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("RedShield AI: Phoenix Incident Report", styles['Title']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph("Environmental Factors", styles['Heading2']))
        env_data = [
            ["Factor", "Value"],
            ["Is Holiday", str(env_factors.is_holiday)],
            ["Weather", env_factors.weather],
            ["Traffic Level", f"{env_factors.traffic_level:.2f}"],
            ["Major Event", str(env_factors.major_event)],
            ["Avg. Population Density", f"{env_factors.population_density:.0f}"],
            ["Air Quality Index", f"{env_factors.air_quality_index:.1f}"],
            ["Heatwave Alert", str(env_factors.heatwave_alert)]
        ]
        env_table = Table(env_data, colWidths=[200, 200])
        env_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black), ('BACKGROUND', (0, 1), (-1, -1), colors.beige)
        ]))
        elements.append(env_table)
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Key Performance Indicators (KPIs)", styles['Heading2']))
        kpi_df_report = kpi_df.round(2)
        kpi_data = [kpi_df_report.columns.tolist()] + kpi_df_report.values.tolist()
        kpi_table = Table(kpi_data, hAlign='LEFT')
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black), ('BACKGROUND', (0, 1), (-1, -1), colors.beige)
        ]))
        elements.append(kpi_table)
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph("Forecast Summary (Combined Risk)", styles['Heading2']))
        forecast_pivot = forecast_df.pivot(index='Zone', columns='Horizon (Hours)', values='Combined Risk').round(2)
        forecast_data = [['Zone'] + forecast_pivot.columns.tolist()] + [[idx] + row for idx, row in forecast_pivot.iterrows()]
        forecast_table = Table(forecast_data, hAlign='LEFT')
        forecast_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black), ('BACKGROUND', (0, 1), (-1, -1), colors.beige)
        ]))
        elements.append(forecast_table)
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph("Ambulance Allocation Recommendations", styles['Heading2']))
        alloc_data = [['Zone', 'Recommended Units']] + list(allocations.items())
        alloc_table = Table(alloc_data, colWidths=[200, 200])
        alloc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black), ('BACKGROUND', (0, 1), (-1, -1), colors.beige)
        ]))
        elements.append(alloc_table)
        
        try:
            doc.build(elements)
            buffer.seek(0)
            logger.info("PDF report generated successfully.")
            return buffer
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}", exc_info=True)
            return io.BytesIO()
