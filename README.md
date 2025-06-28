RedShield AI: Phoenix Architecture v2.3
Overview
RedShield AI: Phoenix Architecture v2.3 is a commercial-grade predictive intelligence engine for urban emergency response. It integrates advanced mathematical models to predict and manage medical emergencies (trauma and disease) with a focus on their heuristic, stochastic, and chaotic nature. The application provides real-time insights, resource allocation recommendations, and exportable reports to enhance public safety and emergency preparedness.
Features

Real-Time Data Integration: Fetches live incident data from an external API, with fallback to synthetic data.
Advanced Mathematical Models:
Marked Hawkes Process: Models clustering of trauma incidents.
Spatio-Temporal SIR Model: Predicts disease-related emergency surges.
Lyapunov Exponent: Detects chaotic dynamics in emergency patterns.
Copula-Based Correlation: Models dependencies between trauma and disease events.


Predictive KPIs: Includes Incident Probability, Trauma Clustering Score, Disease Surge Score, and more.
Resource Optimization: Recommends ambulance reallocations for high-risk zones.
Forecasting: Predicts trauma and disease risks over a 3-hour horizon.
Security: User authentication with role-based access (admin/operator).
Reporting: Generates downloadable PDF reports with KPIs, recommendations, and forecasts.
Visualizations: Interactive dashboards, heatmaps, and forecast trends using Plotly and Mapbox.

Installation

Clone the Repository:
git clone <repository-url>
cd redshield-phoenix


Install Dependencies:
pip install -r requirements.txt


Set Environment Variables:

Set MAPBOX_API_KEY for geospatial visualizations:export MAPBOX_API_KEY="your_mapbox_api_key"




Configure the Application:

Edit config.json to include your real-time API endpoint and key (if available).
Ensure zone polygons, ambulance data, and model parameters suit your use case.


Run the Application:
streamlit run RedShield_Phoenix_Documented.py



Usage

Access the Application:

Open your browser and navigate to http://localhost:8501.
Log in with credentials (default: admin/admin123 or operator/operator123).


Configure Environment Factors:

In the sidebar, adjust settings for holiday status, weather, traffic level, major events, and population density.
These factors influence incident predictions.


Run Predictive Cycle:

Click "Run Predictive Cycle" to generate real-time insights.
View results in the following tabs:
Operational Dashboard: KPI table and resource recommendations.
Geospatial Intelligence: Heatmaps for Incident Probability, Trauma Clustering, or Disease Surge.
Forecasting: Trends for trauma and disease risks over 3 hours.
Methodology & KPIs: Explanations of models and KPIs.




Download Reports:

After running a cycle, download a PDF report summarizing insights.



Configuration

config.json:
mapbox_api_key: Your Mapbox API key for visualizations.
data.zones: Define city zones with polygons, prior risks, and population sizes.
data.ambulances: List ambulances with status and locations.
data.distributions: Probabilities for zones, incident types, and triage levels.
data.road_network: Graph edges for spatial connectivity.
data.real_time_api: Endpoint and key for live incident data.
model_params: Parameters for Hawkes, SIR, Laplacian, and Copula models.
bayesian_network: Structure and CPDs for Bayesian inference.
tcnn_params: Settings for the Temporal Convolutional Network.



Testing with Sample API Data

A sample_api_response.json file is provided for testing real-time data integration.
To simulate API calls locally:
Host sample_api_response.json on a local server (e.g., using python -m http.server).
Update config.json with the local endpoint (e.g., http://localhost:8000/sample_api_response.json).



Dependencies
See requirements.txt for required Python packages. Key dependencies include:

streamlit: Web interface
pandas, numpy, scipy: Data processing
geopandas, shapely: Geospatial analysis
networkx: Graph modeling
plotly: Visualizations
requests: API calls
reportlab: PDF generation
torch (optional): Deep learning
pgmpy (optional): Bayesian networks

Notes

The application assumes a pre-trained TCNN model for forecasting. For production, train the model with historical data.
Replace placeholder API endpoint and key in config.json with actual values.
Ensure MAPBOX_API_KEY is set for heatmap visualizations; otherwise, a fallback style is used.
The application is designed for scalability and reliability, with robust error handling and fallbacks.

License
This project is proprietary and intended for use by authorized emergency management personnel only.
Contact
For support, contact the RedShield AI development team at support@xai.redshield.ai.
