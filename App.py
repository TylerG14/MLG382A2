import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import joblib
import numpy as np
import os

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Load pre-trained models
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
logreg_model = joblib.load("logreg_model.pkl")

# Load dataset for input reference
df = pd.read_csv("Invistico_Airline.csv")
df.drop_duplicates(inplace=True)

# Define features (MUST match training features exactly)
input_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class',
                  'Flight Distance', 'Inflight wifi service',
                  'Departure/Arrival time convenient', 'Ease of Online booking',
                  'Food and drink', 'Online boarding', 'Seat comfort',
                  'Inflight entertainment', 'On-board service', 'Leg room service',
                  'Baggage handling', 'Checkin service', 'Cleanliness',
                  'Departure Delay in Minutes']

# Helper to get dropdown options
def get_dropdown_options(col):
    return [{'label': str(val), 'value': val} for val in sorted(df[col].dropna().unique())]

# Build layout
app.layout = html.Div([
    html.H1("Invistico Airline Satisfaction Classifier", style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            html.Label(f"{feature}"),
            dcc.Input(id=feature, type='number', placeholder="Enter value") if df[feature].dtype in [np.int64, np.float64]
            else dcc.Dropdown(id=feature, options=get_dropdown_options(feature), placeholder="Select value")
        ], style={'marginBottom': '10px'}) for feature in input_features
    ], style={'columnCount': 2, 'padding': '20px'}),

    html.Div([
        html.Button("Predict (Random Forest)", id='rf-button', n_clicks=0),
        html.Button("Predict (XGBoost)", id='xgb-button', n_clicks=0),
        html.Button("Predict (Logistic Regression)", id='logreg-button', n_clicks=0),
    ], style={'marginTop': '20px'}),

    html.H2("Prediction Result"),
    html.Div(id="prediction-output", style={'fontSize': '20px', 'fontWeight': 'bold', 'marginTop': '10px'})
])

# Callback
@app.callback(
    Output("prediction-output", "children"),
    Input("rf-button", "n_clicks"),
    Input("xgb-button", "n_clicks"),
    Input("logreg-button", "n_clicks"),
    *[Input(feature, 'value') for feature in input_features]
)
def predict(n_rf, n_xgb, n_logreg, *values):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "Awaiting input..."
    
    model_name = ctx.triggered[0]['prop_id'].split('.')[0]

    # Check if inputs are filled
    if any(v is None for v in values):
        return "⚠️ Please provide all inputs."

    try:
        X_input = np.array(values).reshape(1, -1)
        if model_name == "rf-button":
            pred = rf_model.predict(X_input)[0]
        elif model_name == "xgb-button":
            pred = xgb_model.predict(X_input)[0]
        elif model_name == "logreg-button":
            pred = logreg_model.predict(X_input)[0]
        else:
            return "Select a model."
        return f"✅ Predicted Satisfaction: {pred}"
    except Exception as e:
        return f"Error: {e}"

# Run app
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host="0.0.0.0", port=port)
