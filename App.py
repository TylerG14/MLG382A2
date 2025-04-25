import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import joblib
import numpy as np
import os

# Initialize app
app = dash.Dash(__name__)
server = app.server

# Load models
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
logreg_model = joblib.load("logreg_model.pkl")

# Load dataset
df = pd.read_csv("Invistico_Airline.csv")
df.drop_duplicates(inplace=True)

# Final 23 input features (excluding 'satisfaction' which is the target)
input_features = [
    'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
    'Seat comfort', 'Departure/Arrival time convenient', 'Gate location',
    'Inflight wifi service', 'Inflight entertainment', 'Online support',
    'Ease of Online booking', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Cleanliness', 'Online boarding',
    'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Food and drink',
    'Inflight service'  # ✅ Added back in
]

# Categorical encodings
cat_maps = {
    'Gender': {'Female': 0, 'Male': 1},
    'Customer Type': {'Loyal Customer': 0, 'disloyal Customer': 1},
    'Type of Travel': {'Personal Travel': 0, 'Business travel': 1},
    'Class': {'Eco': 0, 'Eco Plus': 1, 'Business': 2},
}

# Dropdown values for 0–5 rating inputs
preset_values = {
    'Seat comfort': list(range(0, 6)),
    'Departure/Arrival time convenient': list(range(0, 6)),
    'Gate location': list(range(0, 6)),
    'Inflight wifi service': list(range(0, 6)),
    'Inflight entertainment': list(range(0, 6)),
    'Online support': list(range(0, 6)),
    'Ease of Online booking': list(range(0, 6)),
    'On-board service': list(range(0, 6)),
    'Leg room service': list(range(0, 6)),
    'Baggage handling': list(range(0, 6)),
    'Checkin service': list(range(0, 6)),
    'Cleanliness': list(range(0, 6)),
    'Online boarding': list(range(0, 6)),
    'Food and drink': list(range(0, 6)),
    'Inflight service': list(range(0, 6)),  # ✅ Added back in
}

# Inputs that allow free typing
manual_input_fields = [
    'Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'
]

# Generate input fields dynamically
def generate_input(feature):
    if feature in cat_maps:
        return dcc.Dropdown(
            id=feature,
            options=[{'label': k, 'value': k} for k in cat_maps[feature]],
            placeholder=f"Select {feature}"
        )
    elif feature in manual_input_fields:
        return dcc.Input(id=feature, type='number', placeholder=f"Enter {feature}")
    elif feature in preset_values:
        return dcc.Dropdown(
            id=feature,
            options=[{'label': str(v), 'value': v} for v in preset_values[feature]],
            placeholder=f"Select {feature}"
        )
    else:
        return dcc.Input(id=feature, type='number', placeholder=f"Enter {feature}")

# App layout
app.layout = html.Div([
    html.H1("Invistico Airline Satisfaction Classifier", style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            html.Label(f"{feature}"),
            generate_input(feature)
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

# Prediction callback
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

    if any(v is None for v in values):
        return "Please complete all fields."

    try:
        encoded_values = []
        for i, feature in enumerate(input_features):
            if feature in cat_maps:
                encoded_values.append(cat_maps[feature][values[i]])
            else:
                encoded_values.append(values[i])

        X_input = np.array(encoded_values).reshape(1, -1)

        if model_name == "rf-button":
            pred = rf_model.predict(X_input)[0]
        elif model_name == "xgb-button":
            pred = xgb_model.predict(X_input)[0]
        elif model_name == "logreg-button":
            pred = logreg_model.predict(X_input)[0]
        else:
            return "Unknown model button."

        return f"✅ Predicted Satisfaction: {pred}"

    except Exception as e:
        return f"⚠️ Error: {e}"

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 1000))
    app.run(debug=False, host="0.0.0.0", port=port)
