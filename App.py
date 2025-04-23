import dash
from dash import html, dcc, Input, Output
import pandas as pd
import pickle
import numpy as np

# Load models
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("logreg_model.pkl", "rb") as f:
    logreg_model = pickle.load(f)

# Load dataset (for layout reference)
df = pd.read_csv("Invistico_Airline.csv")

# Prepare dropdown options from dataset
def get_dropdown_options(column):
    return [{'label': str(i), 'value': i} for i in sorted(df[column].dropna().unique())]

# Features you used in your model — adjust to match your pipeline!
input_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'Flight Distance', 'Inflight wifi service',
                  'Departure/Arrival time convenient', 'Ease of Online booking', 'Food and drink', 'Online boarding',
                  'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service',
                  'Baggage handling', 'Checkin service', 'Cleanliness', 'Departure Delay in Minutes']

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Airline Satisfaction Prediction", style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            html.Label(f"{feature}"),
            dcc.Input(id=feature, type='text', placeholder=f"Enter {feature}") if df[feature].dtype in [np.int64, np.float64]
            else dcc.Dropdown(id=feature, options=get_dropdown_options(feature), placeholder=f"Select {feature}")
        ]) for feature in input_features
    ], style={'columnCount': 2, 'padding': '20px'}),

    html.Br(),
    html.Button("Predict (Random Forest)", id='predict-rf', n_clicks=0),
    html.Button("Predict (XGBoost)", id='predict-xgb', n_clicks=0),
    html.Button("Predict (LogReg)", id='predict-logreg', n_clicks=0),
    html.Br(), html.Br(),
    html.Div(id='prediction-output', style={'fontSize': '20px', 'fontWeight': 'bold'})
])

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-rf', 'n_clicks'),
    Input('predict-xgb', 'n_clicks'),
    Input('predict-logreg', 'n_clicks'),
    *[Input(feature, 'value') for feature in input_features]
)
def predict(n_rf, n_xgb, n_log, *inputs):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "Enter input and click a button"
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    try:
        features_array = np.array(inputs).reshape(1, -1)
        if button_id == 'predict-rf':
            pred = rf_model.predict(features_array)[0]
        elif button_id == 'predict-xgb':
            pred = xgb_model.predict(features_array)[0]
        elif button_id == 'predict-logreg':
            pred = logreg_model.predict(features_array)[0]
        else:
            return "Click a prediction button"
        return f"Predicted Satisfaction: {pred}"
    except Exception as e:
        return f"Error during prediction: {e}"

# Run the app
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run_server(debug=True, host="0.0.0.0", port=port)