import dash
from dash import html, dcc, dash_table, Input, Output, State
import pandas as pd
import plotly.express as px
import joblib
import os

# Load pre-trained model and other components
try:
    model = joblib.load("model.pkl")
    target_encoder = joblib.load("target_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    df = pd.read_csv("Invistico_Airline.csv")
    df.drop_duplicates(inplace=True)
    print("Model and data loaded successfully.")
except Exception as e:
    print(f"ERROR LOADING MODEL OR DATA: {e}")
    df = None

app = dash.Dash(__name__)
server = app.server

if df is None:
    app.layout = html.Div("Error loading model or dataset.")
else:
    app.layout = html.Div([
        html.H1("Airline Satisfaction Classifier", style={'textAlign': 'center'}),

        dcc.Tabs([
            dcc.Tab(label='Raw Dataset', children=[
                html.H2("Airline Dataset"),
                dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df.columns],
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'}
                )
            ]),

            dcc.Tab(label='Predict Satisfaction', children=[
                html.H2("Predict New Passenger Satisfaction"),
                html.Div(style={'display': 'flex', 'gap': '40px'}, children=[

                    html.Div([
                        html.Div(id="input-fields", children=[
                            html.Div([
                                html.Label(f"{col}"),
                                dcc.Input(id=f'input-{col}', type='number', placeholder=f"Enter {col}", step=0.01)
                            ], style={'marginBottom': '10px'}) for col in feature_columns
                        ]),
                        html.Button("Predict", id='predict-button', n_clicks=0, style={'marginTop': '10px'}),
                        html.Div(id='prediction-output', style={'marginTop': '20px', 'fontSize': '20px'})
                    ], style={'flex': 1}),

                    html.Div([
                        html.H4("Feature Guide (Examples):"),
                        html.P("Gender_Male: 1 = Male, 0 = Female (if dropped_first=True)"),
                        html.P("Class_Eco: 1 = Economy, 0 = other"),
                        html.P("Checkin service: Ratings from 0–5"),
                        html.P("Departure Delay in Minutes: Number in minutes")
                    ], style={
                        'flex': 1,
                        'backgroundColor': '#f9f9f9',
                        'padding': '15px',
                        'borderRadius': '8px',
                        'fontSize': '14px'
                    })
                ])
            ])
        ])
    ])

    @app.callback(
        Output('prediction-output', 'children'),
        Input('predict-button', 'n_clicks'),
        [State(f'input-{col}', 'value') for col in feature_columns]
    )
    def predict_satisfaction(n_clicks, *values):
        if n_clicks > 0:
            if None in values:
                return "Please fill in all input fields."
            input_df = pd.DataFrame([values], columns=feature_columns)
            prediction = model.predict(input_df)[0]
            result = target_encoder.inverse_transform([prediction])[0]
            return f"Predicted Satisfaction Level: **{result}**"
        return ""

# Start app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)
