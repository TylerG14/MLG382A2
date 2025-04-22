import dash
from dash import html, dcc, dash_table, Input, Output, State
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
import os

# === Load and Prepare Data ===
try:
    df = pd.read_csv("Invistico_Airline.csv")
    print("CSV loaded successfully.")
except Exception as e:
    print(f"ERROR LOADING CSV: {e}")
    df = None

# Continue only if dataset loaded successfully
if df is not None:
    df.drop_duplicates(inplace=True)

    # Fill missing values
    numerical_cols = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())

    ordinal_cols = [
        'Seat comfort', 'Food and drink', 'Gate location', 'Inflight wifi service',
        'Inflight entertainment', 'Online support', 'Ease of Online booking',
        'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service',
        'Cleanliness', 'Online boarding', 'Departure/Arrival time convenient'
    ]
    for col in ordinal_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = df[col].clip(0, 5)

    cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode target and categorical features
    target_encoder = LabelEncoder()
    df['satisfaction'] = target_encoder.fit_transform(df['satisfaction'])

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # One-hot encode
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Split data
    X = df.drop("satisfaction", axis=1)
    y = df["satisfaction"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Classification report
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T.reset_index()
    report.rename(columns={'index': 'Class'}, inplace=True)

# === Dash App ===
app = dash.Dash(__name__)
server = app.server

if df is None:
    app.layout = html.Div("Error loading dataset. Please ensure Invistico_Airline.csv is present.")
else:
    app.layout = html.Div([
        html.H1("Airline Satisfaction Classifier", style={'textAlign': 'center'}),

        dcc.Tabs([
            dcc.Tab(label='Dashboard', children=[
                html.H2("Classification Report"),
                dash_table.DataTable(
                    data=report.round(3).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in report.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'center'},
                    page_size=10
                ),
                html.H2("Feature Importance"),
                dcc.Graph(
                    figure=px.bar(
                        x=model.feature_importances_,
                        y=X_train.columns,
                        orientation='h',
                        title="Feature Importances"
                    )
                )
            ]),

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
                            ], style={'marginBottom': '10px'}) for col in X_train.columns
                        ]),
                        html.Button("Predict", id='predict-button', n_clicks=0, style={'marginTop': '10px'}),
                        html.Div(id='prediction-output', style={'marginTop': '20px', 'fontSize': '20px'})
                    ], style={'flex': 1}),

                    html.Div([
                        html.H4("Feature Guide (Examples):"),
                        html.P("Gender_Male: 1 = Male, 0 = Female (if dropped_first)"),
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
        [State(f'input-{col}', 'value') for col in X_train.columns]
    )
    def predict_satisfaction(n_clicks, *values):
        if n_clicks > 0:
            if None in values:
                return "Please fill in all input fields."
            input_df = pd.DataFrame([values], columns=X_train.columns)
            prediction = model.predict(input_df)[0]
            result = target_encoder.inverse_transform([prediction])[0]
            return f"Predicted Satisfaction Level: **{result}**"
        return ""

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)
