import dash
from dash import dcc, html, Input, Output, callback_context
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = dash.Dash(__name__)
server = app.server

# Add CSS animations
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <title>Stock Price Prediction Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            /* Smooth scrolling for the entire page */
            html {
                scroll-behavior: smooth;
            }
            
            body {
                scroll-behavior: smooth;
                overflow-x: hidden;
            }
            
            /* Enhanced smooth animations */
            @keyframes slideInFromTop {
                0% {
                    transform: translateY(-100px) scale(0.95);
                    opacity: 0;
                    filter: blur(10px);
                }
                100% {
                    transform: translateY(0) scale(1);
                    opacity: 1;
                    filter: blur(0);
                }
            }
            
            @keyframes slideInFromBottom {
                0% {
                    transform: translateY(100px) scale(0.95);
                    opacity: 0;
                    filter: blur(10px);
                }
                100% {
                    transform: translateY(0) scale(1);
                    opacity: 1;
                    filter: blur(0);
                }
            }
            
            @keyframes slideInFromLeft {
                0% {
                    transform: translateX(-100px) scale(0.95);
                    opacity: 0;
                    filter: blur(10px);
                }
                100% {
                    transform: translateX(0) scale(1);
                    opacity: 1;
                    filter: blur(0);
                }
            }
            
            @keyframes slideInFromRight {
                0% {
                    transform: translateX(100px) scale(0.95);
                    opacity: 0;
                    filter: blur(10px);
                }
                100% {
                    transform: translateX(0) scale(1);
                    opacity: 1;
                    filter: blur(0);
                }
            }
            
            @keyframes slideInFromCenter {
                0% {
                    transform: scale(0.8) translateY(50px);
                    opacity: 0;
                    filter: blur(15px);
                }
                50% {
                    transform: scale(0.95) translateY(25px);
                    opacity: 0.7;
                    filter: blur(5px);
                }
                100% {
                    transform: scale(1) translateY(0);
                    opacity: 1;
                    filter: blur(0);
                }
            }
            
            @keyframes fadeIn {
                0% {
                    opacity: 0;
                    transform: scale(0.98);
                    filter: blur(8px);
                }
                100% {
                    opacity: 1;
                    transform: scale(1);
                    filter: blur(0);
                }
            }
            
            /* Smooth hover effects */
            .smooth-hover {
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            .smooth-hover:hover {
                transform: translateY(-5px) scale(1.02);
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            }
            
            /* Enhanced glass effect with smooth transitions */
            .glass-effect {
                backdrop-filter: blur(25px);
                transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
                will-change: transform, backdrop-filter, box-shadow;
            }
            
            .glass-effect:hover {
                backdrop-filter: blur(35px);
                transform: translateY(-3px);
                box-shadow: 0 25px 50px rgba(0, 0, 0, 0.4);
            }
            
            /* Smooth button interactions */
            button {
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                will-change: transform, box-shadow, background-color;
            }
            
            button:hover {
                transform: translateY(-2px) scale(1.05);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            }
            
            button:active {
                transform: translateY(0) scale(0.98);
                transition: all 0.1s ease;
            }
            
            /* Smooth chart transitions */
            .js-plotly-plot {
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            /* Enhanced page transitions */
            .page-transition {
                animation: pageSlideIn 0.8s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            @keyframes pageSlideIn {
                0% {
                    opacity: 0;
                    transform: translateY(30px) scale(0.98);
                    filter: blur(5px);
                }
                100% {
                    opacity: 1;
                    transform: translateY(0) scale(1);
                    filter: blur(0);
                }
            }
            
            /* Smooth scrollbar */
            ::-webkit-scrollbar {
                width: 12px;
            }
            
            ::-webkit-scrollbar-track {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 6px;
            }
            
            ::-webkit-scrollbar-thumb {
                background: rgba(0, 255, 136, 0.3);
                border-radius: 6px;
                transition: background 0.3s ease;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: rgba(0, 255, 136, 0.5);
            }
            
            /* Performance optimizations */
            * {
                backface-visibility: hidden;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }
            
            /* Smooth focus transitions */
            *:focus {
                outline: none;
                transition: all 0.2s ease;
            }
            
            /* Enhanced loading states */
            .loading {
                animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
            }
            
            @keyframes pulse {
                0%, 100% {
                    opacity: 1;
                }
                50% {
                    opacity: 0.5;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Create dummy data for demonstration
print("Creating dummy data for demonstration...")

# Create dummy NSE-TATA data
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
close_prices = np.random.randn(1000).cumsum() + 100
df_nse = pd.DataFrame({
    'Date': dates,
    'Close': close_prices
})

df_nse.index = df_nse['Date']
new_data = df_nse[['Close']].copy()

dataset = new_data.values
train_size = int(len(dataset) * 0.8)
train = dataset[0:train_size, :]
valid = dataset[train_size:, :]

# Create dummy predictions
closing_price = np.random.randn(len(valid)) * 10 + valid[:, 0].mean()
closing_price = closing_price.reshape(-1, 1)

train_data = new_data[:train_size]
valid_data = new_data[train_size:].copy()
valid_data['Predictions'] = closing_price.flatten()

# Create dummy stock data for the second tab
stock_symbols = ['FB', 'AAPL', 'TSLA', 'MSFT']
stock_names = ['Facebook', 'Apple', 'Tesla', 'Microsoft']

# Create more realistic stock data
dates = pd.date_range('2020-01-01', periods=250, freq='D')
df = pd.DataFrame()

for i, symbol in enumerate(stock_symbols):
    symbol_data = pd.DataFrame({
        'Stock': [symbol] * 250,
        'Date': dates,
        'High': np.random.randn(250).cumsum() + 100 + i*20,
        'Low': np.random.randn(250).cumsum() + 90 + i*20,
        'Volume': np.random.randint(1000000, 10000000, 250)
    })
    df = pd.concat([df, symbol_data], ignore_index=True)

# Calculate some statistics for the dashboard
current_price = valid_data['Close'].iloc[-1]
predicted_price = valid_data['Predictions'].iloc[-1]
price_change = predicted_price - current_price
price_change_pct = (price_change / current_price) * 100

# Navigation component
def create_navigation():
    return html.Div([
        html.Div([
            html.Button("Dashboard Overview", id="btn-overview", n_clicks=0, 
                       style={'margin': '5px', 'padding': '10px 20px', 'borderRadius': '8px', 'border': '1px solid rgba(255, 255, 255, 0.2)', 'background': 'rgba(40, 40, 40, 0.8)', 'color': '#00ff88', 'cursor': 'pointer', 'backdropFilter': 'blur(15px)', 'transition': 'all 0.3s ease', 'transform': 'translateY(0px)', 'boxShadow': '0 4px 15px rgba(0, 0, 0, 0.2)'}),
            html.Button("NSE-TATAGLOBAL Analysis", id="btn-nse", n_clicks=0,
                       style={'margin': '5px', 'padding': '10px 20px', 'borderRadius': '8px', 'border': '1px solid rgba(255, 255, 255, 0.2)', 'background': 'rgba(30, 30, 30, 0.8)', 'color': '#cccccc', 'cursor': 'pointer', 'backdropFilter': 'blur(15px)', 'transition': 'all 0.3s ease', 'transform': 'translateY(0px)', 'boxShadow': '0 4px 15px rgba(0, 0, 0, 0.2)'}),
            html.Button("Multi-Stock Analysis", id="btn-multi", n_clicks=0,
                       style={'margin': '5px', 'padding': '10px 20px', 'borderRadius': '8px', 'border': '1px solid rgba(255, 255, 255, 0.2)', 'background': 'rgba(30, 30, 30, 0.8)', 'color': '#cccccc', 'cursor': 'pointer', 'backdropFilter': 'blur(15px)', 'transition': 'all 0.3s ease', 'transform': 'translateY(0px)', 'boxShadow': '0 4px 15px rgba(0, 0, 0, 0.2)'})
        ], style={'textAlign': 'center', 'padding': '20px', 'background': 'rgba(25, 25, 25, 0.9)', 'borderRadius': '10px', 'border': '1px solid rgba(255, 255, 255, 0.1)', 'backdropFilter': 'blur(20px)', 'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.3)', 
        'animation': 'slideInFromTop 0.8s cubic-bezier(0.4, 0, 0.2, 1)'})
    ])

# Dashboard Overview Page
def create_overview_page():
    return html.Div([
        html.Div([
            html.H1("Stock Price Prediction Dashboard", 
                    style={
                        'background': 'linear-gradient(135deg, #00ff88 0%, #00cc6a 100%)',
                        'color': '#1a1a1a',
                        'padding': '30px',
                        'textAlign': 'center',
                        'margin': '0',
                        'fontSize': '2.5em',
                        'fontWeight': '300',
                        'textShadow': '2px 2px 4px rgba(0,0,0,0.3)',
                        'borderRadius': '20px 20px 0 0'
                    })
        ]),
        
        html.Div([
            # Statistics Cards
            html.Div([
                html.Div([
                    html.Div([
                        html.Div(f"${current_price:.2f}", 
                                style={'fontSize': '2em', 'fontWeight': 'bold', 'color': '#00ff88', 'margin': '10px 0'}),
                        html.Div("Current Price", 
                                style={'color': '#cccccc', 'fontSize': '0.9em', 'textTransform': 'uppercase', 'letterSpacing': '1px'})
                    ], style={
                        'background': 'rgba(30, 30, 30, 0.9)',
                        'borderRadius': '10px',
                        'padding': '20px',
                        'textAlign': 'center',
                        'border': '1px solid rgba(255, 255, 255, 0.1)',
                        'backdropFilter': 'blur(15px)',
                        'boxShadow': '0 8px 25px rgba(0, 0, 0, 0.3)',
                        'transition': 'all 0.3s ease',
                        'animation': 'slideInFromLeft 0.8s cubic-bezier(0.4, 0, 0.2, 1)'
                    }),
                    html.Div([
                        html.Div(f"${predicted_price:.2f}", 
                                style={'fontSize': '2em', 'fontWeight': 'bold', 'color': '#00ff88', 'margin': '10px 0'}),
                        html.Div("Predicted Price", 
                                style={'color': '#cccccc', 'fontSize': '0.9em', 'textTransform': 'uppercase', 'letterSpacing': '1px'})
                    ], style={
                        'background': 'rgba(30, 30, 30, 0.9)',
                        'borderRadius': '10px',
                        'padding': '20px',
                        'textAlign': 'center',
                        'border': '1px solid rgba(255, 255, 255, 0.1)',
                        'backdropFilter': 'blur(15px)',
                        'boxShadow': '0 8px 25px rgba(0, 0, 0, 0.3)',
                        'transition': 'all 0.3s ease',
                        'animation': 'slideInFromTop 0.8s cubic-bezier(0.4, 0, 0.2, 1) 0.1s both'
                    }),
                    html.Div([
                        html.Div(f"{price_change:+.2f}", 
                                style={'fontSize': '2em', 'fontWeight': 'bold', 'color': '#00ff88' if price_change > 0 else '#ff4757', 'margin': '10px 0'}),
                        html.Div("Price Change", 
                                style={'color': '#cccccc', 'fontSize': '0.9em', 'textTransform': 'uppercase', 'letterSpacing': '1px'})
                    ], style={
                        'background': 'rgba(30, 30, 30, 0.9)',
                        'borderRadius': '10px',
                        'padding': '20px',
                        'textAlign': 'center',
                        'border': '1px solid rgba(255, 255, 255, 0.1)',
                        'backdropFilter': 'blur(15px)',
                        'boxShadow': '0 8px 25px rgba(0, 0, 0, 0.3)',
                        'transition': 'all 0.3s ease',
                        'animation': 'slideInFromRight 0.8s cubic-bezier(0.4, 0, 0.2, 1) 0.2s both'
                    }),
                    html.Div([
                        html.Div(f"{price_change_pct:+.2f}%", 
                                style={'fontSize': '2em', 'fontWeight': 'bold', 'color': '#00ff88' if price_change_pct > 0 else '#ff4757', 'margin': '10px 0'}),
                        html.Div("Change Percentage", 
                                style={'color': '#5f6368', 'fontSize': '0.9em', 'textTransform': 'uppercase', 'letterSpacing': '1px'})
                    ], style={
                        'background': 'rgba(30, 30, 30, 0.9)',
                        'borderRadius': '10px',
                        'padding': '20px',
                        'textAlign': 'center',
                        'border': '1px solid rgba(255, 255, 255, 0.1)',
                        'backdropFilter': 'blur(15px)',
                        'boxShadow': '0 8px 25px rgba(0, 0, 0, 0.3)',
                        'transition': 'all 0.3s ease',
                        'animation': 'slideInFromBottom 0.6s ease-out 0.3s both'
                    })
                ], style={
                    'display': 'grid',
                    'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))',
                    'gap': '20px',
                    'margin': '20px 0'
                })
            ], style={
                'background': 'rgba(25, 25, 25, 0.9)',
                'borderRadius': '12px',
                'padding': '20px',
                'margin': '15px 0',
                'border': '1px solid rgba(255, 255, 255, 0.1)',
                'backdropFilter': 'blur(20px)',
                'boxShadow': '0 12px 35px rgba(0, 0, 0, 0.3)',
                'animation': 'fadeIn 0.8s ease-out'
            }),

            # Quick Summary Chart
            html.Div([
                html.H2("Market Performance Summary", 
                        style={
                            'color': '#ffffff',
                            'fontSize': '1.8em',
                            'fontWeight': '500',
                            'margin': '20px 0',
                            'textAlign': 'center'
                        }),
                html.Div([
                    dcc.Graph(
                        id="summary-chart",
                        figure={
                            "data": [
                                go.Scatter(
                                    x=train_data.index,
                                    y=train_data["Close"],
                                    mode='lines',
                                    name='Training Data',
                                    line=dict(color='#00ff88', width=3),
                                    fill='tonexty',
                                    fillcolor='rgba(0, 255, 136, 0.1)'
                                ),
                                go.Scatter(
                                    x=valid_data.index,
                                    y=valid_data["Close"],
                                    mode='lines',
                                    name='Validation Data',
                                    line=dict(color='#00cc6a', width=3),
                                    fill='tonexty',
                                    fillcolor='rgba(0, 204, 106, 0.1)'
                                )
                            ],
                            "layout": go.Layout(
                                title='',
                                xaxis={'title': 'Date', 'gridcolor': '#e0e0e0', 'color': '#333333'},
                                yaxis={'title': 'Closing Price ($)', 'gridcolor': '#e0e0e0', 'color': '#333333'},
                                hovermode='x unified',
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#333333'),
                                margin=dict(l=50, r=50, t=30, b=50)
                            )
                        }
                    )
                ], style={
                    'background': 'rgba(30, 30, 30, 0.9)',
                    'borderRadius': '12px',
                    'padding': '20px',
                    'margin': '15px 0',
                    'border': '1px solid rgba(255, 255, 255, 0.1)',
                    'backdropFilter': 'blur(15px)',
                    'boxShadow': '0 8px 25px rgba(0, 0, 0, 0.3)',
                    'animation': 'slideInFromBottom 0.7s ease-out 0.2s both'
                })
            ], style={
                'background': 'rgba(25, 25, 25, 0.9)',
                'borderRadius': '15px',
                'padding': '25px',
                'margin': '15px 0',
                'border': '1px solid rgba(255, 255, 255, 0.1)',
                'backdropFilter': 'blur(20px)',
                'boxShadow': '0 15px 40px rgba(0, 0, 0, 0.3)',
                'animation': 'fadeIn 1s ease-out 0.3s both'
            })
        ], style={
            'background': '#1a1a1a',
            'padding': '20px'
        })
    ])

# NSE-TATAGLOBAL Analysis Page
def create_nse_analysis_page():
    return html.Div([
        html.Div([
            html.H1("NSE-TATAGLOBAL Stock Analysis", 
                    style={
                        'background': 'linear-gradient(135deg, #00ff88 0%, #00cc6a 100%)',
                        'color': '#1a1a1a',
                        'padding': '30px',
                        'textAlign': 'center',
                        'margin': '0',
                        'fontSize': '2.5em',
                        'fontWeight': '300',
                        'textShadow': '2px 2px 4px rgba(0,0,0,0.3)',
                        'borderRadius': '20px 20px 0 0'
                    })
        ]),
        
        html.Div([
            # Training vs Validation Analysis
            html.Div([
                html.H2("Training Data vs Validation Data Comparison", 
                        style={
                            'color': '#ffffff',
                            'fontSize': '1.8em',
                            'fontWeight': '500',
                            'margin': '20px 0',
                            'textAlign': 'center'
                        }),
                html.Div([
                    dcc.Graph(
                        id="training-validation-chart",
                        figure={
                            "data": [
                                go.Scatter(
                                    x=train_data.index,
                                    y=train_data["Close"],
                                    mode='lines',
                                    name='Training Data',
                                    line=dict(color='#00ff88', width=3),
                                    fill='tonexty',
                                    fillcolor='rgba(0, 255, 136, 0.1)'
                                ),
                                go.Scatter(
                                    x=valid_data.index,
                                    y=valid_data["Close"],
                                    mode='lines',
                                    name='Validation Data',
                                    line=dict(color='#00cc6a', width=3),
                                    fill='tonexty',
                                    fillcolor='rgba(0, 204, 106, 0.1)'
                                )
                            ],
                            "layout": go.Layout(
                                title='',
                                xaxis={'title': 'Date', 'gridcolor': '#e0e0e0', 'color': '#333333'},
                                yaxis={'title': 'Closing Price ($)', 'gridcolor': '#e0e0e0', 'color': '#333333'},
                                hovermode='x unified',
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#333333'),
                                margin=dict(l=50, r=50, t=30, b=50)
                            )
                        }
                    )
                ], style={
                    'background': 'rgba(30, 30, 30, 0.9)',
                    'borderRadius': '12px',
                    'padding': '20px',
                    'margin': '15px 0',
                    'border': '1px solid rgba(255, 255, 255, 0.1)',
                    'backdropFilter': 'blur(15px)',
                    'boxShadow': '0 8px 25px rgba(0, 0, 0, 0.3)',
                    'animation': 'slideInFromLeft 0.6s ease-out'
                }),
                
                # Prediction Analysis
                html.H2("Prediction Accuracy Analysis", 
                        style={
                            'color': '#ffffff',
                            'fontSize': '1.8em',
                            'fontWeight': '500',
                            'margin': '20px 0',
                            'textAlign': 'center',
                            'animation': 'fadeIn 0.8s ease-out 0.2s both'
                        }),
                html.Div([
                    dcc.Graph(
                        id="prediction-accuracy-chart",
                        figure={
                            "data": [
                                go.Scatter(
                                    x=valid_data.index,
                                    y=valid_data["Close"],
                                    mode='lines',
                                    name='Actual Values',
                                    line=dict(color='#00ff88', width=3)
                                ),
                                go.Scatter(
                                    x=valid_data.index,
                                    y=valid_data["Predictions"],
                                    mode='lines',
                                    name='Predicted Values',
                                    line=dict(color='#ff4757', width=3, dash='dash')
                                )
                            ],
                            "layout": go.Layout(
                                title='',
                                xaxis={'title': 'Date', 'gridcolor': '#e0e0e0', 'color': '#333333'},
                                yaxis={'title': 'Closing Price ($)', 'gridcolor': '#e0e0e0', 'color': '#333333'},
                                hovermode='x unified',
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#333333'),
                                margin=dict(l=50, r=50, t=30, b=50)
                            )
                        }
                    )
                ], style={
                    'background': 'rgba(30, 30, 30, 0.9)',
                    'borderRadius': '12px',
                    'padding': '20px',
                    'margin': '15px 0',
                    'border': '1px solid rgba(255, 255, 255, 0.1)',
                    'backdropFilter': 'blur(15px)',
                    'boxShadow': '0 8px 25px rgba(0, 0, 0, 0.3)',
                    'animation': 'slideInFromRight 0.6s ease-out 0.3s both'
                }),

                # Performance Metrics
                html.H2("Performance Metrics", 
                        style={
                            'color': '#ffffff',
                            'fontSize': '1.8em',
                            'fontWeight': '500',
                            'margin': '20px 0',
                            'textAlign': 'center',
                            'animation': 'fadeIn 0.8s ease-out 0.4s both'
                        }),
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div(f"{abs(price_change):.2f}", 
                                    style={'fontSize': '2em', 'fontWeight': 'bold', 'color': '#00ff88', 'margin': '10px 0'}),
                            html.Div("Mean Absolute Error", 
                                    style={'color': '#cccccc', 'fontSize': '0.9em', 'textTransform': 'uppercase', 'letterSpacing': '1px'})
                        ], style={
                            'background': 'rgba(30, 30, 30, 0.9)',
                            'borderRadius': '10px',
                            'padding': '20px',
                            'textAlign': 'center',
                            'border': '1px solid rgba(255, 255, 255, 0.1)',
                            'backdropFilter': 'blur(15px)',
                            'boxShadow': '0 8px 25px rgba(0, 0, 0, 0.3)',
                            'animation': 'slideInFromLeft 0.6s ease-out 0.5s both'
                        }),
                        html.Div([
                            html.Div(f"{abs(price_change_pct):.2f}%", 
                                    style={'fontSize': '2em', 'fontWeight': 'bold', 'color': '#00ff88', 'margin': '10px 0'}),
                            html.Div("Mean Percentage Error", 
                                    style={'color': '#cccccc', 'fontSize': '0.9em', 'textTransform': 'uppercase', 'letterSpacing': '1px'})
                        ], style={
                            'background': 'rgba(30, 30, 30, 0.9)',
                            'borderRadius': '10px',
                            'padding': '20px',
                            'textAlign': 'center',
                            'border': '1px solid rgba(255, 255, 255, 0.1)',
                            'backdropFilter': 'blur(15px)',
                            'boxShadow': '0 8px 25px rgba(0, 0, 0, 0.3)',
                            'animation': 'slideInFromRight 0.6s ease-out 0.6s both'
                        })
                    ], style={
                        'display': 'grid',
                        'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))',
                        'gap': '20px',
                        'margin': '20px 0'
                    })
                ], style={
                    'background': 'rgba(25, 25, 25, 0.9)',
                    'borderRadius': '12px',
                    'padding': '20px',
                    'margin': '15px 0',
                    'border': '1px solid rgba(255, 255, 255, 0.1)',
                    'backdropFilter': 'blur(20px)',
                    'boxShadow': '0 12px 35px rgba(0, 0, 0, 0.3)',
                    'animation': 'fadeIn 0.8s ease-out 0.7s both'
                })
            ], style={
                'background': 'rgba(25, 25, 25, 0.9)',
                'borderRadius': '15px',
                'padding': '25px',
                'margin': '15px 0',
                'border': '1px solid rgba(255, 255, 255, 0.1)'
            })
        ], style={
            'background': '#1a1a1a',
            'padding': '20px'
        })
    ])

# Multi-Stock Analysis Page
def create_multi_stock_page():
    return html.Div([
        html.Div([
            html.H1("Multi-Stock Market Analysis", 
                    style={
                        'background': 'linear-gradient(135deg, #00ff88 0%, #00cc6a 100%)',
                        'color': '#1a1a1a',
                        'padding': '30px',
                        'textAlign': 'center',
                        'margin': '0',
                        'fontSize': '2.5em',
                        'fontWeight': '300',
                        'textShadow': '2px 2px 4px rgba(0,0,0,0.3)',
                        'borderRadius': '20px 20px 0 0'
                    })
        ]),
        
        html.Div([
            # Stock Price Comparison
            html.Div([
                html.H2("Stock Price Comparison Analysis", 
                        style={
                            'color': '#ffffff',
                            'fontSize': '1.8em',
                            'fontWeight': '500',
                            'margin': '20px 0',
                            'textAlign': 'center'
                        }),
                
                # Enhanced Stock Selection Bar
                html.Div([
                    html.H3("Quick Stock Selection", 
                            style={
                                'color': '#ffffff',
                                'fontSize': '1.3em',
                                'fontWeight': '500',
                                'margin': '20px 0 15px 0',
                                'textAlign': 'center',
                                'animation': 'fadeIn 0.8s ease-out 0.2s both'
                            }),
                    html.Div([
                        html.Button(stock_names[i], 
                                  id=f"quick-select-{stock_symbols[i]}",
                                  n_clicks=0,
                                  style={
                                      'margin': '10px',
                                      'padding': '15px 25px',
                                      'borderRadius': '30px',
                                      'border': '2px solid rgba(0, 255, 136, 0.3)',
                                      'background': 'rgba(0, 255, 136, 0.1)',
                                      'color': '#00ff88',
                                      'cursor': 'pointer',
                                      'fontWeight': '700',
                                      'fontSize': '0.95em',
                                      'backdropFilter': 'blur(15px)',
                                      'transition': 'all 0.3s ease',
                                      'textTransform': 'uppercase',
                                      'letterSpacing': '1px',
                                      'minWidth': '120px',
                                      'boxShadow': '0 4px 15px rgba(0, 0, 0, 0.2)',
                                      'animation': f'slideInFromTop 0.6s ease-out {0.3 + i * 0.1}s both'
                                  })
                        for i, symbol in enumerate(stock_symbols)
                    ], style={
                        'display': 'flex',
                        'flexWrap': 'wrap',
                        'justifyContent': 'center',
                        'gap': '15px',
                        'margin': '20px 0',
                        'padding': '10px'
                    }),
                ], style={
                    'background': 'rgba(25, 25, 25, 0.9)',
                    'borderRadius': '18px',
                    'padding': '25px',
                    'margin': '25px 0',
                    'border': '2px solid rgba(255, 255, 255, 0.1)',
                    'backdropFilter': 'blur(20px)',
                    'boxShadow': '0 12px 35px rgba(0, 0, 0, 0.3)',
                    'animation': 'slideInFromLeft 0.7s ease-out 0.3s both'
                }),
                
                html.Div([
                    dcc.Graph(id='stock-comparison-chart')
                ], style={
                    'background': 'rgba(30, 30, 30, 0.9)',
                    'borderRadius': '12px',
                    'padding': '20px',
                    'margin': '15px 0',
                    'border': '1px solid rgba(255, 255, 255, 0.1)',
                    'backdropFilter': 'blur(15px)',
                    'boxShadow': '0 8px 25px rgba(0, 0, 0, 0.3)',
                    'animation': 'slideInFromBottom 0.7s ease-out 0.4s both'
                }),

                # Market Volume Analysis
                html.H2("Market Volume Analysis", 
                        style={
                            'color': '#ffffff',
                            'fontSize': '1.8em',
                            'fontWeight': '500',
                            'margin': '20px 0',
                            'textAlign': 'center',
                            'animation': 'fadeIn 0.8s ease-out 0.5s both'
                        }),
                
                # Enhanced Volume Selection Bar
                html.Div([
                    html.H3("Quick Volume Selection", 
                            style={
                                'color': '#ffffff',
                                'fontSize': '1.3em',
                                'fontWeight': '500',
                                'margin': '20px 0 15px 0',
                                'textAlign': 'center',
                                'animation': 'fadeIn 0.8s ease-out 0.6s both'
                            }),
                    html.Div([
                        html.Button(stock_names[i], 
                                  id=f"quick-volume-{stock_symbols[i]}",
                                  n_clicks=0,
                                  style={
                                      'margin': '10px',
                                      'padding': '15px 25px',
                                      'borderRadius': '30px',
                                      'border': '2px solid rgba(255, 71, 87, 0.3)',
                                      'background': 'rgba(255, 71, 87, 0.1)',
                                      'color': '#ff4757',
                                      'cursor': 'pointer',
                                      'fontWeight': '700',
                                      'fontSize': '0.95em',
                                      'backdropFilter': 'blur(15px)',
                                      'transition': 'all 0.3s ease',
                                      'textTransform': 'uppercase',
                                      'letterSpacing': '1px',
                                      'minWidth': '120px',
                                      'boxShadow': '0 4px 15px rgba(0, 0, 0, 0.2)',
                                      'animation': f'slideInFromBottom 0.6s ease-out {0.7 + i * 0.1}s both'
                                  })
                        for i, symbol in enumerate(stock_symbols)
                    ], style={
                        'display': 'flex',
                        'flexWrap': 'wrap',
                        'justifyContent': 'center',
                        'gap': '15px',
                        'margin': '20px 0',
                        'padding': '10px'
                    }),
                ], style={
                    'background': 'rgba(25, 25, 25, 0.9)',
                    'borderRadius': '18px',
                    'padding': '25px',
                    'margin': '25px 0',
                    'border': '2px solid rgba(255, 255, 255, 0.1)',
                    'backdropFilter': 'blur(20px)',
                    'boxShadow': '0 12px 35px rgba(0, 0, 0, 0.3)',
                    'animation': 'slideInFromRight 0.7s ease-out 0.8s both'
                }),
                
                html.Div([
                    dcc.Graph(id='volume-analysis-chart')
                ], style={
                    'background': 'rgba(30, 30, 30, 0.9)',
                    'borderRadius': '12px',
                    'padding': '20px',
                    'margin': '15px 0',
                    'border': '1px solid rgba(255, 255, 255, 0.1)',
                    'backdropFilter': 'blur(15px)',
                    'boxShadow': '0 8px 25px rgba(0, 0, 0, 0.3)',
                    'animation': 'slideInFromBottom 0.7s ease-out 0.9s both'
                }),

                # Stock Performance Summary
                html.H2("Stock Performance Summary", 
                        style={
                            'color': '#ffffff',
                            'fontSize': '1.8em',
                            'fontWeight': '500',
                            'margin': '20px 0',
                            'textAlign': 'center',
                            'animation': 'fadeIn 0.8s ease-out 1s both'
                        }),
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div(f"{len(stock_symbols)}", 
                                    style={'fontSize': '2em', 'fontWeight': 'bold', 'color': '#00ff88', 'margin': '10px 0'}),
                            html.Div("Total Stocks Analyzed", 
                                    style={'color': '#cccccc', 'fontSize': '0.9em', 'textTransform': 'uppercase', 'letterSpacing': '1px'})
                        ], style={
                            'background': 'rgba(30, 30, 30, 0.9)',
                            'borderRadius': '10px',
                            'padding': '20px',
                            'textAlign': 'center',
                            'border': '1px solid rgba(255, 255, 255, 0.1)',
                            'backdropFilter': 'blur(15px)',
                            'boxShadow': '0 8px 25px rgba(0, 0, 0, 0.3)',
                            'animation': 'slideInFromLeft 0.6s ease-out 1.1s both'
                        }),
                        html.Div([
                            html.Div(f"250", 
                                    style={'fontSize': '2em', 'fontWeight': 'bold', 'color': '#00ff88', 'margin': '10px 0'}),
                            html.Div("Data Points per Stock", 
                                    style={'color': '#cccccc', 'fontSize': '0.9em', 'textTransform': 'uppercase', 'letterSpacing': '1px'})
                        ], style={
                            'background': 'rgba(30, 30, 30, 0.9)',
                            'borderRadius': '10px',
                            'padding': '20px',
                            'textAlign': 'center',
                            'border': '1px solid rgba(255, 255, 255, 0.1)',
                            'backdropFilter': 'blur(15px)',
                            'boxShadow': '0 8px 25px rgba(0, 0, 0, 0.3)',
                            'animation': 'slideInFromRight 0.6s ease-out 1.2s both'
                        })
                    ], style={
                        'display': 'grid',
                        'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))',
                        'gap': '20px',
                        'margin': '20px 0'
                    })
                ], style={
                    'background': 'rgba(25, 25, 25, 0.9)',
                    'borderRadius': '12px',
                    'padding': '20px',
                    'margin': '15px 0',
                    'border': '1px solid rgba(255, 255, 255, 0.1)',
                    'backdropFilter': 'blur(20px)',
                    'boxShadow': '0 12px 35px rgba(0, 0, 0, 0.3)',
                    'animation': 'fadeIn 0.8s ease-out 1.3s both'
                })
            ], style={
                'background': 'rgba(25, 25, 25, 0.9)',
                'borderRadius': '15px',
                'padding': '25px',
                'margin': '15px 0',
                'border': '1px solid rgba(255, 255, 255, 0.1)'
            })
        ], style={
            'background': '#1a1a1a',
            'padding': '20px'
        })
    ])

# Main layout
app.layout = html.Div([
    html.Div([
        create_navigation(),
        html.Div(id="page-content")
    ], style={
        'maxWidth': '1400px',
        'margin': '0 auto',
        'background': 'rgba(20, 20, 20, 0.95)',
        'borderRadius': '20px',
        'border': '1px solid rgba(255, 255, 255, 0.1)',
        'backdropFilter': 'blur(25px)',
        'boxShadow': '0 20px 40px rgba(0, 0, 0, 0.4)',
        'overflow': 'hidden',
        'animation': 'slideInFromCenter 1.2s cubic-bezier(0.4, 0, 0.2, 1)',
        'transition': 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)'
    })
], style={
    'fontFamily': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    'background': 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)',
    'margin': '0',
    'padding': '20px',
    'minHeight': '100vh'
})

# Callback to update page content
@app.callback(
    Output("page-content", "children"),
    [Input("btn-overview", "n_clicks"),
     Input("btn-nse", "n_clicks"),
     Input("btn-multi", "n_clicks")]
)
def update_page_content(overview_clicks, nse_clicks, multi_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return create_overview_page()
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "btn-overview":
        return create_overview_page()
    elif button_id == "btn-nse":
        return create_nse_analysis_page()
    elif button_id == "btn-multi":
        return create_multi_stock_page()
    
    return create_overview_page()

# Callback for stock comparison chart
@app.callback(
    Output('stock-comparison-chart', 'figure'),
    [Input(f'quick-select-{symbol}', 'n_clicks') for symbol in stock_symbols]
)
def update_stock_comparison(*clicks):
    ctx = callback_context
    if not ctx.triggered:
        # Default to showing FB stock data
        selected_stocks = ['FB']
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        symbol = button_id.replace('quick-select-', '')
        # Show the clicked stock's data
        selected_stocks = [symbol]
    
    if not selected_stocks:
        return go.Figure()
    
    dropdown = dict(zip(stock_symbols, stock_names))
    colors = ['#00ff88', '#00cc6a', '#ff4757', '#ff6b7a']
    
    traces = []
    for i, stock in enumerate(selected_stocks):
        stock_data = df[df["Stock"] == stock]
        if not stock_data.empty:
            # Calculate if stock is increasing or decreasing
            first_price = stock_data["High"].iloc[0]
            last_price = stock_data["High"].iloc[-1]
            is_increasing = last_price > first_price
            
            # Use green for increasing, red for decreasing
            line_color = '#00ff88' if is_increasing else '#ff4757'
            
            traces.append(go.Scatter(
                x=stock_data["Date"],
                y=stock_data["High"],
                mode='lines',
                opacity=0.8,
                name=f'{dropdown[stock]} - High',
                line=dict(color=line_color, width=2)
            ))
            traces.append(go.Scatter(
                x=stock_data["Date"],
                y=stock_data["Low"],
                mode='lines',
                opacity=0.6,
                name=f'{dropdown[stock]} - Low',
                line=dict(color=line_color, width=2, dash='dot')
            ))
    
    return {
        'data': traces,
        'layout': go.Layout(
            title='',
            xaxis={"title": 'Date', 'gridcolor': '#e0e0e0', 'color': '#333333', 'rangeslider': {'visible': True}, 'type': 'date'},
            yaxis={"title": 'Price (USD)', 'gridcolor': '#e0e0e0', 'color': '#333333'},
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff'),
            margin=dict(l=50, r=50, t=30, b=50)
        )
    }

# Callback for volume analysis chart
@app.callback(
    Output('volume-analysis-chart', 'figure'),
    [Input(f'quick-volume-{symbol}', 'n_clicks') for symbol in stock_symbols]
)
def update_volume_analysis(*clicks):
    ctx = callback_context
    if not ctx.triggered:
        # Default to showing FB volume data
        selected_stocks = ['FB']
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        symbol = button_id.replace('quick-volume-', '')
        # Show the clicked stock's volume data
        selected_stocks = [symbol]
    
    if not selected_stocks:
        return go.Figure()
    
    dropdown = dict(zip(stock_symbols, stock_names))
    colors = ['#00ff88', '#00cc6a', '#ff4757', '#ff6b7a']
    
    traces = []
    for i, stock in enumerate(selected_stocks):
        stock_data = df[df["Stock"] == stock]
        if not stock_data.empty:
            # Calculate if stock volume is increasing or decreasing
            first_volume = stock_data["Volume"].iloc[0]
            last_volume = stock_data["Volume"].iloc[-1]
            is_increasing = last_volume > first_volume
            
            # Use green for increasing, red for decreasing
            line_color = '#00ff88' if is_increasing else '#ff4757'
            
            traces.append(go.Scatter(
                x=stock_data["Date"],
                y=stock_data["Volume"],
                mode='lines',
                opacity=0.8,
                name=f'{dropdown[stock]} Volume',
                line=dict(color=line_color, width=3),
                fill='tonexty',
                fillcolor=f'rgba({int(line_color[1:3], 16)}, {int(line_color[3:5], 16)}, {int(line_color[5:7], 16)}, 0.1)'
            ))
    
    return {
        'data': traces,
        'layout': go.Layout(
            title='',
            xaxis={"title": 'Date', 'gridcolor': '#444444', 'color': '#ffffff', 'rangeslider': {'visible': True}, 'type': 'date'},
            yaxis={"title": 'Volume', 'gridcolor': '#444444', 'color': '#ffffff'},
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff'),
            margin=dict(l=50, r=50, t=30, b=50)
        )
    }

if __name__ == '__main__':
    print("Starting Stock Price Prediction Dashboard...")
    print("Access the dashboard at: http://localhost:8080")
    app.run(debug=False, host='0.0.0.0', port=8080)
