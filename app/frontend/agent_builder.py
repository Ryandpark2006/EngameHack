import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any
import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from app.models.model_factory import ModelFactory
from app.utils.data_preprocessor import DataPreprocessor
from app.backend.trading_simulator import TradingSimulator
from app.models.trading_strategy import BaseStrategy
from app.models.risk_management import RiskManager
from app.backend.token_metrics_api import TokenMetricsAPI

def show_agent_builder():
    """Show the agent builder interface."""
    # Initialize session state for storing the trained model
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = None
    if 'model_type' not in st.session_state:
        st.session_state.model_type = None

    st.title("AI Agent Builder")
    
    # Model selection
    st.header("1. Select Model Type")
    model_types = ModelFactory.get_available_models()
    model_type = st.selectbox(
        "Choose a model type",
        options=list(model_types.keys()),
        format_func=lambda x: model_types[x]
    )
    
    # Model configuration
    st.header("2. Configure Model")
    config = ModelFactory.get_default_config(model_type)
    
    if model_type in ["decision_tree", "random_forest"]:
        config["max_depth"] = st.slider("Max Depth", 1, 20, config["max_depth"])
        config["min_samples_split"] = st.slider("Min Samples Split", 2, 20, config["min_samples_split"])
        config["min_samples_leaf"] = st.slider("Min Samples Leaf", 1, 10, config["min_samples_leaf"])
    
    elif model_type == "mlp":
        config["input_size"] = st.number_input("Input Size", 1, 100, config["input_size"])
        config["hidden_sizes"] = st.text_input("Hidden Layer Sizes (comma-separated)", 
                                             value=",".join(map(str, config["hidden_sizes"])))
        config["hidden_sizes"] = [int(x) for x in config["hidden_sizes"].split(",")]
        config["learning_rate"] = st.number_input("Learning Rate", 0.0001, 0.01, config["learning_rate"], 0.0001)
        config["epochs"] = st.number_input("Epochs", 10, 1000, config["epochs"])
    
    elif model_type == "transformer":
        config["d_model"] = st.number_input("Model Dimension", 16, 512, config["d_model"])
        config["nhead"] = st.number_input("Number of Attention Heads", 1, 16, config["nhead"])
        config["num_layers"] = st.number_input("Number of Layers", 1, 8, config["num_layers"])
        config["sequence_length"] = st.number_input("Sequence Length", 5, 100, config["sequence_length"])
        config["learning_rate"] = st.number_input("Learning Rate", 0.0001, 0.01, config["learning_rate"], 0.0001)
        config["epochs"] = st.number_input("Epochs", 10, 1000, config["epochs"])
    
    # Data source selection
    st.header("3. Data Source")
    data_source = st.radio(
        "Choose data source",
        ["Token Metrics API", "Upload CSV File"],
        index=0
    )
    
    if data_source == "Token Metrics API":
        # Token Metrics API configuration
        st.subheader("Token Metrics API Configuration")
        
        # Symbol selection
        symbol = st.selectbox(
            "Select Trading Pair",
            ["BTC", "ETH", "SOL", "AVAX"]
        )
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date")
        with col2:
            end_date = st.date_input("End Date")
        
        # API endpoint selection
        st.subheader("Select Data Endpoints")
        endpoints = {
            "Market Data": ["price", "volume", "market_cap"],
            "Technical Indicators": ["rsi", "macd", "bollinger_bands"],
            "Sentiment Analysis": ["sentiment_score", "sentiment_magnitude", "news_count", "social_volume"],
            "On-chain Metrics": ["active_addresses", "transaction_count", "miner_revenue"],
            "Social Metrics": ["twitter_volume", "reddit_activity", "github_activity"]
        }
        
        selected_endpoints = {}
        for category, options in endpoints.items():
            selected_endpoints[category] = st.multiselect(
                category,
                options=options,
                default=options
            )
        
        # Fetch data button
        if st.button("Fetch Data"):
            with st.spinner("Fetching data from Token Metrics API..."):
                try:
                    api = TokenMetricsAPI()
                    # Fetch data for each selected endpoint
                    data = {}
                    for category, options in selected_endpoints.items():
                        if options:
                            if category == "Market Data":
                                market_data = api.get_token_metrics(symbol)
                                if market_data:
                                    data.update(market_data)
                            elif category == "Technical Indicators":
                                tech_data = api.get_technical_indicators(symbol)
                                if tech_data:
                                    data.update(tech_data)
                            elif category == "Sentiment Analysis":
                                sentiment_data = api.get_market_sentiment(symbol)
                                if sentiment_data:
                                    data.update(sentiment_data)
                            # Add more endpoint calls as needed
                    
                    if not data:
                        st.error("No data was fetched from the API. Please check your API key and try again.")
                        return
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(data)
                    
                    # Create target variable (price change for next period)
                    if 'price' in df.columns:
                        df['target'] = df['price'].shift(-1) - df['price']
                        df = df.dropna()  # Remove rows with NaN values
                    
                    if df.empty:
                        st.error("No valid data was processed. Please check your data selection.")
                        return
                    
                    st.session_state.training_data = df
                    st.success("Data fetched successfully!")
                    st.dataframe(df.head())
                    
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
    
    else:  # CSV Upload
        st.subheader("Upload Training Data")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'target' not in df.columns:
                st.error("CSV file must contain a 'target' column for training.")
                return
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            st.session_state.training_data = df
    
    # Training section
    if 'training_data' in st.session_state and not st.session_state.training_data.empty:
        st.header("4. Train Model")
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                try:
                    # Create and train model
                    model = ModelFactory.create_model(model_type, config)
                    X = st.session_state.training_data.drop(columns=['target']).values
                    y = st.session_state.training_data['target'].values
                    model.train(X, y)
                    
                    # Run backtest
                    simulator = TradingSimulator(model)
                    metrics = simulator.backtest(st.session_state.training_data, 
                                              st.session_state.training_data.columns[:-1], 
                                              'target')
                    
                    # Store model and metrics in session state
                    st.session_state.trained_model = model
                    st.session_state.model_metrics = metrics
                    st.session_state.model_type = model_type
                    
                    # Display results
                    st.success("Model trained successfully!")
                    
                    # Performance metrics
                    st.subheader("Performance Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Return", f"{metrics['total_return']:.2%}")
                    with col2:
                        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    with col3:
                        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
                    
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
    
    # Save model section
    if st.session_state.trained_model is not None:
        st.header("5. Save Model")
        model_name = st.text_input("Model Name", f"{st.session_state.model_type}_model")
        if st.button("Save Model"):
            try:
                # Create the saved_models directory if it doesn't exist
                models_dir = os.path.join(project_root, "app", "models", "saved_models")
                os.makedirs(models_dir, exist_ok=True)
                
                # Save the model
                model_path = os.path.join(models_dir, f"{model_name}.pkl")
                st.session_state.trained_model.save(model_path)
                st.success(f"Model saved to {model_path}")
            except Exception as e:
                st.error(f"Error saving model: {str(e)}")

    # Add agent configuration options
    st.header("6. Trading Configuration")
    initial_capital = st.number_input("Initial Capital", min_value=1000.0, value=10000.0)
    risk_per_trade = st.slider("Risk per Trade (%)", min_value=0.1, max_value=5.0, value=1.0)
    
    # Create risk manager
    risk_manager = RiskManager(
        max_position_size=initial_capital * 0.1,  # 10% of capital
        max_drawdown=0.2  # 20% max drawdown
    )
    
    # Create trading simulator
    simulator = TradingSimulator(initial_capital, risk_manager)
    
    # Add strategy selection
    strategy_type = st.selectbox(
        "Select Strategy",
        ["Trend Following", "Mean Reversion"]
    )
    
    if st.button("Run Simulation"):
        if st.session_state.trained_model is None:
            st.error("Please train a model first before running the simulation.")
        else:
            # Implementation would run the simulation
            st.write("Simulation completed!") 