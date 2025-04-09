import streamlit as st

def show_learning_interface():
    """Show the learning interface."""
    st.title("Learning Interface")
    
    # Introduction
    st.header("Introduction to Crypto Trading")
    st.write("""
    Welcome to the learning interface! Here you'll find resources to help you understand
    crypto trading fundamentals and how AI can be used in trading strategies.
    """)
    
    # Trading Basics
    st.header("Trading Basics")
    with st.expander("What is Crypto Trading?"):
        st.write("""
        Crypto trading involves buying and selling cryptocurrencies with the goal of making a profit.
        Traders use various strategies and tools to analyze market movements and make decisions.
        """)
    
    with st.expander("Key Concepts"):
        st.write("""
        - **Market Orders**: Buy or sell at the current market price
        - **Limit Orders**: Buy or sell at a specific price
        - **Stop Loss**: Automatically sell if price drops below a certain level
        - **Take Profit**: Automatically sell if price reaches a certain level
        - **Liquidity**: How easily an asset can be bought or sold
        - **Volatility**: How much the price changes over time
        """)
    
    # Technical Analysis
    st.header("Technical Analysis")
    with st.expander("Technical Indicators"):
        st.write("""
        Technical indicators are mathematical calculations based on price, volume, or open interest
        that help traders identify potential trading opportunities.
        
        Common indicators include:
        - Moving Averages
        - Relative Strength Index (RSI)
        - Moving Average Convergence Divergence (MACD)
        - Bollinger Bands
        """)
    
    with st.expander("Chart Patterns"):
        st.write("""
        Chart patterns are formations that appear on price charts and can indicate potential
        future price movements.
        
        Common patterns include:
        - Head and Shoulders
        - Double Top/Bottom
        - Triangles
        - Flags and Pennants
        """)
    
    # AI in Trading
    st.header("AI in Trading")
    with st.expander("How AI is Used in Trading"):
        st.write("""
        AI can be used in trading to:
        - Analyze large amounts of data quickly
        - Identify patterns that humans might miss
        - Make predictions about future price movements
        - Automate trading decisions
        """)
    
    with st.expander("Types of AI Models"):
        st.write("""
        Common AI models used in trading:
        - **Decision Trees**: Simple, interpretable models
        - **Random Forests**: Ensemble of decision trees
        - **Neural Networks**: Complex models that can learn patterns
        - **Transformers**: Advanced models for sequence prediction
        """)
    
    # Risk Management
    st.header("Risk Management")
    with st.expander("Importance of Risk Management"):
        st.write("""
        Risk management is crucial in trading to:
        - Protect your capital
        - Minimize losses
        - Maintain consistent performance
        - Stay in the game long-term
        """)
    
    with st.expander("Risk Management Strategies"):
        st.write("""
        Common risk management strategies:
        - Position sizing
        - Stop-loss orders
        - Diversification
        - Risk-reward ratios
        """)
    
    # Interactive Examples
    st.header("Interactive Examples")
    with st.expander("Try it Yourself"):
        st.write("""
        Here you can experiment with different trading strategies and see how they perform:
        """)
        
        # Example strategy selector
        strategy = st.selectbox(
            "Select a strategy to explore",
            ["Moving Average Crossover", "RSI Strategy", "MACD Strategy"]
        )
        
        if strategy == "Moving Average Crossover":
            st.write("""
            The Moving Average Crossover strategy generates buy signals when a short-term
            moving average crosses above a long-term moving average, and sell signals when
            it crosses below.
            """)
            
            # Interactive parameters
            short_window = st.slider("Short-term MA period", 5, 50, 20)
            long_window = st.slider("Long-term MA period", 50, 200, 100)
            
            # Show example chart
            st.write("Example chart would go here")
        
        elif strategy == "RSI Strategy":
            st.write("""
            The RSI strategy generates buy signals when the RSI is below 30 (oversold)
            and sell signals when it's above 70 (overbought).
            """)
            
            # Interactive parameters
            rsi_period = st.slider("RSI period", 5, 30, 14)
            oversold = st.slider("Oversold level", 10, 40, 30)
            overbought = st.slider("Overbought level", 60, 90, 70)
            
            # Show example chart
            st.write("Example chart would go here")
        
        elif strategy == "MACD Strategy":
            st.write("""
            The MACD strategy generates buy signals when the MACD line crosses above
            the signal line, and sell signals when it crosses below.
            """)
            
            # Interactive parameters
            fast_period = st.slider("Fast EMA period", 5, 20, 12)
            slow_period = st.slider("Slow EMA period", 20, 50, 26)
            signal_period = st.slider("Signal period", 5, 20, 9)
            
            # Show example chart
            st.write("Example chart would go here")
    
    # Additional Resources
    st.header("Additional Resources")
    st.write("""
    - [Token Metrics Documentation](https://docs.tokenmetrics.com)
    - [Crypto Trading Books](https://www.example.com/books)
    - [Online Courses](https://www.example.com/courses)
    - [Trading Communities](https://www.example.com/communities)
    """) 