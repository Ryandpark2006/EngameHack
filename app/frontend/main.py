import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from app.frontend.agent_builder import show_agent_builder
from app.frontend.learning_interface import show_learning_interface
from app.frontend.paper_trading import show_paper_trading

# Load environment variables
load_dotenv()

# Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="AI Crypto Trading Simulator",
    page_icon="📈",
    layout="wide"
)

def main():
    """Main application entry point."""
    # Initialize session state for navigation
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

    # Navigation button callbacks
    def nav_to(page_name):
        st.session_state.page = page_name
        st.rerun()

    # Sidebar
    st.sidebar.title("Navigation")
    # Use session_state.page as the default value for the radio button
    selected_page = st.sidebar.radio(
        "Go to",
        ["Home", "Learning Interface", "Agent Builder", "Paper Trading"],
        index=["Home", "Learning Interface", "Agent Builder", "Paper Trading"].index(st.session_state.page)
    )
    
    # Update session state if radio button changes
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()

    # API Key input
    st.sidebar.markdown("---")
    st.sidebar.header("API Configuration")
    api_key = st.sidebar.text_input("Token Metrics API Key", type="password")
    if api_key:
        os.environ["TOKEN_METRICS_API_KEY"] = api_key

    # Main content
    if st.session_state.page == "Home":
        st.title("AI Crypto Trading Simulator")
        st.write("Welcome to the AI Crypto Trading Simulator! This tool helps you learn trading strategies and explore AI-driven trading models.")
        
        st.header("Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Learning Interface")
            st.write("Learn crypto trading fundamentals and AI applications in finance")
            if st.button("Go to Learning Interface"):
                nav_to("Learning Interface")
        
        with col2:
            st.subheader("Agent Builder")
            st.write("Create and train custom AI trading agents")
            if st.button("Go to Agent Builder"):
                nav_to("Agent Builder")
        
        with col3:
            st.subheader("Paper Trading")
            st.write("Simulate trades with backtesting and live modes")
            if st.button("Go to Paper Trading"):
                nav_to("Paper Trading")
        
        st.header("Getting Started")
        st.write("""
        1. Enter your Token Metrics API key in the sidebar
        2. Explore the Learning Interface to understand trading concepts
        3. Use the Agent Builder to create and train your AI trading model
        4. Test your model in the Paper Trading section
        """)

    elif st.session_state.page == "Learning Interface":
        show_learning_interface()

    elif st.session_state.page == "Agent Builder":
        show_agent_builder()

    elif st.session_state.page == "Paper Trading":
        show_paper_trading()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Token Metrics Innovation Challenge")

if __name__ == "__main__":
    main() 