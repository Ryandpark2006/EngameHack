# AI Crypto Trading Simulator

A locally hosted, cost-free application designed for the Token Metrics Innovation Challenge. This tool helps users learn trading strategies, explore AI-driven trading models, and simulate trades using real-time and historical data from the Token Metrics API.

## Features

- Interactive learning interface for crypto trading fundamentals
- AI agent builder with custom model training
- Paper trading sandbox with backtesting and live simulation
- Real-time dashboard visualizations
- Integration with Token Metrics API

## Project Structure

```
.
├── app/                    # Main application code
│   ├── frontend/          # Streamlit UI components
│   ├── backend/           # Core business logic
│   ├── models/            # AI model implementations
│   └── utils/             # Utility functions
├── data/                  # Data storage and processing
├── tests/                 # Test suite
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Token Metrics API key in the environment variables
4. Run the application:
   ```bash
   streamlit run app/frontend/main.py
   ```

## Development

- Frontend: Streamlit
- Backend: Python, Flask
- AI Models: Scikit-learn, PyTorch, Hugging Face
- Data: Token Metrics API, Local CSV storage

## License

MIT License