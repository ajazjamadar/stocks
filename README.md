# Stock Price Prediction Dashboard

A modern, interactive web dashboard for stock price analysis and prediction using machine learning models with a professional multi-page interface featuring a sleek black/grey theme with glass effects.

## Features

- **Multi-Page Navigation**: Separate pages for different analysis types
- **Professional Dark Theme**: Sleek black/grey color scheme with glass morphism effects
- **Interactive Charts**: Real-time stock price visualization with Plotly
- **Stock Analysis**: Compare multiple stocks (Facebook, Apple, Tesla, Microsoft)
- **Price Prediction**: ML-based stock price forecasting
- **Responsive Layout**: Works on desktop and mobile devices
- **Real-time Updates**: Dynamic data updates and interactive controls
- **Smart Color Coding**: Green for increasing stocks, red for decreasing stocks
- **Error-Free Operation**: All ID-related errors have been resolved for smooth functionality
- **Glass Morphism**: Beautiful backdrop blur effects throughout the interface
- **Smooth Animations**: Sliding and fade-in animations for enhanced user experience

## Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stockpriceprediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:8080
```

## Dependencies

- **Dash**: Web framework for building analytical web applications
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive plotting library
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities
- **TensorFlow**: Deep learning framework (for model loading)

## Dashboard Pages

### 1. Dashboard Overview
- **Key Metrics**: Current price, predicted price, price change, change percentage
- **Market Performance Summary**: Training vs validation data visualization
- **Quick Statistics**: Real-time metrics with smart color coding (green/red)

### 2. NSE-TATAGLOBAL Analysis
- **Training vs Validation Comparison**: Detailed analysis of model performance
- **Prediction Accuracy Analysis**: Actual vs predicted values comparison
- **Performance Metrics**: Mean Absolute Error and Mean Percentage Error
- **Advanced Analytics**: Comprehensive stock performance evaluation

### 3. Multi-Stock Analysis
- **Stock Price Comparison**: High/Low price analysis across multiple stocks
- **Market Volume Analysis**: Trading volume trends and patterns
- **Stock Performance Summary**: Overview of analyzed stocks and data points
- **Interactive Controls**: Quick selection buttons for stock selection (no dropdown)
- **Smart Visualization**: Green lines for increasing trends, red for decreasing trends
- **Quick Selection Buttons**: Interactive button bars for rapid stock selection

## Design Features

- **Glass Morphism**: Modern backdrop blur effects and transparency throughout
- **Smart Color Coding**: Green (#00ff88) for increasing stocks, red (#ff4757) for decreasing
- **Smooth Animations**: Interactive elements with sliding and fade-in animations
- **Responsive Grid**: Adaptive layout for different screen sizes
- **Professional Typography**: Clean, readable fonts with high contrast
- **Navigation System**: Intuitive page switching with visual feedback and animations
- **Enhanced Dropdowns**: Glass-styled multi-select dropdowns
- **Quick Selection Bars**: Interactive button bars for rapid stock selection
- **Modern Visual Effects**: Layered transparency, backdrop filters, and smooth transitions
- **Staggered Animations**: Sequential animation timing for elegant page loading

## Recent Fixes

- **ID Error Resolution**: Fixed all component ID mismatches and callback issues
- **Button Functionality**: Resolved quick selection button click errors
- **Dropdown Stability**: Enhanced dropdown components for better user experience
- **Callback Optimization**: Improved callback functions for smoother interactions
- **Server Bar Removal**: Disabled debug mode to remove the server bar and improve performance
- **Quick Selection Fixes**: Corrected button ID generation and callback logic for error-free operation
- **Dropdown Simplification**: Removed stock comparison dropdown from Multi-Stock Analysis page for cleaner interface
- **Glass Effects & Animations**: Added comprehensive glass morphism effects and sliding animations across all pages

## Customization

### Adding New Stocks
Edit the `stock_symbols` and `stock_names` arrays in `app.py`:

```python
stock_symbols = ['FB', 'AAPL', 'TSLA', 'MSFT', 'GOOGL']
stock_names = ['Facebook', 'Apple', 'Tesla', 'Microsoft', 'Google']
```

### Modifying Colors
Update the color scheme in the styling sections:

```python
# Green for increasing stocks
'color': '#00ff88'

# Red for decreasing stocks  
'color': '#ff4757'

# Glass effect backgrounds
'background': 'rgba(30, 30, 30, 0.9)'
'backdropFilter': 'blur(15px)'
```

### Adding New Pages
Extend the dashboard by creating new page functions and adding navigation buttons.

## Project Structure

```
stockpriceprediction/
├── app.py              # Main application file with multi-page structure
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
├── saved_model.h5     # Pre-trained ML model
├── NSE-TATA.xls       # Sample stock data
├── stock_data.xls     # Additional stock data
└── mlp.ipynb         # Jupyter notebook for model training
```

## Technical Implementation

### Page Navigation
- **Navigation Component**: Professional button-based navigation with glass effects and animations
- **Page Content Management**: Dynamic content loading based on user selection
- **State Management**: Efficient callback system for page updates

### Data Visualization
- **Interactive Charts**: Plotly-based charts with dark theme integration
- **Real-time Updates**: Dynamic chart updates based on user selections
- **Responsive Design**: Charts adapt to different screen sizes
- **Smart Color Logic**: Automatic green/red coloring based on stock performance

### Performance Optimization
- **Efficient Callbacks**: Optimized callback functions for smooth interactions
- **Data Processing**: Streamlined data handling and visualization
- **Memory Management**: Efficient use of resources

### Enhanced UI Components
- **Glass Morphism**: Backdrop blur effects and transparency throughout
- **Quick Selection Buttons**: Interactive button arrays with glass effects
- **Visual Feedback**: Smooth transitions, animations, and responsive interactions
- **Smart Color System**: Automatic color coding for stock performance trends

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Dash and Plotly
- Stock data visualization techniques
- Machine learning model integration
- Modern web design principles
- Glass morphism design trends
- Professional UI/UX standards

## Support

For questions or support, please open an issue in the repository or contact the development team.

---

**Note**: This dashboard uses dummy data for demonstration purposes. For production use, integrate with real-time stock data APIs and ensure proper model validation.
ef0569aabc363546d0fb73ad57066a700bf7c724
# stocks