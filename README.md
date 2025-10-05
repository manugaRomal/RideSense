# 🚗 RideSense - Vehicle Condition Prediction System

A comprehensive Streamlit application that predicts vehicle condition using multiple trained machine learning models.

## 🎯 Features

- **Multiple Model Support**: Use Random Forest, Decision Tree, Gradient Boosting, Logistic Regression, or XGBoost
- **Interactive UI**: Beautiful Streamlit interface with real-time predictions
- **Model Comparison**: Compare predictions across different models
- **Confidence Scores**: View prediction probabilities and confidence levels
- **Visual Analytics**: Interactive charts and graphs
- **Easy Deployment**: Ready for Streamlit Cloud or any hosting platform

## 📊 Supported Models

Your application includes these trained models:
- **Random Forest** (`random_forest.pkl`)
- **Decision Tree** (`decision_tree.pkl`) 
- **Gradient Boosting** (`gradient_boosting.pkl`)
- **Logistic Regression** (`logistic_regression.pkl`)
- **XGBoost Classifier** (`xgboost_classifier.pkl`)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Your Models
```bash
python test_models.py
```

### 3. Run the Application
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 📝 Input Features

The system accepts these vehicle specifications (matching your cleaned dataset):

### Basic Information
- **Price**: Vehicle asking price ($100-$100,000)
- **Year**: Manufacturing year (1900-2025)
- **Manufacturer**: ford, chevrolet, toyota, honda, bmw, mercedes, audi, nissan, hyundai, other
- **Model**: Vehicle model (e.g., camry, accord, f150)

### Vehicle Specifications
- **Cylinders**: Number of engine cylinders (1-16)
- **Fuel Type**: gas, diesel, hybrid, electric, other
- **Odometer**: Total mileage (0-500,000 miles)
- **Title Status**: clean, lien, rebuilt, salvage, missing, other
- **Transmission**: automatic, manual, other
- **Drive Type**: fwd, rwd, 4wd, awd, other
- **Vehicle Type**: sedan, suv, truck, coupe, hatchback, convertible, wagon, other
- **Paint Color**: black, white, silver, red, blue, green, other

### Ownership & Location
- **State**: State abbreviation (e.g., CA)
- **Previous Owners**: Number of previous owners (0-10, fewer is generally better)
- **Location Cluster**: Geographic location cluster identifier (0-100)

## 🎯 Prediction Output

The system predicts vehicle condition into categories like:
- **Excellent/Like New**: Minimal wear, excellent condition
- **Good**: Well-maintained with minor wear
- **Fair**: Average condition with expected wear
- **Poor**: Significant wear or potential issues

## 🔧 Model Integration

### How It Works
1. **Model Loading**: All `.pkl` files in the `model/` directory are automatically loaded
2. **Feature Preprocessing**: Input data is preprocessed to match training format
3. **Prediction**: Selected model makes prediction with confidence scores
4. **Visualization**: Results displayed with interactive charts

### Adding New Models
1. Train your model using scikit-learn
2. Save it as a `.pkl` file in the `model/` directory
3. The app will automatically detect and load it

```python
import joblib
from sklearn.ensemble import RandomForestClassifier

# Train your model
model = RandomForestClassifier()
# ... training code ...

# Save the model
joblib.dump(model, 'model/my_new_model.pkl')
```

## 🌐 Deployment Options

### Streamlit Cloud (Recommended)
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set main file: `app.py`
5. Deploy!

### Docker
```bash
# Build and run
docker build -t ridesense .
docker run -p 8501:8501 ridesense
```

### Local Server
```bash
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

## 📁 Project Structure

```
RideSense/
├── app.py                 # Main application entry point
├── test_models.py         # Model testing script
├── test_architecture.py   # Architecture testing script
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── model/                 # Trained models directory
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   ├── gradient_boosting.pkl
│   ├── logistic_regression.pkl
│   └── xgboost_classifier.pkl
├── src/                   # Source code modules
│   ├── __init__.py       # Package initialization
│   ├── logic.py          # Business logic and ML operations
│   ├── ui.py             # Streamlit UI components
│   └── Ridesense.py      # Original app (backup)
└── README.md             # This file
```

## 🧪 Testing

### Test Your Models
```bash
python test_models.py
```

This script will:
- Load all available models
- Test predictions with sample data
- Display model information
- Show prediction probabilities

### Test Architecture
```bash
python test_architecture.py
```

This script will:
- Test module imports
- Verify logic module functionality
- Check UI module integration
- Ensure separated architecture works correctly

### Expected Output
```
🧪 Testing models with sample data:
   price  year manufacturer model  cylinders fuel  odometer title_status transmission drive   type paint_color state  owners location_cluster
0  15000  2018      toyota camry         4  gas     75000       clean    automatic  fwd sedan       white    ca       1               5

==================================================

🔍 Testing Random Forest...
✅ Model loaded successfully
🎯 Prediction: good
📊 Probabilities: {'excellent': 0.1, 'good': 0.7, 'fair': 0.15, 'poor': 0.05}
🌳 Estimators: 100
🏷️  Classes: ['excellent' 'fair' 'good' 'poor']
```

## 🔍 Troubleshooting

### Common Issues

1. **Models not loading**:
   - Check that `.pkl` files are in the `model/` directory
   - Verify file permissions
   - Run `python test_models.py` to debug

2. **Prediction errors**:
   - Ensure input data matches training format
   - Check feature preprocessing
   - Verify model classes match expected output

3. **Deployment issues**:
   - Check `requirements.txt` includes all dependencies
   - Verify model files are included in deployment
   - Check Streamlit Cloud logs for errors

### Debug Mode
```bash
streamlit run app.py --logger.level=debug
```

## 📈 Performance Tips

1. **Caching**: Models are cached using `@st.cache_resource`
2. **Batch Predictions**: Process multiple vehicles efficiently
3. **Model Selection**: Choose the best performing model for your use case
4. **Feature Engineering**: Optimize preprocessing for your specific models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and development purposes.

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section
2. Run the test script
3. Review Streamlit documentation
4. Check model compatibility

---

**Built with ❤️ using Streamlit and Scikit-learn**
