# 📈 Sales Prediction AI — Advertising ROI & Sales Forecaster

A modern, full-featured **Streamlit** web application powered by **Scikit-Learn** Multiple Linear Regression. The application models and forecasts sales revenue based on multi-channel advertising expenditures across **TV**, **Radio**, and **Newspaper**.

---

## 🚀 Key Features

1. **Machine Learning Model (Multiple Linear Regression)**:
   - Fitted dynamically using `scikit-learn` on train/test split.
   - Comprehensive performance metrics: **$R^2$ Score**, **RMSE**, **MAE**, **MSE**, and **Adjusted $R^2$**.
   - Displays the mathematical regression equation with exact coefficients.
   - **Zero-Failure Startup**: Automatically initializes realistic synthetic sample data if no external dataset file is present, ensuring immediate operation out-of-the-box.
   - Supports uploading custom CSV datasets with automatic schema detection.

2. **Interactive UI Design (Streamlit)**:
   - Polished, responsive layout with custom metric cards, badges, and clean styling.
   - **Two-way synchronized input controls**: Interactive sliders and precision number inputs for TV, Radio, and Newspaper spend.
   - **Quick Campaign Presets**: "Balanced Mix", "TV Blitz", "Radio Boost", and "Low Budget" for one-click scenario testing.
   - Real-time advertising budget distribution and percentage allocation tracker.
   - Prominent **"Predict Sales Revenue"** action button.

3. **Rich Outputs & Visualizations**:
   - **Prominent Metric Cards**:
     - Predicted Sales Volume (with delta comparison to historical mean).
     - Total Advertising Spend.
     - 95% Confidence / Prediction Interval ($\pm 1.96 \times \text{RMSE}$).
     - Sales-to-Spend Multiplier (ROAS).
   - **Visualizations**:
     - **Feature Contribution Breakdown**: Bar chart illustrating how Baseline Intercept + TV + Radio + Newspaper contribute to total predicted sales.
     - **Marginal Channel Regression Lines**: Individual channel trend curves highlighting the user's active spend point with a star marker.
     - **Actual vs. Predicted Diagnostics**: Test set scatter plot with ideal $y = \hat{y}$ reference line.
     - **Residual Distribution Plot**: Histogram and KDE curve verifying normal distribution of error terms.
     - **Correlation Heatmap & Boxplots**: Inter-channel correlations and spend distribution overview.

4. **Scenario Comparison Manager**:
   - Bookmark and compare multiple advertising spend scenarios side-by-side.
   - Export scenario comparison tables directly to CSV.

---

## 📦 Installation & Quickstart

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Install Dependencies
Run the following command in your terminal:
```bash
pip install streamlit pandas scikit-learn matplotlib seaborn numpy
```
*Or install from `requirements.txt`:*
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the Streamlit development server:
```bash
streamlit run app.py
```
The application will launch automatically in your default browser at `http://localhost:8501`.

---

## 🗂️ Project Structure

```
sales_prediction/
├── app.py                  # Main Streamlit application script (self-contained)
├── advertising.csv         # Benchmark dataset (TV, Radio, Newspaper, Sales)
├── Sales_Prediction.ipynb  # Exploratory Data Analysis & Notebook experiments
├── requirements.txt        # Project dependencies
└── README.md               # Documentation & usage guide
```

---

## 📐 Mathematical Formulation

The Multiple Linear Regression model predicts sales volume ($\hat{Y}$) using the equation:

$$\widehat{\text{Sales}} = \beta_0 + \beta_{\text{TV}} \cdot \text{TV} + \beta_{\text{Radio}} \cdot \text{Radio} + \beta_{\text{Newspaper}} \cdot \text{Newspaper}$$

### Sample Benchmark Coefficients:
- **Baseline Intercept ($\beta_0$)**: $\approx 4.625$ (Expected sales with zero advertising expenditure)
- **TV ($\beta_{\text{TV}}$)**: $\approx 0.0544$ (+$54.4$ sales units per $\$1,000$ spent)
- **Radio ($\beta_{\text{Radio}}$)**: $\approx 0.1070$ (+$107.0$ sales units per $\$1,000$ spent)
- **Newspaper ($\beta_{\text{Newspaper}}$)**: $\approx 0.0003$ (Minimal direct impact, diminishing returns)
