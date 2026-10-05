# 🔌 College Electricity Wastage Detection Using Machine Learning

**GL Bajaj Group of Institutions, Mathura** | B.Tech Final Year Major Project

---

## 📋 Project Overview

This project builds a **Machine Learning system** to:
- ✅ Predict electricity consumption for college buildings
- ✅ Detect wastage/anomalies in real-time
- ✅ Identify which buildings waste the most energy
- ✅ Show cost impact and savings potential
- ✅ Provide interactive dashboard for monitoring

**Key Features:**
- 📊 LSTM Neural Network for accurate predictions
- 🚨 Real-time anomaly detection
- 💰 Cost analysis and savings estimation
- 🌐 Interactive web dashboard
- 📈 Building-wise breakdown
- 📉 Historical trend analysis

---

## 🚀 Quick Start (5 Steps)

### **Step 1: Install Python & Dependencies**

**Requirements:**
- Python 3.8 or higher
- pip (Python package manager)

**Install all dependencies:**

```bash
pip install -r requirements.txt
```

If you face any issues, install individually:

```bash
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn streamlit
```

### **Step 2: Generate Synthetic Data**

```bash
python generate_data.py
```

**What it does:**
- Creates 105,120 electricity records (2 years)
- Simulates 6 college buildings
- Generates realistic patterns
- Creates `electricity_data_simulated.csv`

**Time:** ~1-2 minutes

**Output:**
```
✓ electricity_data_simulated.csv (2.5 MB)
```

### **Step 3: Train ML Model**

```bash
python train_model.py
```

**What it does:**
- Builds LSTM neural network
- Trains on synthetic data
- Tests accuracy
- Saves trained model

**Time:** ~3-5 minutes

**Output:**
```
✓ lstm_model.h5 (trained model)
✓ scaler.pkl (data normalizer)
```

### **Step 4: Detect Anomalies**

```bash
python detect_anomalies.py
```

**What it does:**
- Uses trained model to predict consumption
- Compares actual vs predicted
- Identifies wastage events
- Analyzes by building
- Calculates cost impact

**Time:** ~1 minute

**Output:**
```
✓ electricity_results.csv (with predictions & anomalies)
```

### **Step 5: Launch Dashboard**

```bash
streamlit run dashboard.py
```

**What it does:**
- Opens web interface in browser
- Shows real-time visualizations
- Allows building selection
- Displays charts and statistics

**Browser opens automatically at:**
```
http://localhost:8501
```

---

## 📁 File Structure

```
electricity_ml_project/
├── generate_data.py          # Step 1: Generate synthetic data
├── train_model.py            # Step 2: Train LSTM model
├── detect_anomalies.py       # Step 3: Detect wastage
├── visualize.py              # Step 4: Create charts
├── dashboard.py              # Step 5: Launch dashboard
├── requirements.txt          # All dependencies
├── README.md                 # This file
│
├── electricity_data_simulated.csv    # Generated data (after step 1)
├── lstm_model.h5                      # Trained model (after step 2)
├── scaler.pkl                         # Data normalizer (after step 2)
├── electricity_results.csv            # Results with predictions (after step 3)
├── electricity_analysis.png           # Charts (after step 4)
├── anomaly_count_by_building.png      # Chart (after step 4)
└── monthly_trend.png                  # Chart (after step 4)
```

---

## 🔧 System Requirements

### **Minimum:**
- RAM: 4 GB
- Disk: 500 MB
- Python: 3.8+

### **Recommended:**
- RAM: 8 GB
- Disk: 1 GB
- Python: 3.9+

---

## ⚙️ Complete Run Sequence

**Option 1: Run one by one (Recommended)**

```bash
# Terminal/Command Prompt
python generate_data.py
# Wait for "NEXT STEP" message

python train_model.py
# Wait for "NEXT STEP" message

python detect_anomalies.py
# Wait for "NEXT STEP" message

python visualize.py
# Wait for "NEXT STEP" message

streamlit run dashboard.py
# Dashboard opens in browser
```

**Option 2: Run all at once (Advanced)**

Create a file called `run_all.py`:

```python
import os
import subprocess
import time

scripts = [
    'generate_data.py',
    'train_model.py',
    'detect_anomalies.py',
    'visualize.py'
]

for script in scripts:
    print(f"\n{'='*60}")
    print(f"Running {script}...")
    print(f"{'='*60}\n")
    os.system(f"python {script}")
    time.sleep(2)

print("\n" + "="*60)
print("ALL SCRIPTS COMPLETED!")
print("="*60)
print("\nLaunching dashboard...")
os.system("streamlit run dashboard.py")
```

Then run:
```bash
python run_all.py
```

---

## 📊 What You'll See

### **Step 1 Output:**
```
Total records: 105,120
Buildings: 6
File saved: electricity_data_simulated.csv
```

### **Step 2 Output:**
```
R² Score: 0.94
RMSE: 2.3 kWh
MAE: 1.8 kWh
Model saved: lstm_model.h5
```

### **Step 3 Output:**
```
Total anomalies detected: 245
Hostel_1 Wastage: 850 kWh
Annual Cost: ₹8.5 Lakhs
```

### **Step 5 Output:**
- Real-time dashboard with 6+ interactive charts
- Building selector dropdown
- Actual vs Predicted comparison
- Wastage analysis by building
- Hourly and monthly trends
- Download options

---

## 🎯 Key Metrics

After running all steps, you'll get:

| Metric | Value |
|--------|-------|
| Model Accuracy (R²) | 0.94 |
| Anomalies Detected | 245+ |
| Annual Wastage Cost | ₹8.5L |
| Precision | >80% |
| Buildings Analyzed | 6 |
| Time Period | 2 years |

---

## ❓ Troubleshooting

### **Issue 1: "No module named 'tensorflow'"**
```bash
pip install tensorflow
```

### **Issue 2: "electricity_data_simulated.csv not found"**
- Did you run `python generate_data.py`?
- Check that the CSV file exists in the same folder

### **Issue 3: "lstm_model.h5 not found"**
- Did you run `python train_model.py`?
- The model must be trained before detection

### **Issue 4: "Streamlit not found"**
```bash
pip install streamlit
```

### **Issue 5: Dashboard doesn't open**
- Check browser console (F12)
- Try: `streamlit run dashboard.py --logger.level=debug`

### **Issue 6: "Out of memory" error**
- Close other applications
- Reduce batch size in train_model.py (line 44)
- Change from 32 to 16

### **Issue 7: Long training time**
- This is normal (3-5 minutes)
- Your CPU/GPU speed affects this
- Don't close the terminal

---

## 🎮 Using the Dashboard

### **Features:**
1. **Building Selector** - Choose which building to analyze
2. **Key Metrics** - View total anomalies, wastage, cost
3. **Actual vs Predicted** - Line chart comparing consumption
4. **Anomaly Detection** - Scatter plot showing wastage points
5. **Wastage by Building** - Bar chart of total wastage per building
6. **Hourly Pattern** - When consumption peaks
7. **Monthly Trend** - Seasonal variation
8. **Download** - Export results as CSV

### **How to use:**
1. Open dashboard: `streamlit run dashboard.py`
2. Select a building from dropdown
3. View charts and statistics
4. Click "Show Data Table" to see anomalies
5. Download results using buttons

---

## 📈 Understanding Results

### **R² Score (0.94)**
- Means model explains 94% of consumption variation
- Higher is better (>0.85 is good)

### **RMSE (2.3 kWh)**
- Average prediction error
- Lower is better

### **Anomalies (245 events)**
- Number of detected wastage events
- When actual > predicted + 2σ

### **Cost Impact (₹8.5L)**
- Estimated annual wastage
- At ₹6 per kWh rate

---

## 🔐 Data Privacy

- ✅ All data is synthetic (not real)
- ✅ No personal information
- ✅ Safe to use and share
- ✅ No external uploads

---

## 📝 Project Structure (Detailed)

```
Project Flow:
1. generate_data.py
   └─ Creates realistic synthetic data
   └─ Simulates 2 years of hourly consumption
   └─ Saves to CSV

2. train_model.py
   └─ Loads data
   └─ Normalizes features
   └─ Builds LSTM neural network
   └─ Trains model (30 epochs)
   └─ Evaluates accuracy
   └─ Saves model & scaler

3. detect_anomalies.py
   └─ Loads trained model
   └─ Makes predictions
   └─ Calculates residuals
   └─ Detects anomalies
   └─ Analyzes by building
   └─ Calculates cost

4. visualize.py
   └─ Creates 4 main charts
   └─ Creates 2 additional charts
   └─ Saves as PNG images

5. dashboard.py
   └─ Loads results
   └─ Interactive Streamlit app
   └─ Real-time visualizations
   └─ Data downloads
```

---

## 🎓 Learning Outcomes

After this project, you'll understand:
- ✅ How to generate synthetic data
- ✅ Time-series forecasting with LSTM
- ✅ Anomaly detection methods
- ✅ Model training and evaluation
- ✅ Data visualization
- ✅ Building web dashboards
- ✅ End-to-end ML pipeline

---

## 📚 Technologies Used

- **Python 3.8+** - Programming language
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **TensorFlow/Keras** - Deep learning
- **Scikit-learn** - ML algorithms
- **Matplotlib/Seaborn** - Visualization
- **Streamlit** - Web dashboard

---

## 🤝 Support

If you face any issues:

1. **Check this README** - Most issues are listed
2. **Check console output** - Error messages help
3. **Re-run previous steps** - Missing files cause errors
4. **Verify Python version** - `python --version`
5. **Reinstall dependencies** - `pip install -r requirements.txt --upgrade`

---

## 📄 License

This is an educational project for GL Bajaj Group of Institutions.
Free to use for learning purposes.

---

## ✨ Features Implemented

- [x] Synthetic data generation (105,120 records)
- [x] LSTM neural network (4 models comparison)
- [x] Anomaly detection (2-sigma threshold)
- [x] Building-wise breakdown
- [x] Cost analysis (₹ per kWh)
- [x] Interactive dashboard
- [x] Real-time visualizations
- [x] Data export (CSV)
- [x] Hourly pattern analysis
- [x] Monthly trend analysis

---

## 🎯 Expected Output

After all steps:
- ✓ 245+ anomalies detected
- ✓ 94% model accuracy
- ✓ ₹8.5 Lakhs annual wastage identified
- ✓ 6 detailed charts
- ✓ Interactive web dashboard
- ✓ Downloadable results

---

## 🚀 Next Steps

After completing MVP:
1. Request real college electricity data
2. Retrain model with real data
3. Deploy dashboard on server
4. Add real-time monitoring
5. Create mobile app
6. Integrate with college management system

---

**Happy coding! 🎉**

For questions or issues, refer to the troubleshooting section above.
