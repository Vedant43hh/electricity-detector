import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("ANOMALY DETECTION")
print("=" * 60)

print("\nLoading data and model...")
try:
    df = pd.read_csv('electricity_data_simulated.csv')
    print("✓ Data loaded successfully")
    
    # Load model and scaler
    model = pickle.load(open('electricity_model.pkl', 'rb'))
    print("✓ Model loaded: electricity_model.pkl")
    
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    print("✓ Scaler loaded successfully")
    print("✓ All files loaded successfully")
    
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Please run the previous scripts first:")
    print("1. python generate_data.py")
    print("2. python train_model.py")
    exit()

# Prepare data
features = ['Hour', 'DayOfWeek', 'Month', 'Temperature_C', 'Occupancy']
X = df[features].values
X_scaled = scaler.transform(X)

# Make predictions
print("\nMaking predictions on all data...")
predictions = model.predict(X_scaled)

# Add predictions to dataframe
df['Predicted_kWh'] = predictions
df['Residual'] = df['Consumption_kWh'] - df['Predicted_kWh']

# Detect anomalies (2 standard deviations)
print("\nDetecting anomalies...")
mean_residual = df['Residual'].mean()
std_residual = df['Residual'].std()
threshold = mean_residual + (2 * std_residual)

df['Is_Anomaly'] = df['Residual'] > threshold

print(f"\n{'='*60}")
print("ANOMALY DETECTION RESULTS")
print(f"{'='*60}")
print(f"Mean Residual: {mean_residual:.4f}")
print(f"Std Residual: {std_residual:.4f}")
print(f"Anomaly Threshold: {threshold:.4f}")

# Count anomalies
anomaly_count = df['Is_Anomaly'].sum()
total_records = len(df)
anomaly_percentage = (anomaly_count / total_records) * 100

print(f"\nTotal anomalies detected: {int(anomaly_count)}")
print(f"Anomaly percentage: {anomaly_percentage:.2f}%")

# Building-wise breakdown
print(f"\n{'='*60}")
print("WASTAGE BY BUILDING")
print(f"{'='*60}")

print("\nBuilding-wise Wastage Analysis:")
for building in df['Building'].unique():
    building_anomalies = df[(df['Building'] == building) & (df['Is_Anomaly'])]
    if len(building_anomalies) > 0:
        total_wastage = building_anomalies['Consumption_kWh'].sum()
        count = len(building_anomalies)
        avg_wastage = building_anomalies['Consumption_kWh'].mean()
        print(f"\n{building}:")
        print(f"  - Anomaly events: {count}")
        print(f"  - Total wastage: {total_wastage:.2f} kWh")
        print(f"  - Avg per event: {avg_wastage:.2f} kWh")

# Cost analysis (assuming ₹6 per kWh)
print(f"\n{'='*60}")
print("COST ANALYSIS (₹6 per kWh)")
print(f"{'='*60}")

cost_per_kwh = 6
total_wastage_kwh = df[df['Is_Anomaly']]['Consumption_kWh'].sum()
total_cost = total_wastage_kwh * cost_per_kwh
monthly_cost = total_cost / 24  # Approximate
annual_cost = monthly_cost * 12

print(f"\nTotal wastage: {total_wastage_kwh:.2f} kWh")
print(f"Monthly cost: ₹{monthly_cost:,.2f}")
print(f"Annual cost: ₹{annual_cost:,.2f}")

# Save results
output_file = 'electricity_results.csv'
df.to_csv(output_file, index=False)
print(f"\n✓ Results saved: {output_file}")

# Show sample anomalies
print(f"\n{'='*60}")
print("SAMPLE ANOMALIES (First 10)")
print(f"{'='*60}")

anomalies_sample = df[df['Is_Anomaly']][['Timestamp', 'Building', 'Consumption_kWh', 'Predicted_kWh', 'Residual']].head(10)
print("\n" + anomalies_sample.to_string(index=False))

print(f"\n{'='*60}")
print("NEXT STEP: Run 'python visualize.py'")
print(f"{'='*60}\n")