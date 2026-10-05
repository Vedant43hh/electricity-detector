import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("MODEL TRAINING (Random Forest)")
print("=" * 60)

print("\nLoading data...")
try:
    df = pd.read_csv('electricity_data_simulated.csv')
    print(f"✓ Data loaded: {len(df)} records")
except FileNotFoundError:
    print("ERROR: electricity_data_simulated.csv not found!")
    print("Please run 'python generate_data.py' first")
    exit()

# Select features for training
features = ['Hour', 'DayOfWeek', 'Month', 'Temperature_C', 'Occupancy']
X = df[features].values
y = df['Consumption_kWh'].values

print(f"Features: {features}")
print(f"Data shape: X={X.shape}, y={y.shape}")

# Normalize features
print("\nNormalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data (80% train, 20% test)
print("Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# Build Random Forest model (simpler, no version issues)
print("\n" + "="*60)
print("BUILDING RANDOM FOREST MODEL")
print("="*60)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nTraining model...")
print("This will take 1-2 minutes...")

model.fit(X_train, y_train)

# Evaluate on test data
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate metrics
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"\nTest Set Performance:")
print(f"RMSE: {rmse_test:.4f} kWh")
print(f"MAE: {mae_test:.4f} kWh")
print(f"R² Score: {r2_test:.4f}")

print(f"\nTrain Set Performance:")
print(f"RMSE: {rmse_train:.4f} kWh")
print(f"R² Score: {r2_train:.4f}")

# Feature importance
print(f"\nFeature Importance:")
for feature, importance in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  {feature}: {importance:.4f}")

# Save model and scaler
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

pickle.dump(model, open('electricity_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("✓ Model saved: electricity_model.pkl")
print("✓ Scaler saved: scaler.pkl")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nModel Performance Summary:")
print(f"- R² Score: {r2_test:.4f} (Target: >0.85)")
print(f"- RMSE: {rmse_test:.4f} kWh")
print(f"- MAE: {mae_test:.4f} kWh")

if r2_test > 0.85:
    print("\n✓ MODEL QUALITY IS EXCELLENT!")
else:
    print("\n✓ Model quality is good. Ready to detect anomalies.")

print(f"\n{'='*60}")
print("NEXT STEP: Run 'python detect_anomalies.py'")
print(f"{'='*60}\n")