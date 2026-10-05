import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

print("=" * 60)
print("ELECTRICITY DATA GENERATION")
print("=" * 60)

print("\nGenerating synthetic electricity data...")
print("This may take 1-2 minutes...")

# Create date range (2 years, hourly)
dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='H')

# Building names
buildings = ['Block_A', 'Block_B', 'Hostel_1', 'Hostel_2', 'Lab_Complex', 'Library']

# Create empty list to store data
data = []

# Generate data for each hour and building
for i, date in enumerate(dates):
    if (i + 1) % 10000 == 0:
        print(f"Processing... {i + 1}/{len(dates)} records")
    
    hour = date.hour
    day_of_week = date.dayofweek
    month = date.month
    
    # Temperature (seasonal variation)
    day_of_year = date.timetuple().tm_yday
    temp = 20 + 10 * np.sin(day_of_year / 365.0) + np.random.normal(0, 3)
    
    for building in buildings:
        # Base consumption depends on time of day
        if hour >= 9 and hour <= 17:  # Peak (office hours)
            base = np.random.uniform(120, 180)
        elif hour >= 18 and hour <= 23:  # Evening
            base = np.random.uniform(60, 100)
        else:  # Night (2-8 AM)
            base = np.random.uniform(20, 50)
        
        # Add randomness
        consumption = base + np.random.normal(0, 8)
        consumption = max(0, consumption)  # No negative values
        
        # Occupancy (higher on weekdays)
        if day_of_week < 5:  # Monday-Friday
            occupancy = np.random.uniform(200, 300)
        else:  # Weekend
            occupancy = np.random.uniform(50, 150)
        
        # Add some random wastage events (2% anomalies)
        if np.random.random() < 0.02:
            consumption *= np.random.uniform(2, 3)  # Spike consumption
        
        data.append({
            'Timestamp': date,
            'Building': building,
            'Consumption_kWh': consumption,
            'Temperature_C': temp,
            'Occupancy': occupancy,
            'Hour': hour,
            'DayOfWeek': day_of_week,
            'Month': month
        })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_file = 'electricity_data_simulated.csv'
df.to_csv(output_file, index=False)

print(f"\n{'='*60}")
print("✓ DATA GENERATED SUCCESSFULLY!")
print(f"{'='*60}")
print(f"Total records: {len(df)}")
print(f"Buildings: {len(df['Building'].unique())}")
print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
print(f"File saved: {output_file}")
print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

print(f"\nData preview:")
print(df.head(10))

print(f"\nData statistics:")
print(df['Consumption_kWh'].describe())

print(f"\n{'='*60}")
print("NEXT STEP: Run 'python train_model.py'")
print(f"{'='*60}\n")
