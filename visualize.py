import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

print("\nLoading results...")
try:
    df = pd.read_csv('electricity_results.csv')
    print(f"✓ Data loaded: {len(df)} records")
except FileNotFoundError:
    print("ERROR: electricity_results.csv not found!")
    print("Please run 'python detect_anomalies.py' first")
    exit()

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

print("\nGenerating charts...")

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Actual vs Predicted (first 2000 records)
print("  - Actual vs Predicted chart...")
sample = df.head(2000)
axes[0, 0].plot(sample.index, sample['Consumption_kWh'], label='Actual', alpha=0.8, linewidth=1.5, color='blue')
axes[0, 0].plot(sample.index, sample['Predicted_kWh'], label='Predicted', alpha=0.8, linewidth=1.5, color='orange')
axes[0, 0].set_title('Actual vs Predicted Consumption (First 2000 hours)', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Time (hours)', fontsize=11)
axes[0, 0].set_ylabel('Consumption (kWh)', fontsize=11)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Anomalies scatter
print("  - Anomaly detection chart...")
anomalies = df[df['Is_Anomaly']]
axes[0, 1].scatter(df.index, df['Consumption_kWh'], alpha=0.2, s=5, label='Normal', color='green')
axes[0, 1].scatter(anomalies.index, anomalies['Consumption_kWh'], color='red', s=15, label='Anomaly', alpha=0.8)
axes[0, 1].set_title('Anomaly Detection Results', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Time (hours)', fontsize=11)
axes[0, 1].set_ylabel('Consumption (kWh)', fontsize=11)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Wastage by Building
print("  - Wastage by building chart...")
wastage_by_building = df[df['Is_Anomaly']].groupby('Building')['Consumption_kWh'].sum().sort_values(ascending=False)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
axes[1, 0].barh(wastage_by_building.index, wastage_by_building.values, color=colors[:len(wastage_by_building)])
axes[1, 0].set_title('Wastage by Building (Anomaly Consumption)', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Total Consumption (kWh)', fontsize=11)
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Add values on bars
for i, v in enumerate(wastage_by_building.values):
    axes[1, 0].text(v + 10, i, f'{v:.0f}', va='center', fontsize=10)

# Plot 4: Hourly Pattern
print("  - Hourly pattern chart...")
hourly_avg = df.groupby('Hour')['Consumption_kWh'].mean()
axes[1, 1].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2.5, color='green', markersize=6)
axes[1, 1].fill_between(hourly_avg.index, hourly_avg.values, alpha=0.3, color='green')
axes[1, 1].set_title('Average Hourly Consumption Pattern', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Hour of Day', fontsize=11)
axes[1, 1].set_ylabel('Average Consumption (kWh)', fontsize=11)
axes[1, 1].set_xticks(range(0, 24, 2))
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
output_file = 'electricity_analysis.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✓ Chart saved: {output_file}")

# Additional detailed charts
print("\nGenerating detailed charts...")

# Chart 5: Building-wise anomaly count
fig, ax = plt.subplots(figsize=(10, 6))
anomaly_count_by_building = df[df['Is_Anomaly']].groupby('Building').size().sort_values(ascending=False)
ax.bar(anomaly_count_by_building.index, anomaly_count_by_building.values, color='coral')
ax.set_title('Number of Anomalies by Building', fontsize=13, fontweight='bold')
ax.set_ylabel('Count', fontsize=11)
ax.set_xlabel('Building', fontsize=11)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('anomaly_count_by_building.png', dpi=150, bbox_inches='tight')
print("✓ Chart saved: anomaly_count_by_building.png")

# Chart 6: Monthly consumption trend
fig, ax = plt.subplots(figsize=(12, 6))
monthly_avg = df.groupby('Month')['Consumption_kWh'].mean()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2.5, markersize=8, color='navy')
ax.fill_between(monthly_avg.index, monthly_avg.values, alpha=0.3, color='navy')
ax.set_title('Average Monthly Consumption Trend', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Consumption (kWh)', fontsize=11)
ax.set_xlabel('Month', fontsize=11)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(months)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('monthly_trend.png', dpi=150, bbox_inches='tight')
print("✓ Chart saved: monthly_trend.png")

print(f"\n{'='*60}")
print("VISUALIZATIONS COMPLETE!")
print(f"{'='*60}")
print("\nCharts generated:")
print("  - electricity_analysis.png (4 main charts)")
print("  - anomaly_count_by_building.png")
print("  - monthly_trend.png")

print(f"\n{'='*60}")
print("NEXT STEP: Run 'streamlit run dashboard.py'")
print(f"{'='*60}\n")
