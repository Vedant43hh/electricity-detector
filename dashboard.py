import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(
    page_title="Electricity Wastage Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .metric-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔌 College Electricity Wastage Detection System")
st.markdown("**GL Bajaj Group of Institutions, Mathura** | Real-time ML-powered Energy Monitoring")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    try:
        return pd.read_csv('electricity_results.csv')
    except FileNotFoundError:
        st.error("Error: electricity_results.csv not found!")
        st.info("Please run the following commands first:")
        st.code("""
python generate_data.py
python train_model.py
python detect_anomalies.py
        """)
        st.stop()

df = load_data()

# Calculate metrics
anomaly_count = df['Is_Anomaly'].sum()
total_wastage = df[df['Is_Anomaly']]['Consumption_kWh'].sum()
avg_consumption = df['Consumption_kWh'].mean()
max_consumption = df['Consumption_kWh'].max()

# Cost calculation (₹6 per kWh)
cost_per_kwh = 6
total_cost = total_wastage * cost_per_kwh
monthly_cost = total_cost / 24
annual_cost = monthly_cost * 12

# Display metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Anomalies",
        f"{int(anomaly_count):,}",
        delta=f"{(anomaly_count/len(df)*100):.2f}%",
        delta_color="inverse"
    )

with col2:
    st.metric(
        "Wastage (kWh)",
        f"{total_wastage:,.0f}",
        delta="Over normal"
    )

with col3:
    st.metric(
        "Annual Cost Loss",
        f"₹{annual_cost:,.0f}",
        delta="Potential savings"
    )

with col4:
    st.metric(
        "Avg Consumption",
        f"{avg_consumption:.1f}",
        delta=f"Max: {max_consumption:.1f}"
    )

st.markdown("---")

# Sidebar controls
st.sidebar.header("🔧 Controls")

selected_building = st.sidebar.selectbox(
    "Select Building",
    df['Building'].unique(),
    index=0
)

show_data_table = st.sidebar.checkbox("Show Data Table", value=False)

st.sidebar.markdown("---")
st.sidebar.header("📊 About")
st.sidebar.info(
    """
    This system predicts electricity consumption and detects anomalies in real-time.
    
    **How it works:**
    1. LSTM model predicts expected consumption
    2. Actual consumption is compared with prediction
    3. Anomalies (wastage) are detected
    4. Results shown in real-time
    """
)

# Filter data for selected building
building_data = df[df['Building'] == selected_building].reset_index(drop=True)

st.markdown(f"### 📍 Building: {selected_building}")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(building_data))
with col2:
    st.metric("Anomalies", int(building_data['Is_Anomaly'].sum()))
with col3:
    st.metric("Anomaly %", f"{(building_data['Is_Anomaly'].sum()/len(building_data)*100):.2f}%")

st.markdown("---")

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Actual vs Predicted Consumption")
    sample = building_data.head(500)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sample.index, sample['Consumption_kWh'], label='Actual', alpha=0.8, linewidth=2, color='blue')
    ax.plot(sample.index, sample['Predicted_kWh'], label='Predicted', alpha=0.8, linewidth=2, color='orange')
    ax.set_xlabel('Time (hours)', fontsize=10)
    ax.set_ylabel('Consumption (kWh)', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("🚨 Anomaly Detection")
    fig, ax = plt.subplots(figsize=(10, 5))
    normal = building_data[~building_data['Is_Anomaly']]
    anomalies = building_data[building_data['Is_Anomaly']]
    ax.scatter(normal.index, normal['Consumption_kWh'], alpha=0.3, s=10, label='Normal', color='green')
    ax.scatter(anomalies.index, anomalies['Consumption_kWh'], color='red', s=20, label='Anomaly', alpha=0.8)
    ax.set_xlabel('Time (hours)', fontsize=10)
    ax.set_ylabel('Consumption (kWh)', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")

# Wastage by building
st.subheader("📊 Wastage by Building")
wastage_by_building = df[df['Is_Anomaly']].groupby('Building')['Consumption_kWh'].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
ax.barh(wastage_by_building.index, wastage_by_building.values, color=colors[:len(wastage_by_building)])
ax.set_xlabel('Total Wastage (kWh)', fontsize=10)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
st.pyplot(fig)

# Add values on bars
for i, v in enumerate(wastage_by_building.values):
    st.text(f"{wastage_by_building.index[i]}: {v:.0f} kWh")

st.markdown("---")

# Hourly pattern
col1, col2 = st.columns(2)

with col1:
    st.subheader("⏰ Average Hourly Pattern")
    hourly_avg = building_data.groupby('Hour')['Consumption_kWh'].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, color='green', markersize=6)
    ax.fill_between(hourly_avg.index, hourly_avg.values, alpha=0.3, color='green')
    ax.set_xlabel('Hour of Day', fontsize=10)
    ax.set_ylabel('Average Consumption (kWh)', fontsize=10)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("📅 Monthly Trend")
    monthly_avg = df.groupby('Month')['Consumption_kWh'].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2, color='navy', markersize=6)
    ax.fill_between(monthly_avg.index, monthly_avg.values, alpha=0.3, color='navy')
    ax.set_xlabel('Month', fontsize=10)
    ax.set_ylabel('Average Consumption (kWh)', fontsize=10)
    ax.set_xticks(range(1, 13))
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticklabels(months)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")

# Statistics
st.subheader("📊 Detailed Statistics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Avg Consumption", f"{building_data['Consumption_kWh'].mean():.2f} kWh")
    st.metric("Max Consumption", f"{building_data['Consumption_kWh'].max():.2f} kWh")

with col2:
    st.metric("Min Consumption", f"{building_data['Consumption_kWh'].min():.2f} kWh")
    st.metric("Std Dev", f"{building_data['Consumption_kWh'].std():.2f} kWh")

with col3:
    anomaly_kwh = building_data[building_data['Is_Anomaly']]['Consumption_kWh'].sum()
    st.metric("Anomaly kWh", f"{anomaly_kwh:.2f} kWh")
    st.metric("Anomaly %", f"{(building_data['Is_Anomaly'].sum()/len(building_data)*100):.2f}%")

st.markdown("---")

# Anomaly details table
if show_data_table:
    st.subheader("🔍 Anomaly Details")
    anomalies_detail = building_data[building_data['Is_Anomaly']][
        ['Timestamp', 'Consumption_kWh', 'Predicted_kWh', 'Residual']
    ].head(20)
    st.dataframe(anomalies_detail, use_container_width=True)

st.markdown("---")

# Download data
st.subheader("📥 Download Results")
col1, col2 = st.columns(2)

with col1:
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Full Results (CSV)",
        data=csv,
        file_name="electricity_results.csv",
        mime="text/csv"
    )

with col2:
    building_csv = building_data.to_csv(index=False)
    st.download_button(
        label="Download Building Data (CSV)",
        data=building_csv,
        file_name=f"{selected_building}_data.csv",
        mime="text/csv"
    )

st.markdown("---")

st.footer = """
🔌 College Electricity Wastage Detection System | 
Built with Streamlit & TensorFlow | 
GL Bajaj Group of Institutions, Mathura
"""
