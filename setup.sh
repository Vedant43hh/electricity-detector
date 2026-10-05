#!/bin/bash

echo "============================================================"
echo "ELECTRICITY WASTAGE DETECTION SYSTEM - SETUP"
echo "============================================================"
echo ""

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "============================================================"
echo "INSTALLATION COMPLETE!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Run: python generate_data.py"
echo "2. Run: python train_model.py"
echo "3. Run: python detect_anomalies.py"
echo "4. Run: python visualize.py"
echo "5. Run: streamlit run dashboard.py"
echo ""
echo "Or simply run: python run_all.py"
echo ""
