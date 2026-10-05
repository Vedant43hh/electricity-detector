#!/usr/bin/env python3
"""
Electricity Wastage Detection - Automated Runner
Runs all steps sequentially
"""

import os
import subprocess
import sys
import time

def run_script(script_name):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR in {script_name}")
        print(f"Return code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ ERROR: {script_name} not found!")
        return False

def main():
    print("\n" + "="*60)
    print("🔌 ELECTRICITY WASTAGE DETECTION SYSTEM")
    print("Automated Setup & Execution")
    print("="*60 + "\n")
    
    # List of scripts to run in order
    scripts = [
        ('generate_data.py', 'Generate Synthetic Data'),
        ('train_model.py', 'Train LSTM Model'),
        ('detect_anomalies.py', 'Detect Anomalies'),
        ('visualize.py', 'Create Visualizations'),
    ]
    
    completed = 0
    failed = 0
    
    # Run each script
    for script, description in scripts:
        print(f"\n[{completed + failed + 1}/{len(scripts)}] {description}")
        
        if run_script(script):
            completed += 1
            print(f"✓ {script} completed successfully")
            time.sleep(1)
        else:
            failed += 1
            print(f"✗ {script} failed")
            print("\nPlease fix the error and try again.")
            sys.exit(1)
    
    # Summary
    print("\n" + "="*60)
    print("✓ ALL SCRIPTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  ✓ Completed: {completed}")
    print(f"  ✗ Failed: {failed}")
    
    # Check for required files
    print("\n" + "="*60)
    print("Checking generated files...")
    print("="*60)
    
    required_files = [
        'electricity_data_simulated.csv',
        'lstm_model.h5',
        'scaler.pkl',
        'electricity_results.csv',
        'electricity_analysis.png'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            file_size = os.path.getsize(file) / (1024*1024)
            print(f"✓ {file} ({file_size:.2f} MB)")
        else:
            print(f"✗ {file} NOT FOUND")
    
    # Launch dashboard
    print("\n" + "="*60)
    print("NEXT STEP: Launch Dashboard")
    print("="*60)
    print("\nRun the following command in your terminal:")
    print("\n  streamlit run dashboard.py\n")
    print("The dashboard will open in your browser at:")
    print("  http://localhost:8501\n")
    
    # Ask user if they want to launch dashboard
    response = input("Launch dashboard now? (y/n): ").strip().lower()
    
    if response == 'y':
        print("\nLaunching dashboard...")
        os.system("streamlit run dashboard.py")
    else:
        print("\nYou can launch the dashboard anytime by running:")
        print("  streamlit run dashboard.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)
