import pandas as pd
try:
    df = pd.read_excel("evaluation/scientific_report.xlsx")
    print("✅ Report loaded successfully!")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
except Exception as e:
    print(f"❌ Error reading report: {e}")
