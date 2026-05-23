@echo off
echo [*] Dang khoi dong giao dien Thu thap du lieu (Data Collector)...
cd /d "%~dp0"
conda run -n signlang streamlit run collect_data_ui.py --browser.gatherUsageStats=false
pause
