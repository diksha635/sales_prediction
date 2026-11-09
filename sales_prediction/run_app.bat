@echo off
cd /d "D:\sales_prediction"
echo ========================================================
echo Starting Sales Prediction Streamlit App from D: Drive...
echo ========================================================

REM Try py launcher first, then direct python path
py -m streamlit run app.py 2>nul
if %errorlevel% neq 0 (
    "C:\Users\Diksha\AppData\Local\Programs\Python\Python314\python.exe" -m streamlit run app.py
)

pause
