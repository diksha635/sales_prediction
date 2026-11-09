Set-Location "D:\sales_prediction"
Write-Host "Starting Sales Prediction Streamlit App from D:\sales_prediction..." -ForegroundColor Cyan

$pythonPath = "C:\Users\Diksha\AppData\Local\Programs\Python\Python314\python.exe"

if (Test-Path $pythonPath) {
    & $pythonPath -m streamlit run app.py
} else {
    python -m streamlit run app.py
}
