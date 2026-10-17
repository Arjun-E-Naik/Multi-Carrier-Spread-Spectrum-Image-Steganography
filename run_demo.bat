@echo off
title MC-SS Image Steganography Showcase
echo ========================================================
echo  Multi-Carrier Spread Spectrum (MC-SS) Steganography
echo ========================================================
echo.

if not exist .venv (
    echo [ERROR] Virtual environment .venv not found!
    pause
    exit /b
)

echo [*] Activating dedicated virtual environment...
call .venv\Scripts\activate.bat

echo [*] Launching Streamlit Interactive Showcase...
streamlit run app.py

pause
