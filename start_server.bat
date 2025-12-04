@echo off
echo ========================================
echo   RAG Chat API Server
echo ========================================
echo.
echo Starting FastAPI server...
echo.
echo API will be available at:
echo   - API: http://localhost:8000
echo   - Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.
uvicorn app:app --reload --host 0.0.0.0 --port 8000

