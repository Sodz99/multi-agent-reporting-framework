@echo off
cd /d "C:\Users\House\OneDrive\Desktop\Sohan_Arun\Career\Portfolio Projects\Fault Diagnosis Multi Agent\Automated Reporting"
..\venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'src'); from automated_reporting.cli import main; main(['generate-reports', '--session', 'demo'])"
pause