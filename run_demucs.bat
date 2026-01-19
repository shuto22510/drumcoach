@echo off
echo Running Demucs drum separation...
python -m demucs --two-stems=drums -o output/demucs data/soramotoberuhazu.wav
echo Done!
pause
