@echo off
REM Windows batch script to train MSQNet
set DATASET=%1
if "%DATASET%"=="" set DATASET=animalkingdom

set MODEL=%2
if "%MODEL%"=="" set MODEL=msqnet

set DATA_DIR=%3
if "%DATA_DIR%"=="" set DATA_DIR=./datasets

echo === Training MSQNet ===
echo Dataset : %DATASET%
echo Model   : %MODEL%
echo Data Dir: %DATA_DIR%

python run.py --dataset %DATASET% --model %MODEL% --data_dir %DATA_DIR% --batch_size 16 --epochs 100 --total_length 16 --train True
