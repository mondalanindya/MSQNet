@echo off
REM Windows batch script to evaluate MSQNet
set CHECKPOINT=%1
if "%CHECKPOINT%"=="" set CHECKPOINT=./checkpoints/msqnet_msqnet_animalkingdom.pth

set DATASET=%2
if "%DATASET%"=="" set DATASET=animalkingdom

set DATA_DIR=%3
if "%DATA_DIR%"=="" set DATA_DIR=./datasets

echo === Evaluating MSQNet ===
echo Checkpoint: %CHECKPOINT%
echo Dataset   : %DATASET%

python run.py --dataset %DATASET% --model msqnet --data_dir %DATA_DIR% --checkpoint %CHECKPOINT% --total_length 16 --train False
