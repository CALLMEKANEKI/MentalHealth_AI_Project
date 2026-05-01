@echo off
title CHUONG TRINH HUAN LUYEN AI - PHAN TRUONG GIANG
echo Dang kich hoat moi truong ao Py.12...
call venv\Scripts\activate

echo Dang kiem tra GPU...
python -c "import torch; print('CUDA Ready:', torch.cuda.is_available())"

echo Bat dau huan luyen mo hinh PhoBERT...
python scripts/module1/train_multitask.py

echo Da hoan thanh! Nhan phim bat ky de thoat.
pause