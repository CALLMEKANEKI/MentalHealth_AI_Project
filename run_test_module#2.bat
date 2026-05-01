@echo off
title CHUONG KIEM TRA AI - PHAN TRUONG GIANG
echo Dang kich hoat moi truong ao Py.12...
call venv\Scripts\activate

echo Dang kiem tra GPU...
python -c "import torch; print('CUDA Ready:', torch.cuda.is_available())"

echo Bat dau kiem tra mo hinh PhoBERT...
python scripts/module2/test_module#2.py

echo Da hoan thanh! Nhan phim bat ky de thoat.
pause