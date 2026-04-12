import os
import sys
import pandas as pd

# 1. Xác định đường dẫn thư mục gốc
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocess import TextCleaner

def prepare_all_data():
    # Sử dụng os.path.join để tránh lỗi gạch chéo trên Windows/Linux
    dict_path = os.path.join(project_root, 'data', 'external', 'Xử lý teencode.xlsx')
    cleaner = TextCleaner(dict_path)
    
    data_files = [
        {'raw': 'data/raw/UIT-VSEC/train_nor_811.xlsx', 'clean': 'data/processed/uit_train_clean.xlsx', 'col': 'Sentence'},
        {'raw': 'data/raw/UIT-VSEC/test_nor_811.xlsx', 'clean': 'data/processed/uit_test_clean.xlsx', 'col': 'Sentence'},
        {'raw': 'data/raw/UIT-VSEC/valid_nor_811.xlsx', 'clean': 'data/processed/uit_valid_clean.xlsx', 'col': 'Sentence'},
        {'raw': 'data/raw/ViHSD/xlsx/train_df.xlsx', 'clean': 'data/processed/vihsd_train_clean.xlsx', 'col': 'cmt_col'},
        {'raw': 'data/raw/ViHSD/xlsx/val_df.xlsx', 'clean': 'data/processed/vihsd_val_clean.xlsx', 'col': 'cmt_col'},
        {'raw': 'data/raw/ViHSD/xlsx/test_df.xlsx', 'clean': 'data/processed/vihsd_test_clean.xlsx', 'col': 'cmt_col'}
    ]

    for file in data_files:
        raw_path = os.path.join(project_root, file['raw'])
        clean_path = os.path.join(project_root, file['clean'])
        
        if os.path.exists(raw_path):
            print(f"🧹 Đang xử lý: {file['raw']}")
            df = pd.read_excel(raw_path)
            
            # Làm sạch dữ liệu
            df['cleaned_text'] = df[file['col']].apply(cleaner.clean)
            
            # --- ĐOẠN FIX LỖI: Tự tạo folder nếu chưa có ---
            os.makedirs(os.path.dirname(clean_path), exist_ok=True)
            
            df.to_excel(clean_path, index=False)
            print(f"✅ Đã lưu file sạch tại: {file['clean']}")
        else:
            print(f"⚠️ Không tìm thấy file: {raw_path}")

if __name__ == "__main__":
    prepare_all_data()