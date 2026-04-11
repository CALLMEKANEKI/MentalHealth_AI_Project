import sys
import os
import pandas as pd

# Thiết lập đường dẫn gốc (như đã hướng dẫn ở bước trước)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'src'))

from preprocess import TextCleaner


dict_path = os.path.join(project_root, 'data', 'external', 'Xử lý teencode.xlsx')

# Khởi tạo
try:
    cleaner = TextCleaner(dict_path)
    print("✅ Đã kết nối và nạp từ điển thành công!")
    
    # Test thử 1 câu
    test_sent = "t mún đi chơi"
    print(f"Gốc: {test_sent}")
    print(f"Sạch: {cleaner.clean(test_sent)}")
except Exception as e:
    print(f"❌ Lỗi: {e}")