import pandas as pd
import re

class TextCleaner:
    def __init__(self, excel_path):
        df_dict = pd.read_excel(excel_path)
        
        # Tạo mapping: Cột 'Teencode - GenZ' -> Cột 'Ý nghĩa'
        # Lưu ý: Kiểm tra chính xác tên cột trong file của Vũ nhé
        self.mapping = dict(zip(df_dict['Teencode - GenZ'], 
                                df_dict['Ý nghĩa']))

    def clean(self, text):
        if not isinstance(text, str): return ""
        text = text.lower()
        words = text.split()
        # Thay thế teencode
        cleaned_words = [str(self.mapping.get(w, w)) for w in words]
        return " ".join(cleaned_words)