import re
import pandas as pd
import emoji

class TextCleaner:
    def __init__(self, dict_path):
        # Load file Excel teencode
        df_dict = pd.read_excel(dict_path)
        # Ép kiểu để tránh lỗi dữ liệu số hoặc ô trống
        self.teencode_dict = {str(k).lower().strip(): str(v).lower().strip() 
                             for k, v in zip(df_dict['Teencode - GenZ'], df_dict['Ý nghĩa']) 
                             if pd.notna(k)}

    def clean(self, text):
        if not text or pd.isna(text):
            return ""
            
        # 1. Chuyển đổi Emoji sang dạng chữ tiếng Anh
        # Ví dụ: 😊 -> :smiling_face:
        text = emoji.demojize(text, delimiters=(" ", " "))
        
        # 2. Làm sạch cơ bản
        text = text.lower().strip()
        
        # 3. Thay thế Teencode và Icon ký tự (như :), :v, <3)
        # Sắp xếp từ điển theo độ dài giảm dần để tránh thay thế nhầm (ví dụ: ':((' trước ':(' )
        sorted_keys = sorted(self.teencode_dict.keys(), key=len, reverse=True)
        
        for teencode in sorted_keys:
            standard = self.teencode_dict[teencode]
            # Nếu là icon ký tự (có ký tự đặc biệt)
            if not teencode.isalnum():
                text = text.replace(teencode, f" {standard} ")
            else:
                # Nếu là chữ viết tắt (đr, ko)
                pattern = r'\b' + re.escape(teencode) + r'\b'
                text = re.sub(pattern, standard, text)
        
        # 4. Xử lý các tag emoji từ bước 1 (Xóa dấu ':' và '_' để PhoBERT dễ đọc)
        # Ví dụ: :smiling_face: -> smiling face
        text = text.replace(":", " ").replace("_", " ")
        
        # 5. Xóa các ký tự đặc biệt còn sót lại (giữ lại chữ cái và số)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 6. Chuẩn hóa khoảng trắng
        text = " ".join(text.split())
        
        return text
    
