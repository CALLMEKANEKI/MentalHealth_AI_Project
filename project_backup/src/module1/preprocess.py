import re
import pandas as pd
import emoji

class TextCleaner:
    def __init__(self, dict_path):
        df_dict = pd.read_excel(dict_path)

        # CHỈ nạp những từ có Phân loại là 'Viết tắt' vào từ điển replace
        # Những từ 'Viết lóng' mang cảm xúc sẽ KHÔNG bị replace
        self.teencode_dict = {
            str(k).lower().strip(): str(v).lower().strip()
            for k, v, t in zip(df_dict['Teencode - GenZ'], df_dict['Ý nghĩa'], df_dict['Phân loại'])
            if pd.notna(k) and str(t).strip() == 'Viết tắt'
        }

        # Map emoji và ký tự đặc biệt sang text cảm xúc
        # Thứ tự quan trọng: các chuỗi dài hơn phải được xử lý trước
        self.emotion_map = {
            # === BUỒN / KHÓC ===
            '😭': ' buồn ',
            '😢': ' buồn ',
            '😔': ' buồn ',
            '😞': ' buồn ',
            '😟': ' buồn ',
            '🥺': ' buồn ',
            ':(((':  ' buồn ',
            ':((': ' buồn ',
            ':(': ' buồn ',
            'huhu': ' buồn ',
            'hix': ' buồn ',
            'hixx': ' buồn ',
            'hixxx': ' buồn ',
            'hu hu': ' buồn ',

            # === VUI / CƯỜI ===
            '😊': ' vui ',
            '😄': ' vui ',
            '😁': ' vui ',
            '🥰': ' vui ',
            '😍': ' vui ',
            '=)))': ' cười ',
            '=))': ' cười ',
            ':)))': ' vui ',
            ':))': ' vui ',
            ':)': ' vui ',
            'haha': ' cười ',
            'hahaha': ' cười ',
            'hihi': ' cười ',
            'hihihi': ' cười ',
            'kkk': ' cười ',
            'kkkk': ' cười ',
            'kkkkk': ' cười ',
            'hehe': ' cười ',
            'hehehehe': ' cười ',
            'hhha': ' cười ',

            # === TỨC GIẬN ===
            '😡': ' tức giận ',
            '😤': ' tức giận ',
            '🤬': ' tức giận ',
            '👿': ' tức giận ',

            # === SỢ HÃI ===
            '😱': ' sợ hãi ',
            '😨': ' sợ hãi ',
            '😰': ' sợ hãi ',
            '😧': ' sợ hãi ',

            # === NGẠC NHIÊN ===
            '😮': ' ngạc nhiên ',
            '😲': ' ngạc nhiên ',
            '🤯': ' ngạc nhiên ',
            '@@': ' ngạc nhiên ',

            # === CHÁN / THỞ DÀI ===
            'haizzz': ' thở dài ',
            'haizz': ' thở dài ',
            'haiz': ' thở dài ',
            'haiizzz': ' thở dài ',
            '😑': ' chán ',
            '😒': ' chán ',
            '🙄': ' chán ',

            # === GHÉT / CHÁN GHÉT ===
            '🤢': ' ghét ',
            '🤮': ' ghét ',

            # === YÊU THƯƠNG ===
            '❤️': ' yêu thương ',
            '❤': ' yêu thương ',
            '💔': ' đau lòng ',
            '🥰': ' yêu thương ',
            '😘': ' yêu thương ',

            # === KHÁC ===
            '😴': ' buồn ngủ ',
            '🤔': ' suy nghĩ ',
            '🙏': ' cầu xin ',
        }

    def clean(self, text):
        
        # Thêm vào đầu hàm clean() trong preprocess.py
# BƯỚC 0: Xóa HTML tags trước tất cả
# BƯỚC 0b: Xóa ký tự zero‑width và chuẩn hoá dấu câu
#   - Loại bỏ ​, ﻿ (zero‑width space, BOM)
#   - Chuyển các dấu chấm, ?, ! liên tiếp thành một dấu '.'
#   - Đảm bảo không còn ký tự HTML entity lẻ
#   - Các bước này giúp giảm noise cho các nhãn hiếm
        # BƯỚC 0: Xóa HTML tags trước tất cả
        text = re.sub(r'<br\s*/?>', ' ', text)   # <br> → space
        # BƯỚC 0b: Xóa ký tự zero‑width (U+200B, U+FEFF) và chuẩn hoá dấu câu
        text = text.replace('​', '').replace('﻿', '')
        # Chuẩn hoá các dấu chấm, ?, ! liên tiếp thành một dấu '.'
        text = re.sub(r'[.!?]{2,}', '.', text)
        text = re.sub(r'&quot;', '"', text)       # &quot; → "
        text = re.sub(r'&lt;', '<', text)         # &lt; → 
        text = re.sub(r'&gt;', '>', text)         # &gt; → >
        text = re.sub(r'&amp;', '&', text)        # &amp; → &
        text = re.sub(r'&#39;', "'", text)        # &#39; → '
        text = re.sub(r'<[^>]+>', ' ', text)      # các tag HTML khác
        
        if not text or pd.isna(text):
            return ""

        text = str(text).lower().strip()

        # ========================================
        # BƯỚC 1: MAP EMOJI & KÝ TỰ CẢM XÚC
        # Phải làm TRƯỚC khi xóa ký tự đặc biệt
        # Sắp xếp theo độ dài giảm dần để tránh replace nhầm
        # ========================================
        sorted_emotion_keys = sorted(self.emotion_map.keys(), key=len, reverse=True)
        for symbol in sorted_emotion_keys:
            text = text.replace(symbol, self.emotion_map[symbol])

        # Xử lý các emoji còn lại chưa được map (xóa bỏ)
        try:
            text = emoji.replace_emoji(text, replace=' ')
        except Exception:
            pass

        # ========================================
        # BƯỚC 2: DỊCH TỪ VIẾT TẮT
        # Chỉ dịch 'Viết tắt', bypass 'Viết lóng'
        # ========================================
        sorted_keys = sorted(self.teencode_dict.keys(), key=len, reverse=True)
        for teencode in sorted_keys:
            pattern = r'\b' + re.escape(teencode) + r'\b'
            text = re.sub(pattern, self.teencode_dict[teencode], text)

        # ========================================
        # BƯỚC 3: XÓA KÝ TỰ RÁC
        # Chỉ giữ lại chữ cái, số và khoảng trắng
        # (emoji và ký tự cảm xúc đã được xử lý ở bước 1)
        # ========================================
        text = re.sub(r'[^\w\s]', ' ', text)

        # ========================================
        # BƯỚC 4: CHUẨN HÓA KHOẢNG TRẮNG
        # ========================================
        text = ' '.join(text.split())

        return text


if __name__ == "__main__":
    # Test nhanh
    cleaner = TextCleaner(
        r'C:\Users\ASPIRE A715 - 42G\Downloads\MentalHealth_AI_Project\data\external\Xử lý teencode.xlsx'
    )

    test_cases = [
        ("Hôm nay t cảm thấy vl :)))", "Vui (vl giữ nguyên, :))) → cười)"),
        ("buồn quá huhu 😭😭", "Buồn (huhu + emoji buồn)"),
        ("tức quá dm 😡", "Tức giận"),
        ("haizzz chán vcl", "Thở dài + chán"),
        ("kkk cute vãi", "Cười (kkk giữ nguyên cute)"),
        ("ko biết làm sao nữa :(((", "Buồn (ko→không, :((( → buồn)"),
        ("bjo mày đang làm gì vậy @@", "Ngạc nhiên (bjo→bây giờ)"),
        ("t yêu per lắm ❤️❤️", "Yêu thương"),
    ]

    print("=" * 60)
    print("TEST PREPROCESS")
    print("=" * 60)
    for text, expected in test_cases:
        result = cleaner.clean(text)
        print(f"\nInput:    {text}")
        print(f"Output:   {result}")
        print(f"Expected: {expected}")