import os
import sys
import ast
import pandas as pd
import numpy as np

# ==================== ĐƯỜNG DẪN ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(scripts_dir)

if project_root not in sys.path:
    sys.path.append(project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

print(f"📁 project_root: {project_root}")

from module1.preprocess import TextCleaner

# ==================== MAPPING ====================
# VSMEC 7 nhãn → index trong ViGoEmotions (28 nhãn)
VSMEC_TO_VIGO_INDEX = {
    'Anger':     [24],   # anger
    'Disgust':   [23],   # disgust
    'Enjoyment': [2],    # joy
    'Fear':      [16],   # fear
    'Sadness':   [21],   # sadness
    'Surprise':  [13],   # surprise
    'Other':     [27],   # neutral
}


def parse_labels(label_str):
    """Parse chuỗi '[2, 8, 3]' → list [2, 8, 3]"""
    try:
        return ast.literal_eval(str(label_str))
    except Exception:
        return [27]  # fallback: neutral


def prepare_emotion_data():
    dict_path = os.path.join(project_root, 'data', 'external', 'Xử lý teencode.xlsx')
    cleaner = TextCleaner(dict_path)

    # Đường dẫn raw
    vigo_train = os.path.join(project_root, 'data', 'raw', 'ViGoEmotions', 'train.csv')
    vigo_valid = os.path.join(project_root, 'data', 'raw', 'ViGoEmotions', 'valid.csv')
    vigo_test  = os.path.join(project_root, 'data', 'raw', 'ViGoEmotions', 'test.csv')

    vsmec_train = os.path.join(project_root, 'data', 'raw', 'UIT-VSEMC', 'train_nor_811.xlsx')
    vsmec_valid = os.path.join(project_root, 'data', 'raw', 'UIT-VSEMC', 'valid_nor_811.xlsx')
    vsmec_test  = os.path.join(project_root, 'data', 'raw', 'UIT-VSEMC', 'test_nor_811.xlsx')

    output_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)

    splits = [
        ('train', vigo_train, vsmec_train),
        ('valid', vigo_valid, vsmec_valid),
        ('test',  vigo_test,  vsmec_test),
    ]

    for split_name, vigo_path, vsmec_path in splits:
        print(f"\n{'='*50}")
        print(f"🔄 Xử lý split: {split_name.upper()}")
        rows = []

        # ── 1. ViGoEmotions ───────────────────────────────────
        if os.path.exists(vigo_path):
            df_vigo = pd.read_csv(vigo_path)
            before = len(df_vigo)

            for _, row in df_vigo.iterrows():
                text_clean = cleaner.clean(str(row['text']))
                if not text_clean.strip():
                    continue
                labels = parse_labels(row['labels'])
                rows.append({'text': text_clean, 'labels': labels})

            print(f"✅ ViGoEmotions: {before} → {len(rows)} samples")
        else:
            print(f"⚠️  Không tìm thấy: {vigo_path}")

        # ── 2. VSMEC ─────────────────────────────────────────
        # Maping Lable của VSMEC -> Label của ViGoEMotion
        if os.path.exists(vsmec_path):
            df_vsmec = pd.read_excel(vsmec_path)
            vsmec_count = 0
            vsmec_skip  = 0

            for _, row in df_vsmec.iterrows():
                emotion = str(row['Emotion']).strip()

                # Tạm thời cho phép Other - Neutral
                # # Bỏ Other — không có nhãn tương đương
                # if emotion == 'Other' or emotion not in VSMEC_TO_VIGO_INDEX:
                #     vsmec_skip += 1
                #     continue

                text_clean = cleaner.clean(str(row['Sentence']))
                if not text_clean.strip():
                    vsmec_skip += 1
                    continue

                labels = VSMEC_TO_VIGO_INDEX[emotion]
                rows.append({'text': text_clean, 'labels': labels})
                vsmec_count += 1

            print(f"✅ VSMEC: {vsmec_count} samples thêm vào | {vsmec_skip} bỏ qua (Other/rỗng)")
        else:
            print(f"⚠️  Không tìm thấy: {vsmec_path}")

        # ── 3. Merge & lưu ────────────────────────────────────
        df_merged = pd.DataFrame(rows)

        # Convert list → string để lưu CSV dễ đọc lại
        df_merged['labels'] = df_merged['labels'].apply(str)

        # Shuffle
        df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)

        out_path = os.path.join(output_dir, f'emotion_{split_name}.csv')
        df_merged.to_csv(out_path, index=False, encoding='utf-8-sig')

        print(f"💾 Đã lưu: emotion_{split_name}.csv ({len(df_merged)} samples)")

        # Thống kê phân bố nhãn
        all_labels = []
        for label_str in df_merged['labels']:
            all_labels.extend(parse_labels(label_str))

        from collections import Counter
        label_counts = Counter(all_labels)
        VIGO_EMOTIONS = [
            'amusement','excitement','joy','love','desire','optimism',
            'caring','pride','admiration','gratitude','relief','approval',
            'realization','surprise','curiosity','confusion','fear',
            'nervousness','remorse','embarrassment','disappointment',
            'sadness','grief','disgust','anger','annoyance',
            'disapproval','neutral'
        ]
        print(f"\n📊 Phân bố nhãn top 10:")
        for idx, cnt in label_counts.most_common(10):
            name = VIGO_EMOTIONS[idx] if idx < len(VIGO_EMOTIONS) else f'idx_{idx}'
            print(f"   {name:15s} (idx={idx:2d}): {cnt}")


def prepare_hate_data():
    """Giữ nguyên pipeline ViHSD như cũ."""
    dict_path = os.path.join(project_root, 'data', 'external', 'Xử lý teencode.xlsx')
    cleaner = TextCleaner(dict_path)

    data_files = [
        {'raw': 'data/raw/ViHSD/xlsx/train_df.xlsx', 'clean': 'data/processed/vihsd_train_clean.xlsx', 'col': 'cmt_col'},
        {'raw': 'data/raw/ViHSD/xlsx/val_df.xlsx',   'clean': 'data/processed/vihsd_valid_clean.xlsx', 'col': 'cmt_col'},
        {'raw': 'data/raw/ViHSD/xlsx/test_df.xlsx',  'clean': 'data/processed/vihsd_test_clean.xlsx',  'col': 'cmt_col'},
    ]

    print(f"\n{'='*50}")
    print("🔄 Xử lý ViHSD (Hate Speech)")
    for file in data_files:
        raw_path   = os.path.join(project_root, file['raw'])
        clean_path = os.path.join(project_root, file['clean'])

        if os.path.exists(raw_path):
            print(f"🧹 Đang xử lý: {file['raw']}")
            df = pd.read_excel(raw_path)
            df['cleaned_text'] = df[file['col']].apply(cleaner.clean)
            os.makedirs(os.path.dirname(clean_path), exist_ok=True)
            df.to_excel(clean_path, index=False)
            print(f"✅ Đã lưu: {file['clean']} ({len(df)} samples)")
        else:
            print(f"⚠️  Không tìm thấy: {raw_path}")


if __name__ == "__main__":
    print("🚀 Bắt đầu chuẩn bị dữ liệu Module 1\n")
    prepare_emotion_data()
    prepare_hate_data()
    print("\n✅ Hoàn tất!")