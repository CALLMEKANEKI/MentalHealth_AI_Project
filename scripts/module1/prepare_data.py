# =============================================================================
# Module 1 - Prepare Data (Ver 2.6)
# =============================================================================
# Thay đổi so với ver 2.5:
#   - Thêm OVERSAMPLING cho nhãn hiếm trong ViGoEmotions
#     Nhãn hiếm: disapproval(F1=0.27), realization(0.35), confusion(0.34)
#                disappointment(0.36), desire(0.38), excitement(0.41)
#     → Duplicate các sample có nhãn hiếm × OVERSAMPLE_RATIO lần
#   - Giữ nguyên HTML cleanup và teencode từ ver 2.4/2.5
# =============================================================================

import os
import sys
import ast
import pandas as pd
import numpy as np
from collections import Counter

current_dir  = os.path.dirname(os.path.abspath(__file__))
scripts_dir  = os.path.dirname(current_dir)
project_root = os.path.dirname(scripts_dir)

if project_root not in sys.path:
    sys.path.append(project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

print(f"📁 project_root: {project_root}")

from module1.preprocess import TextCleaner

# ==================== CẤU HÌNH ====================
VIGO_EMOTIONS = [
    'amusement','excitement','joy','love','desire','optimism',
    'caring','pride','admiration','gratitude','relief','approval',
    'realization','surprise','curiosity','confusion','fear',
    'nervousness','remorse','embarrassment','disappointment',
    'sadness','grief','disgust','anger','annoyance',
    'disapproval','neutral'
]

# Nhãn hiếm cần oversample — dựa trên F1 thấp từ test ver 2.5
# disapproval(0.27), confusion(0.34), realization(0.35),
# disappointment(0.36), desire(0.38), excitement(0.41)
RARE_LABEL_INDICES = [26, 15, 12, 20, 4, 1]  # index trong VIGO_EMOTIONS
OVERSAMPLE_RATIO   = 3   # duplicate mỗi sample có nhãn hiếm × 3 lần
MIN_LABEL_COUNT    = 500 # nếu nhãn có < 500 mẫu → oversample

# Mapping VSMEC → ViGoEmotions index
VSMEC_TO_VIGO = {
    'Anger':     [24],
    'Disgust':   [23],
    'Enjoyment': [2],
    'Fear':      [16],
    'Sadness':   [21],
    'Surprise':  [13],
    # 'Other' → bỏ
}


def parse_labels(label_str):
    try:
        return ast.literal_eval(str(label_str))
    except Exception:
        return [27]


def oversample_rare_labels(rows, rare_indices, min_count, ratio):
    """
    Duplicate các sample có nhãn hiếm để cân bằng distribution.

    Args:
        rows:         list of dict {'text': ..., 'labels': [...]}
        rare_indices: list of label indices cần oversample
        min_count:    ngưỡng — nhãn có < min_count mẫu mới oversample
        ratio:        số lần duplicate

    Returns:
        rows mới (có thêm duplicated samples)
    """
    # Đếm số lần mỗi nhãn xuất hiện
    label_counts = Counter()
    for row in rows:
        for idx in row['labels']:
            label_counts[idx] += 1

    print("\n📊 Phân bố nhãn trước oversampling (top 10 ít nhất):")
    for idx, cnt in sorted(label_counts.items(), key=lambda x: x[1])[:10]:
        name = VIGO_EMOTIONS[idx] if idx < len(VIGO_EMOTIONS) else f'idx_{idx}'
        print(f"   {name:20s} (idx={idx}): {cnt}")

    # Tìm sample có nhãn hiếm
    augmented = []
    for row in rows:
        labels = row['labels']
        should_oversample = any(
            idx in rare_indices and label_counts[idx] < min_count
            for idx in labels
        )
        if should_oversample:
            for _ in range(ratio - 1):  # -1 vì bản gốc đã có
                augmented.append(row)

    rows_new = rows + augmented
    print(f"\n✅ Oversample: thêm {len(augmented)} samples")
    print(f"   Tổng sau oversample: {len(rows_new)}")

    # In lại phân bố sau oversample
    label_counts_after = Counter()
    for row in rows_new:
        for idx in row['labels']:
            label_counts_after[idx] += 1

    print("\n📊 Phân bố nhãn sau oversampling (các nhãn được oversample):")
    for idx in rare_indices:
        name = VIGO_EMOTIONS[idx] if idx < len(VIGO_EMOTIONS) else f'idx_{idx}'
        before = label_counts[idx]
        after  = label_counts_after[idx]
        print(f"   {name:20s}: {before} → {after}")

    return rows_new


def prepare_emotion_data():
    dict_path = os.path.join(project_root, 'data', 'external', 'Xử lý teencode.xlsx')
    cleaner   = TextCleaner(dict_path)

    vigo_train = os.path.join(project_root, 'data', 'raw', 'ViGoEmotions', 'train.csv')
    vigo_valid = os.path.join(project_root, 'data', 'raw', 'ViGoEmotions', 'valid.csv')
    vigo_test  = os.path.join(project_root, 'data', 'raw', 'ViGoEmotions', 'test.csv')

    vsmec_train = os.path.join(project_root, 'data', 'raw', 'UIT-VSEMC', 'train_nor_811.xlsx')
    vsmec_valid = os.path.join(project_root, 'data', 'raw', 'UIT-VSEMC', 'valid_nor_811.xlsx')
    vsmec_test  = os.path.join(project_root, 'data', 'raw', 'UIT-VSEMC', 'test_nor_811.xlsx')

    output_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)

    splits = [
        ('train', vigo_train, vsmec_train, True),   # True = có oversample
        ('valid', vigo_valid, vsmec_valid, False),  # False = không oversample val/test
        ('test',  vigo_test,  vsmec_test,  False),
    ]

    for split_name, vigo_path, vsmec_path, do_oversample in splits:
        print(f"\n{'='*55}")
        print(f"🔄 Xử lý split: {split_name.upper()}")
        rows = []

        # ── 1. ViGoEmotions ───────────────────────────────────────
        if os.path.exists(vigo_path):
            df_vigo = pd.read_csv(vigo_path)
            for _, row in df_vigo.iterrows():
                text_clean = cleaner.clean(str(row['text']))
                if not text_clean.strip():
                    continue
                labels = parse_labels(row['labels'])
                rows.append({'text': text_clean, 'labels': labels})
            print(f"✅ ViGoEmotions: {len(rows)} samples")
        else:
            print(f"⚠️  Không tìm thấy: {vigo_path}")

        # ── 2. VSMEC ─────────────────────────────────────────────
        if os.path.exists(vsmec_path):
            df_vsmec    = pd.read_excel(vsmec_path)
            vsmec_count = 0
            for _, row in df_vsmec.iterrows():
                emotion = str(row['Emotion']).strip()
                if emotion not in VSMEC_TO_VIGO:
                    continue
                text_clean = cleaner.clean(str(row['Sentence']))
                if not text_clean.strip():
                    continue
                rows.append({'text': text_clean, 'labels': VSMEC_TO_VIGO[emotion]})
                vsmec_count += 1
            print(f"✅ VSMEC: thêm {vsmec_count} samples")
        else:
            print(f"⚠️  Không tìm thấy: {vsmec_path}")

        # ── 3. Oversample nhãn hiếm (chỉ train) ──────────────────
        if do_oversample:
            rows = oversample_rare_labels(
                rows,
                rare_indices=RARE_LABEL_INDICES,
                min_count=MIN_LABEL_COUNT,
                ratio=OVERSAMPLE_RATIO
            )

        # ── 4. Merge & lưu ────────────────────────────────────────
        df_merged = pd.DataFrame(rows)
        df_merged['labels'] = df_merged['labels'].apply(str)
        df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)

        out_path = os.path.join(output_dir, f'emotion_{split_name}.csv')
        df_merged.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"💾 Saved: emotion_{split_name}.csv ({len(df_merged)} samples)")


def prepare_hate_data():
    dict_path = os.path.join(project_root, 'data', 'external', 'Xử lý teencode.xlsx')
    cleaner   = TextCleaner(dict_path)

    data_files = [
        {'raw': 'data/raw/ViHSD/xlsx/train_df.xlsx', 'clean': 'data/processed/vihsd_train_clean.xlsx', 'col': 'cmt_col'},
        {'raw': 'data/raw/ViHSD/xlsx/val_df.xlsx',   'clean': 'data/processed/vihsd_valid_clean.xlsx', 'col': 'cmt_col'},
        {'raw': 'data/raw/ViHSD/xlsx/test_df.xlsx',  'clean': 'data/processed/vihsd_test_clean.xlsx',  'col': 'cmt_col'},
    ]

    print(f"\n{'='*55}")
    print("🔄 Xử lý ViHSD (Hate Speech)")
    for file in data_files:
        raw_path   = os.path.join(project_root, file['raw'])
        clean_path = os.path.join(project_root, file['clean'])
        if os.path.exists(raw_path):
            df = pd.read_excel(raw_path)
            df['cleaned_text'] = df[file['col']].apply(cleaner.clean)
            os.makedirs(os.path.dirname(clean_path), exist_ok=True)
            df.to_excel(clean_path, index=False)
            print(f"✅ {os.path.basename(clean_path)} ({len(df)} samples)")
        else:
            print(f"⚠️  Không tìm thấy: {raw_path}")


if __name__ == "__main__":
    print("🚀 Bắt đầu chuẩn bị dữ liệu Module 1 Ver 2.6\n")
    prepare_emotion_data()
    prepare_hate_data()
    print("\n✅ Hoàn tất!")