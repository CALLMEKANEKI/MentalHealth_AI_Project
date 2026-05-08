import torch
import numpy as np
from torch.utils.data import Dataset

# 28 nhãn ViGoEmotions (index 0-27)
VIGO_EMOTIONS = [
    'amusement', 'excitement', 'joy', 'love', 'desire', 'optimism',
    'caring', 'pride', 'admiration', 'gratitude', 'relief', 'approval',
    'realization', 'surprise', 'curiosity', 'confusion', 'fear',
    'nervousness', 'remorse', 'embarrassment', 'disappointment',
    'sadness', 'grief', 'disgust', 'anger', 'annoyance',
    'disapproval', 'neutral'
]
NUM_VIGO_LABELS = len(VIGO_EMOTIONS)  # 28

# Mapping VSMEC 7 nhãn → index trong ViGoEmotions
VSMEC_TO_VIGO = {
    'Anger':     [24],   # anger
    'Disgust':   [23],   # disgust
    'Enjoyment': [2],    # joy
    'Fear':      [16],   # fear
    'Sadness':   [21],   # sadness
    'Surprise':  [13],   # surprise
    'Other':     [27],   # neutral
}


def make_multilabel_vector(label_indices, num_labels=NUM_VIGO_LABELS):
    """
    Chuyển list index nhãn thành multi-hot vector.
    Ví dụ: [2, 8, 3] → [0,0,1,1,0,0,0,0,1,0,...] (28 chiều)
    """
    vec = np.zeros(num_labels, dtype=np.float32)
    for idx in label_indices:
        if 0 <= idx < num_labels:
            vec[idx] = 1.0
    return vec


class MultiTaskDataset(Dataset):
    """
    Dataset cho bài toán multi-task:
    - Emotion: multi-label (28 nhãn ViGoEmotions) → BCEWithLogitsLoss
    - Hate Speech: single-label (3 nhãn) → CrossEntropyLoss

    Cách dùng:
    - Với ViGoEmotions: truyền emotion_labels là list of lists, ví dụ [[2,8],[0,1,3],...]
    - Với VSMEC:        truyền emotion_labels là list of strings, ví dụ ['Anger','Joy',...]
                        hoặc list of ints theo VSMEC_TO_VIGO
    - Với ViHSD:        emotion_labels=None, hate_labels là list of ints
    """

    def __init__(self, texts, emotion_labels, hate_labels, tokenizer, max_len,
                 emotion_source='vigo'):
        """
        Args:
            texts:          list of str
            emotion_labels: list of (list of int) cho ViGoEmotions
                            hoặc list of str cho VSMEC
                            hoặc None nếu không có emotion label (ViHSD)
            hate_labels:    list of int hoặc None nếu không có hate label
            tokenizer:      PhoBERT tokenizer
            max_len:        int
            emotion_source: 'vigo' | 'vsmec' | 'none'
        """
        self.texts = texts
        self.hate_labels = hate_labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.emotion_source = emotion_source

        # Xử lý emotion labels thành multi-hot vector
        if emotion_source == 'vigo':
            # emotion_labels là list of list of int
            # Ví dụ: [[2, 8, 3], [27], [0, 1]]
            self.emotion_vectors = [
                make_multilabel_vector(indices)
                for indices in emotion_labels
            ]
        elif emotion_source == 'vsmec':
            # emotion_labels là list of string nhãn VSMEC
            # Ví dụ: ['Anger', 'Joy', 'Other']
            self.emotion_vectors = [
                make_multilabel_vector(VSMEC_TO_VIGO.get(label, [27]))
                for label in emotion_labels
            ]
        else:
            # Không có emotion (ViHSD rows)
            self.emotion_vectors = [
                np.full(NUM_VIGO_LABELS, -1.0, dtype=np.float32)
                for _ in range(len(texts))
            ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors=None
        )

        hate_label = int(self.hate_labels[idx]) if self.hate_labels is not None else -100

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'emotion_labels': torch.tensor(self.emotion_vectors[idx], dtype=torch.float),
            'hate_labels': torch.tensor(hate_label, dtype=torch.long),
            'has_emotion': torch.tensor(
                1.0 if self.emotion_source != 'none' else 0.0,
                dtype=torch.float
            )
        }