Dang kich hoat moi truong ao Py.12...
Dang kiem tra GPU...
CUDA Ready: True
Bat dau huan luyen mo hinh PhoBERT...
Loaded 70 labels
Test samples: 1348
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 3093.29it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.layer_norm.bias         | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

✅ Test F1 micro: 0.8186, macro: 0.8316
   Top-1 accuracy: 0.8361
   Top-5 accuracy: 0.9414
   Top-10 accuracy: 0.9607
   Top-20 accuracy: 0.9770

=== PER-LABEL F1 REPORT ===

⚠️  Nhãn có F1 thấp nhất (bottom 10):
  Rối loạn tâm lý                               F1=0.375
  Rối loạn phát triển                           F1=0.424
  Rối loạn hành vi                              F1=0.522
  Rối loạn hệ thần kinh thực vật ở trẻ em       F1=0.632
  Rối loạn tâm thần                             F1=0.638
  Rối loạn nhân dạng phân ly (DID)              F1=0.640
  Tâm thần phân liệt                            F1=0.651
  Rối loạn căng thẳng cấp tính (ASD)            F1=0.667
  Trầm cảm sau sinh                             F1=0.667
  Rối loạn lo âu                                F1=0.697

✅ Nhãn có F1 cao nhất (top 10):
  Hội chứng Folie à Deux                        F1=0.963
  Ảo Thanh                                      F1=0.980
  Bệnh CreutzfeldtJakob                         F1=1.000
  Bệnh Parkinson                                F1=1.000
  Hội chứng Lima                                F1=1.000
  Hội chứng Stendhal                            F1=1.000
  Hội chứng hoang tưởng được yêu (Erotomania)   F1=1.000
  Rối loạn nhân cách phân liệt (ScPD)           F1=1.000
  Rối loạn nhân cách phụ thuộc (DPD)            F1=1.000
  Rối loạn phân ly (Dissosiative Disorder)      F1=1.000

🔍 'Khác' riêng: Precision=0.707 | Recall=0.783 | F1=0.743

📝 Dự đoán mẫu (top 5):

Text: Tôi cảm thấy buồn bã, mất ngủ, không muốn gặp ai, nghĩ đến cái chết....
Top 5: [('Rối loạn trầm cảm', 0.9987762570381165), ('Trầm cảm sau sinh', 0.13619178533554077), ('Rối loạn điều chỉnh (AD)', 0.024944864213466644), ('Rối loạn giấc ngủ (Hypersomnia)', 0.016018096357584), ('Rối loạn tâm lý', 0.008602556772530079)]

Text: Tôi sợ đi thang máy, mỗi lần thấy nhện là hoảng hốt, tay run....
Top 5: [('Chứng sợ không gian hẹp (Claustrophobia)', 0.8587087988853455), ('Hội chứng sợ khoảng rộng (agoraphobia)', 0.06574676185846329), ('Khác', 0.05250207707285881), ('Rối loạn hoảng sợ (Panic disorder)', 0.005311579909175634), ('Rối loạn căng thẳng cấp tính (ASD)', 0.0032111809123307467)]
Da hoan thanh! Nhan phim bat ky de thoat.
Press any key to continue . . .