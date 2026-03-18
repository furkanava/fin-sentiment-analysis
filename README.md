# Finansal Metin Duyarlılık Analizi (Sentiment Analysis)

[![Türkçe](https://img.shields.io/badge/dil-türkçe-red.svg)](#finansal-metin-duyarlılık-analizi-sentiment-analysis) [![English](https://img.shields.io/badge/language-english-blue.svg)](#financial-text-sentiment-analysis)

## Proje Hakkında

Bu proje, finansal haber metinlerinde duyarlılık (sentiment) analizi gerçekleştirerek metinlerin **pozitif**, **negatif** veya **nötr** olduğunu sınıflandıran bir NLP sistemidir. Üç farklı model yaklaşımını aynı test seti üzerinde karşılaştırır:

1. **Temel RandomForest** — TF-IDF özellik çıkarımı + dengeli sınıf ağırlıkları
2. **SMOTE + RandomForest** — Sentetik azınlık örnekleme ile veri dengeleme
3. **FinBERT** — Finansal metinlere özel ön eğitimli transformer modeli (ProsusAI/finbert)

## Veri Kümesi

[Kaggle — Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news/data) veri seti kullanılmaktadır.

| Özellik | Değer |
|---------|-------|
| Format | CSV (`all-data.csv`) |
| Sütunlar | Sentiment, Text |
| Toplam Örnek | ~4,846 |
| Nötr | %59.4 |
| Pozitif | %28.1 |
| Negatif | %12.5 |

## Proje Mimarisi

```
fin-sentiment-analysis/
├── main.py              # Tüm pipeline: veri yükleme → eğitim → değerlendirme
├── all-data.csv         # Veri seti
├── requirements.txt     # Bağımlılıklar
├── images/              # Çalıştırıldığında oluşan grafikler
│   ├── model_comparison.png
│   ├── temel_randomforest_confusion_matrix.png
│   ├── smote_randomforest_confusion_matrix.png
│   └── finbert_confusion_matrix.png
└── README.md
```

## Kullanılan Teknolojiler

- **Python 3.11+**
- pandas, numpy — Veri manipülasyonu
- matplotlib, seaborn — Görselleştirme
- scikit-learn — Geleneksel ML modelleri ve metrikler
- imbalanced-learn — SMOTE veri dengeleme
- transformers, torch — FinBERT dil modeli

## Kurulum ve Çalıştırma

```bash
# Depoyu klonlayın
git clone https://github.com/furkanava/fin-sentiment-analysis.git
cd fin-sentiment-analysis

# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Çalıştırın
python main.py
```

Script çalıştırıldığında sırasıyla:
1. Veri yüklenir ve ön işlemden geçirilir
2. Temel RandomForest modeli eğitilip değerlendirilir
3. SMOTE ile dengelenmiş model eğitilip değerlendirilir
4. FinBERT modeli indirilir ve test seti üzerinde değerlendirilir
5. Tüm modellerin karşılaştırma grafikleri `images/` klasörüne kaydedilir
6. 5-fold çapraz doğrulama (SMOTE pipeline) çalıştırılır
7. Örnek finansal metinler üzerinde tahminler gösterilir

> **Not:** FinBERT modelinin ilk çalıştırmada indirilmesi gerekir (~400 MB). İnternet bağlantısı yoksa FinBERT adımı atlanır ve sadece RandomForest modelleri değerlendirilir.

## Çalışma Akışı

```
CSV Verisi
    │
    ▼
Metin Temizleme (lowercase, özel karakter temizliği)
    │
    ▼
Etiket Dönüşümü (positive→2, neutral→1, negative→0)
    │
    ├──────────────────────────────┐
    ▼                              ▼
TF-IDF Vektörizasyon          Ham Metin (FinBERT için)
    │                              │
    ├─► Temel RandomForest         ├─► FinBERT Tokenizer
    ├─► SMOTE + RandomForest       └─► FinBERT Model
    │                              │
    └──────────────┬───────────────┘
                   ▼
         Değerlendirme & Karşılaştırma
         (Accuracy, Precision, Recall, F1)
```

## Önemli Teknik Detaylar

- **FinBERT Label Mapping:** ProsusAI/finbert `0=positive, 1=negative, 2=neutral` çıktısı verir. Projede `0=negative, 1=neutral, 2=positive` kullanıldığı için dönüşüm uygulanır.
- **SMOTE Cross-Validation:** Çapraz doğrulama `imblearn.pipeline.Pipeline` kullanarak her fold içinde SMOTE uygular — data leakage önlenir.
- **TF-IDF:** Unigram + Bigram, max 5000 özellik.

## Gelecek Çalışmalar

- Daha büyük ve çeşitli finansal haber veri setleriyle eğitim
- SHAP değerleri ile model yorumlanabilirliği
- Çok dilli finansal metin desteği
- Zaman serisi üzerinde duyarlılık değişimi analizi

## Kaynaklar

1. Liu, Y., Wang, J., Long, L., Li, X., Ma, R., Wu, Y., & Chen, X. (2025). "A Multi-Level Sentiment Analysis Framework for Financial Texts". arXiv:2504.02429
2. Mun, Y., & Kim, N. (2025). "Leveraging Large Language Models for Sentiment Analysis and Investment Strategy Development in Financial Markets". JTAECR, 20(2), 77.
3. Bhargava, N., Radaideh, M. I., et al. (2025). "On the Impact of Language Nuances on Sentiment Analysis with LLMs". arXiv:2504.05603

---

# Financial Text Sentiment Analysis

[![Türkçe](https://img.shields.io/badge/dil-türkçe-red.svg)](#finansal-metin-duyarlılık-analizi-sentiment-analysis) [![English](https://img.shields.io/badge/language-english-blue.svg)](#financial-text-sentiment-analysis)

## About

This project performs sentiment analysis on financial news headlines, classifying them as **positive**, **negative**, or **neutral**. It compares three model approaches on the same test set:

1. **Base RandomForest** — TF-IDF features + balanced class weights
2. **SMOTE + RandomForest** — Synthetic minority oversampling for class balance
3. **FinBERT** — Pre-trained transformer model specialized for financial text (ProsusAI/finbert)

## Dataset

Uses the [Kaggle — Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news/data) dataset.

| Property | Value |
|----------|-------|
| Format | CSV (`all-data.csv`) |
| Columns | Sentiment, Text |
| Total Samples | ~4,846 |
| Neutral | 59.4% |
| Positive | 28.1% |
| Negative | 12.5% |

## Installation & Usage

```bash
git clone https://github.com/furkanava/fin-sentiment-analysis.git
cd fin-sentiment-analysis
pip install -r requirements.txt
python main.py
```

When executed, the script will:
1. Load and preprocess the data
2. Train and evaluate a base RandomForest model
3. Train and evaluate a SMOTE-balanced RandomForest model
4. Download and evaluate the FinBERT model on the test set
5. Save comparison charts to `images/`
6. Run 5-fold cross-validation with a proper SMOTE pipeline
7. Show sample predictions on example financial texts

> **Note:** FinBERT model needs to be downloaded on first run (~400 MB). If no internet connection is available, the FinBERT step is skipped and only RandomForest models are evaluated.

## Key Technical Details

- **FinBERT Label Mapping:** ProsusAI/finbert outputs `0=positive, 1=negative, 2=neutral`. The project uses `0=negative, 1=neutral, 2=positive`, so proper label conversion is applied.
- **SMOTE Cross-Validation:** Uses `imblearn.pipeline.Pipeline` to apply SMOTE within each fold, preventing data leakage.
- **TF-IDF:** Unigram + Bigram features, max 5,000 features.

## Future Work

- Training with larger and more diverse financial news datasets
- Model interpretability with SHAP values
- Multilingual financial text support
- Sentiment trend analysis over time series

## References

1. Liu, Y., Wang, J., Long, L., Li, X., Ma, R., Wu, Y., & Chen, X. (2025). "A Multi-Level Sentiment Analysis Framework for Financial Texts". arXiv:2504.02429
2. Mun, Y., & Kim, N. (2025). "Leveraging Large Language Models for Sentiment Analysis and Investment Strategy Development in Financial Markets". JTAECR, 20(2), 77.
3. Bhargava, N., Radaideh, M. I., et al. (2025). "On the Impact of Language Nuances on Sentiment Analysis with LLMs". arXiv:2504.05603
