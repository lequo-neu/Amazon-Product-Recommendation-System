# Models Directory

## 📁 Directory Structure

```
models/
├── user/              # User-based CF models (*.pkl)
├── item/              # Item-based CF models (*.pkl)
├── content/           # Content-based models (*.pkl)
├── model/             # SVD/ALS models (*.pkl)
├── trending/          # Trending algorithm data (*.pkl)
├── hybrid/            # Hybrid ensemble (*.pkl)
├── comparison_all_models.csv          # ✅ Tracked in Git
├── comparison_all_metrics.png         # ✅ Tracked in Git (moved to images/)
├── comparison_per_category.png        # ✅ Tracked in Git (moved to images/)
├── best_algorithms_per_category.csv   # ✅ Tracked in Git
└── best_per_metric.csv                # ✅ Tracked in Git
```

## ⚠️ Important Notes

### Models Are NOT Included in Git Repository

The pre-trained model files (*.pkl, *.joblib) are **NOT uploaded to GitHub** due to:
- Large file sizes (hundreds of MBs)
- GitHub file size limits (100MB/file)
- Models can be easily regenerated from notebooks

### How to Get the Models

#### Option 1: Download Pre-trained Models (Recommended for Quick Demo)
**Coming soon:** Pre-trained models will be available via:
- Google Drive link: [To be added]
- GitHub Release: [To be added]

Download and extract to this `models/` directory.

#### Option 2: Train Models Yourself (Recommended for Learning)

Run the notebooks in order:

```bash
# 1. Data collection
jupyter notebook notebooks/data_collection.ipynb

# 2. Train individual models
jupyter notebook notebooks/collaborative_filtering/user_based.ipynb
jupyter notebook notebooks/collaborative_filtering/item_based.ipynb
jupyter notebook notebooks/collaborative_filtering/content_based.ipynb
jupyter notebook notebooks/collaborative_filtering/model_based.ipynb
jupyter notebook notebooks/collaborative_filtering/trending_based.ipynb

# 3. Train hybrid system
jupyter notebook notebooks/hybrid_system.ipynb
```

**Estimated time:** 1-2 hours depending on hardware and sample size

---

## 📊 What's Tracked in Git?

✅ **Included in repository:**
- CSV files with model comparison results
- Performance metric summaries
- This README file

❌ **NOT included in repository:**
- Binary model files (*.pkl, *.joblib)
- Large numpy arrays (*.npy, *.npz)
- Temporary training files

---

## 🔍 Model Details

### User-Based CF Models
- **Location:** `models/user/`
- **Files:** Similarity matrices, KNN models per category
- **Size:** ~50-100MB per category

### Item-Based CF Models
- **Location:** `models/item/`
- **Files:** Item similarity matrices, KNN models
- **Size:** ~50-100MB per category

### Content-Based Models
- **Location:** `models/content/`
- **Files:** TF-IDF vectorizers, cosine similarity matrices
- **Size:** ~30-50MB per category

### Model-Based (SVD/ALS)
- **Location:** `models/model/`
- **Files:** Matrix factorization models
- **Size:** ~20-30MB per category

### Trending Algorithm
- **Location:** `models/trending/`
- **Files:** Popularity scores, trending item lists
- **Size:** ~5-10MB per category

### Hybrid Ensemble
- **Location:** `models/hybrid/`
- **Files:** Combined model with learned weights
- **Size:** ~10-20MB per category

---

## 💡 Tips

1. **First time setup:** Run Option 2 (train yourself) to understand the pipeline
2. **Quick demo:** Use Option 1 (download pre-trained) if available
3. **Custom categories:** Modify `utilities/configuration.py` and retrain
4. **Storage optimization:** Delete old models before retraining

---

## 🆘 Troubleshooting

**Issue:** Models not found error
```
Solution: Run the training notebooks to generate models
```

**Issue:** Out of memory during training
```
Solution: Reduce sample size in utilities/configuration.py
```

**Issue:** Models don't match current data
```
Solution: Delete models/ and retrain with current data
```

---

For more information, see main [README.md](../README.md)
