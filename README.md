# Smart-Product-Pricing_MLM_2
#### **Data Preparation**

* **Dataset**: Training set (75k samples) includes sample\_id, catalog\_content (concatenated text), image\_link (URL), and price (target). Test set mirrors this without price.  
* **Preprocessing**:  
  * Text: No tokenization needed; fed directly to embedding model.  
  * Images: Downloaded via requests with 3-retry logic and timeout=5s. Failures fallback to zero tensors (blank images) to ensure 100% coverage.  
  * Target: log\_price \= log(1 \+ price) to normalize skewness (prices range \~$1-500+). Predictions reverse via expm1 and clip to ≥0.  
* **Environment**: Google Colab (T4 GPU for acceleration). Batched processing (batch\_size=32 for images, 128 for text) minimizes latency.  
* **Efficiency Measures**: Intermediate features saved as .npy files to avoid re-extraction on restarts. Total runtime: \~10-15 mins for 75k samples.

---

## **2\. Model Architecture and Algorithms**

#### **2.1 Text Processing**

* **Algorithm**: Sentence-BERT (all-MiniLM-L6-v2) – A distilled BERT variant for semantic embeddings.  
* **Architecture**: Transformer encoder (6 layers, 384 hidden dims) producing dense vectors capturing semantic similarity (e.g., "premium leather bag" → high-price cluster).  
* **Why Selected**: Fast inference (22M params), outperforms TF-IDF on descriptive text. Batch-encoded for efficiency.  
* **Output**: 384-dimensional vectors per sample.

#### **2.2 Image Processing**

* **Visual Features**:  
  * **Algorithm**: Pre-trained MobileNetV2 (ImageNet).  
  * **Architecture**: Depthwise separable convolutions (53 layers, 3.5M params, 1280-dim output from avg-pool). Frozen; removes classifier head.  
  * **Why Selected**: 2x faster than ResNet-18 on mobile/GPU, robust to e-commerce image variations (lighting, angles). GPU-accelerated via PyTorch.  
* **Color Features** (Novel Addition):  
  * **Algorithm**: KMeans clustering (scikit-learn) on downsampled pixels (64x64 RGB).  
  * **Architecture**: Unsupervised; clusters into 3 dominant colors, extracts centroids (normalized \[0,1\]).  
  * **Why Selected**: Colors drive pricing (e.g., metallic tones signal luxury). Adds 9 dims (3 RGB x 3 colors) without overhead—computed once per PIL image.  
* **Preprocessing**: Resize to 224x224, normalize (ImageNet stats). Retries handle \~5-10% download failures.  
* **Output**: Visual (1280 dims) \+ Colors (9 dims).

#### **2.3 Fusion and Prediction**

* **Fusion**: Horizontal stack (np.hstack) of text (384) \+ colors (9) \+ visual (1280) \= 1673 features. No dimensionality reduction (LightGBM handles high-dim well).  
* **Algorithm**: LightGBM (Gradient Boosting Decision Trees).  
  * **Architecture**: 200 boosting rounds, gbdt booster, num\_leaves=31, learning\_rate=0.05, feature\_fraction=0.9. Objective: regression (l1 metric for MAE approximation to SMAPE).  
  * **Why Selected**: Faster/more memory-efficient than XGBoost; native early stopping prevents overfitting. Trees capture non-linear interactions (e.g., brand text \+ black color → premium price).  
  * **Training**: 80/20 split; early\_stopping(10) on val set. Full fit on train post-val.  
* **Output**: Log-price predictions; denormalized for final CSV.  
* dular for custom datasets (e.g., add metadata cols).

---

## **Feature Engineering Techniques**

### **3.1 Text Features**

* **Technique**: Semantic embedding via contrastive learning (MiniLM pre-trained on 1B **3.2 Image Features**  
* **Visual**: Transfer learning—extract bottleneck features post-conv layers. Handles variations without augmentation (assumes clean URLs).  
* **Color**: Pixel-level clustering on resized image (avoids full-res compute). Normalizes to \[0,1\] for scale-invariance.  
  * **Formula**: Cluster RGB pixels → arg⁡min⁡∑∥xi−μc∥2 \\arg\\min \\sum \\| x\_i \- \\mu\_c \\|^2 argmin∑∥xi​−μc​∥2 for 3 centroids μ \\mu μ.

### **3.3 Fusion and Augmentation**

* **Technique**: Simple concatenation \+ tree-based learner (no neural fusion layer—keeps it lightweight).

## **4\. Other Relevant Information**

### **Limitations and Improvements**

* **Limitations**: Relies on public image URLs (failures \~5%); assumes English text. No temporal/competitor data.  
* **Potential Enhancements**:  
  * Ensemble: Average 3 LightGBM seeds for \+1% SMAPE.  
  * Advanced: Fine-tune MobileNet end-to-end (if \>30 mins budget); add OCR for image text.  
  * Deployment: Export LightGBM to ONNX for API serving; monitor drift on new products.  
* **Reproducibility**: Random seeds (42) throughout. Full code in Colab notebook (6 cells). Dependencies: sentence-transformers, lightgbm, torch, scikit-learn.
