#  PPG_Model  
###  Live Interactive Demo  
https://ppgmodel-8a9aov53t3uyhqrf6wpn6a.streamlit.app/

---

#  Signal Preprocessing Pipeline

The raw PPG signals go through several preprocessing steps to enhance quality and extract meaningful features.

##  **1. Detrending**
Removes baseline wander and slow-moving drifts that obscure real cardiac activity.

##  **2. Bandpass Filter (0.5–8 Hz)**
- Keeps frequencies relevant to heart rate (30–180 BPM).  
- Removes low-frequency drift + high-frequency noise.

##  **3. Notch Filter (50 Hz)**
Eliminates electrical powerline interference.

## **4. Normalization (0–1)**
Ensures signals from different people/sensors are comparable.

##  **5. Smoothing**
Applies a small moving-average window to reduce jitter while preserving waveform structure.

 These preprocessing steps are widely used in biomedical signal processing and help reveal the true cardiac rhythm hidden in noisy raw PPG.

---

#  Feature Extraction

From each preprocessed PPG segment, **20 features** were extracted to capture waveform quality.

##  Statistical Features
- Mean  
- Standard deviation  
- Range  
- Skewness  
- Kurtosis  

##  Peak-Based Features
- Heart rate  
- Heart rate variability (HRV)  
- Peak regularity  

##  Frequency Domain Features
- Dominant frequency  
- Spectral entropy  
- SNR estimate  

##  Time Domain Features
- Zero crossing rate  
- Signal energy  

These features help ML models distinguish **clean** vs **noisy** signals.

---

#  Model Performance

Three ML models were evaluated:

| Model                     | Accuracy | F1 Score |
|---------------------------|----------|----------|
| **XGBoost Classifier**    | **100%** | **1.00** |
| Random Forest Classifier  | 95%      | 0.96     |
| Logistic Regression       | 85%      | 0.86     |

---
## most likely it is overfitting.

When we have:

Very small dataset → only 100 rows

Very powerful model → XGBoost (high-capacity, tree-based ensemble)

Accuracy = 100% on training set

…it almost always means the model has simply memorized the training data, not learned generalizable patterns.


---

# How to Use the Project

### **1. Install dependencies**
```bash
pip install -r requirements.txt

```

2. Run the Streamlit app
```
streamlit run streamlit_ppg_app.py
```
