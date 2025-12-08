
import streamlit as st

st.set_page_config(layout="wide")

st.title("PPG Signal Quality Classifier")
st.write("Upload your PPG signal data to classify its quality as clean or noisy.")


import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def notch_filter(data, f0, Q, fs):
    nyquist = 0.5 * fs
    w0 = f0 / nyquist
    b, a = signal.iirnotch(w0, Q)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def detrend_signal(data):
    detrended_data = signal.detrend(data)
    return detrended_data

def smooth_signal(data, window_size):
    if window_size <= 0:
        return data
    window = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, window, mode='same')
    return smoothed_data

def normalize_signal(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return np.zeros_like(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def standardize_signal(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    if std_val == 0:
        return np.zeros_like(data)
    return (data - mean_val) / std_val


def preprocess_ppg_signal(data: np.ndarray, fs: int = 100,
                          verbose: bool = False) -> Dict[str, np.ndarray]:
    results = {'raw': data.copy()}

    if verbose:
        print("Starting PPG Signal Preprocessing Pipeline")
        print(f"Signal length: {len(data)} samples ({len(data)/fs:.1f} seconds)")
        print("-" * 50)

    detrended = detrend_signal(data)
    results['detrended'] = detrended

    bandpassed = bandpass_filter(detrended, lowcut=0.5, highcut=8.0, fs=fs, order=4)
    results['bandpassed'] = bandpassed

    notched = notch_filter(bandpassed, f0=50, Q=30, fs=fs)
    results['notched'] = notched

    normalized = normalize_signal(notched)
    results['normalized'] = normalized

    smoothed = smooth_signal(normalized, window_size=5)
    results['smoothed'] = smoothed
    results['final'] = smoothed

    if verbose:
        print("-" * 50)
        print("Preprocessing complete!")

    return results


def extract_features(signal_data: np.ndarray, fs: int = 100) -> Dict[str, float]:
    features = {}

    # Statistical features
    features['mean'] = np.mean(signal_data)
    features['std'] = np.std(signal_data)
    features['min'] = np.min(signal_data)
    features['max'] = np.max(signal_data)
    features['range'] = features['max'] - features['min']
    features['peak_to_peak'] = np.ptp(signal_data)

    # Skewness and Kurtosis
    features['skewness'] = np.mean(((signal_data - features['mean']) / features['std']) ** 3)
    features['kurtosis'] = np.mean(((signal_data - features['mean']) / features['std']) ** 4)

    # Signal energy
    features['energy'] = np.sum(signal_data ** 2)
    features['power'] = features['energy'] / len(signal_data)

    # Peak detection for heart rate estimation
    min_distance = int(0.5 * fs)
    peaks, properties = find_peaks(signal_data, distance=min_distance, prominence=0.1)

    features['peak_count'] = len(peaks)

    if len(peaks) > 1:
        avg_rr_interval = np.mean(np.diff(peaks)) / fs
        features['heart_rate'] = 60 / avg_rr_interval if avg_rr_interval > 0 else 0

        rr_intervals = np.diff(peaks) / fs
        features['hrv_sdnn'] = np.std(rr_intervals)

        features['peak_regularity'] = np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0

        features['mean_peak_prominence'] = np.mean(properties['prominences']) if 'prominences' in properties else 0
    else:
        features['heart_rate'] = 0
        features['hrv_sdnn'] = 0
        features['peak_regularity'] = 0
        features['mean_peak_prominence'] = 0

    # Frequency domain features
    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=min(256, len(signal_data)))
    features['dominant_frequency'] = freqs[np.argmax(psd)]

    # Spectral entropy
    psd_norm = psd / np.sum(psd)
    features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

    # SNR estimation
    cardiac_band = (freqs >= 0.5) & (freqs <= 3.0)
    signal_power = np.sum(psd[cardiac_band])
    total_power = np.sum(psd)
    features['snr_estimate'] = 10 * np.log10(signal_power / (total_power - signal_power + 1e-10))

    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.sign(signal_data - np.mean(signal_data))) != 0)
    features['zero_crossing_rate'] = zero_crossings / len(signal_data)

    return features


def plot_preprocessing_steps(results: Dict[str, np.ndarray], fs: int = 100,
                             segment_id: str = ""):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle(f'PPG Signal Preprocessing Steps - Segment {segment_id}', fontsize=16)

    time = np.arange(len(results['raw'])) / fs

    steps = [
        ('raw', 'Raw Signal'),
        ('detrended', 'After Detrending'),
        ('bandpassed', 'After Bandpass (0.5-8 Hz)'),
        ('notched', 'After Notch Filter (50 Hz)'),
        ('normalized', 'After Normalization'),
        ('final', 'Final Processed Signal')
    ]

    for idx, (key, title) in enumerate(steps):
        ax = axes[idx // 2, idx % 2]
        ax.plot(time, results[key], linewidth=1)
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


@st.cache_data
def load_data():
    df = pd.read_csv("mock_ppg_dataset.csv")
 
    df_ppg = df['signal'].str.split(' ', expand=True)
    df_ppg = df_ppg.apply(pd.to_numeric, errors='coerce')
    df_ppg = df_ppg.add_prefix('ppg_')
    df = pd.concat([df, df_ppg], axis=1)
    return df

@st.cache_resource
def load_model_and_encoder():
    model_xgb = joblib.load('xgboost_model.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    return model_xgb, label_encoder

df = load_data()
model_xgb, label_encoder = load_model_and_encoder()

fs = 100 
st.sidebar.header("Signal Selection")
segment_ids = df['id'].tolist()
selected_segment_id = st.sidebar.selectbox(
    "Select a PPG Segment ID",
    segment_ids
)


selected_row = df[df['id'] == selected_segment_id].iloc[0]
raw_signal_data = selected_row.filter(like='ppg_').to_numpy().astype(float)

st.header(f"Analysis for Segment ID: {selected_segment_id}")
st.subheader("Raw PPG Signal")
fig_raw, ax_raw = plt.subplots(figsize=(12, 4))
ax_raw.plot(raw_signal_data)
ax_raw.set_xlabel("Sample Index")
ax_raw.set_ylabel("Amplitude")
ax_raw.set_title("Raw PPG Signal")
st.pyplot(fig_raw)


processed_results = preprocess_ppg_signal(raw_signal_data, fs=fs, verbose=False)
extracted_features = extract_features(processed_results['final'], fs=fs)

features_df_display = pd.DataFrame([extracted_features]).T.rename(columns={0: "Value"})


st.subheader("Preprocessing Steps")
st.pyplot(plot_preprocessing_steps(processed_results, fs=fs, segment_id=selected_segment_id))

st.subheader("Extracted Features")
st.dataframe(features_df_display)

features_for_prediction = pd.DataFrame([extracted_features])
prediction_numeric = model_xgb.predict(features_for_prediction)

prediction_label = label_encoder.inverse_transform(prediction_numeric)[0]

st.subheader("Signal Quality Prediction")
if prediction_label == 'clean':
    st.success(f"Predicted Quality: **{prediction_label.upper()}**")
else:
    st.error(f"Predicted Quality: **{prediction_label.upper()}**")

st.markdown("--- ")
st.markdown("**Note:** This application demonstrates PPG signal preprocessing, feature extraction, and classification. The model's performance on this dataset is highly accurate, but real-world performance may vary.")
