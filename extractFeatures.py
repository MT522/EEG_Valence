from sklearn.linear_model import LinearRegression
from scipy.signal import welch
from scipy.integrate import simps
import numpy as np

def calc_variance(data):
    return np.std(data, axis=1).T

def calc_amp_hist(data, num_bins):
    num_channels = data.shape[0]
    num_trials = data.shape[2]
    features = np.zeros((num_trials, num_channels*(num_bins+1)))
    for i in range(num_trials):
        for j in range(num_channels):
            features[i, j*(num_bins+1):(j+1)*(num_bins+1)] = np.histogram(data[j, :, i], num_bins)[1]

    return features

def calc_ar(data, model_size, fit_intercept):
    num_channels = data.shape[0]
    num_trials = data.shape[2]
    signal_length = data.shape[1]

    features = np.zeros((num_trials, num_channels*(model_size + (lambda x: 1 if x else 0)(fit_intercept))))
    for i in range(num_trials):
        for j in range(num_channels):
            signal = data[j, :, i]

            feature_mat = np.zeros((signal_length-model_size, model_size))
            for k in range(model_size, signal_length):
                feature_mat[k-model_size, :] = signal[k-model_size:k]

            lr = LinearRegression(fit_intercept=fit_intercept)
            lr.fit(feature_mat, signal[model_size:])

            if fit_intercept:
                features[i, j*(model_size+1):(j+1)*(model_size+1)] = np.insert(lr.coef_, 0, lr.intercept_)
            else:
                features[i, j*(model_size):(j+1)*(model_size)] = lr.coef_

    return features

def calc_correlation(data):
    num_channels = data.shape[0]
    num_trials = data.shape[2]

    features = np.zeros((num_trials, num_channels**2))
    for i in range(num_trials):
        for j in range(num_channels):
            for k in range(num_channels):
                mean1 = np.mean(data[j, :, i])
                mean2 = np.mean(data[k, :, i])

                features[i, j*num_channels + k] = np.mean((data[j, :, i] - mean1)*(data[k, :, i])-mean2)

    return features

def calc_max_freq(data, fs): 
    num_channels = data.shape[0]
    num_trials = data.shape[2]

    features = np.zeros((num_trials, num_channels))
    for i in range(num_trials):
        for j in range(num_channels):      
            frequencies, psd = welch(data[j, :, i], fs=fs, nperseg=2048)
            frequencies = frequencies.ravel()
            features[i, j] = frequencies[np.argmax(psd)]

    return features

def calc_mean_freq(data, fs): 
    num_channels = data.shape[0]
    num_trials = data.shape[2]

    features = np.zeros((num_trials, num_channels))
    for i in range(num_trials):
        for j in range(num_channels):      
            frequencies, psd = welch(data[j, :, i], fs=fs, nperseg=2048)
            frequencies = frequencies.ravel()
            features[i, j] = np.sum(frequencies * psd) / np.sum(psd)

    return features

def calc_median_freq(data, fs): 
    num_channels = data.shape[0]
    num_trials = data.shape[2]

    features = np.zeros((num_trials, num_channels))
    for i in range(num_trials):
        for j in range(num_channels):      
            frequencies, psd = welch(data[j, :, i], fs=fs, nperseg=2048)
            frequencies = frequencies.ravel()
            cumulative_psd = np.cumsum(psd)
            median_index = np.where(cumulative_psd >= cumulative_psd[-1] / 2)[0][0]

            features[i, j] = frequencies[median_index]

    return features

def calc_rel_energy(data, fs, bands):
    num_channels = data.shape[0]
    num_trials = data.shape[2]
    
    features = np.zeros((num_trials, num_channels*len(bands.keys())))
    for i in range(num_trials):
        for j in range(num_channels):
            frequencies, psd = welch(data[j, :, i], fs=fs, nperseg=2048)
            frequencies = frequencies.ravel()

            total_energy = simps(psd, frequencies)
            
            for k, (band, (low, high)) in enumerate(bands.items()):
                idx_band = np.logical_and(frequencies >= low, frequencies <= high)
                band_energy = simps(psd[idx_band], frequencies[idx_band], axis=0)
                relative_band_energy = band_energy / total_energy

                features[i, j*len(bands.keys()) + k] = relative_band_energy

    return features