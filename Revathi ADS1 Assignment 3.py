# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 21:24:28 2024

@author: Revathi Nanubala
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.optimize import curve_fit
from sklearn.metrics import silhouette_score

#Read weather data from a CSV file.
def read_weather_data(file_path, nrows=None, usecols=None):
    return pd.read_csv(file_path, nrows=nrows, usecols=usecols)

def logistic_model(x, a, b, c):
    return c / (1 + np.exp(-(x - b) / a))

def err_ranges(x, y, fit, confidence=0.95):
    """
    Calculate the error ranges for the polynomial fit.
    """
    y_pred = np.polyval(fit, x)
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    n = len(y)
    p = len(fit)
    std_error = np.sqrt(ss_res / (n - p)) * np.sqrt(1 + 1/n + ((x - np.mean(x))**2 / np.sum((x - np.mean(x))**2)))
    t_val = t.ppf((1 + confidence) / 2, n - p)
    ci = t_val * std_error
    return y_pred - ci, y_pred + ci

def calculate_wcss(data, max_k):
    """
    Calculate the sum of squared distances for different clusters.
    """
    wcss = []
    for n in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

# File path and constants
file_path_weather = 'C:\\Users\\Revathi Nanubala\\Downloads\\minute_weather\\minute_weather.csv'
chunk_size = 50000
max_k = 10
poly_degree = 2


# Data normalization
scaler = StandardScaler()

# Reading and preprocessing data
sample_data = read_weather_data(file_path_weather, nrows=100000, usecols=['air_pressure', 'air_temp'])
sample_normalized = scaler.fit_transform(sample_data)

# WCSS calculation for Elbow Method
wcss = calculate_wcss(sample_normalized, max_k)

# Optimal number of clusters determination
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(sample_normalized)
labels = kmeans.labels_
silhouette_avg = silhouette_score(sample_normalized, labels)
print("The average silhouette_score is :", silhouette_avg)

# Elbow curve plotting
plt.figure(figsize=(12, 7))
plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='-', color='blue', label='WCSS')
plt.axvline(x=n_clusters, color='green', linestyle='--', label=f'Optimal clusters: {n_clusters}')
plt.text(n_clusters, max(wcss)/2, f'Silhouette: {silhouette_avg:.2f}', color='red')
plt.title('Elbow Curve')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.legend()
plt.show()

#Processing and clustering data in chunks
all_clusters = []
for chunk in pd.read_csv(file_path_weather, usecols=['air_pressure', 'air_temp', 'hpwren_timestamp'], chunksize=chunk_size):
    chunk_normalized = scaler.transform(chunk[['air_pressure', 'air_temp']])
    chunk['cluster'] = kmeans.predict(chunk_normalized)
    all_clusters.append(chunk)

#Combining chunked data
df_combined = pd.concat(all_clusters)

#Cluster centers calculation
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
sorted_centers = cluster_centers[cluster_centers[:, 0].argsort()]

#Polynomial fitting
coefficients = np.polyfit(df_combined['air_pressure'], df_combined['air_temp'], poly_degree)
sorted_pressures = np.sort(df_combined['air_pressure'])
polynomial = np.polyval(coefficients, sorted_pressures)

#Error range calculation
lower_bound, upper_bound = err_ranges(sorted_pressures, df_combined['air_temp'], coefficients)

#Cluster center plotting
plt.figure(figsize=(12, 7))
plt.plot(sorted_centers[:, 0], sorted_centers[:, 1], '-o', color='green', label='Cluster Centers')
plt.xlabel('Air Pressure')
plt.ylabel('Air Temperature')
plt.title('K-Means Cluster Centers')
plt.legend()
plt.show()

#Curve fit and cluster center plotting
plt.figure(figsize=(12, 7))
plt.plot(sorted_pressures, polynomial, color='orange', label='Polynomial Fit', linestyle='--')
plt.plot(sorted_centers[:, 0], sorted_centers[:, 1], '-o', color='green', label='Cluster Centers')
plt.xlabel('Air Pressure')
plt.ylabel('Air Temperature')
plt.title('Curve Fit and K-Means Cluster Centers')
plt.legend()
plt.show()

# Plotting Cluster Centers, Curve Fit with Error Ranges
plt.figure(figsize=(12, 7))
plt.plot(sorted_pressures, polynomial, color='orange', label='Polynomial Fit', linestyle='--')
plt.fill_between(sorted_pressures, lower_bound, upper_bound, color='orange', alpha=0.3, label='Confidence Interval')
plt.plot(sorted_centers[:, 0], sorted_centers[:, 1], '-o', color='green', label='Cluster Centers')
plt.xlabel('Air Pressure')
plt.ylabel('Air Temperature')
plt.title('Curve Fit and K-Means Cluster Centers with Confidence Interval')
plt.legend()
plt.show()

# Bar Plot for Air Temperature and Air Pressure Over Years
df_combined['year'] = pd.to_datetime(df_combined['hpwren_timestamp'], format='%Y-%m-%d %H:%M:%S').dt.year
yearly_data = df_combined[df_combined['year'].between(2011, 2014)]
yearly_avg = yearly_data.groupby('year')['air_pressure', 'air_temp'].mean()

#Data transposition and cleaning
yearly_avg_transposed = yearly_avg.transpose()
print(yearly_avg_transposed.head())
yearly_avg_transposed.fillna(method='ffill', inplace=True)
yearly_avg_transposed.index = ['Air Pressure', 'Air Temperature']
yearly_avg_transposed.reset_index(inplace=True)
yearly_avg_transposed.columns = yearly_avg_transposed.columns.astype(str)

# Bar Plot for Air Temperature and Air Pressure Over Years
plt.figure(figsize=(12, 7))
yearly_avg.plot(kind='bar')
plt.xlabel('Year')
plt.ylabel('Average Values')
plt.title('Average Air Temperature and Air Pressure (2011-2014)')
plt.show()

# Logistic Fit
# Using air pressure as the independent variable for logistic fit
popt, pcov = curve_fit(logistic_model, df_combined['air_pressure'], df_combined['air_temp'], maxfev=10000)

air_pressure_to_predict = 1015
predicted_temp = logistic_model(air_pressure_to_predict, *popt)
perr = np.sqrt(np.diag(pcov))
temp_error = np.sqrt(sum([(perr[i] * logistic_model(air_pressure_to_predict, *(popt + np.eye(3)[i] * perr[i]) - predicted_temp)**2) for i in range(3)]))

print(f"Predicted air temperature for air pressure {air_pressure_to_predict} is {predicted_temp:.2f}±{temp_error:.2f}")

plt.figure(figsize=(12, 7))
plt.scatter(df_combined['air_pressure'], df_combined['air_temp'], color='brown', label='Data', alpha=0.5)
plt.plot(sorted_pressures, logistic_model(sorted_pressures, *popt), color='black', label='Logistic Fit')
plt.errorbar(air_pressure_to_predict, predicted_temp, yerr=temp_error, fmt='o', color='black', label='Prediction with Uncertainty')
plt.xlabel('Air Pressure')
plt.ylabel('Air Temperature')
plt.title('Logistic Fit for Air Temperature vs. Air Pressure with Prediction')
plt.legend()
plt.show()

