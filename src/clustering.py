import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from data_preprocessing import load_and_preprocess_data

def run_clustering():
    df = load_and_preprocess_data('data/Sleep_health_and_lifestyle_dataset.csv')
    X_unsup = df[['Sleep Duration', 'Stress Level', 'Physical Activity Level', 'Daily Steps', 'Age']]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_unsup)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df_cluster = X_unsup.copy()
    df_cluster['Cluster'] = clusters
    df_cluster['PCA1'] = X_pca[:, 0]
    df_cluster['PCA2'] = X_pca[:, 1]
    
    joblib.dump({'scaler': scaler, 'kmeans': kmeans, 'pca': pca}, 'models/clustering_pipeline.pkl')
    df_cluster.to_csv('data/clustered_data.csv', index=False)

if __name__ == "__main__":
    run_clustering()