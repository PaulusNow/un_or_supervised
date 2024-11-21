import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

def load_model_and_cluster(model_path, data_path, new_data, algorithm):
    try:
        # Load the original dataset
        original_data = pd.read_csv(data_path)

        # Create a DataFrame for the new data point
        new_data_point = pd.DataFrame(new_data)

        # Combine the new data point with the original data
        updated_data = pd.concat([original_data, new_data_point], ignore_index=True)

        # Standardizing the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(updated_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

        # Choose the algorithm based on user selection
        if algorithm == "KNN (K-Means)":
            # Load the KMeans model
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            clusters = model.predict(scaled_data)  # Use predict for already trained model
        elif algorithm == "DBSCAN":
            # Load the DBSCAN model
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            clusters = model.fit_predict(scaled_data)  # DBSCAN uses fit_predict directly

        # Add the cluster labels to the updated data
        updated_data['Cluster'] = clusters

        # Identify the cluster of the new data point (last row in the updated data)
        new_data_cluster = updated_data.iloc[-1]['Cluster']

        # Display the predicted cluster for the new data
        st.subheader(f"Prediksi Cluster untuk Data Baru: Cluster {new_data_cluster}")

        # Matplotlib scatterplot visualization (2D)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x='Annual Income (k$)', 
            y='Spending Score (1-100)', 
            hue='Cluster', 
            data=updated_data, 
            palette='Set1'
        )
        plt.title('Updated Clusters with New Data Point')
        plt.scatter(
            new_data_point['Annual Income (k$)'], 
            new_data_point['Spending Score (1-100)'], 
            marker='*', 
            s=200, 
            c='red', 
            label='New Data Point'
        )
        plt.legend()
        st.pyplot(plt)

        # Plotly 3D scatterplot visualization (3D)
        fig = go.Figure()
        for cluster in updated_data['Cluster'].unique():
            cluster_data = updated_data[updated_data['Cluster'] == cluster]
            fig.add_trace(
                go.Scatter3d(
                    x=cluster_data['Age'],
                    y=cluster_data['Spending Score (1-100)'],
                    z=cluster_data['Annual Income (k$)'],
                    mode='markers',
                    marker=dict(size=5),
                    name=f'Cluster {cluster}'
                )
            )

        # Add the new data point in 3D plot
        fig.add_trace(
            go.Scatter3d(
                x=new_data_point['Age'],
                y=new_data_point['Spending Score (1-100)'],
                z=new_data_point['Annual Income (k$)'],
                mode='markers',
                marker=dict(size=10, color='red'),
                name='New Data Point'
            )
        )

        # Layout settings
        fig.update_layout(
            title='Clusters by K-Means or DBSCAN (3D)',
            scene=dict(
                xaxis=dict(title='Age'),
                yaxis=dict(title='Spending Score'),
                zaxis=dict(title='Annual Income')
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        # Display 3D plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        return updated_data

    except Exception as e:
        st.error(f"Error during clustering or visualization: {e}")
        return None


def main():
    st.title("Unsupervised Clustering")
    st.write("Masukkan data untuk clustering:")

    # Select algorithm (K-Means or DBSCAN)
    algo = st.radio("Pilih Algoritma:", ["KNN (K-Means)", "DBSCAN"])

    # Input fields for user data
    age = st.number_input("Masukkan Umur:", min_value=0, max_value=120, value=25)
    annual_income = st.number_input("Masukkan Annual Income (k$):", min_value=0, max_value=1000, value=60)
    spending_score = st.number_input("Masukkan Spending Score (1-100):", min_value=1, max_value=100, value=50)

    if st.button("Clustering"):
        # Create dictionary for new input
        new_data = {
            'Age': [age],
            'Annual Income (k$)': [annual_income],
            'Spending Score (1-100)': [spending_score]
        }

        # Path to the saved models and dataset
        if algo == "KNN (K-Means)":
            model_file = '..//unsupervised//knn_model.sav'
        elif algo == "DBSCAN":
            model_file =  '..//unsupervised//dbscan_model.sav'
        
        data_file =  '..//unsupervised//Mall_Customers.csv'

        # Perform clustering and show results
        clustered_data = load_model_and_cluster(model_file, data_file, new_data, algo)
        if clustered_data is not None:
            st.write("Hasil Clustering (Data Terbaru):")
            st.write(clustered_data.tail())


# Run the application
if __name__ == "__main__":
    main()
