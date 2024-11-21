import streamlit as st
import pickle
import numpy as np

# Fungsi untuk memuat model dan vectorizer
def load_model_and_vectorizer(model_choice):
    try:
        if model_choice == "KNN":
            model = pickle.load(open('C:/Kuliah/APLAI/Code/model/knn_model.sav', 'rb'))
            vectorizer = pickle.load(open('C:/Kuliah/APLAI/Code/model/vectorizer_knn.sav', 'rb'))
        elif model_choice == "Logistic Regression":
            model = pickle.load(open('C:/Kuliah/APLAI/Code/model/logistic_model.sav', 'rb'))
            vectorizer = pickle.load(open('C:/Kuliah/APLAI/Code/model/vectorizer_logistic.sav', 'rb'))
        elif model_choice == "Support Vector Machine ***(SVM)***":
            model = pickle.load(open('C:/Kuliah/APLAI/Code/model/svm_model.sav', 'rb'))
            vectorizer = pickle.load(open('C:/Kuliah/APLAI/Code/model/vectorizer_svm.sav', 'rb'))
        else:
            st.error("Model choice is invalid.")
            return None, None

        
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model or vectorizer file not found.")
        return None, None

# Fungsi untuk memproses data dan membuat prediksi
def preprocess_and_predict(input_message, model, vectorizer):
    # Transform input message menggunakan vectorizer
    input_vectorized = vectorizer.transform([input_message]).toarray()
    
    # Pad data jika jumlah fitur kurang dari 2500
    current_features = input_vectorized.shape[1]
    padding_size = 2500 - current_features
    if padding_size > 0:
        padding = np.zeros((input_vectorized.shape[0], padding_size))
        input_vectorized = np.concatenate((input_vectorized, padding), axis=1)
    
    # Prediksi menggunakan model
    prediction = model.predict(input_vectorized)
    
    # Return the prediction message
    if prediction[0] == 0:
        return "The message is not spam."
    else:
        return "The message is spam."

# Streamlit App
def main():
    st.title("Message Spam Detection App")
    
    # Pilihan algoritma
    model_choice = st.radio(
        "Choose 1 Algorithm",
        ["KNN", "Logistic Regression", "Support Vector Machine ***(SVM)***"]
    )
    
    # Input dari pengguna
    input_message = st.text_area("Enter your message here:")
    
    if st.button("Check Spam"):
        if input_message.strip() == "":
            st.warning("Please input a message to check.")
        else:
            # Load model dan vectorizer berdasarkan pilihan
            model, vectorizer = load_model_and_vectorizer(model_choice)
            
            if model and vectorizer:  # Memastikan model dan vectorizer berhasil dimuat
                # Prediksi
                result = preprocess_and_predict(input_message, model, vectorizer)
                st.success(result)

# Jalankan aplikasi
if __name__ == "__main__":
    main()
