#1 Setup , import packages
import tensorflow as tf
from tensorflow import keras
import numpy as np
import streamlit as st
import os,pickle,re

#2.Function to load the pickle objects and keras model
def load_pickle_file(filepath):
    with open(filepath,"rb") as f:
        pickle_object = pickle.load(f)
    return pickle_object

@st.cache_resource
def load_model(filepath):
    model_object = keras.models.load_model(filepath)
    return model_object


#3. Define the file paths to the resources we want to load
tokenizer_filepath = r"C:\Users\youth\OneDrive\Desktop\YPAI08(Deep Learning)\Deeplearninglatest\nlp\assessment3\tokenizer_assessment.pkl"
label_encoder_filepath = r"C:\Users\youth\OneDrive\Desktop\YPAI08(Deep Learning)\Deeplearninglatest\nlp\assessment3\label_encoder_assessment.pkl"
model_filepath = r"C:\Users\youth\OneDrive\Desktop\YPAI08(Deep Learning)\Deeplearninglatest\nlp\assessment3\model_nlp"

#4. Load the tokenizer,label encoder and model
tokenizer = load_pickle_file(tokenizer_filepath)
label_encoder = load_pickle_file(label_encoder_filepath)
model = load_model(model_filepath)

#5 Build the components of streamlit app
#(A) A title text to display the app name
st.title("Classification for E-Commerce")
#(B) Text box to input description
with st.form("input_form"):
    text_area = st.text_area("Please input your description here")
    submitted = st.form_submit_button("Show result!")

text_inputs = [text_area]
#(C) Process the input string
# Remove unwanted string from the text input
def remove_unwanted_string(text_input):
    for index,data in enumerate(text_inputs):
        text_inputs[index] = re.sub('<.*?>'," ",data)
        text_inputs[index] = re.sub("[^a-zA-Z]"," ",data).lower()
    return text_inputs

#A. Use the function to filter unwanted string
text_filtered = remove_unwanted_string(text_inputs)
#B. Tokenize the string
text_token = tokenizer.texts_to_sequences(text_filtered)
#C. Padding and truncating
text_padded = keras.utils.pad_sequences(text_token,maxlen=200,padding = "post",truncating="post")

#D. Use the model to do prediction
y_score = model.predict(text_padded)
y_pred = np.argmax(y_score,axis=1)

#E Display the result
label_map = {i:classes for i,classes in enumerate(label_encoder.classes_)}
result = label_map[y_pred[0]]

#F Write the prediction into Streamlit
st.header("Label list")
st.write(label_encoder.classes_)
st.header("Prediction Score")
st.write(y_score)
st.header("Final Prediction")
st.write(f"The category is : {result}")