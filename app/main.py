import streamlit as st 
from model.model import model, encode
import cv2
import numpy as np

st.title('Apple Disease Detection')

upload_file = st.file_uploader('Upload a file', type = ['JPG', 'PNG', 'JPEG'])
if upload_file:
    st.write('File Uploaded Sucessfully!')
    st.divider()
    st.image(upload_file)

    def preprocessing(uploaded_file):
        ''' 
        Transform image to array
        '''
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  
        return img

    image_pixels = preprocessing(uploaded_file = upload_file)
    
    def preprocess_predict(image):
        ''' 
        Preprocess image for the model
        '''
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image / 255
        image = np.expand_dims(image, axis = 0)
        return image
    
    image_pixels = preprocess_predict(image = image_pixels)

    def predict(image):
        index_pred = np.argmax(model.predict(image))
        predict_name = encode.inverse_transform([index_pred])[0]
        return predict_name
    
    prediction = predict(image = image_pixels)
    st.write(f"The predicted apple disease is: **{prediction}**")






