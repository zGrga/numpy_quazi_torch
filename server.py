import streamlit as st
from numpy_quazi_torch.custom_models.MNIST_model_2 import MNIST
from numpy_quazi_torch.models.SoftmaxLoss import SoftmaxLoss
import numpy as np
import cv2

softmax = SoftmaxLoss()
st.title('Hello')

st.markdown("> In this section you can try MNIST detector. Try to upload image of handwritten digit.")

if 'model' not in st.session_state:
    model = MNIST()
    model.load_parameters('best/weights')
    st.session_state['model'] = model

uploaded_file = st.file_uploader('Choose a file')

submit = st.button('Submit')

if submit:
    if 'model' not in st.session_state:
        st.error('Error on loading model')
    else:
        if uploaded_file is not None:
            with open('image.png', 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            st.image('image.png')
            img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))
            img = np.array([img])[:, :, :, None]
            print(img.shape)
            got = st.session_state['model'](img)
            print(softmax(got))
            ind = np.argmax(got[0])
            print()
            st.write(f'Preddicted digit: {ind}, with confidence: {softmax(got)[0][ind]}')
        else:
            st.error('Please upload file')