import streamlit as st
import pandas as pd
import time
import cv2
import requests
from utils import save_img, detect_image
from PIL import Image
from tf_keras.models import load_model
import os

st.title("Age and Gender Prediction")

st.header("나이와 성별을 예측할 사진을 업로드해주세요")

st.caption("choose a photo....")

file = st.file_uploader("파일 선택(jpg or jpeg or png)", type = ['jpg', 'jpeg', 'png'])


def save_uploaded_file(uploaded_file):
    # Save the uploaded file to a temporary directory
    try:
        os.makedirs('tempDir', exist_ok=True)
        file_path = os.path.join('tempDir', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


##def call_model_server(file_path, saved_path):
    url = "http://localhost:8501//predict"
    response = requests.post(url, json={'file_path': file_path, 'saved_path': saved_path})
    return response.json()


if file is not None:
    img_path = save_img(file)
    img = Image.open(img_path)
    st.image(img)
    file_path = save_uploaded_file(file)

    age_model_path = "./weight/model_age.hdf5"
    gender_model_path = "./weight/model_gender.hdf5"

    model_age = load_model(age_model_path)
    model_gender = load_model(gender_model_path)

    
    with st.spinner('Classifying...'):
        progress_bar = st.progress(0)
    
        crop_img_arr, age_arr, gender_arr, box_img =  detect_image(img_path, model_age, model_gender)
    
        saved_path = os.path.join(os.getcwd(),'tempDir')
        img_name = file_path.split('\\')[-1]
        
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.01)  # Simulate some work being done
    
    st.markdown("---")

    st.markdown("<h2 style='text-align: center;'>성별 및 얼굴 나이 예측 결과</h2>", unsafe_allow_html=True)
    
    st.title("결과 사진")
    st.image(box_img, caption='Predicted Result', use_column_width=True)

    with open(os.path.join(saved_path, img_name), "rb") as file:
        st.download_button(
            label="Download Result Image",
            data=file,
            file_name=f"result_{img_name}",
            mime="image/png"
        )
         
    st.markdown(
        """
        <style>
        div.stDownloadButton { display: flex; justify-content: center; }
        div.st-emotion-cache-1kyxreq { display: flex; justify-content: center; }
        h3 { text-align: center; }
        </style>
        """,
        unsafe_allow_html=True
    )

      # Prepare data for the table
    data = []
    for i in range(len(crop_img_arr)):
        age = age_arr[i]
        gender = gender_arr[i]
        face_id = f"ID{i + 1}"
        data.append({"ID": face_id, "Age": age, "Gender": gender})
    
    # Create a dataframe
    df = pd.DataFrame(data)
    
    # Display the table
    st.markdown("<h3 style='text-align: center;'>예측 결과 표</h3>", unsafe_allow_html=True)
    st.table(df[['ID', 'Age', 'Gender']])


    if crop_img_arr:
        for i in range(len(crop_img_arr)):
            age = age_arr[i]
            gender = gender_arr[i]
            crop_img = crop_img_arr[i]
            cap = str(age) + "_" + str(gender)
            st.image(crop_img, caption= cap,  width=300)
            
    else:
        st.write("사람 얼굴을 탐색하지 못했습니다. 다른 사진을 업로드해주세요.")

    st.markdown("""
        <style>
        
        div.st-emotion-cache-1kyxreq.e115fcil2 {
            display: flex;
            justify-content: center;
        }
        
        </style>
        """, unsafe_allow_html=True)
