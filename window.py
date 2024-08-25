import numpy as np
import streamlit as st
import pandas as pd
import os
import time
import cv2
import requests
from PIL import Image
from tf_keras.models import load_model
from yoloface import face_analysis


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


def save_img(file):
    dir_path = os.path.join(os.getcwd(),'image')
    os.makedirs(dir_path, exist_ok=True)
    img_path = os.path.join(dir_path, file.name)
    with open(img_path, 'wb') as f:
        f.write(file.getbuffer())
    return img_path


def detect_image(img_path, model_age, model_gender):
  age_arr = []
  gender_arr = []
  crop_img_arr=[]

  img_array = np.fromfile(img_path, np.uint8)
  img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
  img2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
  
  #얼굴 탐지 모델 불러오기
  face=face_analysis()
  
  try:
    _,box,_=face.face_detection(image_path=img_path,model='full')
  except:
     print("사람 얼굴을 탐지하지 못했습니다.")
     return ['', '', '']

  for idx, (x,y,w,h) in enumerate(box):
    cv2.rectangle(img2, (x, y), (x + h, y + w), (0, 255, 0), 2)
    crop_img = img[y:y + w, x:x + h]
    crop_img_arr.append(crop_img)
    x_img = cv2.resize(crop_img, (50,50)).reshape(1,50,50,3)
    age_out = model_age.predict(x_img/255)
    age = int(np.round(age_out[0][0]))
    age_arr.append(age)
  
    gender = model_gender.predict(x_img/255)
    gender_dict = {0:"남자", 1:"여자"}
    gender_arr.append(gender_dict[int(np.round(gender[0][0]))])
  
  return crop_img_arr, age_arr, gender_arr, img2

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
