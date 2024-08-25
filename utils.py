import os
import cv2
import numpy as np
from yoloface import face_analysis
 
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