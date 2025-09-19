from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

def redact_faces_using_yolov8(images):

    try:
      face_detect_model = YOLO('./models/yolov8/custom_face_detect.pt') #* initailize the model for face detect
      sign_detect_model = YOLO('./models/yolov8/custom_signature_detect.pt') #* initailize the model for signature detect

      results_face_detect = face_detect_model(images) #* feed the images
      results_sign_detect = sign_detect_model(images) #* feed the images      

      id2label_for_face_detect = {0: 'human face'}
      id2label_for_sign_detect = {0: 'barcode', 1: 'signature'}

      if len(results_face_detect[0].boxes) > 0: #* apply redaction for faces
        for r,image in zip(results_face_detect, images): 
          draw = ImageDraw.Draw(image)
          for box in r.boxes:
            if id2label_for_face_detect.get(int(box.cls.tolist()[0])) == 'human face':
              x1, y1, x2, y2 = [int(num) for num in box.xyxy.tolist()[0]]
              draw.rectangle([x1,y1,x2,y2], fill="black")
          
      if len(results_sign_detect[0].boxes) > 0:#* apply redaction for sign
        for r,image in zip(results_sign_detect, images):
          draw = ImageDraw.Draw(image)
          for box in r.boxes:
            if id2label_for_sign_detect.get(int(box.cls.tolist()[0])) in ['signature', 'barcode']:
              x1, y1, x2, y2 = [int(num) for num in box.xyxy.tolist()[0]]
              draw.rectangle([x1,y1,x2,y2], fill="black") 

      return images
      
    except Exception as e:
       print("Error:", e)

if __name__ == "__main__":
  _ = ''
  # images = [Image.open('./images/download.jpg'), Image.open('./images/vicks.jpg')]
  # images = [Image.open('./images/aam09c00.png'), Image.open('./images/adp7aa00.png'), Image.open('./images/adh36e00_2.png')]
  # redact_faces_using_yolov8(images)