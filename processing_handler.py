from using_custom_model_for_user_input import using_custom_model_for_user_input
from pdf2image import convert_from_bytes, convert_from_path
from sql_handler import add_user_interaction_db
from datetime import datetime
from PIL import Image
import os
from shared import bytes_storage, log_data, event_queue
from face_detection import redact_faces_using_yolov8

def handle_input(file_path, input_time, output_file_path, user_id):
    try:
        log_data[user_id]['input_dt'] = input_time #* get input time
        log_data[user_id]['no_of_files'] = 1 #* set number of files
        extension = os.path.splitext(log_data[user_id]['uploaded_file_name'])[1].lower() #* get file extension
        log_data[user_id]['file_type'] = extension #* set file type
        
        if extension in ('.png', '.jpeg', '.jpg'):            
            img_file = Image.open(file_path)
            file_format_map = {".png":"PNG",".jpeg":"JPEG",".jpg":"JPEG"}
            processed_image = image_input_process(img_file)
            processed_image.save(output_file_path, format = file_format_map[extension])#* save the image in bytes_storage
                    
        elif extension in ('.pdf'):
            processed_pdf_images = pdf_input_process(file_path)
            processed_pdf_images[0].save(output_file_path, format="PDF", save_all = True, append_images = processed_pdf_images[1:]) #* save the images in pdf format in bytes_storage

        else:
            raise Exception("Error")
        
        log_data[user_id]['output_dt'] = datetime.now() #* get output time
        add_user_interaction_db(user_id) #* log the data to mysql
       
        event_queue.task_done() #* mark 1 job as finished
        return    
        
    except Exception as e:
        print(e)

def image_input_process(image_file):
    processed_image = redact_faces_using_yolov8(using_custom_model_for_user_input(image_file))[0]
    return processed_image

def pdf_input_process(pdf_file_path):
    images_from_pdf = convert_from_path(pdf_file_path, dpi=300) #* pdf to images
    processed_pdf_images_from_layoutlmv3 = []
    processed_pdf_images_from_yolov8 = []

    def images_batch_generator_for_layoutlmv3(images_list, batch_size): 
        for i in range(0, len(images_list), batch_size):#* layoutlmv3 processing
            yield using_custom_model_for_user_input(images_list[i:i+batch_size]) #* getting model output

    def images_batch_generator_for_yolov8(images_list, batch_size): 
        for i in range(0, len(images_list), batch_size):#* yolov8 processing
            yield redact_faces_using_yolov8(images_list[i:i+batch_size]) #* getting model output

    for i, images in enumerate(images_batch_generator_for_layoutlmv3(images_from_pdf, 4), start=1):
        processed_pdf_images_from_layoutlmv3.extend(images)
    
    for i, images in enumerate(images_batch_generator_for_yolov8(processed_pdf_images_from_layoutlmv3, 4), start=1):
        processed_pdf_images_from_yolov8.extend(images)

    return processed_pdf_images_from_yolov8