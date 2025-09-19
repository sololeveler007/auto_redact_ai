from transformers import AutoModelForTokenClassification, AutoProcessor
import torch
from PIL import ImageDraw, Image, ImageFont
import numpy as np

def unnormalize_boxes(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000)
    ]

def flabel_2_nlabel(label):
    label = label[2:]
    if not label:
        return "other"
    else:
        return label
    
def preprocess_images(images):
    # if image.mode == "1":
    #     print("modified")
    #     img_array = np.array(image).astype(np.float32)
    #     img_array = np.expand_dims(img_array, axis=-1)
    #     img_array = np.repeat(img_array, 3, axis=-1)
    #     modified_pillow_image = Image.fromarray((img_array*255).astype(np.uint8))
    #     return modified_pillow_image
    # else:
    #     print("not modified")
    #     return image
    if not isinstance(images, list): #* (check) if 1 image then add to list 
        images = [images]
    processed_images = [img.convert("RGB") if img.mode != "RGB" else img for img in images] #* (check) convert to RGB if not

    return images, processed_images


def using_custom_model_for_user_input(images):
    images, preprocessed_images = preprocess_images(images) #* preprocess and confirmations
    try:
        repo_id = "dark1007/layoutlmv3"
        configured = False
        while not configured:
            try:
                model = AutoModelForTokenClassification.from_pretrained(repo_id) #* loading trained model
                processor = AutoProcessor.from_pretrained(repo_id, apply_ocr = True) #* loading trained processor
                configured = True
            except Exception as e:
                print("Error in model config:",str(e))

        #* setting the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #* moving the model to device
        model.to(device)

        encodings = processor(preprocessed_images, return_tensors='pt', padding = 'max_length', truncation = True)

        for k,v in encodings.items():
            encodings[k] = v.to(device)
        id2labels = {0: 'O', 1: 'B-NAME', 2: 'I-NAME', 3: 'B-ADDRESS', 4: 'I-ADDRESS', 5: 'B-ID', 6: 'I-ID', 7: 'B-EMAIL', 8: 'I-EMAIL', 9: 'B-PHONE', 10: 'I-PHONE', 11: 'B-DOB',12: 'I-DOB'}
        outputs = model(**encodings)

        label2color = {'name':'blue', 'address':'green', 'id':'orange',"email":"red", "phone":"yellow","dob":"pink",'other':'violet'}
        predictions = outputs.logits.argmax(-1).tolist()
        boxes = encodings['bbox'].tolist()
        attention_masks = encodings['attention_mask'].tolist()

        for image, prediction, box, attention_mask in zip(images, predictions, boxes, attention_masks):

            width, height = image.size
            draw = ImageDraw.Draw(image)
            
            valid_indices = [i for i,indicator in enumerate(attention_mask) if indicator != 0]
            
            true_predictions = [id2labels.get(prediction[i],"O") for i in valid_indices]
            true_boxes = [unnormalize_boxes(box[i], width, height) for i in valid_indices]

            for individual_prediction, bounding_box in zip(true_predictions, true_boxes):
                nlabel = flabel_2_nlabel(individual_prediction).lower()
                if nlabel != 'other':
                    draw.rectangle(bounding_box, outline="black", fill="black")
                # draw.text((bounding_box[0]+10, bounding_box[1]-10), text = nlabel, font=ImageFont.load_default(), fill=label2color[nlabel])
            # image.show()
           
        return images  #* return modified images

    except Exception as e:
        print("Error:", str(e))
        return images

if __name__ == "__main__":
    pass