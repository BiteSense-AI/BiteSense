import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PIL import ImageFile, Image
from numpy import expand_dims
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
ImageFile.LOAD_TRUNCATED_IMAGES = True

model = load_model('final.h5')

classes=["Ant","Bedbug","Chigger","Flea","Mosquito","No_Bite","Spider","Tick"]

def getPrediction(img_bytes, model):
    original_image = Image.open(img_bytes)
    original_image = original_image.convert('RGB')
    original_image = original_image.resize((224, 224), Image.NEAREST)
    numpy_image = image.img_to_array(original_image)
    image_batch = expand_dims(numpy_image, axis=0)
    processed_image = preprocess_input(image_batch, mode='caffe')
    preds = model.predict(processed_image)
    return preds

def classifyImage(file):
    preds = getPrediction(file, model)
    result = preds.tolist()[0]
    mIndex = 0
    for i in range(8):
        if result[i]>result[mIndex]:
            mIndex = i
    return classes[mIndex]