import tensorflow.compat.v2 as tf
import numpy as np
from numpy import asarray
from PIL import Image
from os import listdir
import numpy 
import argparse


model_dir = "webpage_stonybrook_baseline"
model = tf.saved_model.load(model_dir)
SALICON_IM_MEAN = [0.485, 0.456, 0.406] 
SALICON_IM_STD = [0.229, 0.224, 0.225]
MODEL_INPUT_IMAGE_SIZE = [360, 640]
normalization = tf.keras.layers.Normalization( axis=-1,  mean=SALICON_IM_MEAN, variance=[cur**2 for cur in SALICON_IM_STD])


def preprocess_image(image, model_img_size):
    model_height, model_width = model_img_size 
    image = tf.image.convert_image_dtype(image, tf.float32) 
    image = tf.image.resize_with_pad(image, model_height, model_width) 
    image = normalization(image) 
    return image

def get_saliency(image_dir):
    image = Image.open(image_dir)
    image = asarray(image)
    if np.issubdtype(image.dtype, np.integer):
        input_image = image.astype(np.float32) / 255.0
    processed_image = preprocess_image(input_image, MODEL_INPUT_IMAGE_SIZE)
    processed_image=tf.expand_dims(processed_image, axis=0)
    pred = model(processed_image).numpy()
    pred = numpy.reshape(pred,(360,640))
    pred= pred / pred.sum()
    return pred


def get_saliency_per_grid(image_directory):
    files = [ image_directory + "/"+i for i in sorted(listdir(image_directory)) ]
    new_im = Image.new('RGB', (360,640),(255, 255, 255))
    index = 0
    for j in range(0,640,160):
        for i in range(15,340,170):
            im = Image.open(files[index])
            im.thumbnail((180,160))
            new_im.paste(im, (i,j))
            index += 1
    print('grid of images generated and saved as ' + "grids/grid_" + image_directory + ".png")
    new_im.save("grids/grid_" + image_directory + ".png")
    saliency_prediction= get_saliency("grids/grid_" +  image_directory + ".png")
    image_saliency=[]

    for j in range(0,640,160):
        for i in range(0,360,180):
            saliency  = []
            for x in range(i,i+180):
                for y in range(j,j+160):
                    saliency.append(saliency_prediction[x][y]) 
            image_saliency.append(numpy.sum(saliency))
    return(image_saliency)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-image_dir', type=str ,default='generated_images/i1.png') 
    args = parser.parse_args()
    print('saliency of the imnage is predicted as' , get_saliency(args.image_dir))
