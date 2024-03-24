from PIL import Image
from saliency import get_saliency
from os import listdir
import numpy 

image_direcotry = "generated_images"
files = [ image_direcotry + "/"+i for i in sorted(listdir("generated_images")) ]

new_im = Image.new('RGB', (360,640),(255, 255, 255))

index = 0
for j in range(0,640,160):
    for i in range(15,340,170):
        im = Image.open(files[index])
        im.thumbnail((180,160))
        new_im.paste(im, (i,j))
        index += 1

new_im.save("grids/" + image_direcotry + ".png")
saliency_prediction= get_saliency("grids/" +  image_direcotry".png")
img_idx=0
image_saliency=[]

for j in range(0,640,160):
    for i in range(0,360,180):
        saliency  = []
        for x in range(i,i+180):
            for y in range(j,j+160):
                saliency.append(saliency_prediction[x][y]) 
        image_saliency.append(numpy.sum(saliency))

print(image_saliency)
 