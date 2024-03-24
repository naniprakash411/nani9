This repository contains the code for running offline evaluation of Set-Based Text-to-Image Generation.

## Evaluation on a set of generated images
To run the set of proposed evaluation metrics on a set of generated images, first clone this repository and then run  ```eval.py``` as follows:

```
python eval.py \ 
  -image_dir </path/to/folder/including/generated_images< 
  -target_image </path/to/gold/standard/target/image<
  -metric <choice of ['rbp','err']>
  -trajectory <choice of ['saliency','order']>
  -gamma <user persistency parameter default=0.8>
  -n_samples <number of sampled trajectories, default=50>
  -variety <if vairety needs to be considered when measuring relevance scores, choice of [True, False]>
```
### Example 1 - RBP 
```
python eval.py  \
  -image_dir example1 \
  -target_image targets/example1.png  \
  -metric rbp   \
  -gamma 0.8  \
  -n_samples 50 \
  -variety False
```
This script will generate the following grid from [example1](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/tree/main/generated_images) give you the following outputs:

![alt text](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/blob/main/grids/grid_example1.png)

Given [this target image](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/blob/main/targets/example1.png),the script will evaluate RBP as explained in the paper and show the following outputs:

```
grid of images generated and saved as grids/grid_generated_images.png
1/1 [==============================] - 2s 2s/step
1/1 [==============================] - 0s 297ms/step
1/1 [==============================] - 0s 291ms/step
1/1 [==============================] - 0s 308ms/step
1/1 [==============================] - 0s 307ms/step
1/1 [==============================] - 0s 324ms/step
1/1 [==============================] - 0s 300ms/step
1/1 [==============================] - 0s 342ms/step
1/1 [==============================] - 0s 312ms/step
1/1 [==============================] - 0s 307ms/step
1/1 [==============================] - 0s 282ms/step
1/1 [==============================] - 0s 283ms/step
1/1 [==============================] - 0s 307ms/step
1/1 [==============================] - 0s 298ms/step
1/1 [==============================] - 0s 316ms/step
1/1 [==============================] - 0s 341ms/step
1/1 [==============================] - 0s 304ms/step
1/1 [==============================] - 0s 335ms/step
saliency [0.00225529 0.00182395 0.2671824  0.2021625  0.28705123 0.23540027 0.00211734 0.00200697]
The quality of the gird of generated images in example1 directory is evaluated as :
metric rbp
variety True
trajectory saliency
evaluation: 0.6345379112701999
```

### Example 2 - Novelty ERR - Saliency-based Trajectories:
```
python eval.py  \
  -image_dir example2 \
  -target_image targets/example2.png  \
  -metric err   \
  -trajectory saliency   \
  -gamma 0.8  \
  -n_samples 50 \
  -variety True
```
 This script will generate the following grid from [example2](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/tree/main/generated_images) give you the following outputs:

![alt text](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/blob/main/grids/grid_example2.png)

Given [this target image](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/blob/main/targets/example2.png),the script will evaluate ERR based on saliency trajectories as explained in the paper and show the following outputs:

```
grid of images generated and saved as grids/grid_example2.png
1/1 [==============================] - 2s 2s/step
1/1 [==============================] - 0s 291ms/step
1/1 [==============================] - 0s 314ms/step
1/1 [==============================] - 0s 277ms/step
1/1 [==============================] - 0s 296ms/step
1/1 [==============================] - 0s 295ms/step
1/1 [==============================] - 0s 291ms/step
1/1 [==============================] - 0s 290ms/step
1/1 [==============================] - 0s 322ms/step
1/1 [==============================] - 0s 283ms/step
1/1 [==============================] - 0s 287ms/step
1/1 [==============================] - 0s 284ms/step
1/1 [==============================] - 0s 307ms/step
1/1 [==============================] - 0s 309ms/step
1/1 [==============================] - 0s 304ms/step
1/1 [==============================] - 0s 288ms/step
1/1 [==============================] - 0s 302ms/step
saliency [0.00303167 0.00243593 0.25322178 0.18516748 0.29889885 0.25214726 0.002624   0.00247295]
The quality of the gird of generated images in example2 directory is evaluated as :
metric err
variety True
trajectory saliency
evaluation: 0.7194848886004105
```

### Saliency Prediction
We use the [trained visual saliency model on the web pages](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/tree/main/webpage_stonybrook_baseline) in order to predict the saliency of an image or a grid of images.
[```saliency.py```](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/blob/main/saliency.py) provide neccessary functions to preprocess an image and predict the visual saliency.

For example, the following command, will predict the saliency of a single image:

```  
python saliency.py -image_dir example1/i1.png
```
the output will look like this which will be a 2darray with the size of the image:
```
saliency of the imnage is predicted as
[[1.1226661e-07 1.1226661e-07 1.0985102e-07 ... 4.4218339e-08
  4.5917613e-08 4.5917613e-08]
 [1.1226661e-07 1.1226661e-07 1.0985102e-07 ... 4.4218339e-08
  4.5917613e-08 4.5917613e-08]
 [1.1417994e-07 1.1417994e-07 1.1223193e-07 ... 4.2988301e-08
  4.4331880e-08 4.4331880e-08]
 ...
 [3.6094601e-08 3.6094601e-08 3.8221611e-08 ... 2.0949896e-07
  1.9440878e-07 1.9440878e-07]
 [3.5043357e-08 3.5043357e-08 3.7084359e-08 ... 1.9273388e-07
  1.7636771e-07 1.7636771e-07]
 [3.5043357e-08 3.5043357e-08 3.7084359e-08 ... 1.9273388e-07
  1.7636771e-07 1.7636771e-07]]
```
### Relevance
[```inception.py```](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/blob/main/inception.py) provide neccessary function to embed the images using InceptionV3 model and find the relevance score w.r.t a given target image. 

### Metrics
[```metrics.py```](https://github.com/Narabzad/Set-Based-Text-to-ImageGeneration/blob/main/metrics.py) provide necessary functions to measure ERR, RBP and their different variations on a given list of relevance scores from a ranked list/grid. 

