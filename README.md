# Finding a segment containing a given image pixel within a given region-of-interest (ROI) mask using a convolutional net (CNN), and its application for sequential region-by-region instance-aware segmentation of images with unfamiliar categories.


This work examines the use of fully convolutional nets (FCN) for finding a binary mask corresponding to a single image segment given a single pixel within this segment region (Figure 1). The target segment region is  also limited by  and given a region of interest (ROI) mask that limits the area were the segment can be found (Figure 1). The net receives an image and a  single arbitrary point inside the target segment. An additional input is an ROI mask that restricts to the region of the image where the segment can be found. It returns a binary mask of the segment on which the pointer point is located (Figure 1).

The method is explained in [this document](https://arxiv.org/pdf/1902.07810.pdf) 

![](/Figure1.png)
Figure 1. Pointer based segmentation

This net can also achieve full image segmentation by running sequentially, one segment at a time on the image and stitching the output segments into a single segmentation map (Figure 2).
The net is class independent. Hence, it can segment even things corresponding to unfamiliar categories it did not encounter in the training stage.
The net is capable of instance aware segmentation of individual objects in a group (Figure 1.b).
In addition, the net can  also segment stuff none object classes such as sky and ocean( Figure 1.a)

![](/Figure2.png)
Figure 2. Sequential region by region Pointer point based segmentation


# Using the net
# Setup
This network was run with [Python 3.7 Anaconda](https://www.anaconda.com/distribution/)  package and [Pytorch 1.0](https://pytorch.org/).
The net was trained using the [COCO panoptic data set](http://cocodataset.org/#download)
Trained model can be download from  [here](https://drive.google.com/file/d/1c2aAH_sf2kynwbkOiN-iXY7esm2wcrvM/view?usp=sharing)

# Run segmentation for full image
## In Run_Segmentation.py:
1) Train the net or download the pre-trained model from [here](https://drive.google.com/file/d/1c2aAH_sf2kynwbkOiN-iXY7esm2wcrvM/view?usp=sharing)
2) Set the path to the pre-trained model in the Trained_model_path parameter
3) Set path for test image in the InputImagePath parameter (or leave as is)
4) Set the path where the output overlay annotation map  in the OutputFile parameter
5) Run the script.

# Train the  net 
## In train.py
1) Download COCO panoptic dataset and train images  from [here](http://cocodataset.org/#download)
2) Set the path to COCO train images folder in the ImageDir parameter
3) Set the path to COCO panoptic annotations folder in the AnnotationDir parameter 4) Set the path to COCO panoptic data .json file in the DataFile parameter
5) Run script.
Trained model weight and data will appear in the path given by the TrainedModelWeightDir parameter

# Evaluate trained model full image segmentation accuracy 
## In Evalauate_FullImageSequentialSegmentation.py for full image sequential segmentation
## In Evalauate_SingleSegmentPrediction.py for single segmentation
1) Download COCO panoptic dataset and eval images  from [here](http://cocodataset.org/#download)
2) Train the net or download the pre-trained model from [here](https://drive.google.com/file/d/1c2aAH_sf2kynwbkOiN-iXY7esm2wcrvM/view?usp=sharing)
3) Set the path to the pre-trained model in the Trained_model_path parameter
4) Set the path to COCO eval images folder in the ImageDir parameter
5) Set the path to COCO panoptic eval annotations folder in the AnnotationDir parameter
6) Set the path to COCO panoptic data .json file in the DataFile parameter
7) Set path to output statistics file in the Statistics_File_Path parameter
8) Run script.


![](/Image3.png)
Figure 3. Net architecture. a) Standart fully convolutional net (FCN) for semantic segmentation. b) FCN with additional input of pointer point. c) FCN with two addtional inputs of ROI mask and Pointer point mask 

# Net architecture
 The net architecture is shown in Figure 3.c   based on standard fully convolutional neural (FCN) for semantic segmentation (Figure 3.a) with  two additional inputs:
Pointer point mask: A binary mask with a  single point within the target segment marked on it as 1 (Figure 3.b). 
ROI Mask: A binary mask that covers (or limits) the region of the image where the segment can be found (Figure 3.c). 

The Pointer mask and ROI mask are each processed using a convolution layer to generate two distinct attentions map in the same shape as the feature map generates from the image after the first convolutional layer (Figure 3c).
The attention map generated from the Pointer mask is merged with the feature map generated from the image using elementwise multiplication to the generate merged feature map 1 (Figure 3) 
Merged map  1 is then merged with the attention map generated from the ROI mask using elementwise addition to generate a second merged map which is used as input for the next layers of the FCN (Figure 3c). 




