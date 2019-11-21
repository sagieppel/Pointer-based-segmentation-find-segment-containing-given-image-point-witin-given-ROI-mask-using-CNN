# Evaluate segmentation accuracy for full image sequential segmentation using the COCO panoptic evaluation set
# Evaluate trained model full image segmentatation accuracy
# 1) Download COCO panoptic dataset and eval images  from [here](http://cocodataset.org/#download)
# 2) Set the path to the pre-trained model in the Trained_model_path parameter
# 3) Set the path to COCO eval images folder in the ImageDir parameter
# 4) Set the path to COCO panoptic eval annotations folder in the AnnotationDir parameter
# 5) Set the path to COCO panoptic data .json file in the DataFile parameter
# 6) Set path to output statitics file in the Statistics_File_Path parameter
# 7) Run script.

import torch
import numpy as np
import CocoPanoptic_Reader as Data_Reader
import FCN_NetModel as NET_FCN# The net Class
#.....................................Input parametrs..................................................................................................................
ImageDir="/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/COCO/val2017" # image folder (coco training)  evaluation set
AnnotationDir="/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/COCO/COCO_panoptic/panoptic_val2017/panoptic_val2017" # annotation maps from coco panoptic evaluation set
DataFile="/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/COCO/COCO_panoptic/panoptic_val2017.json" # Json Data file coco panoptic  evaluation set
Trained_model_path="logs/PointerSegmentationNetWeights.torch"# Path of trained model
Statistics_File_Path=Trained_model_path.replace(".torch",".xls") # Name od statistic file
#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
print("Loadin model")
Net=NET_FCN.Net(NumClasses=2) # Create net and load pretrained encoder path
Net.AddAttententionLayer() # Load attention layer
Net=Net.cuda()
Net.load_state_dict(torch.load(Trained_model_path)) # load traine model
Net.eval()
print("Model Loaded")
#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
Reader=Data_Reader.Reader(ImageDir=ImageDir,AnnotationDir=AnnotationDir, DataFile=DataFile,TrainingMode=False)
#---------------------------  Create statitics table----------------------------------------------------------------------------------------------------------
#Statistics per case
IOUCase=np.array([0,0,0],dtype=np.float)# Stuff Things crowds
PrecisionCase=np.array([0,0,0],dtype=np.float)
RecallCase=np.array([0,0,0],dtype=np.float)
SumCase=np.array([0,0,0],dtype=np.float)+0.0000001
#Statistics per pixel
IOUPixel=np.array([0,0,0],dtype=np.float)# Stuff Things crowds
PrecisionPixel=np.array([0,0,0],dtype=np.float)
RecallPixel=np.array([0,0,0],dtype=np.float)
SumPixel=np.array([0,0,0],dtype=np.float)+0.000000001
#-----------------------------------Run net on evaluation set and generate statistic--------------------------------------------------------------------------------------------------------------------
iii=0
for Reader.itr in range(len(Reader.FileList)):
    iii+=1
    print(iii)
    for ii in range(100):
        PointerMask, Images, ROIMask=Reader.LoadNextGivenROI(NewImg=(ii==0))
      #  Reader.DisplayTrainExample(Images[0], ROIMask[0], SegmentMask[0], PointerMask[0])
        with torch.autograd.no_grad():
                 Prob, PredLb = Net.forward(Images=Images, Pointer=PointerMask,ROI=ROIMask)  # Run net inference and get prediction
        PredLb=PredLb.data.cpu().numpy()
        Reader.BROIMask[PredLb == 1] = 0 # Remove predicted segment from the ROI mask

        IOU, Precision, Recall, SegType, SegmentMask=Reader.FindCorrespondingSegmentMaxIOU( PredLb) # Find IOU of segment with the closest segment on the GT mask
#===============display prediction===============================================================================================================================
        # import cv2
        # I=Images[0].copy()
        # I[:,:,0]*=1-PredLb[0]
        # I[:,:,1]*=1-SegmentMask
        # I[:,:,2]*=1-ROIMask[0]
        # #
        # print(SegType)
        # print("IOU="+str(IOU)+" Precision"+str(Precision))
        # misc.imshow(Images[0])
        # misc.imshow(PredLb[0])
        # misc.imshow(SegmentMask)
        # misc.imshow(I)
#===========Stiched the predicted to full segmentation mask========================================================
        if ii==0:
            SegViz=np.zeros(Images[0].shape,dtype=np.uint8)
        else:
            SegViz[:,:, 0] += np.uint8(PredLb[0]*(ii+1)*21%255)
            SegViz[:,:, 1] += np.uint8(PredLb[0]*((ii+1)*67) % 255)
            SegViz[:,:, 2] += np.uint8(PredLb[0]*((ii+1) * 111) % 255)
        # misc.imshow(SegViz)
#======================Add prediction accuracy to statistical tables====================================================================================
        Area=PredLb.sum()/PredLb.shape[1]/PredLb.shape[2]
        if not SegType=="Unlabeled":
            if not SegType == "Unlabeled":
                if SegType == "stuff": ind = 0
                if SegType == "thing": ind = 1
                if SegType == "crowd": ind = 2
              #  print(ind)
                if not (np.isnan(IOU) or  np.isnan(Precision) or  np.isnan( Recall) or  np.isnan(Area) ):
                    IOUCase[ind] += IOU
                    PrecisionCase[ind] += Precision
                    RecallCase[ind] += Recall
                    SumCase[ind] += 1
                    # Statistics per pixel
                    IOUPixel[ind] += IOU*Area
                    PrecisionPixel[ind] += Precision*Area
                    RecallPixel[ind] += Recall*Area
                    SumPixel[ind] += Area
        if (Reader.BROIMask.sum()/Reader.BROIMask.shape[1]/Reader.BROIMask.shape[2])<0.05:break

    print("Average IOU "+str((IOUPixel[0:2]/SumPixel[0:2]).mean()))
    print("Average Precission " + str((PrecisionPixel[0:2] / SumPixel[0:2]).mean()))
    print("Average Recall " + str((RecallPixel[0:2] / SumPixel[0:2]).mean()))

    print("Things IOU " + str((IOUPixel[1] / SumPixel[1]).mean()))
    print("Things Precission " + str((PrecisionPixel[1] / SumPixel[1]).mean()))
    print("Things Recall " + str((RecallPixel[1] / SumPixel[1]).mean()))

    print("Stuff IOU " + str((IOUPixel[0] / SumPixel[0]).mean()))
    print("Stuff Precission " + str((PrecisionPixel[0] / SumPixel[0]).mean()))
    print("Stuff Recall " + str((RecallPixel[0] / SumPixel[0]).mean()))
    # misc.imshow(Images[0])
    # misc.imshow(SegViz)
    # misc.imshow((SegViz + Images[0]) / 2)
# --------------Save and display evaluation Tables------------------------------------------------------------------------------------------------------------------------------------------
f = open(Statistics_File_Path, "w")
print("Pixel")
txt="\tIOU all\tIOU stuff\tIOU things\tIOU Crowds\tPrecision all All\tPrecision stuff\tPrecision things\tPrecision Crowds\tRecall all All\tRecall stuff\tRecall things\tRecall Crowds\r\n"
print(txt)
f.write(txt)
txt="Per pixel:\t"
txt=str((IOUPixel[0:2]/SumPixel[0:2]).mean())+"\t"
txt+=str((IOUPixel[0]/SumPixel[0]).mean())+"\t"
txt+=str((IOUPixel[1]/SumPixel[1]).mean())+"\t"
txt+=str((IOUPixel[2]/SumPixel[2]).mean())+"\t"

txt+=str((PrecisionPixel[0:2]/SumPixel[0:2]).mean())+"\t"
txt+=str((PrecisionPixel[0]/SumPixel[0]).mean())+"\t"
txt+=str((PrecisionPixel[1]/SumPixel[1]).mean())+"\t"
txt+=str((PrecisionPixel[2]/SumPixel[2]).mean())+"\t"

txt+=str((RecallPixel[0:2]/SumPixel[0:2]).mean())+"\t"
txt+=str((RecallPixel[0]/SumPixel[0]).mean())+"\t"
txt+=str((RecallPixel[1]/SumPixel[1]).mean())+"\t"
txt+=str((RecallPixel[2]/SumPixel[2]).mean())+"\t\r\n"
print(txt)
f.write(txt)

txt="Per segment:\t"
txt+=str((IOUCase[0:2]/SumCase[0:2]).mean())+"\t"
txt+=str((IOUCase[0]/SumCase[0]).mean())+"\t"
txt+=str((IOUCase[1]/SumCase[1]).mean())+"\t"
txt+=str((IOUCase[2]/SumCase[2]).mean())+"\t"

txt+=str((PrecisionCase[0:2]/SumCase[0:2]).mean())+"\t"
txt+=str((PrecisionCase[0]/SumCase[0]).mean())+"\t"
txt+=str((PrecisionCase[1]/SumCase[1]).mean())+"\t"
txt+=str((PrecisionCase[2]/SumCase[2]).mean())+"\t"

txt+=str((RecallCase[0:2]/SumCase[0:2]).mean())+"\t"
txt+=str((RecallCase[0]/SumCase[0]).mean())+"\t"
txt+=str((RecallCase[1]/SumCase[1]).mean())+"\t"
txt+=str((RecallCase[2]/SumCase[2]).mean())+"\t"
print(txt)
f.write(txt)
f.close()
