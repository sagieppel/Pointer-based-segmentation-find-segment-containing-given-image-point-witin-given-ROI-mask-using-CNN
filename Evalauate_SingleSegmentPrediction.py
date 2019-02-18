# Evaluate trained model for single segment segmentation
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
#import scipy.misc as misc
import FCN_NetModel as NET_FCN # The net Class

#......................................................................Input parametrs..................................................................................................
ImageDir="/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/COCO/val2017" # image folder (coco training)  evaluation set
AnnotationDir="/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/COCO/COCO_panoptic/panoptic_val2017/panoptic_val2017" # annotation maps from coco panoptic evaluation set
DataFile="/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/COCO/COCO_panoptic/panoptic_val2017.json" # Json Data file coco panoptic  evaluation set
Trained_model_path="logs/PointerSegmentationNetWeights.torch"# Path of trained model
Statistics_File_Path=Trained_model_path.replace(".torch",".xls") # Name od statistic file
#--------------------------------------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
print("Loadin model")
Net=NET_FCN.Net(NumClasses=2) # Create net and load pretrained encoder path
Net.AddAttententionLayer() # Load attention layer
Net=Net.cuda()
Net.load_state_dict(torch.load(Trained_model_path)) # load traine model
Net.eval()
print("Model Loaded")
#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
Reader=Data_Reader.Reader(ImageDir=ImageDir,AnnotationDir=AnnotationDir, DataFile=DataFile,TrainingMode=False)
#--------------------------- Create statistics table----------------------------------------------------------------------------------------------------------
Sizes=[1000,2000,4000,8000,16000,32000,64000,128000,256000,500000,1000000] #Size range for bins
NumSizes=len(Sizes)

SumIOU=np.zeros([NumSizes,11,2])
SumPrecision=np.zeros([NumSizes,11,2])
SumRecall=np.zeros([NumSizes,11,2])
SumMeasurment=np.zeros([NumSizes,11,2])

#--------------------------------Go over the evaluation data  generate IOU statitics------------------------------
iii=0
while(Reader.Epoch<10):
    iii+=1
    print("Epoch "+str(Reader.Epoch)+") Step:  "+str(Reader.itr))
   # if iii==500: break
    Imgs, SegmentMask, ROIMask, PointerMap,IsThing = Reader.LoadSingleClean()
    b,h,w=ROIMask.shape
#=============================================================================================
    LbSize = SegmentMask[0].sum()
    SzInd = -1
    for f, sz in enumerate(Sizes):  # Find size range of the ROI region
        if LbSize < sz:
            SzInd = f
            break
#================Check what fraction of the image covered by the ROI=============================================================================
    ROIfrac = ROIMask[0].sum()/h/w

    if ROIMask[0].min()==1: ROIind=10
    else:
        ROIind =int(np.floor(10*ROIMask[0].sum() / h / w))

#=============================================================================================
    # print(Imgs.shape)
   # Reader.DisplayTrainExample(Imgs[0], ROIMask[0], SegmentMask[0], PointerMap[0])
    Prob, Lb=Net.forward(Images=Imgs,Pointer=PointerMap,ROI=ROIMask) # Run net inference and get prediction
    Pred=Lb[0].data.cpu().numpy()
#.........Generate statitics...........................................................................
    IOU=(SegmentMask[0]*Pred).sum()/(SegmentMask[0].sum()+Pred.sum()-(SegmentMask[0]*Pred).sum())
    Precision = (SegmentMask[0] * Pred).sum() / (Pred.sum())
    Recall = (SegmentMask[0] * Pred).sum() / (SegmentMask[0].sum())

    SumIOU[SzInd, ROIind, int(IsThing)] += IOU
    SumPrecision[SzInd, ROIind, int(IsThing)] += Precision
    SumRecall[SzInd, ROIind, int(IsThing)] += Recall
    SumMeasurment[SzInd, ROIind, int(IsThing)]+=1
    print("IOU="+str(IOU))
    # I=Imgs[0]
    # I[:,:,0]*=1-Pred
    # I[:,:,1]*=1-SegmentMask[0]
    # I[:,:,2]*=1-ROIMask[0]
    # misc.imshow(I)

# --------------Save  and display statistic Tables------------------------------------------------------------------------------------------------------------------------------------------
f = open(Statistics_File_Path, "w")
txt="\r\n\r\nIOU Per Segment Size\r\n"
print(txt)
f.write(txt)
txt="Segment Size<\tIOU all\tIOU things\tIOU stuff\tPrecision all\tPrecision things\tPrecision stuff\tRecall all\tRecall things\tRecall stuff\tNum All\tNum things\tNum Stuff\r\n"
print(txt)
f.write(txt)
for i in range(SumIOU.shape[0]):
        txt = str(Sizes[i]) + "\t" + \
              str(SumIOU[i].sum() / SumMeasurment[i].sum()) + "\t" + \
              str(SumIOU[i, :, 1].sum() / SumMeasurment[i, :, 1].sum()) + "\t" +\
              str(SumIOU[i, :, 0].sum() / SumMeasurment[i, :, 0].sum()) + "\t" + \
              str(SumPrecision[i].sum() / SumMeasurment[i].sum()) + "\t" + \
              str(SumPrecision[i, :, 1].sum() / SumMeasurment[i, :, 1].sum()) + "\t" + \
              str(SumPrecision[i, :, 0].sum() / SumMeasurment[i, :, 0].sum()) + "\t" + \
              str(SumRecall[i].sum() / SumMeasurment[i].sum()) + "\t" + \
              str(SumRecall[i, :, 1].sum() / SumMeasurment[i, :, 1].sum()) + "\t" + \
              str(SumRecall[i, :, 0].sum() / SumMeasurment[i, :, 0].sum()) + "\t" + \
              str(SumMeasurment[i].sum()) + "\t" + \
              str(SumMeasurment[i, :, 1].sum()) + "\t" + \
              str(SumMeasurment[i, :, 0].sum()) + "\r\n"
        print(txt)
        f.write(txt)
#....................................................................................................
txt="\r\n\r\n\r\nIOU Per ROI fraction\r\n"
print(txt)
f.write(txt)
txt="ROI Fraction<\tIOU all\tIOU things\tIOU stuff\tPrecision all\tPrecision things\tPrecision stuff\tRecall all\tRecall things\tRecall stuff\tNum All\tNum things\tNum Stuff\r\n"
print(txt)
f.write(txt)
for i in range(SumIOU.shape[1]):
        txt = str(i*10) + "%\t" + \
              str(SumIOU[:,i,:].sum() / SumMeasurment[:,i,:].sum()) + "\t" + \
              str(SumIOU[:, i, 1].sum() / SumMeasurment[:, i, 1].sum()) + "\t" +\
              str(SumIOU[:, i, 0].sum() / SumMeasurment[:, i, 0].sum()) + "\t" + \
              str(SumPrecision[:, i, :].sum() / SumMeasurment[:, i, :].sum()) + "\t" + \
              str(SumPrecision[:, i, 1].sum() / SumMeasurment[:, i, 1].sum()) + "\t" + \
              str(SumPrecision[:, i, 0].sum() / SumMeasurment[:, i, 0].sum()) + "\t" + \
              str(SumRecall[:, i, :].sum() / SumMeasurment[:, i, :].sum()) + "\t" + \
              str(SumRecall[:, i, 1].sum() / SumMeasurment[:, i, 1].sum()) + "\t" + \
              str(SumRecall[:, i, 0].sum() / SumMeasurment[:, i, 0].sum()) + "\t" + \
              str(SumMeasurment[:,i,:].sum()) + "\t" + \
              str(SumMeasurment[:, i, 1].sum()) + "\t" + \
              str(SumMeasurment[:, i, 0].sum()) + "\r\n"
        print(txt)
        f.write(txt)
#...........................................................................................................
txt="\r\n\r\n\r\nGeneral IOU\r\n ROI/Size\t"
for i in range(len(Sizes)): txt+=str(Sizes[i]) + "\t"
txt+="\r\n"
print(txt)
f.write(txt)
for ri in range(SumIOU.shape[1]):
    txt=str(ri*10)+"%\t"
    for si in range(SumIOU.shape[0]):
        txt+=str(SumIOU[si, ri, :].sum() / SumMeasurment[si, ri, :].sum()) + "\t"
    txt+= "\r\n"
    print(txt)
    f.write(txt)
#...........................................................................................................
txt="\r\n\r\n\r\nThings IOU\r\n ROI/Size\t"
for i in range(len(Sizes)): txt+=str(Sizes[i]) + "\t"
txt+="\r\n"
print(txt)
f.write(txt)
for ri in range(SumIOU.shape[1]):
    txt=str(ri*10)+"%\t"
    for si in range(SumIOU.shape[0]):
        txt+=str(SumIOU[si, ri, 1].sum() / SumMeasurment[si, ri, 1].sum()) + "\t"
    txt+= "\r\n"
    print(txt)
    f.write(txt)
#...........................................................................................................
txt="\r\n\r\n\r\nStuff IOU\r\n ROI/Size\t"
for i in range(len(Sizes)): txt+=str(Sizes[i]) + "\t"
txt+="\r\n"
print(txt)
f.write(txt)
for ri in range(SumIOU.shape[1]):
    txt=str(ri*10)+"%\t"
    for si in range(SumIOU.shape[0]):
        txt+str(SumIOU[si, ri, 0].sum() / SumMeasurment[si, ri, 0].sum()) + "\t"
    txt+ "\r\n"
    print(txt)
    f.write(txt)
#...........................................................................................................
txt="\r\n\r\n\r\nGeneral Sum\r\n ROI/Size\t"
for i in range(len(Sizes)): txt+=str(Sizes[i]) + "\t"
txt+="\r\n"
print(txt)
f.write(txt)
for ri in range(SumIOU.shape[1]):
    txt=str(ri*10)+"%\t"
    for si in range(SumIOU.shape[0]):
        txt+=str(SumMeasurment[si, ri, :].sum()) + "\t"
    txt+= "\r\n"
    print(txt)
    f.write(txt)
#...........................................................................................................
txt="\r\n\r\n\r\nThings Sum\r\n ROI/Size\t"
for i in range(len(Sizes)): txt+=str(Sizes[i]) + "\t"
txt+="\r\n"
print(txt)
f.write(txt)
for ri in range(SumIOU.shape[1]):
    txt=str(ri*10)+"%\t"
    for si in range(SumIOU.shape[0]):
        txt+=str( SumMeasurment[si, ri, 1].sum()) + "\t"
    txt+= "\r\n"
    print(txt)
    f.write(txt)
#...........................................................................................................
txt="\r\n\r\n\r\nStuff Sum\r\n ROI/Size\t"
for i in range(len(Sizes)): txt+=str(Sizes[i]) + "\t"
txt+="\r\n"
print(txt)
f.write(txt)
for ri in range(SumIOU.shape[1]):
    txt=str(ri*10)+"%\t"
    for si in range(SumIOU.shape[0]):
        txt+=str(SumMeasurment[si, ri, 0].sum()) + "\t"
    txt+= "\r\n"
    print(txt)
    f.write(txt)
f.close()