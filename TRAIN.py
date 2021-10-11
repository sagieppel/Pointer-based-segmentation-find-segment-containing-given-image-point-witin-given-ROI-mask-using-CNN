# Train the  net (in train.py)
# 1) Download COCO panoptic dataset and train images  from [here](http://cocodataset.org/#download)
# 2) Set the path to COCO train images folder in the ImageDir parameter
# 3) Set the path to COCO panoptic train annotations folder in the AnnotationDir parameter
# 4) Set the path to COCO panoptic data .json file in the DataFile parameter
# 5) Run script.
# Trained model weight and data will appear in the path given by the TrainedModelWeightDir parameter


#...............................Imports..................................................................
import ConvertLabelToOneHotEncoding
import os
import torch
import numpy as np
import CocoPanoptic_Reader as Data_Reader
import FCN_NetModel as NET_FCN # The net Class
############################################################################################################################333
#...............................Fractal learning rate update................................................
def UpdateFractaleLearninRate(LearningRate,ind=1):
    LearningRate[ind]*=0.7
    if LearningRate[ind] < 5e-8:UpdateFractaleLearninRate(LearningRate, ind+1)
    LearningRate[ind-1]=LearningRate[ind]
##################################Input paramaters#########################################################################################
#.................................Main Input parametrs...........................................................................................
#os.environ["CUDA_VISIBLE_DEVICES"]="0" # Select GPU
ImageDir="/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/COCO/train2017" # image folder (coco training) train set
AnnotationDir="/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/COCO/COCO_panoptic/panoptic_train2017/panoptic_train2017" # annotation maps from coco panoptic train set
DataFile="/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/COCO/COCO_panoptic/panoptic_train2017.json" # Json Data file coco panoptic train set
TrainedModelWeightDir="logs/" # Folder where trained model weight and information will be stored"
#...............Other training paramters..............................................................................

MaxBatchSize=7 # Max images in batch
MinSize=250 # Min image Height/Width
MaxSize=900# Max image Height/Width
Trained_model_path="" # Path of trained model weights If you want to return to trained model, else should be =""

Learning_Rate=np.ones(100)*2e-5 # Initial learning rate
#Learning_Rate_Decay=Learning_Rate[0]/40 # Used for standart
Learning_Rate_Decay=Learning_Rate[0]/20
MaxPixels=340000*4# Max pixel in batch can have (to keep oom out of memory problems) if the image larger it will be resized.
TrainLossTxtFile=TrainedModelWeightDir+"TrainLoss.txt" #Where train losses will be writen
Weight_Decay=1e-5# Weight for the weight decay loss function
MAX_ITERATION = int(100000010) # Max  number of training iteration
#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
Reader=Data_Reader.Reader(ImageDir=ImageDir,AnnotationDir=AnnotationDir, DataFile=DataFile, MaxBatchSize=MaxBatchSize,MinSize=MinSize,MaxSize=MaxSize,MaxPixels=MaxPixels,TrainingMode=True)
#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
Net=NET_FCN.Net(NumClasses=2) # Create net and load pretrained encoder path
Net.AddAttententionLayer() # Create attention later
Net=Net.cuda()
if Trained_model_path!="": # Optional initiate full net by loading a pretrained net
    Net.load_state_dict(torch.load(Trained_model_path))
#optimizer=torch.optim.SGD(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay,momentum=0.5)
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate[0],weight_decay=Weight_Decay) # Create adam optimizer
#--------------------------- Create logs files for saving loss during training----------------------------------------------------------------------------------------------------------
if not os.path.exists(TrainedModelWeightDir): os.makedirs(TrainedModelWeightDir) # Create folder for trained weight
f = open(TrainLossTxtFile, "w+")# Training loss log file
f.write("Iteration\tloss\t Learning Rate=")
f.close()
#..............Start Training loop: Main Training....................................................................
AVGLoss=-1# running average loss
print("Start Training")
for itr in range(1,MAX_ITERATION): # Main training loop
    Imgs, SegmentMask, ROIMask, PointerMap = Reader.LoadBatch()
    #print(ROIMask.shape)
    # for i in range(1):  # Imgs.shape[0]):
    #   Reader.DisplayTrainExample(Imgs[i], ROIMask[i], SegmentMask[i], PointerMap[i])

    OneHotLabels = ConvertLabelToOneHotEncoding.LabelConvert(SegmentMask, 2) #Convert labels map to one hot encoding pytorch
    #print("RUN PREDICITION")
    Prob, Lb=Net.forward(Images=Imgs,Pointer=PointerMap,ROI=ROIMask) # Run net inference and get prediction
    Net.zero_grad()
    Loss = -torch.mean((OneHotLabels * torch.log(Prob + 0.0000001)))  # Calculate loss between prediction and ground truth label
    Loss.backward() # Backpropogate loss
    optimizer.step() # Apply gradient descent change to weight
    if AVGLoss==-1:  AVGLoss=float(Loss.data.cpu().numpy()) #Calculate average loss for display
    else: AVGLoss=AVGLoss*0.999+0.001*float(Loss.data.cpu().numpy()) # Intiate runing average loss
# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
    if itr % 10000 == 0 and itr>0: #Save model weight once every 10k steps
        print("Saving Model to file in "+TrainedModelWeightDir)
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/" + str(itr) + ".torch")
        print("model saved")
#......................Write and display train loss..........................................................................
    if itr % 10==0: # Display train loss
        torch.cuda.empty_cache()  #Empty cuda memory to avoid memory leaks
        print("Step "+str(itr)+" Runnig Average Loss="+str(AVGLoss)+" Learning Rate="+str(Learning_Rate[0])+" "+str(Learning_Rate[1]))
        #Write train loss to file
        with open(TrainLossTxtFile, "a") as f:
            f.write("\n"+str(itr)+"\t"+str(float(Loss.data))+"\t"+str(AVGLoss)+"\t"+str(Learning_Rate[0])+" "+str(Learning_Rate[1]))
            f.close()
#----------------Update learning rate fractal manner-------------------------------------------------------------------------------
    if itr%5000==0:
        Learning_Rate[0]-= Learning_Rate_Decay
        if Learning_Rate[0]<=4e-7:
            UpdateFractaleLearninRate(Learning_Rate)
            Learning_Rate_Decay=Learning_Rate[0]/30
        print(Learning_Rate[0:5])
        print("======================================================================================================================")
        optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate[0],weight_decay=Weight_Decay)  # Create adam optimizer
