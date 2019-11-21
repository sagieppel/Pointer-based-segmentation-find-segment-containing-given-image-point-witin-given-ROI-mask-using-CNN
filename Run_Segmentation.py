# Run segmentation for full image  (In Run_Segmentation.py)
# 1) Train the net or download pre-trained model from:
# 2) Set the path to the pre-trained model in the Trained_model_path parameter
# 3) Set path for test image in the InputImagePath parameter (or leave as is)
# 4) Set path where the output overlay annotation map  in the OutputFile parameter
# 5) Run script



import torch
import numpy as np
import FCN_NetModel as NET_FCN# The net Class
from PIL import Image
#.....................................Input parametrs..................................................................................................................
InputImagePath="TestImages/Image3.jpg"
Trained_model_path="logs/TrainedModePointerImageSegmentation1.8m.torch"# Path of trained model
OutputFile="TestImages/Label.png"
#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
print("Loadin model")
Net=NET_FCN.Net(NumClasses=2) # Create net and load pretrained encoder path
Net.AddAttententionLayer() # Load attention layer
Net=Net.cuda()
Net.load_state_dict(torch.load(Trained_model_path)) # load traine model
Net.eval()
print("Model Loaded")
#----------------------------------------Read image and generate initial ROI mask--------------------------------------------------------------------------------------------------------------
Im=Image.open(InputImagePath)
Im.show()
I=np.array(Im)[:,:,0:3]
H,W,d=I.shape
I=np.expand_dims(I,0)
ROIMask=np.ones(I.shape[:3]) # Generate ROI mask that cover the full image
#-------------------------------Start sequential segmentation--------------------------------------------------------------
for ii in range(100):

#--------------pick random point in ROI mask----------------------------------------------------
        print("picking next point point")
        while (True):
            X = np.random.randint(W)
            Y = np.random.randint(H)
            if ROIMask[0,Y,X]==1: break
#==============Generate Pointer mask==================================================================================
        PointerMask=np.zeros(ROIMask.shape)
        PointerMask[0, Y, X] = 1

#==============run inference predict segment==================================================================================
        with torch.autograd.no_grad():
               Prob, PredLb = Net.forward(Images=I, Pointer=PointerMask,ROI=ROIMask)  # Run net inference and get prediction
        PredLb=PredLb.data.cpu().numpy()
        ROIMask[PredLb == 1] = 0 # Remove predicted segment from the ROI mask

#===========Stiched the predicted segment to full segmentation mask========================================================
        if ii==0:
            SegViz=np.zeros(I[0].shape,dtype=np.uint8)
        SegViz[:,:, 0] += np.uint8(PredLb[0]*(ii+1)*21%255)
        SegViz[:,:, 1] += np.uint8(PredLb[0]*((ii+1)*67) % 255)
        SegViz[:,:, 2] += np.uint8(PredLb[0]*((ii+1) * 111) % 255)
#============================break when 95% of the image have been segmented==================================================================================
        if (ROIMask.sum() / W / H) < 0.05: break
        print(str(ii)+") ROI as fraction of image"+str(ROIMask.sum() / W / H))
#=============================Display and save reslut====================================================================================
OverLay=(I[0]*0.2+SegViz*0.8).astype(np.uint8)
disp1 = Image.fromarray(OverLay, 'RGB')
disp1.show()
disp1.save(OutputFile)


