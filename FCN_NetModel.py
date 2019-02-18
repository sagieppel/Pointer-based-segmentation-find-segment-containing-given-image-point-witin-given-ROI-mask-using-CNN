
import scipy.misc as misc
import torch
import copy
import torchvision.models as models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# FCN Net model class for semantic segmentation
######################################################################################################################333
class Net(nn.Module):
########################################################################################################################
    def __init__(self, NumClasses=2):
        # Generate standart FCN net for image segmentation with only image as input (attention layers will be added at next function
        # --------------Build layers for standart FCN with only image as input------------------------------------------------------
            super(Net, self).__init__()
            # ---------------Load pretrained  Resnet 50 encoder----------------------------------------------------------
            self.Encoder = models.resnet50(pretrained=True)
            # ---------------Create Pyramid Scene Parsing PSP layer -------------------------------------------------------------------------
            self.PSPScales = [1, 1 / 2, 1 / 4, 1 / 8]

            self.PSPLayers = nn.ModuleList()  # [] # Layers for decoder
            for Ps in self.PSPScales:
                self.PSPLayers.append(nn.Sequential(
                    nn.Conv2d(2048, 1024, stride=1, kernel_size=3, padding=1, bias=True)))
                # nn.BatchNorm2d(1024)))
            self.PSPSqueeze = nn.Sequential(
                nn.Conv2d(4096, 512, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()
            )
            # ------------------Skip conncetion layers for upsampling-----------------------------------------------------------------------------
            self.SkipConnections = nn.ModuleList()
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(1024, 512, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()))
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(512, 256, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()))
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(256, 128, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU()))
            # ------------------Skip squeeze applied to the (concat of upsample+skip conncection layers)-----------------------------------------------------------------------------
            self.SqueezeUpsample = nn.ModuleList()
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(1024, 512, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()))
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(256 + 512, 256, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()))
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(256 + 128, 128, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU()))


            # ----------------Final prediction layer predict class per region/pixel------------------------------------------------------------------------------------------
            self.FinalPrdiction = nn.Conv2d(128, NumClasses, stride=1, kernel_size=3, padding=1, bias=False)


####################################################################################################################################################
# Add attention layer that convert the pointer point and ROI mask into an attention mask
    def AddAttententionLayer(self):
                self.AttentionLayers = nn.ModuleList()
                self.ROIEncoder = nn.Conv2d(1, 64, stride=1, kernel_size=3, padding=1, bias=True)
                self.ROIEncoder.bias.data = torch.zeros(self.ROIEncoder.bias.data.shape)
                self.ROIEncoder.weight.data = torch.zeros(self.ROIEncoder.weight.data.shape)

                self.PointerEncoder = nn.Conv2d(1, 64, stride=1, kernel_size=3, padding=1, bias=True)
                self.PointerEncoder.bias.data = torch.zeros(self.ROIEncoder.bias.data.shape)
                self.PointerEncoder.weight.data = torch.ones(self.ROIEncoder.weight.data.shape)
##########################################################################################################################################################
    def forward(self,Images,Pointer,ROI,UseGPU=True):

#----------------------Convert image to pytorch and normalize values-----------------------------------------------------------------
                RGBMean = [123.68,116.779,103.939]
                RGBStd = [65,65,65]
                InpImages = torch.autograd.Variable(torch.from_numpy(Images.astype(float)), requires_grad=False).transpose(2,3).transpose(1, 2).type(torch.FloatTensor)

#-------------------Convert ROI mask and pointer point mask into pytorch format----------------------------------------------------------------
                ROImap = torch.autograd.Variable(torch.from_numpy(ROI.astype(np.float)), requires_grad=False).unsqueeze(dim=1).type(torch.FloatTensor)
                Pointermap = torch.autograd.Variable(torch.from_numpy(Pointer.astype(np.float)), requires_grad=False).unsqueeze(dim=1).type(torch.FloatTensor)
#---------------Convert to cuda gpu-------------------------------------------------------------------------------------------------------------------
                if UseGPU:
                    ROImap=ROImap.cuda()
                    Pointermap=Pointermap.cuda()
                    InpImages=InpImages.cuda()
#----------------Normalize image values-----------------------------------------------------------------------------------------------------------
                for i in range(len(RGBMean)): InpImages[:, i, :, :]=(InpImages[:, i, :, :]-RGBMean[i])/RGBStd[i] # normalize image values
                x=InpImages
#--------------------------------------------------------------------------------------------------------------------------
                SkipConFeatures=[] # Store features map of layers used for skip connection
#---------------Run Encoder first layer-----------------------------------------------------------------------------------------------------
                x = self.Encoder.conv1(x)
                x = self.Encoder.bn1(x)
#------------------------Convery ROI mask and pointer map into attention layer and merge with image feature mask-----------------------------------------------------------
                r = self.ROIEncoder(ROImap) # Generate attention map from ROI mask
                pt = self.PointerEncoder(Pointermap) # Generate attention Mask from Pointer point
                sp = (x.shape[2], x.shape[3])
                pt = nn.functional.interpolate(pt, size=sp, mode='bilinear')  #
                r = nn.functional.interpolate(r, size=sp, mode='bilinear')  # Resize
                x = x* pt + r # Merge feature mask and attention maps
#-------------------------Run remaining encoder layer------------------------------------------------------------------------------------------
                x = self.Encoder.relu(x)
                x = self.Encoder.maxpool(x)
                x = self.Encoder.layer1(x)
                SkipConFeatures.append(x)
                x = self.Encoder.layer2(x)
                SkipConFeatures.append(x)
                x = self.Encoder.layer3(x)
                SkipConFeatures.append(x)
                x = self.Encoder.layer4(x)
#------------------Run psp  Layers----------------------------------------------------------------------------------------------
                PSPSize=(x.shape[2],x.shape[3]) # Size of the original features map

                PSPFeatures=[] # Results of various of scaled procceessing
                for i,PSPLayer in enumerate(self.PSPLayers): # run PSP layers scale features map to various of sizes apply convolution and concat the results
                      NewSize=(np.array(PSPSize)*self.PSPScales[i]).astype(np.int)
                      if NewSize[0] < 1: NewSize[0] = 1
                      if NewSize[1] < 1: NewSize[1] = 1

                      # print(str(i)+")"+str(NewSize))
                      y = nn.functional.interpolate(x, tuple(NewSize), mode='bilinear')
                      #print(y.shape)
                      y = PSPLayer(y)
                      y = nn.functional.interpolate(y, PSPSize, mode='bilinear')

                #      if np.min(PSPSize*self.ScaleRates[i])<0.4: y*=0
                      PSPFeatures.append(y)
                x=torch.cat(PSPFeatures,dim=1)
                x=self.PSPSqueeze(x)
#----------------------------Upsample features map  and combine with layers from encoder using skip  connection-----------------------------------------------------------------------------------------------------------
                for i in range(len(self.SkipConnections)):
                  sp=(SkipConFeatures[-1-i].shape[2],SkipConFeatures[-1-i].shape[3])
                  x=nn.functional.interpolate(x,size=sp,mode='bilinear') #Resize
                  x = torch.cat((self.SkipConnections[i](SkipConFeatures[-1-i]),x), dim=1)
                  x = self.SqueezeUpsample[i](x)
#---------------------------------Final prediction-------------------------------------------------------------------------------
                x = self.FinalPrdiction(x) # Make prediction per pixel
                x = nn.functional.interpolate(x,size=InpImages.shape[2:4],mode='bilinear') # Resize to original image size
#********************************************************************************************************
                #x = nn.UpsamplingBilinear2d(size=InpImages.shape[2:4])(x)
                Prob=F.softmax(x,dim=1) # Calculate class probability per pixel
                tt,Labels=x.max(1) # Find label per pixel
                return Prob,Labels







