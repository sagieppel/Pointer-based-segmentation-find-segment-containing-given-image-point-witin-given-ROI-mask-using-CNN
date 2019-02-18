
import scipy.misc as misc
import torch
import copy
import torchvision.models as models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import densenet_cosine_264_k32
class Net(nn.Module):# FCN Net class for semantic segmentation init generate net layers and forward run the inference
    def __init__(self, NumClasses=2):  # Load pretrained encoder and prepare net layers
            super(Net, self).__init__()
            # ---------------Load pretrained encoder----------------------------------------------------------
            self.Encoder = models.resnet101(pretrained=True)
            # ----------------Fully convolutional final encoder layers -------------------------------------------------------------------------
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
            # ------------------Skip conncetion pass layers from the encoder to layer from the decoder/upsampler after convolution-----------------------------------------------------------------------------
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
            # ------------------Skip squeeze concat of upsample+skip conncecion-----------------------------------------------------------------------------
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
    def AddAttententionLayer(self):
                self.AttentionLayers = nn.ModuleList()
                self.ROIEncoder = nn.Conv2d(1, 64, stride=1, kernel_size=3, padding=1, bias=True)
                self.ROIEncoder.bias.data = torch.zeros(self.ROIEncoder.bias.data.shape)
                self.ROIEncoder.weight.data = torch.zeros(self.ROIEncoder.weight.data.shape)

                self.PointerEncoder = nn.Conv2d(1, 64, stride=1, kernel_size=3, padding=1, bias=True)
                self.PointerEncoder.bias.data = torch.zeros(self.ROIEncoder.bias.data.shape)
                self.PointerEncoder.weight.data = torch.ones(self.ROIEncoder.weight.data.shape)
                # self.AttentionLayers.append(self.ROIEncoder)
                # self.AttentionLayers.append(self.PointerEncoder)
##########################################################################################################################################################
    def forward(self,Images,Pointer,ROI,UseGPU=True):

#----------------------Convert image to pytorch and normalize values-----------------------------------------------------------------
                RGBMean = [123.68,116.779,103.939]
                RGBStd = [65,65,65]
                InpImages = torch.autograd.Variable(torch.from_numpy(Images.astype(float)), requires_grad=False).transpose(2,3).transpose(1, 2).type(torch.FloatTensor)


                ROImap = torch.autograd.Variable(torch.from_numpy(ROI.astype(np.float)), requires_grad=False).unsqueeze(dim=1).type(torch.FloatTensor)
                Pointermap = torch.autograd.Variable(torch.from_numpy(Pointer.astype(np.float)), requires_grad=False).unsqueeze(dim=1).type(torch.FloatTensor)
                if UseGPU:
                    ROImap=ROImap.cuda()
                    Pointermap=Pointermap.cuda()
                    InpImages=InpImages.cuda()

                for i in range(len(RGBMean)): InpImages[:, i, :, :]=(InpImages[:, i, :, :]-RGBMean[i])/RGBStd[i] # normalize image values
                x=InpImages
#--------------------Run Encoder------------------------------------------------------------------------------------------------------
#--------------------Run Encoder------------------------------------------------------------------------------------------------------
                SkipConFeatures=[] # Store features map of layers used for skip connection
#--------------------------------------------------------------------------------------------------------------------
                x = self.Encoder.conv1(x)
                x = self.Encoder.bn1(x)
#------------------------Add ROI map and pointer map-----------------------------------------------------------
                r = self.ROIEncoder(ROImap)
                # print("Pointer")
                # print(np.array(self.PointerEncoder.weight.data).mean())
                # print("ROI")
                # print(np.array(self.ROIEncoder.weight.data).mean())
                # print("Conv")
                # print(np.array(self.Encoder.conv1.weight.data).mean())
                pt = self.PointerEncoder(Pointermap)
                sp = (x.shape[2], x.shape[3])
                pt = nn.functional.interpolate(pt, size=sp, mode='bilinear')  #
                r = nn.functional.interpolate(r, size=sp, mode='bilinear')  # Resize
                x = x + r + pt
#-------------------------------------------------------------------------------------------------------------------



                x = self.Encoder.relu(x)
                x = self.Encoder.maxpool(x)
                x = self.Encoder.layer1(x)
                SkipConFeatures.append(x)
                x = self.Encoder.layer2(x)
                SkipConFeatures.append(x)
                x = self.Encoder.layer3(x)
                SkipConFeatures.append(x)
                x = self.Encoder.layer4(x)
#------------------Run psp  decoder Layers----------------------------------------------------------------------------------------------
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
                  #x=selnn.functional.interpolateLayers[i](x) # Apply transpose convolution
                  # print("Skip")
                  # print(sp)
                  # print("Layer")
                  # print(x.shape)
                  x=nn.functional.interpolate(x,size=sp,mode='bilinear') #Resize
                 # print(x.shape)
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



#
#                 SkipConFeatures=[] # Store features map of layers used for skip connection
#                 for i in range(147): # run all layers of Encoder
#                     x=self.Encoder[i](x)
#                     if i==3:
#                         r=self.ROIEncoder(ROImap)
#                         pt=self.PointerEncoder(Pointermap)
#                         sp = (x.shape[2], x.shape[3])
#                         pt = nn.functional.interpolate(pt, size=sp, mode='bilinear')  #
#                         r = nn.functional.interpolate(r, size=sp, mode='bilinear')  # Resize
#                         x=x+r+pt
#                     if i in self.SkipConnectionLayers: # save output of specific layers used for skip conncections
#                          SkipConFeatures.append(x)
#                          #print("skip")
# #------------------Run psp  decoder Layer resize the feature to various of sizes and apply convolution----------------------------------------------------------------------------------------------
#                 PSPSize=(x.shape[2],x.shape[3]) # Size of the original features map
#
#                 PSPFeatures=[] # Results of various of scaled procceessing
#                 for i,Layer in enumerate(self.PSPLayers): # run PSP layers scale features map to various of sizes apply convolution and concat the results
#                       NewSize=np.ceil(np.array(PSPSize)*self.PSPScales[i]).astype(np.int)
#                       y = nn.functional.interpolate(x, tuple(NewSize), mode='bilinear')
#                       #y = nn.functional.interpolate(x, torch.from_numpy(NewSize), mode='bilinear')
#                       y = Layer(y)
#                       y = nn.functional.interpolate(y, PSPSize, mode='bilinear')
#                 #      if np.min(PSPSize*self.ScaleRates[i])<0.4: y*=0
#                       PSPFeatures.append(y)
#                 x=torch.cat(PSPFeatures,dim=1)
#                 x=self.PSPSqueeze(x)
# #----------------------------Upsample features map  and combine with layers from encoder using skip  connection-----------------------------------------------------------------------------------------------------------
#                 for i in range(len(self.SkipConnections)):
#                   sp=(SkipConFeatures[-1-i].shape[2],SkipConFeatures[-1-i].shape[3])
#                   x=nn.functional.interpolate(x,size=sp,mode='bilinear') #Resize
#                   x = torch.cat((self.SkipConnections[i](SkipConFeatures[-1-i]),x), dim=1)
#                   x = self.SqueezeUpsample[i](x)
# #---------------------------------Final prediction-------------------------------------------------------------------------------
#                 x = self.FinalPrdiction(x) # Make prediction per pixel
#                 x = nn.functional.interpolate(x,size=InpImages.shape[2:4],mode='bilinear') # Resize to original image size
#                 Prob=F.softmax(x,dim=1) # Calculate class probability per pixel
#                 tt,Labels=x.max(1) # Find label per pixel
#                 return Prob,Labels
# ###################################################################################################################################
#
# # nt=Net(12).cuda()
# # #torch.save(nt,"tt.torch")
# # #nt.save_state_dict("aa.pth")
# # inp=np.ones((1,3,1000,1000)).astype(np.float32)
# # inp=torch.autograd.Variable(torch.from_numpy(inp).cuda(),requires_grad=False)
# # x=nt.forward(inp)






