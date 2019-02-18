#
#  Read image segment region and classes from the COCO data set  (need the coco API to run)

# Getting COCO dataset and API
# Download and extract the [COCO 2014 train images and Train/Val annotations](http://cocodataset.org/#download)
# Download and make the COCO python API base on the instructions in (https://github.com/cocodataset/cocoapi).
# Copy the pycocotools from cocodataset/cocoapi to the code folder (replace the existing pycocotools folder in the code).
# Note that the code folder already contain pycocotools folder with a compiled API that may or may not work as is.
#
#
import numpy as np
import os
import scipy.misc as misc
import random
import cv2
import json
import threading
#------------------------Class for reading training and  validation data---------------------------------------------------------------------
def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.uint32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return color[0] + 256 * color[1] + 256 * 256 * color[2]
#-----------------------------------------------
class Reader:
################################Initiate folders were files are and list of train images############################################################################
    def __init__(self, ImageDir="/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/COCO/train2017",AnnotationDir="/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/COCO/COCO_panoptic/panoptic_train2017/panoptic_train2017", DataFile="/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/COCO/COCO_panoptic/panoptic_train2017.json",MaxBatchSize=100,MinSize=250,MaxSize=800,MaxPixels=800*800*5, AnnotationFileType="png", ImageFileType="jpg",UnlabeledTag=0,Suffle=True,MultiThread=True):
        self.ImageDir=ImageDir # Image dir
        self.AnnotationDir=AnnotationDir # File containing image annotation
        self.MaxBatchSize=MaxBatchSize # Max number of image in batch
        self.MinSize=MinSize # Min image width and hight
        self.MaxSize=MaxSize #MAx image width and hight
        self.MaxPixels=MaxPixels # Max number of pixel in all the batch (reduce to solve out of memory issues)
        self.AnnotationFileType=AnnotationFileType # What is the the type (ending) of the annotation files
        self.ImageFileType=ImageFileType # What is the the type (ending) of the image files
        self.DataFile=DataFile # Json File that contain data on the annotation
        self.UnlabeledTag=UnlabeledTag # Value of unlabled region usuall 0
        self.ReadStuff = True # Read things that are not instace object (like sky or grass)
        self.SplitThings = False#True # Split instance of things (object) to connected component region and use each connected region as instance
        self.SplitStuff = True # Split instance of things (object) to connected component region and use each connected region as instance
        self.SplitCrowd = True # Split areas marked as Crowds using connected componennt
        self.IgnoreCrowds = True # Ignore areas marked as crowd
        self.PickBySize = True  # Pick instances of with probablity proportional to their sizes
        self.StuffAreaFactor=0.225 # Since we pick segments according to their size stuf segments (wall ground sky) will have higher  probability to be chose compare to things  this factor balance this
        self.MinSegSize=100
        self.Epoch = 0 # Training Epoch
        self.itr = 0
        self.suffle=Suffle # Suffle list of file
        # self.SumThings = 0
        # self.SumStuff = 0

#........................Read data file................................................................................................................
        with open(DataFile) as json_file:
            self.AnnData=json.load(json_file)

#-------------------Get All files in folder--------------------------------------------------------------------------------------
        self.FileList=[]
        for FileName in os.listdir(AnnotationDir):
            if AnnotationFileType in FileName:
                self.FileList.append(FileName)
        if self.suffle:
            random.shuffle(self.FileList)
        if MultiThread: self.StartLoadBatch()
##############################################################################################################################################
##############################################################################################################################################
    def GetAnnnotationData(self, AnnFileName):
            for item in self.AnnData['annotations']:  # Get Annotation Data
                if (item["file_name"] == AnnFileName):
                    return(item['segments_info'])
############################################################################################################################################
    def GetCategoryData(self,ID):
                for item in self.AnnData['categories']:
                    if item["id"]==ID:
                        return item["name"],item["isthing"]
##########################################################################################################################################3333

    def GetConnectedSegment(self, Seg):

            [NumCCmp, CCmpMask, CCompBB, CCmpCntr] = cv2.connectedComponentsWithStats(Seg.astype(np.uint8))  # apply connected component


            Mask=np.zeros([NumCCmp,Seg.shape[0],Seg.shape[1]],dtype=bool)
            BBox=np.zeros([NumCCmp,4])
            Sz=np.zeros([NumCCmp],np.uint32)
            # if NumCCmp>2:
            #     print("lllll")
            for i in range(1,NumCCmp):
                Mask[i-1] = (CCmpMask == i)
                BBox[i-1] = CCompBB[i][:4]
                Sz[i-1] = CCompBB[i][4] #segment Size
            return Mask,BBox,Sz,NumCCmp-1
#################################################################################################################################################
    def PickRandomSegment(self,Sgs,SumAreas): # Pick and return random segment and remove it from the segment list
            if self.PickBySize: # Pick random segment with probability proportional to size
                r = np.random.randint(SumAreas) + 1
                TotAreas=0
                for ind in range(Sgs.__len__()):
                    TotAreas+=Sgs[ind]['Area']
                    if TotAreas>=r:
                        break
            else: ind=np.random.randint(SumAreas) #Pick Random segment with equal probability
           # print("ind" + str(ind))
            SelectedSg=Sgs.pop(ind)
            SumAreas-=SelectedSg["Area"]
            return SelectedSg,SumAreas

#################################################################################################################################################
    def PickRandomSegmentODD(self,Sgs,SumAreas): # Pick and return random segment and remove it from the segment list
            if self.PickBySize: # Pick random segment with probability proportional to size
                r = np.random.randint(SumAreas) + 1
                TotAreas=0
                for ind in range(Sgs.__len__()):
                    TotAreas+=Sgs[ind]['Area']
                    if TotAreas>=r:
                        break
            else: ind=np.random.randint(SumAreas) #Pick Random segment with equal probability
            if Sgs[ind]["CatId"]%5==0:
                SelectedSg=Sgs.pop(ind)
                SumAreas-=SelectedSg["Area"]
                return SelectedSg,SumAreas
            else:
                return Sgs[ind], SumAreas
##########################################################################################################################
    def GenerateRandomROIMask(self, Sgs, SumAreas): # Pick set of segments and generate random ROI map

            ROI = np.ones(Sgs[0]["Mask"].shape)
            if SumAreas<=0 and np.random.randint(6)==0: return ROI
            r = np.random.randint(SumAreas) + 1

            while (SumAreas>r):
                SumAreasOld=SumAreas
                SelectedSg, SumAreas=self.PickRandomSegment( Sgs, SumAreas)
               # misc.imshow(SelectedSg["Mask"].astype(float))
                if SumAreas>r:
                    ROI[SelectedSg["Mask"]]=0
                #    misc.imshow(ROI.astype(float))
                else:
                    if np.random.randint(SumAreas,SumAreasOld)>r:# and (SumAreas>1000):
                        ROI[SelectedSg["Mask"]] = 0
                    else:
                        Sgs.append(SelectedSg)
            #print("F")
            #misc.imshow(ROI.astype(float))

            return(ROI)
#############################################################################################################################
############################################################################################################################
    def PickRandomPointInSegment(self,Seg,ErodeMask=10): # Pick Random point from

            x0 = int(np.floor(Seg["BBox"][0]))  # Bounding box x position
            Wbox = int(np.floor(Seg["BBox"][2]))  # Bounding box width
            y0 = int(np.floor(Seg["BBox"][1]))  # Bounding box y position
            Hbox = int(np.floor(Seg["BBox"][3]))  # Bounding box height
            if ErodeMask:
                Msk = cv2.erode(Seg["Mask"].astype(np.uint8), np.ones((3, 3), np.uint8), iterations=ErodeMask)
                if Msk.sum()==0: Msk=Seg["Mask"]
            else:
                Msk = Seg["Mask"]

            while(True):
                x = np.random.randint(Wbox) + x0
                y = np.random.randint(Hbox) + y0
                if (Msk[y,x])==1:
                    return x,y
##############################################################################################################################
    def DisplayTrainExample(self,Img2,ROI2,Segment2,SelectedPoint2):
        Img=Img2.copy()
        ROI=ROI2.copy()
        Segment=Segment2.copy()
        SelectedPoint=SelectedPoint2.copy()
        misc.imshow(Img)
        SelectedPoint = cv2.dilate(SelectedPoint.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        Img[SelectedPoint][:]=[255,0,0]
        Img[:, :, 0] = SelectedPoint.astype(np.uint8)*255+ (1-SelectedPoint.astype(np.uint8))*Img[:, :, 0]
        Img[:, :, 1] *= 1-SelectedPoint.astype(np.uint8)
        Img[:, :, 2] *= 1-SelectedPoint.astype(np.uint8)
        Img[ :, :, 0] *= 1-(ROI.astype(np.uint8)-Segment.astype(np.uint8))
        #Img[:, :, 1] += ROI.astype(np.uint8)*40
        Img[ :, :, 2] *= 1 - Segment.astype(np.uint8)

      #  misc.imshow(Img)
        #print(ROI.mean())
        ROI[0,0]=0
        misc.imshow(ROI.astype(float))
        misc.imshow( Segment.astype(float))
        misc.imshow(SelectedPoint.astype(float))
        misc.imshow(Img)


#############################################################################################################################
    def CropResize(self,Img, Mask,bbox,ROImask,Px,Py,Hb,Wb): # Crop and resize image and mask and ROI to feet batch size
        # ========================resize image if it two small to the batch size==================================================================================
        [h, w, d] = Img.shape

        Rs = np.max((Hb / h, Wb / w))
        Wbox = int(np.floor(bbox[2]))  # Bounding box width
        Hbox = int(np.floor(bbox[3]))  # Bounding box height


        Bs = np.min((Hb / Hbox, Wb / Wbox))
        if Rs > 1 or Bs<1 or np.random.rand()<0.3:  # Resize image and mask to batch size if mask is smaller then batch or if segment bounding box larger then batch size
            h = int(np.max((h * Rs, Hb)))
            w = int(np.max((w * Rs, Wb)))
            Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            Mask = cv2.resize(Mask.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            ROImask = cv2.resize(ROImask.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            bbox = (np.float32(bbox) * Rs.astype(np.float)).astype(np.int64)
            Px =  int(float(Px) * Rs)
            Py =  int(float(Py) * Rs)
            if Px>=w:
                Px=w-1
            if Py>=h:
                Py=h-1

 # =======================Crop image to fit batch size===================================================================================
        x1 = int(np.floor(bbox[0]))  # Bounding box x position
        Wbox = int(np.floor(bbox[2]))  # Bounding box width
        y1 = int(np.floor(bbox[1]))  # Bounding box y position
        Hbox = int(np.floor(bbox[3]))  # Bounding box height

        if Wb > Wbox:
            Xmax = np.min((w - Wb, x1))
            Xmin = np.max((0, x1 - (Wb - Wbox)-1))
        else:
            Xmin = x1
            Xmax = np.min((w - Wb, x1 + (Wbox - Wb)+1))

        if Hb > Hbox:
            Ymax = np.min((h - Hb, y1))
            Ymin = np.max((0, y1 - (Hb - Hbox)-1))
        else:
            Ymin = y1
            Ymax = np.min((h - Hb, y1 + (Hbox - Hb)+1))
        # if Xmax < Xmin:
        #     print("waaa")
        # if Ymax < Ymin:
        #     print("dddd")
        if Ymax<=Ymin: y0=Ymin
        else:
            while(True):
                y0 = np.random.randint(low=Ymin, high=Ymax + 1)
                if (y0 <= Py) and Py < (y0 + Hb):  break
        if Xmax<=Xmin: x0=Xmin
        else:
            while (True):
                x0 = np.random.randint(low=Xmin, high=Xmax + 1)
                if (x0 <= Px) and Px < (x0 + Wb):  break


        # Img[:,:,1]*=Mask
        # misc.imshow(Img)
        Px-=x0
        Py-=y0
        Img = Img[y0:y0 + Hb, x0:x0 + Wb, :]
        Mask = Mask[y0:y0 + Hb, x0:x0 + Wb]
        ROImask = ROImask[y0:y0 + Hb, x0:x0 + Wb]
#------------------------------------------Verify shape change completed----------------------------------------------------------------------------------------
        if not (Img.shape[0] == Hb and Img.shape[1] == Wb): Img = cv2.resize(Img, dsize=(Wb, Hb),interpolation=cv2.INTER_LINEAR)
        if not (Mask.shape[0] == Hb and Mask.shape[1] == Wb):Mask = cv2.resize(Mask.astype(float), dsize=(Wb, Hb), interpolation=cv2.INTER_NEAREST)
        if not (ROImask.shape[0] == Hb and ROImask.shape[1] == Wb): ROImask = cv2.resize(ROImask.astype(float), dsize=(Wb, Hb), interpolation=cv2.INTER_NEAREST)

        #-----------------------------------------------------------------------------------------------------------------------------------
        return Img,Mask,ROImask,Px,Py
        # misc.imshow(Img)
###################################Generate list of all  segments in the image###################################################################
#--------------------------Generate list of all segments--------------------------------------------------------------------------------
    def GeneratListOfAllSegments(self,Ann,Ann_name,AddUnLabeled=False,IgnoreSmallSeg=True):
        AnnList = self.GetAnnnotationData(Ann_name)
        Sgs = []  # List of segments
        SumAreas=0 # Sum areas of all segments up to this element
        for an in AnnList:
            an["name"], an["isthing"] = self.GetCategoryData(an["category_id"])
            if (an["iscrowd"] and self.IgnoreCrowds) or (not an["isthing"] and not self.ReadStuff):
                Ann[Ann == an['id']] = self.UnlabeledTag
                continue
            if (an["isthing"] and self.SplitThings) or (an["isthing"]==False and self.SplitStuff) or (an["iscrowd"] and self.SplitCrowd):
                TMask, TBBox, TSz, TNm = self.GetConnectedSegment(Ann == an['id']) # Split to connected components
                for i in range(TNm):
                    seg={}
                    seg["Mask"]=TMask[i]
                    seg["BBox"]=TBBox[i]
                    seg["Area"]=TSz[i]
                    if (not an["isthing"]): seg["Area"]*=self.StuffAreaFactor
                    if seg["Area"] < self.MinSegSize and IgnoreSmallSeg:
                        Ann[Ann == an['id']] = self.UnlabeledTag
                        continue
                    seg["NumParts"] =TNm
                    seg["IsSplit"]=TNm>1
                    seg["IsThing"]=an["isthing"]
                    seg["Name"]=an["name"]
                    seg["IsCrowd"]=an["iscrowd"]
                    seg["CatId"]=an["category_id"]
                    seg["IsLabeled"] = True
                    SumAreas+=seg["Area"]
                    Sgs.append(seg)
            else:
                    seg = {}
                    seg["Mask"] = (Ann == an['id'])
                    seg["BBox"] = an["bbox"]
                    seg["Area"] = an["area"]
                    if (not an["isthing"]): seg["Area"] *= self.StuffAreaFactor
                    if seg["Area"] < self.MinSegSize and IgnoreSmallSeg: # Ignore very small segments
                        Ann[Ann == an['id']] = self.UnlabeledTag
                        continue
                    seg["NumParts"] = 1
                    seg["IsSplit"] = False
                    seg["IsThing"] = an["isthing"]
                    seg["Name"] = an["name"]
                    seg["IsCrowd"] = an["iscrowd"]
                    seg["CatId"] = an["category_id"]
                    seg["IsLabeled"]=True
                    SumAreas += seg["Area"]
                    Sgs.append(seg)

        if AddUnLabeled: #Add unlabeled region as additional segments
            TMask, TBBox, TSz, TNm = self.GetConnectedSegment(Ann == self.UnlabeledTag)  # Split to connected components
            for i in range(TNm):
                seg = {}
                seg["Mask"] = TMask[i]
                seg["BBox"] = TBBox[i]
                seg["Area"] = TSz[i]
                seg["NumParts"] = TNm
                seg["Name"] ="unlabeled"
                seg["CatId"] = self.UnlabeledTag

                seg["IsLabeled"] = False
                Sgs.append(seg)


        return Sgs,SumAreas
##################################################################################################################################################
    def LoadNextGivenROI(self,NewImg=True):
        # ==========================Read image annotation and data===============================================================================================

            if NewImg:
                Img_name=self.FileList[self.itr].replace(self.AnnotationFileType,self.ImageFileType)
                Ann_name=self.FileList[self.itr] # Get label image name
              #  print(Ann_name)

                # print(Img_name)
                # print(Ann_name)
                Img = cv2.imread(self.ImageDir + "/" + Img_name)  # Load Image
                Img = Img[...,:: -1]
                if (Img.ndim == 2):  # If grayscale turn to rgb
                    Img = np.expand_dims(Img, 3)
                    Img = np.concatenate([Img, Img, Img], axis=2)
                Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more

                Ann = cv2.imread(self.AnnotationDir + "/" + Ann_name)  # Load Annotation
                Ann = Ann[..., :: -1]
                Ann=rgb2id(Ann)
                # misc.imshow((Ann==0).astype(float))
                # misc.imshow(Img)
                H,W=Ann.shape

                ROIMap=np.ones([H,W])
              #  AnnList = self.GetAnnnotationData(Ann_name)
                Sgs, SumAreas = self.GeneratListOfAllSegments(Ann, Ann_name,AddUnLabeled=True,IgnoreSmallSeg=False)
                self.Sgs=Sgs
                self.BImgs = np.expand_dims(Img, axis=0).astype(np.float32)
            #    self.BAnnList = AnnList
                self.BROIMask = np.expand_dims(ROIMap, axis=0).astype(np.float32)
                self.BAnn = Ann.astype(np.float32)
            else:
             #    Img = self.BImgs[0]

             #    AnnList = self.BAnnList
                 ROIMap = self.BROIMask[0]
                 Ann = self.BAnn
                 H, W = Ann.shape


                # self.BCat = np.zeros((BatchSize
            while (True):
                x = np.random.randint(W)
                y = np.random.randint(H)
                if (ROIMap[y, x]) == 1: break
            # Id=Ann[y,x]


            # SegmentMask=(Ann==Id).astype(float)
            # ConnectedMask=SegmentMask
            # if Id==self.UnlabeledTag:
            #          SegType = "Unlabeled"
            # else:
            #     for seg in Sgs:
            #         if (seg["Mask"][y,x]>0):
            #             SegmentMask=seg["Mask"]
            #             # if an["isthing"]:
            #             #     SegType="thing"
            #             # else:
            #             #     SegType="stuff"
            #             # if an["iscrowd"]:
            #             #     SegType = "crowd"
            #             # TMask, TBBox, TSz, TNm = self.GetConnectedSegment(Ann == an['id'])  # Split to connected components
            #             # for i in range(TNm):
            #             #     if TMask[i][y, x]:
            #             #         ConnectedMask = TMask[i]
            #             #         break
            #             # break

            PointerMask=np.zeros(Ann.shape,dtype=float)
            PointerMask[y,x]=1
            PointerMask=np.expand_dims(PointerMask, axis=0).astype(float)

            return  PointerMask, self.BImgs ,self.BROIMask
#########################################################################################################################################3
    def FindCorrespondingSegmentMaxIOU(self,SegMask): # Find image segment with the highest IOU correlation  to SegMask
        MaxIOU=-1
        TopSeg=0
        for seg in self.Sgs:
            IOU=(seg["Mask"] * SegMask).sum() / (seg["Mask"].sum() + SegMask.sum() - (seg["Mask"] * SegMask).sum())
            if IOU>MaxIOU:
                MaxIOU=IOU
                TopSeg=seg
        IOU = (TopSeg["Mask"] * SegMask).sum() / (TopSeg["Mask"].sum() + SegMask.sum() - (TopSeg["Mask"] * SegMask).sum())
        Precision = (TopSeg["Mask"] * SegMask).sum() / SegMask.sum()
        Recall = (TopSeg["Mask"] * SegMask).sum() / TopSeg["Mask"].sum()
        if not TopSeg["IsLabeled"]: SegType = "Unlabeled"
        elif TopSeg["IsCrowd"]:SegType = "crowd"
        elif TopSeg["IsThing"]: SegType = "thing"
        else: SegType = "stuff"
        return IOU,Precision,Recall,SegType,TopSeg["Mask"].astype(float)


        # if an["isthing"]:
                        #
                        # else:
                        #     SegType="stuff"
                        # if an["iscrowd"]:
                        #     SegType = "crowd"
                        # TMask, TBBox, TSz, TNm = self.GetConnectedSegment(Ann == an['id'])  # Split to connected components
                        # for i in range(TNm):
                        #     if TMask[i][y, x]:
                        #         ConnectedMask = TMask[i]
                        #         break
                        # break


######################################Read next batch. given an image number and a class the batch conssit on all the instance of the input class in the input image######################################################################################################
    def LoadNext(self,batch_pos,itr_pos, Hb=-1,Wb=-1):
        # ==========================Read image annotation and data===============================================================================================


            Img_name=self.FileList[itr_pos].replace(self.AnnotationFileType,self.ImageFileType)
            Ann_name=self.FileList[itr_pos] # Get label image name
          #  print(Ann_name)

            # print(Img_name)
            # print(Ann_name)
            Img = cv2.imread(self.ImageDir + "/" + Img_name)  # Load Image
            Img = Img[...,:: -1]
            if (Img.ndim == 2):  # If grayscale turn to rgb
                Img = np.expand_dims(Img, 3)
                Img = np.concatenate([Img, Img, Img], axis=2)
            Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more

            Ann = cv2.imread(self.AnnotationDir + "/" + Ann_name)  # Load Annotation
            Ann = Ann[..., :: -1]
            Ann=rgb2id(Ann)
            # misc.imshow((Ann==0).astype(float))
            # misc.imshow(Img)



#--------------------------Generate list of all segments--------------------------------------------------------------------------------

            Sgs,SumAreas= self.GeneratListOfAllSegments(Ann, Ann_name)
#----------------------Check if there even labels------------------------------------------------------------------------------------------------------------------
            Evens=False
            for sg in Sgs:
              if sg["CatId"]%5==0:
                  Evens=True
                  break
            SegmentSelected=False
            if Sgs.__len__()>0 and Evens:
               for t in range(10):
                 SelectedSg, SumAreas = self.PickRandomSegmentODD(Sgs, SumAreas)
                 if SelectedSg["CatId"]%5==0:
                     SegmentSelected=True
                     break

# -------------------------------------------------------------------------------------------
            if not SegmentSelected:
                print("No Segments to pick")
                itr_pos=np.random.randint(len(self.FileList))
                return self.LoadNext(batch_pos,itr_pos,Hb,Wb)
            if Sgs.__len__()>0:
                ROIMask = self.GenerateRandomROIMask(Sgs, SumAreas)
            else:
                ROIMask = np.ones(Ann.shape)

            print(SelectedSg["CatId"])

        # misc.imshow(SelectedSg["Mask"].astype(float))
            # misc.imshow(Img)


            Px, Py = self.PickRandomPointInSegment( SelectedSg)



#-----------------------------Crop and resize--------------------------------------------------------------------------------------------------------

            # self.SumThings += SelectedSg["IsThing"]
            # self.SumStuff += 1-SelectedSg["IsThing"]
            # print(self.SumThings)
            # print("stuff")
            # print(self.SumStuff)
            if not Hb==-1:
               Img, SegMask, ROIMask, Px, Py=self.CropResize(Img, SelectedSg["Mask"], SelectedSg["BBox"], ROIMask, Px, Py, Hb, Wb)
            # else:
            #     SegMask=SelectedSg["Mask"]
#---------------------------------------------------------------------------------------------------------------------------------
            PointerMap = np.zeros(SegMask.shape)
            PointerMap[Py, Px] = 1



  #          self.DisplayTrainExample(Img, ROIMask, SegMask, PointerMap)
           # print("______")
           # print(batch_pos)
            self.BImgs[batch_pos] = Img
            self.BSegmentMask[batch_pos] = SegMask
            self.BROIMask[batch_pos] = ROIMask
            self.BPointerMap[batch_pos] =  PointerMap
            self.BIsThing[batch_pos] = SelectedSg["IsThing"]
            self.BCat[batch_pos] = SelectedSg["CatId"]
          #  print("CAT_ID "+str(SelectedSg["CatId"]))


############################################################################################################################################################
############################################################################################################################################################
    def StartLoadBatch(self):
        # =====================Initiate batch=============================================================================================
        while True:
            Hb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # Batch hight
            Wb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # batch  width
            if Hb*Wb<self.MaxPixels: break
        BatchSize =  np.int(np.min((np.floor(self.MaxPixels / (Hb * Wb)), self.MaxBatchSize)))
        self.BImgs = np.zeros((BatchSize, Hb, Wb, 3))  #
        self.BSegmentMask = np.zeros((BatchSize, Hb, Wb))
        self.BROIMask = np.zeros((BatchSize, Hb, Wb))  #
        self.BPointerMap = np.zeros((BatchSize, Hb, Wb))
        self.BIsThing = np.zeros((BatchSize))
        self.BCat= np.zeros((BatchSize))

        if self.itr+BatchSize >= len(self.FileList):
            if self.suffle: random.shuffle(self.FileList)
            self.itr = 0
            self.Epoch += 1

            # print("No More files to read")
            # return
        self.thread_list = []
        for pos in range(BatchSize):
            th=threading.Thread(target=self.LoadNext,name="thread"+str(pos),args=(pos,self.itr+pos,Hb,Wb))
            self.thread_list.append(th)
            th.start()
        self.itr+=BatchSize
###########################################################################################################
    def WaitLoadBatch(self):
            for th in self.thread_list:
                 th.join()

########################################################################################################################################################################################
    def LoadBatch(self):
            self.WaitLoadBatch()
            Imgs=self.BImgs
            SegmentMask=self.BSegmentMask
            ROIMask=self.BROIMask
            PointerMap=self.BPointerMap
            self.StartLoadBatch()
            return Imgs, SegmentMask,ROIMask,PointerMap
########################################################################################################################################################################################
    def LoadSingleClean(self):
        if self.itr >= len(self.FileList):
            self.itr = 0
            self.Epoch += 1
        Hb, Wb, d = cv2.imread(self.AnnotationDir + "/" + self.FileList[self.itr]).shape
        self.BImgs = np.zeros((1, Hb, Wb, 3))  #
        self.BSegmentMask = np.zeros((1, Hb, Wb))
        self.BROIMask = np.zeros((1, Hb, Wb))  #
        self.BPointerMap = np.zeros((1, Hb, Wb))
        self.BIsThing = np.zeros((1))
        self.BCat = np.zeros((1))

        self.LoadNext(0,self.itr, Hb,Wb)

        self.itr += 1
        Imgs = self.BImgs
        SegmentMask = self.BSegmentMask
        ROIMask = self.BROIMask
        PointerMap = self.BPointerMap
        IsThing = self.BIsThing[0]
        return Imgs, SegmentMask, ROIMask, PointerMap,IsThing

