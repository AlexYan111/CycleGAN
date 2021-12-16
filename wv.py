import torch
import torch.optim as optim
import cv2
import numpy as np
from main import Generator

MODEL = "gen_b.wyd"
WEBCAM_VIDEO = True
GEN1 = Generator().to("cuda")
GEN2 = Generator().to("cuda")

#we need a function to convert filtered tensor into an image
import torchvision.transforms as transforms
unloader = transforms.ToPILImage()

#Now load out existing model
optimizer = optim.Adam(list(GEN1.parameters()) + list(GEN2.parameters()),lr=0.002,betas=(0.5, 0.999),)
GEN1.load_state_dict(torch.load(MODEL))

#This is output of our model
res = cv2.VideoWriter('output',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (512,512))

#If WEBCAM_VIDEO == TRUE, we set it as webcam or wv
if WEBCAM_VIDEO:
    wv = cv2.VideoCapture(0)
    if not wv.isOpened():
        raise IOError("Cannot open webcam")
else:
    wv = cv2.VideoCapture('video')


#Read until webcam is closed or all frams of wv have been iterated 
while(True):
    cur, before = wv.read()
    if not cur:
        break
    before = cv2.resize(before, (512,512), interpolation=cv2.INTER_AREA)
    before = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
    before = np.array([before])
    before = before.transpose([0,3,1,2])
    before = torch.FloatTensor(before)

    before = before.to("cuda")

    after = GEN1.call(before)
    after = after.squeeze(0)
    after = unloader(after)
    after = cv2.cvtColor(np.array(after), cv2.COLOR_BGR2RGB)  
    after = cv2.resize(after, (512, 512))      

    res.write(after)

    cv2.imshow('frame', after)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

wv.release()
cv2.destroyAllWindows()