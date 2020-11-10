import os
import utils
import cv2
import numpy as np
import argparse
import torch
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import modellist

parser = argparse.ArgumentParser(description='GradCAM')
parser.add_argument('-m', '--modelname', metavar='ARCH', default='resnet18', help='model architecture: ')
parser.add_argument('-a', '--attention', default='', help='attention')
parser.add_argument('path',type=str, help='image path')
parser.add_argument('state_dict_path',type=str, help='state_dict_path')

args = parser.parse_args()

if args.attention == 'se':
    attention_module='se_layer'
elif args.attention == 'cbam':
    attention_module='cbam_layer'
else:
    attention_module = None 
        
model = modellist.Modellist(args.modelname, args.numclasses, attention_module)
model.load_state_dict(torch.load(args.state_dict_path))

image_path = args.path
img = cv2.imread(image_path, 1)
img = np.float32(cv2.resize(img, (224, 224))) / 255
input = utils.preprocess_image(img)
use_cuda = torch.cuda.is_available()
target_index = None

grad_cam = utils.GradCam(model=model, feature_module=model.layer4, \
                       target_layer_names=["2"], use_cuda=use_cuda)
mask = grad_cam(input, target_index)
utils.show_cam_on_image(img, mask)
gb_model = utils.GuidedBackpropReLUModel(model=model, use_cuda=use_cuda)
gb = gb_model(input, index=target_index)
gb = gb.transpose((1, 2, 0))
cam_mask = cv2.merge([mask, mask, mask])
cam_gb = utils.deprocess_image(cam_mask*gb)
gb = utils.deprocess_image(gb)

cv2.imwrite('gb.jpg', gb)
cv2.imwrite('cam_gb.jpg', cam_gb)
