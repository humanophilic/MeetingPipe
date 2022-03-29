import os, sys, cv2, torch
import numpy as np
from torch import nn
from torchvision import transforms
from sklearn import preprocessing
from PIL import Image

from opts import parse_opts
args = parse_opts()
sys.path.append('./nod')
from nodnet import RNN
from model import SixDRepNet
import utils

snapshot_path = args.snapshot
checkpoint = torch.load(args.nodModel)

transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def nod_main(vpath, gpu_id):
    # extract head pitch
    model = SixDRepNet(gpu_id=gpu_id,
                       backbone_name='RepVGG-B1g2',
                       backbone_file='',
                       deploy=True,
                       pretrained=False)

    saved_state_dict = torch.load(os.path.join(
        snapshot_path), map_location='cpu')

    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)
    model.cuda(gpu_id)
    model.eval()

    cap = cv2.VideoCapture(vpath)
    fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ccount = ((fcount-20)//10) + 1
    pitchs = [[] for _ in range(ccount)]
    frame_th = -1
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            frame_th += 1
            if (not ret) or frame_th>=(ccount-1)*10+20:
                break
            img = Image.fromarray(frame)
            img = img.convert('RGB')
            img = transformations(img)
            img = torch.Tensor(img[None, :]).cuda(gpu_id)
            R_pred = model(img)
            euler = utils.compute_euler_angles_from_rotation_matrices(R_pred)*180/np.pi
            pitch = euler[:, 0].cpu()
            if frame_th//10==0:
                pitchs[frame_th//10].append(-pitch)
            elif frame_th//10>0 and frame_th//10<ccount:
                pitchs[(frame_th//10)-1].append(-pitch)
                pitchs[(frame_th//10)].append(-pitch)
            else:
                pitchs[(frame_th//10)-1].append(-pitch)
    
    pitchs = preprocessing.normalize(pitchs, norm='l2')
    pitchs = np.expand_dims(pitchs, -1)
    pitchs = torch.FloatTensor(pitchs)

    # recognize nodding
    params = {
            'batch_size':8,
            'num_workers':8,
            'conv_hidden_size':1,
            'rnn_hidden_size':16,
            'num_layers':2,
            'dropout':0.2,
        }

    network = RNN(
        gpu_id = gpu_id,
        conv_hidden_size=params['conv_hidden_size'],
        rnn_hidden_size=params['rnn_hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    )
    # network = nn.DataParallel(network)
    network.load_state_dict(checkpoint['state_dict'])
    network.cuda(gpu_id)
    network.eval()

    preds = []
    for idx, X in enumerate(pitchs):              
        X = X.unsqueeze(0).cuda(gpu_id)
        pred = network(X)
        preds.extend([int(i) for i in torch.argmax(pred, axis=1)])

    preds_20 = []
    for i in preds:
        preds_20.extend([i, i])
    preds_10 = []
    preds_10.append(preds_20[0])
    for idx in range(1, len(preds_20)-2, 2):
        if preds_20[idx]==0 or preds_20[idx+1]==0:
            preds_10.append(0)
        else:
            preds_10.append(1)
    preds_10.append(preds_20[-1])

    return [i for i in preds_10 for _ in range(10)]
