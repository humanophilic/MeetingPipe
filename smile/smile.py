import torch, sys, cv2

from opts import parse_opts
args = parse_opts()
sys.path.append('./smile_v2')
from smilenet import smileNet

checkpoint = torch.load(args.smileModel)

def smile_main(vpath, gpu_id):
    device = torch.device('cuda:{}'.format(gpu_id))
    
    model = smileNet(dropout=0.1)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(vpath)
    preds = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_cp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_cp = cv2.resize(frame_cp, (224,224))
        frame_cp = frame_cp/255.
        frame_cp = torch.FloatTensor(frame_cp).unsqueeze(0).unsqueeze(0).to(device)
        pred = model(frame_cp)
        if int(torch.argmax(pred))==1:
            preds.append(1)
        else:
            preds.append(0)
    return preds