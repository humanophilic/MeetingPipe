import cv2
from glob import glob
import os
import pickle

from opts import parse_opts
args = parse_opts()

def smile_main():
    smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + args.smileModel)

    framesp = glob(os.path.join(args.cropDir, os.path.basename(args.videoPath).split('.')[0], 'pyframes', '*.jpg'))
    framesp.sort()
    frames = [cv2.cvtColor(cv2.imread(framep), cv2.COLOR_BGR2GRAY) for framep in framesp]

    with open(os.path.join(args.cropDir, os.path.basename(args.videoPath).split('.')[0], 'pywork', 'pre_tracks.pckl'), 'rb') as f:
        faces = pickle.load(f, encoding='latin1')
    
    scores = [[] for _ in range(args.nGuest)]
    for id in range(args.nGuest):
        for i, coord in enumerate(faces[id]['bbox']):
            x1 = int(coord[0])
            y1 = int(coord[1])
            x2 = int(coord[2])
            y2 = int(coord[3])
            face = frames[i][int((y1+y2)/2):y2, x1:x2]
            smiles = smile_detector.detectMultiScale(face, scaleFactor=1.8, minNeighbors=20)
            if len(smiles)>0:
                scores[id].append(1)
            else:
                scores[id].append(0)
                
    return scores