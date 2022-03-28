import os, sys, cv2, torch, math, python_speech_features
import numpy as np
from scipy.io import wavfile

from opts import parse_opts
args = parse_opts()
sys.path.append('./speak')
from talkNet import talkNet

def speak_main(vpath, gpu_id):
    s = talkNet(gpu_id)
    s.loadParameters(args.speakModel)
    s.eval()
    durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result

    apath = os.path.join(args.cropDir, os.path.basename(args.videoPath).split('.')[0], 'pycrop', os.path.basename(vpath).split('.')[0]+'.wav')
    _, audio = wavfile.read(apath)
    audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
    video = cv2.VideoCapture(vpath)
    videoFeature = []
    while video.isOpened():
        ret, frames = video.read()
        if ret == True:
            face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (224,224))
            face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
            videoFeature.append(face)
        else:
            break
    video.release()
    videoFeature = np.array(videoFeature)
    length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
    audioFeature = audioFeature[:int(round(length * 100)),:]
    videoFeature = videoFeature[:int(round(length * 25)),:,:]
    allScore = []
    for duration in durationSet:
        batchSize = int(math.ceil(length / duration))
        scores = []
        with torch.no_grad():
            for i in range(batchSize):
                inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda(gpu_id)
                inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda(gpu_id)
                embedA = s.model.forward_audio_frontend(inputA)
                embedV = s.model.forward_visual_frontend(inputV)	
                embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                out = s.model.forward_audio_visual_backend(embedA, embedV)
                score = s.lossAV.forward(out, labels = None)
                scores.extend(score)
        allScore.append(scores)
    allScore = np.round((np.mean(np.array(allScore), axis = 0)), 1).astype(float)
    
    return allScore





