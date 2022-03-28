import os, sys, warnings, pickle, cv2, time
from re import I
from curses.textpad import rectangle
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from statistics import mean
import torch.backends.cudnn as cudnn
from moviepy.editor import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from tabulate import tabulate

from opts import parse_opts
args = parse_opts()
from face_track.face_track import face_track_main
from speak.speak import speak_main
from nod.nod import nod_main
from smile.smile import smile_main

warnings.filterwarnings('ignore')

def visualization(fpath):
    data = pd.read_csv(fpath)
    data_arr = np.array(data)
    trackp = os.path.join(args.cropDir, os.path.basename(args.videoPath).split('.')[0], 'pywork', 'tracks.pckl')
    with open(trackp, 'rb') as f:
        tracks = pickle.load(f, encoding='latin1')
    
    videop = os.path.join(args.cropDir, os.path.basename(args.videoPath).split('.')[0], 'pyavi', 'video.avi')
    cap = cv2.VideoCapture(videop)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vp = os.path.join(args.cropDir, os.path.basename(args.videoPath).split('.')[0], 'pyavi', 'tmp.mp4')
    writer = cv2.VideoWriter(vp, fourcc, fps, (w,h))
    ap = os.path.join(args.cropDir, os.path.basename(args.videoPath).split('.')[0], 'pyavi', 'audio.wav')
    outp = os.path.join(args.cropDir, os.path.basename(args.videoPath).split('.')[0], 'pyavi', 'output.mp4')

    frame_th = -1
    while cap.isOpened():
        ret, frame = cap.read()
        frame_th += 1
        if (not ret) or frame_th>=data_arr.shape[0]:
            break
        frame_cp = frame      
        for i in range(args.nGuest):
            if data['speak_{}'.format(i)][frame_th]>0 or data['nod_{}'.format(i)][frame_th]==1 or data['smile_{}'.format(i)][frame_th]==1:
                x = int(tracks[i]['proc_track']['x'][frame_th])
                y = int(tracks[i]['proc_track']['y'][frame_th])
                s = int(tracks[i]['proc_track']['s'][frame_th])
                cv2.rectangle(frame_cp, (x-s,y-s), (x+s,y+s), (0,0,255), 2)
                state = 0
                if data['speak_{}'.format(i)][frame_th]>0:
                    cv2.putText(frame_cp, 'speak', (x-s,y-s-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    state += 1
                if data['nod_{}'.format(i)][frame_th]==1:
                    if state==0:
                        cv2.putText(frame_cp, 'nod', (x-s,y-s-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    else:
                        cv2.putText(frame_cp, 'nod', (x-s,y-s-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    state += 1
                if data['smile_{}'.format(i)][frame_th]==1:
                    if state==0:
                        cv2.putText(frame_cp, 'smile', (x-s,y-s-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                    elif state==1:
                        cv2.putText(frame_cp, 'smile', (x-s,y-s-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                    else:
                        cv2.putText(frame_cp, 'smile', (x-s,y-s-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        writer.write(frame_cp)
    cap.release()
    writer.release()
    vf = VideoFileClip(vp)
    af = AudioFileClip(ap)
    audio = CompositeAudioClip([af])
    vf.audio = audio
    vf.write_videofile(outp, audio_codec='aac')
    os.remove(vp)

speaks, nods, smiles = {}, {}, {}
_tspeak, _tnod = [], []
def mp_detection(id, vfile, gpu_id):
    t5 = time.time()
    score_speak = speak_main(vfile, gpu_id)
    t6 = time.time()
    _tspeak.append(round((t6-t5), 2))
    score_nod = nod_main(vfile, gpu_id)
    t7 = time.time()
    _tnod.append(round((t7-t6), 2))

    n_frame = min(len(score_speak), len(score_nod))
    score_speak = score_speak[: n_frame]
    score_nod = score_nod[: n_frame]

    speaks['{}'.format(id)] = score_speak
    nods['{}'.format(id)] = score_nod

def main():
    t1 = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    cudnn.enabled = True

    face_track_main()
    t2 = time.time()
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + ' Face track has finished.  \r\n\n')

    fdir = os.path.join(args.cropDir, os.path.basename(args.videoPath).split('.')[0], 'pycrop')
    flist = glob(os.path.join(fdir, '*.avi'))
    flist.sort()
    vr = VideoFileClip(flist[0])
    tr = vr.duration
    assert len(flist)==args.nGuest, 'The number of faces detected is not equal to the input nGuest.'

    csvdir = os.path.join(args.csvDir, os.path.basename(args.videoPath).split('.')[0])
    if not os.path.exists(csvdir):
        os.mkdir(csvdir)

    if args.gpus>1:
        p = ThreadPoolExecutor(args.gpus)
        futures = [p.submit(mp_detection, i, f, i%args.gpus) for i, f in enumerate(flist)]
        _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]
    else:
        for id, vfile in tqdm(enumerate(flist), total=args.nGuest):
            t5 = time.time()
            score_speak = speak_main(vfile, 0)
            t6 = time.time()
            _tspeak.append(round((t6-t5), 2))
            score_nod = nod_main(vfile, 0)
            t7 = time.time()
            _tnod.append(round((t7-t6), 2))

            n_frame = min(len(score_speak), len(score_nod))
            score_speak = score_speak[: n_frame]
            score_nod = score_nod[: n_frame]

            speaks['{}'.format(id)] = score_speak
            nods['{}'.format(id)] = score_nod

    t8 = time.time()
    score_smile = smile_main()
    for i in range(args.nGuest):
        smiles['{}'.format(i)] = score_smile[i][:len(speaks['0'])]

    t3 = time.time()
    tsmile = round((t3-t8)/args.nGuest, 2)
    tspeak = mean(_tspeak)
    tnod = mean(_tnod)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + ' Indicator detection has finished.  \r\n\n')

    scores = {}
    for i in range(args.nGuest):
        scores['speak_{}'.format(i)] = speaks['{}'.format(i)]
        scores['nod_{}'.format(i)] = nods['{}'.format(i)]
        scores['smile_{}'.format(i)] = smiles['{}'.format(i)]

    dataframe = pd.DataFrame(scores)
    dataframe.to_csv(os.path.join(csvdir, 'scores.csv'))

    visualization(os.path.join(csvdir, 'scores.csv'))

    t4 = time.time()
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + ' Visualization has finished.  \r\n\n')

    print(tabulate([['Face track', round((t2-t1),2)], ['Indicator detection', '{:.2f} ({:.2f},{:.2f},{:.2f})*{}'.format((t3-t2),tspeak,tnod,tsmile,args.nGuest)], ['Visualization', round((t4-t3),2)], ['Total/Raw video', '{:.2f}/{:.2f}'.format((t4-t1),tr)]], headers=['Operation', 'Time(sec)'], stralign='center'), '\n')
    
if __name__ == '__main__':
    main()