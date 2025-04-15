import numpy as np
import cv2
import glob
import os
import subprocess
import librosa
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


#vid = np.load('trainval/xcR1SWhjZys/50001.npy')
#vid = np.load('trainval/ZzugJPASNB8/50001.npy')
#vid = np.load('trainval/FI8c14giiWQ/50001.npy')
#vid = np.load('test/wzkFoetpSSM/00002.npy')
'''
vid = np.load('test/zuYzOn0U2PY/00005.npy')
for i in range(len(vid)):
    cv2.imshow('image', vid[i])
    cv2.waitKey(0)
cv2.destroyAllWindows()
'''


def audio_files(root_dir):
    # trainval/FI8c14giiWQ/50001.mp4
    vid_paths = glob.glob(os.path.join(root_dir, '*', '*.mp4'))
    print(len(vid_paths))
    for vp in vid_paths:
        out_wav = vp.replace('.mp4', '.wav')
        if os.path.exists(out_wav) and is_valid_wav(out_wav):
            continue
        try:
            #os.system(' '.join(['ffmpeg', '-y', '-i', vp, '-ac', '1', '-ar', '16000', '-vn', '-loglevel', 'quiet', out_wav]))
            #subprocess.run(['ffmpeg', '-y', '-i', vp, '-ac', '1', '-ar', '16000', '-f', 'wav', '-vn', out_wav])
            subprocess.run(['ffmpeg', '-y', '-i', vp, '-ac', '1', '-ar', '16000', '-vn', '-loglevel', 'quiet', out_wav])
            #subprocess.run(['ffmpeg', '-y', '-i', vp, '-ac', '1', '-ar', '16000', '-vn', '-loglevel', 'quiet', out_wav], timeout=15)
        except Exception as e:
            #pass
            print(f"processing file {vp}, occur error !!!", flush=True)
    print('Done')


def save_files(root_dir):
    # trainval/FI8c14giiWQ/50001.mp4
    vid_paths = glob.glob(os.path.join(root_dir, '*', '*.npy'))
    print(len(vid_paths))
    max_vid_len = -1
    max_txt_len = -1
    for vp in vid_paths:
        l = len(np.load(vp))
        max_vid_len = max(max_vid_len, l)
        with open(vp.replace('.npy', '.txt'), 'r', encoding='utf-8') as f:
            txt = f.readline().strip()   # 读第一行
        txt = txt.replace("Text:", "", -1).strip().lower()
        max_txt_len = max(max_txt_len, len(txt))
    print(max_vid_len, max_txt_len)
    print('Done')


def is_valid_wav(fn):
    try:
        y, sr0 = librosa.load(fn, sr=None)
        if len(y) == 0:
            return False
        if sr0 != 16000:
            return False
        return True
    except Exception as e:
        #print(f"loading wav file {fn}, occur error !!!")
        return False


def save_npy_to_imgs(inp_path, out_path='align_faces'):
    rois = np.load(inp_path)
    for i, mouth in enumerate(rois): 
        cv2.imwrite(os.path.join(out_path, f'my{i+1}.jpg'), mouth)  # 保存成视频帧
    print('Done')


if __name__ == '__main__':
    #save_files('trainval')
    #audio_files('trainval')
    #audio_files('test')
    #audio_files('pretrain')

    #save_npy_to_imgs('test/kwYxHPXIaao/00001.npy')
    save_npy_to_imgs('trainval/4yyPSZSGLOI/50004.npy')

    #is_valid_wav('../LRS3/trainval/VMwjscSCcf0/50002.mp4')
    #is_valid_wav('../LRS3/trainval/lU1ISu1LOv4/50010.wav')
    #is_valid_wav('../LRS3/trainval/VT3XyORCFDA/50001.wav')

