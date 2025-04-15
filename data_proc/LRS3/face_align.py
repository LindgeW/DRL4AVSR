import dlib, cv2, os
import numpy as np
# import skvideo
# import skvideo.io
from tqdm import tqdm
import pickle, shutil, tempfile
import math
import glob
import subprocess
from collections import deque
from skimage import transform as tf
from torch.utils.data import DataLoader, Dataset

## Based on: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/master/preprocessing/crop_mouth_from_video.py

""" Crop Mouth ROIs from videos for lipreading"""

# -- Landmark interpolation:
def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks

# -- Face Transformation
def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # warp
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped, tform

def apply_transform(transform, img, std_size):
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255  # note output from warp is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped

def get_frame_count(filename):
    cap = cv2.VideoCapture(filename)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total

def read_video(filename, as_grey=True):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {filename}")
    while(cap.isOpened()):                                                 
        ret, frame = cap.read()   # BGR
        if ret:                      
            if as_grey:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield frame                                                    
        else:                                                              
            break                                                         
    cap.release()


# def vread(filename, as_grey=False, num_frames=None, start_time=None, duration=None):
#     """
#     Read video file and return a generator that yields frames.
#     :param filename: Path to the video file.
#     :param as_grey: If True, convert frames to grayscale.
#     :param num_frames: Maximum number of frames to read.
#     :param start_time: Start reading from this time (in seconds).
#     :param duration: Duration of video to read (in seconds).
#     :yield: Frames as NumPy arrays.
#     """
#     cap = cv2.VideoCapture(filename)
#     if not cap.isOpened():
#         raise IOError(f"Cannot open video file {filename}")
    
#     # Calculate start frame and end frame
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     start_frame = 0 if start_time is None else int(start_time * fps)
#     end_frame = total_frames if duration is None else int(start_frame + duration * fps)
    
#     if num_frames is not None:
#         end_frame = min(end_frame, start_frame + num_frames)
    
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
#     frame_count = 0
#     while cap.isOpened() and frame_count < end_frame - start_frame:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if as_grey:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         yield frame
#         frame_count += 1
#     cap.release()


# -- Crop
def cut_patch(img, landmarks, height, width, threshold=5):
    center_x, center_y = np.mean(landmarks, axis=0)
    if center_y - height < 0:                                                
        center_y = height                                                    
    if center_y - height < 0 - threshold:                                    
        raise Exception('too much bias in height')                           
    if center_x - width < 0:                                                 
        center_x = width                                                     
    if center_x - width < 0 - threshold:                                     
        raise Exception('too much bias in width')                            
                                                                             
    if center_y + height > img.shape[0]:                                     
        center_y = img.shape[0] - height                                     
    if center_y + height > img.shape[0] + threshold:                         
        raise Exception('too much bias in height')                           
    if center_x + width > img.shape[1]:                                      
        center_x = img.shape[1] - width                                      
    if center_x + width > img.shape[1] + threshold:                          
        raise Exception('too much bias in width')                            
                                                                             
    cutted_img = np.copy(img[int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                         int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    return cutted_img

def write_video_ffmpeg(rois, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    decimals = 10
    fps = 25
    tmp_dir = tempfile.mkdtemp()
    for i_roi, roi in enumerate(rois):
        cv2.imwrite(os.path.join(tmp_dir, str(i_roi).zfill(decimals)+'.png'), roi)
    list_fn = os.path.join(tmp_dir, "list")
    with open(list_fn, 'w') as fo:
        fo.write("file " + "'" + tmp_dir+'/%0'+str(decimals)+'d.png' + "'\n")
    ## ffmpeg
    if os.path.isfile(target_path):
        os.remove(target_path)
    cmd = ['ffmpeg', "-f", "concat", "-safe", "0", "-i", list_fn, "-q:v", "1", "-r", str(fps), '-y', '-crf', '20', target_path]
    pipe = subprocess.run(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    # rm tmp dir
    shutil.rmtree(tmp_dir)
    return


def crop_patch(video_pathname, landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, window_margin, start_idx, stop_idx, crop_height, crop_width):
    """Crop mouth patch
    :param str video_pathname: pathname for the video_dieo
    :param list landmarks: interpolated landmarks
    """
    frame_idx = 0
    num_frames = get_frame_count(video_pathname)
    frame_gen = read_video(video_pathname)
    margin = min(num_frames, window_margin)
    while True:
        try:
            frame = frame_gen.__next__() ## -- BGR
        except StopIteration:
            break
        if frame_idx == 0:
            q_frame, q_landmarks = deque(), deque()
            sequence = []

        q_landmarks.append(landmarks[frame_idx])
        q_frame.append(frame)
        if len(q_frame) == margin:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            # -- affine transformation
            trans_frame, trans = warp_img(smoothed_landmarks[stablePntsIDs, :],
                                          mean_face_landmarks[stablePntsIDs, :],
                                          cur_frame,
                                          STD_SIZE)
            trans_landmarks = trans(cur_landmarks)
            # -- crop mouth patch
            sequence.append(cut_patch(trans_frame,
                                      trans_landmarks[start_idx:stop_idx],
                                      crop_height//2,
                                      crop_width//2,))
        if frame_idx == len(landmarks)-1:
            while q_frame:
                cur_frame = q_frame.popleft()
                # -- transform frame
                trans_frame = apply_transform(trans, cur_frame, STD_SIZE)
                # -- transform landmarks
                trans_landmarks = trans(q_landmarks.popleft())
                # -- crop mouth patch
                sequence.append(cut_patch(trans_frame, 
                                          trans_landmarks[start_idx:stop_idx],
                                          crop_height//2,
                                          crop_width//2,))
            return np.array(sequence)
        frame_idx += 1
    return None


def landmarks_interpolate(landmarks):
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks


def detect_landmark(image, detector, cnn_detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    rects = detector(gray, 1)
    if len(rects) == 0 and cnn_detector is not None:   # using cnn detector  mmod_human_face_detector.dat
        print('using cnn detector!!')
        rects = cnn_detector(gray, 0)
        rects = [d.rect for d in rects]
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def preprocess_video(input_video_path, output_video_path, detector, cnn_detector, predictor, mean_face_path):
    #if os.path.exists(output_video_path):
    #    return
    STD_SIZE = (256, 256)
    mean_face_landmarks = np.load(mean_face_path)
    stablePntsIDs = [33, 36, 39, 42, 45]   # 鼻尖、眼角关键点
    # videogen = skvideo.io.vread(input_video_path)
    videogen = read_video(input_video_path)
    frames = np.array([frame for frame in videogen])
    landmarks = []
    for frame in frames:
        landmark = detect_landmark(frame, detector, cnn_detector, predictor)
        landmarks.append(landmark)
    preprocessed_landmarks = landmarks_interpolate(landmarks)
    rois = crop_patch(input_video_path, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, 
                      window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
    
    #write_video_ffmpeg(rois, output_video_path)  # 保存成视频
    #for i, mouth in enumerate(rois): cv2.imwrite(os.path.join('align_faces', f'{i+1}.jpg'), mouth)  # 保存成视频帧
    roi_arr = np.array(rois)   # (T, H, W)
    np.save(output_video_path, roi_arr)   # 自动添加.npy后缀
    return


def save_files(root_dir, saved_path):
    # trainval/FI8c14giiWQ/50001.mp4
    #vid_paths = glob.glob(os.path.join(root_dir, '*', '*.npy'))
    #vid_paths = glob.glob(os.path.join(root_dir, '*', '*.mp4'))
    #ID = -1
    #pre_spk = None
    #with open(saved_path, 'w', encoding='utf-8') as fw:
    #    for vp in vid_paths:
    #        cur_spk = os.path.basename(os.path.dirname(vp))
    #        if cur_spk != pre_spk:
    #            ID += 1
    #        fw.write(str(ID)+' '+vp)
    #        fw.write('\n')
    #        pre_spk = cur_spk
    ID = 0
    with open(saved_path, 'w', encoding='utf-8') as fw:
        for dir0 in os.listdir(root_dir):
            dir1 = os.path.join(root_dir, dir0)
            for fn in os.listdir(dir1):
                pat = os.path.join(dir1, fn)
                if pat.endswith('.npy'):
                #if pat.endswith('.mp4'):
                    fw.write(str(ID) + ' ' + pat)
                    fw.write('\n')
            ID += 1
    print('Done')


class MyDataset(Dataset):
    def __init__(self, root_dir, dlib_path, cnn_face_path, mean_face_path):
        # trainval/FI8c14giiWQ/50001.mp4
        self.video_paths = glob.glob(os.path.join(root_dir, '*', '*.mp4'))
        self.mouth_paths = [os.path.splitext(p)[0]+'.npy' for p in self.video_paths]
        self.mean_face_path = mean_face_path
        print('Total Num:', len(self.video_paths))
        
        self.dlib_path = dlib_path
        self.cnn_face_path = cnn_face_path
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(dlib_path)
        self.cnn_detector = dlib.cnn_face_detection_model_v1(cnn_face_path) if cnn_face_path is not None else None

    def __getitem__(self, idx):
        try:
            preprocess_video(self.video_paths[idx], self.mouth_paths[idx], self.detector, self.cnn_detector, self.predictor, self.mean_face_path)
            #preprocess_video(self.video_paths[idx], self.mouth_paths[idx], self.dlib_path, self.cnn_face_path, self.mean_face_path)
        except:
            print('>>>>> Bad Video:', self.video_paths[idx])
        return 0

    def __len__(self):
        return len(self.video_paths)

def run():
    #root_dir = './trainval'
    root_dir = './test'
    dlib_path = "./shape_predictor_68_face_landmarks.dat"
    cnn_face_path = "./mmod_human_face_detector.dat"
    mean_face_path = "./20words_mean_face.npy"
    loader = DataLoader(MyDataset(root_dir, dlib_path, cnn_face_path, mean_face_path),
                        batch_size=128,
                        num_workers=8,
                        shuffle=False,
                        drop_last=False,
                        pin_memory=True)
    for _ in tqdm(loader):
        pass
    print('Done!!!')


if __name__ == "__main__":
    '''
    face_predictor_path = "shape_predictor_68_face_landmarks.dat"
    mean_face_path = "20words_mean_face.npy"
    #origin_clip_path = "trainval/0af00UcTOSc/50002.mp4"
    origin_clip_path = "trainval/ZzugJPASNB8/50001.mp4"
    #origin_clip_path = "test/zuYzOn0U2PY/00001.mp4"
    #origin_clip_path = "test/wzkFoetpSSM/00002.mp4"
    mouth_roi_path = "./roi.mp4"
    preprocess_video(origin_clip_path, mouth_roi_path, face_predictor_path, mean_face_path)
    '''

    run()
    #save_files('trainval', 'trainval_id.txt')
    #save_files('test', 'test_id.txt')
    #save_files('pretrain', 'pretrain_id.txt')

