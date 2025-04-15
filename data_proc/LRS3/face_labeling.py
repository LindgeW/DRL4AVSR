# 等间隔采样(抽k帧) -> 人脸特征提取，利用深度学习模型（如FaceNet, ArcFace等）从视频帧中提取人脸特征。
# 这些模型能够生成固定长度的向量（嵌入），用于表示每张人脸的独特特征  -> 使用聚类算法（例如K-means，DBSCAN等）对提取出的特征向量进行聚类，
# 目的是将属于同一个说话人的不同视频片段聚集在一起 -> 根据聚类结果为每个群集分配一个唯一的ID作为说话人身份标签。
# 对于一些边界情况或难以分类的样本，可能需要人工审查和修正 -> 模型迭代与优化


from sklearn.cluster import DBSCAN
import numpy as np
import cv2
import os
import dlib
import glob


def extract_frames(video_path, frame_interval=5):
    """
    Extract frames from a video at a fixed interval.
    Args:
        video_path: Path to the input video file.
        frame_interval: Number of frames to skip between extractions.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    key_frames = []
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            key_frames.append(frame)
        frame_count += 1
    cap.release()
    return key_frames[:5]


def get_face_embeddings(frames):
    features = []
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    for frame in frames:
        grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        dets = detector(grey, 1)
        for det in dets:
            # 关键点检测
            shape = sp(grey, det)
            # 特征提取
            face_descriptor = facerec.compute_face_descriptor(frame, shape)
            features.append(np.array(face_descriptor))
    return np.array(features) / len(features)


def cluster_embeddings(embeddings, eps=0.5, min_samples=5):
    """
    Cluster face embeddings using DBSCAN.
    Args:
        embeddings: A numpy array of face embeddings.
        eps: Maximum distance between two samples to be considered in the same cluster.
        min_samples: Minimum number of samples in a cluster.
    Returns:
        labels: Cluster labels for each embedding.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = clustering.fit_predict(embeddings)
    return labels


if __name__ == '__main__':
    #vid_files = glob.glob(os.path.join('vid', '*.mp4'))
    vid_files = []
    with open('test_id.txt', 'r') as fin:
        for f in fin:
            ID, path = f.strip().split(' ')
            vid_files.append(path.replace('.npy', '.mp4'))
    print(len(vid_files))
    face_embs = []
    for vid_file in vid_files:
        frames = extract_frames(vid_file)
        face_emb = get_face_embeddings(frames)
        face_embs.append(face_emb)
    labels = cluster_embeddings(np.array(face_emb), eps=0.5, min_samples=5)
    print(labels)
