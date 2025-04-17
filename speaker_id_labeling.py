# 等间隔采样(抽k帧) -> 人脸特征提取，利用深度学习模型（如FaceNet, ArcFace等）从视频帧中提取人脸特征。
# 这些模型能够生成固定长度的向量（嵌入），用于表示每张人脸的独特特征  -> 使用聚类算法（例如K-means，DBSCAN等）对提取出的特征向量进行聚类，
# 目的是将属于同一个说话人的不同视频片段聚集在一起 -> 根据聚类结果为每个群集分配一个唯一的ID作为说话人身份标签。
# 对于一些边界情况或难以分类的样本，可能需要人工审查和修正 -> 模型迭代与优化
# http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
# http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import dlib
import glob
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

def extract_frames(video_path, k=None):
    """
    Extract frames from a video at a fixed interval.
    Args:
        video_path: Path to the input video file.
        k: Number of frames to sample.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    key_frames = []
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        # if frame_count % frame_interval == 0:
        #     key_frames.append(frame)
        key_frames.append(frame)
        frame_count += 1
    cap.release()
    if k is None or k == 0:
        return key_frames
    # return key_frames[:k]
    # return [key_frames[i] for i in np.random.choice(range(len(key_frames)), k, replace=False)]
    return [key_frames[i] for i in np.linspace(0, len(key_frames) - 1, k, dtype=int)]


## using dlib
# def get_face_embeddings(frames):
#     features = []
#     detector = dlib.get_frontal_face_detector()
#     sp = dlib.shape_predictor("dlib/shape_predictor_68_face_landmarks.dat")
#     facerec = dlib.face_recognition_model_v1("dlib/dlib_face_recognition_resnet_model_v1.dat")
#     for frame in frames:
#         # grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         dets = detector(frame, 0)  # 上采样次数为0
#         for det in dets:
#             # 关键点检测
#             shape = sp(frame, det)
#             # 特征提取
#             face_descriptor = facerec.compute_face_descriptor(frame, shape)  # 128维的人脸描述子向量
#             features.append(np.array(face_descriptor))
#     if len(features) == 0:
#         return None
#     return np.array(features).mean(axis=0)


## using facenet
def get_face_embeddings(frames):
    features = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=160, margin=0, device=device)   # 模型预训练的默认尺寸160x160 
    resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_crop = mtcnn(frame)
        if img_crop is None:
            continue
        img_emb = resnet(img_crop.unsqueeze(0).cuda()).detach().cpu().numpy()   # (1, 512)
        features.append(img_emb.squeeze(0))
    if len(features) == 0:
        return None
    return np.array(features).mean(axis=0)


def cluster_embeddings(embeddings, eps=0.5, min_samples=5, n_clusters=5, cluster_alg='dbscan'):
    """
    Cluster face embeddings using AgglomerativeClustering.
    Args:
        embeddings: A numpy array of face embeddings.
        n_clusters: Number of clusters to form.
        affinity: Metric used to compute the linkage.
        linkage: Linkage criterion to use.
        cluster_alg: Clustering algorithm to use ('dbscan', 'kmeans', or 'agglomerative').
    Returns:
        labels: Cluster labels for each embedding.
    """
    # print(embeddings.shape)
     # 标准化特征向量
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    if cluster_alg == 'dbscan':
        # 如果两个点之间的距离小于或等于eps，则它们被认为是邻居
        # eps决定了聚类的密度和范围。较小的eps值会导致更紧密的聚类，而较大的eps值则会将更多的点视为邻居，从而形成更大的聚类。
        # min_samples是一个核心点必须拥有的最小邻居数（包括自身），才能被视为核心点。核心点是聚类的中心，它们周围有足够的密度。
        # min_samples决定了聚类的最小密度。较大的min_samples值会形成更密集的聚类，而较小的min_samples值则会允许更稀疏的聚类。
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')   # euclidean
    elif cluster_alg == 'kmeans':
        clustering = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    elif cluster_alg == 'agglomerative':
        metric = 'euclidean'  # l1, l2, euclidean, manhattan, cosine(余弦距离[0, 2])
        linkage = 'ward'  # ward, average, complete, single
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)
        # clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, metric=metric, linkage=linkage)
    else:
        raise ValueError('Invalid cluster algorithm')
    labels = clustering.fit_predict(embeddings_scaled)
    return labels


def assign_labels_to_videos(video_files, cluster_labels):
    """
    Assign cluster labels to videos based on their frames.
    Args:
        video_files: Directory containing frames for each video.
        cluster_labels: Cluster labels for all the embeddings.
    Returns:
        video_labels: A dictionary mapping video names to cluster labels.
    """
    video_labels = {}
    for video_name, label in zip(video_files, cluster_labels):
        if video_name not in video_labels:
            video_labels[video_name] = []
        video_labels[video_name].append(label)
    # Assign the most frequent label to the video
    for video_name in video_labels:
        video_labels[video_name] = max(set(video_labels[video_name]), key=video_labels[video_name].count)
    return video_labels


def visualize_embeddings(embeddings, labels):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    # tsne = TSNE(n_components=2, perplexity=30)
    # embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', marker='o')
    plt.title('Clustering')
    plt.xlabel('PCA Component1')
    plt.ylabel('PCA Component2')
    plt.colorbar(label='Cluster Label')
    plt.show()


# 为了更好地选择eps和min_samples，可以绘制K-距离图(K-distance plot)
# 计算每个点的第K个最近邻的距离。
# 绘制K-距离图，选择一个“肘部”点作为eps
from sklearn.neighbors import NearestNeighbors
def plot_k_distance_graph(embeddings, k=4):
    """
    Plot the K-distance graph to help choose the eps parameter for DBSCAN.
    Args:
        embeddings: A numpy array of face embeddings.
        k: The number of neighbors to consider.
    """
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    distances = np.sort(distances[:, k-1], axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title('K-distance Graph')
    plt.xlabel('Points sorted by distance')
    plt.ylabel('k-distance')
    plt.grid(True)
    plt.show()


# 对于包含多个说话人的簇，可以使用子聚类算法(如KMeans或DBSCAN)进行进一步的拆分
def split_clusters(embeddings, labels, video_indices, threshold=2):
    """
    Split clusters that contain multiple speakers.
    Args:
        embeddings: A numpy array of face embeddings.
        labels: Cluster labels for each embedding.
        video_indices: Video indices corresponding to each embedding.
        threshold: Minimum number of unique speakers in a cluster to consider splitting.
    Returns:
        new_labels: New cluster labels after splitting.
    """
    unique_clusters = np.unique(labels)
    new_labels = np.copy(labels)
    new_label_counter = max(unique_clusters) + 1
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue
        # Get indices of samples in the current cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_embeds = embeddings[cluster_indices]
        cluster_video_indices = [video_indices[i] for i in cluster_indices]
        # Count unique speakers in the cluster
        unique_speakers = set(cluster_video_indices)
        if len(unique_speakers) >= threshold:  # 一个簇中包含的唯一说话人数量的阈值，超过该阈值的簇将被拆分
            # Perform sub-clustering
            sub_labels = cluster_embeddings(cluster_embeds, eps=0.3, min_samples=1, cluster_alg='dbscan')
            for sub_cluster_id in np.unique(sub_labels):
                if sub_cluster_id == -1:
                    continue
                new_label = new_label_counter
                new_label_counter += 1
                sub_cluster_indices = np.where(sub_labels == sub_cluster_id)[0]
                new_labels[cluster_indices[sub_cluster_indices]] = new_label   # 更新新的聚类标签
    return new_labels



# 对于每个簇，计算其特征向量的平均值作为簇中心。
# 使用余弦相似度计算每个簇中心之间的相似度。如果两个簇中心的相似度超过similarity_threshold，则在邻接矩阵中标记为连接。
# 使用并查集(Union-Find)算法来检测图的连通分量。将属于同一连通分量的簇合并为一个新的簇。
# 根据并查集的结果，生成新的簇标签。
from scipy.spatial.distance import cosine
def merge_clusters(embeddings, labels, video_indices, similarity_threshold=0.7):
    unique_clusters = np.unique(labels)
    cluster_centers = {}
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_embeds = embeddings[cluster_indices]
        cluster_center = np.mean(cluster_embeds, axis=0)
        cluster_centers[cluster_id] = cluster_center

    # 构建邻接矩阵
    num_clusters = len(cluster_centers)
    adjacency_matrix = np.zeros((num_clusters, num_clusters))
    cluster_ids = list(cluster_centers.keys())
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            center_i = cluster_centers[cluster_ids[i]]
            center_j = cluster_centers[cluster_ids[j]]
            similarity = 1 - cosine(center_i, center_j)
            if similarity > similarity_threshold:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    # 使用并查集合并簇
    parent = {cluster_id: cluster_id for cluster_id in cluster_ids}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            if adjacency_matrix[i, j] == 1:
                union(cluster_ids[i], cluster_ids[j])

    # 生成新的簇标签
    new_labels = np.copy(labels)
    new_label_counter = 0
    new_label_map = {}
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue
        root_id = find(cluster_id)
        if root_id not in new_label_map:
            new_label_map[root_id] = new_label_counter
            new_label_counter += 1
        new_labels[labels == cluster_id] = new_label_map[root_id]

    return new_labels




if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)

    vid_files = glob.glob(os.path.join(r'D:\BaiduNetdiskDownload\lrs3_test_v0.4\test', '*', '*.mp4'))
    print(len(vid_files))
    face_embs = []
    video_indices = []
    for vid_file in vid_files[:200]:
        frames = extract_frames(vid_file, 10)
        face_emb = get_face_embeddings(frames)
        if face_emb is None:
            continue
        print(vid_file)
        face_embs.append(face_emb)
        video_indices.append(os.path.basename(os.path.dirname(vid_file)))

    embs = np.vstack(face_embs)
    # clusters = cluster_embeddings(embs, n_clusters=6, cluster_alg='kmeans')
    # clusters = cluster_embeddings(embs, n_clusters=6, cluster_alg='agglomerative')
    clusters = cluster_embeddings(embs, eps=0.5, min_samples=1, cluster_alg='dbscan')   
    print(clusters, len(set(video_indices)))

    spk_ids = dict()
    for cluster_id in np.unique(clusters):
        if cluster_id == -1:
            continue
        # 获取该簇的所有视频文件名
        cluster_vids = [video_indices[i] for i, label in enumerate(clusters) if label == cluster_id]
        # 为该簇分配一个说话人ID
        for vid_file in cluster_vids:
            spk_ids[vid_file] = f"Speaker_{cluster_id}"
    print(spk_ids, len(spk_ids), len(set(spk_ids.keys())), len(set(spk_ids.values())))
    
    # spk_ids = assign_labels_to_videos(video_indices, clusters)
    # print(spk_ids, len(spk_ids))

    # 可视化聚类结果
    # visualize_embeddings(embs, clusters)

    
    # 后处理：同一个簇中包含不同说话人(split)；同一个说话人包含于不同簇中(merge)
    
    # 找到样本数量最多的簇
    # unique_labels, counts = np.unique(clusters, return_counts=True)
    # largest_cluster_label = unique_labels[np.argmax(counts)]
    # print('largest_cluster_label:', largest_cluster_label)
    # print('largest_cluster_samples:', [video_indices[i] for i in range(len(clusters)) if clusters[i] == largest_cluster_label])

    new_clusters = split_clusters(embs, clusters, video_indices, 2)
    print(new_clusters)
    spk_ids = dict()
    for cluster_id in np.unique(new_clusters):
        if cluster_id == -1:
            continue
        # 获取该簇的所有视频文件名
        cluster_vids = [video_indices[i] for i, label in enumerate(new_clusters) if label == cluster_id]
        # 为该簇分配一个说话人ID
        for vid_file in cluster_vids:
            spk_ids[vid_file] = f"Speaker_{cluster_id}"
    print(spk_ids, len(spk_ids), len(set(spk_ids.keys())), len(set(spk_ids.values())))
