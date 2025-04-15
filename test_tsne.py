import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import umap

# 生成一些示例高维数据
np.random.seed(42)
X = np.random.randn(100, 128)  # 100个样本，每个样本128维


def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - sum(np.dot(v, b) * b for b in basis)
        if (w > 1e-10).any():
            basis.append(w / np.linalg.norm(w))
    return np.array(basis)


def gram_schmidt2(vectors):
    # 确保输入是NumPy数组
    vectors = np.asarray(vectors)
    # 初始化正交向量集
    u = np.zeros_like(vectors)
    # 正交化过程
    for i in range(vectors.shape[0]):
        u[i] = vectors[i]
        for j in range(i):
            proj = np.dot(vectors[i], u[j]) / np.dot(u[j], u[j])
            u[i] -= proj * u[j]
        # 归一化（可选）
        u[i] /= np.linalg.norm(u[i])
    return u


# # 进行 Gram-Schmidt 正交化
# orthogonal_vectors = gram_schmidt(X)
# # 验证正交性
# print("正交向量是否正交:", np.allclose(orthogonal_vectors.T @ orthogonal_vectors, np.eye(500)))
#
# # 应用格拉姆-施密特正交化
# orth_vectors = gram_schmidt2(X)
# # 验证正交性
# for i in range(orth_vectors.shape[0]):
#     for j in range(i + 1, orth_vectors.shape[0]):
#         print(f"Inner product between u_{i + 1} and u_{j + 1}: {np.dot(orth_vectors[i], orth_vectors[j])}")


# reducer = TSNE(n_components=2,
#               learning_rate=100.0,
#               n_iter=3000,
#               init='pca',
#               random_state=42)

# reducer = umap.UMAP(n_components=2)

# embedding = reducer.fit_transform(X)
# print(embedding.shape, type(embedding))

# plt.figure(figsize=(8, 6))
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c='blue', marker='o', label='visual', alpha=0.5)
# plt.scatter(np.random.rand(100), np.random.rand(100), c='red', marker='o', label='audio', alpha=0.5)
# plt.title('t-SNE Visualization')
# # plt.xlabel('Component 1')
# # plt.ylabel('Component 2')
# # plt.minorticks_on()
# plt.legend()
# plt.tight_layout()
# plt.savefig('tsne.pdf', bbox_inches='tight')
# plt.show()
# plt.close()

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot()
# ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c='blue', marker='o', label='visual',  alpha=0.5)
# ax.scatter(np.random.rand(100), np.random.rand(100), c='brown', marker='^', label='audio', alpha=0.5)
# ax.scatter(np.random.rand(100)*2, np.random.rand(100)*2, c='green', marker='d', label='text', alpha=0.5)
# # ax.set_title('t-SNE Visualization')
# ax.set_xticks([])
# ax.set_yticks([])
# plt.legend()
# plt.tight_layout()
# plt.savefig('tsne.pdf', bbox_inches='tight')
# plt.show()
# plt.close()



def tsne_feats(feat_files):    # repr: (num_feat, feat_size)
    # reducer = TSNE(n_components=3,
    #               learning_rate=100.0,
    #               n_iter=3000,
    #               init='pca',
    #               random_state=42)
    reducer = umap.UMAP(n_components=2,
                        metric='cosine'
                       )

    all_feats = []
    for feat_file in feat_files:
        feats = np.loadtxt(feat_file)[:2000]  # (N, D)
        r_tsne = reducer.fit_transform(X=feats)  # (B, 2)
        all_feats.append(r_tsne)
    del reducer

    fig, ax = plt.subplots(figsize=(8, 6))
    ## 'vc.txt', 'ac.txt', 'vp.txt', 'ap.txt', 'vid.txt', 'aid.txt'
    labels = feat_files
    # labels = ['V shared', 'A shared', 'V private', 'A private', 'V id', 'A id'][:len(tsne_files)]
    colors = ['blue', 'red', 'green', 'cyan', 'orange', 'purple'][:len(feat_files)]
    markers = ['o', 's', '^', 'v', '+', 'x'][:len(feat_files)]
    for i, X_tsne in enumerate(all_feats):
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors[i], marker=markers[i], label=labels[i], alpha=0.6)
    # ax.set_title('t-SNE Visualization')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.legend()
    plt.tight_layout()
    # plt.savefig('tsne.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


def tsne_multi_feats(feat_files):
    reducer = TSNE(n_components=2,
                  learning_rate='auto',
                  # learning_rate=100.0,
                  # n_iter=2000,
                  metric='cosine',
                  init='pca',
                  random_state=42)
    # reducer = umap.UMAP(n_components=2,
    #                     # init='pca',   # 默认谱聚类, pca对应euclidean距离
    #                     min_dist=0.3,
    #                     metric='cosine',    # euclidean, cosine, correlation (1 - Pearson或Spearman相关系数)
    #                     random_state=42)

    all_feats = []
    feat_labels = []
    for i, feat_file in enumerate(feat_files):
        feats = np.loadtxt(feat_file)[:1000]   # (N, D)
        all_feats.append(feats)
        feat_labels.append([i] * len(feats))
    all_feat = np.vstack(all_feats)
    feat_label = np.hstack(feat_labels)
    print(all_feat.shape, feat_label.shape)

    X_tsne = reducer.fit_transform(X=all_feat)   # (B, 2)
    print(X_tsne.shape)
    del reducer

    fig, ax = plt.subplots(figsize=(8, 6))
    ## 'vc.txt', 'ac.txt', 'vp.txt', 'ap.txt', 'vid.txt', 'aid.txt'
    labels = list(feat_files.values())
    # labels = ['V shared', 'A shared', 'V private', 'A private', 'V id', 'A id'][:len(feat_files)]
    # colors = ['blue', 'red', 'green', 'cyan', 'orange', 'purple', 'brown', 'yellow'][:len(feat_files)]
    colors = plt.cm.get_cmap('Paired')(range(len(feat_files)+6))
    markers = ['o', 's', '^', 'v', '+', 'x', 'd', '.'][:len(feat_files)]
    for i in range(len(feat_files)):
        idx = np.where(feat_label == i)[0]
        if i >= 4:  # 跳过红色
            ax.scatter(X_tsne[idx, 0], X_tsne[idx, 1], c=colors[i+2], marker=markers[i], label=labels[i], alpha=0.6)
        else:
            ax.scatter(X_tsne[idx, 0], X_tsne[idx, 1], c=colors[i], marker=markers[i], label=labels[i], alpha=0.6)
    # ax.set_title('t-SNE Visualization')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.legend(prop={'size': 13})
    plt.tight_layout()
    plt.savefig('tsne_zh_cosine-s2-1000-noorth.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


def draw_feats(tsne_file='tsne.txt'):
    X_tsne = np.loadtxt(tsne_file)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c='blue', marker='o', label='visual',  alpha=0.5)
    # ax.scatter(np.random.rand(100), np.random.rand(100), c='brown', marker='^', label='audio', alpha=0.5)
    # ax.scatter(np.random.rand(100)*2, np.random.rand(100)*2, c='green', marker='d', label='text', alpha=0.5)
    # ax.set_title('t-SNE Visualization')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig('tsne.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


def draw_multi_feats(tsne_files):
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot()
    fig, ax = plt.subplots(figsize=(8, 6))
    ## 'vc.txt', 'ac.txt', 'vp.txt', 'ap.txt', 'vid.txt', 'aid.txt'
    labels = tsne_files
    # labels = ['V shared', 'A shared', 'V private', 'A private', 'V id', 'A id'][:len(tsne_files)]
    colors = ['blue', 'red', 'green', 'cyan', 'orange', 'purple'][:len(tsne_files)]
    markers = ['o', 's', '^', 'v', '+', 'x'][:len(tsne_files)]
    for i, f in enumerate(tsne_files):
        X_tsne = np.loadtxt(f)
        # X_tsne = X_tsne / np.linalg.norm(X_tsne)
        # X_tsne = (X_tsne - np.mean(X_tsne, axis=0)) / np.std(X_tsne, axis=0, keepdims=True)
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors[i], marker=markers[i], label=labels[i], alpha=0.5)
    # ax.set_title('t-SNE Visualization')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.legend()
    plt.tight_layout()
    # plt.savefig('tsne.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


def draw_multi_3Dfeats(tsne_files):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ## 'vc.txt', 'ac.txt', 'vp.txt', 'ap.txt', 'vid.txt', 'aid.txt'
    labels = ['V shared', 'A shared', 'V private', 'A private', 'V id', 'A id']
    colors = ['blue', 'red', 'green', 'cyan', 'orange', 'purple']
    markers = ['o', 's', '^', 'v', '+', 'x']
    # labels = ['visual shared', 'visual private', 'visual id']
    # colors = ['blue', 'red', 'green']
    # markers = ['o', 's', 'd']
    for i, f in enumerate(tsne_files):
        X_tsne = np.loadtxt(f)
        # X_tsne = X_tsne / np.linalg.norm(X_tsne, axis=-1, keepdims=True)
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=colors[i], marker=markers[i], label=labels[i], alpha=0.5)
    # ax.set_title('t-SNE Visualization')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.legend()
    plt.tight_layout()
    # plt.savefig('tsne.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    # tsne_feats([os.path.join('dump_feats', f) for f in ['vc.txt', 'ac.txt', 'vp.txt', 'ap.txt', 'vid.txt', 'aid.txt']])
    # tsne_multi_feats([os.path.join('dump_feats', f) for f in ['vc.txt', 'ac.txt', 'vp.txt', 'ap.txt', 'vid.txt', 'aid.txt']])

    # file_dics = dict(zip([os.path.join('dump_feats', f) for f in ['vc_zh_s2.txt', 'ac_zh_s2.txt', 'vp_zh_s2.txt', 'ap_zh_s2.txt', 'vid_zh_s2.txt', 'aid_zh_s2.txt']],
    #                  ['visual shared', 'audio shared', 'visual specific', 'audio specific', 'visual ID', 'audio ID']))
    file_dics = dict(zip([os.path.join('dump_feats', f) for f in ['vc_zh_s2_noorth.txt', 'ac_zh_s2_noorth.txt', 'vp_zh_s2_noorth.txt', 'ap_zh_s2_noorth.txt', 'vid_zh_s2_noorth.txt', 'aid_zh_s2_noorth.txt']],
                         ['visual shared', 'audio shared', 'visual specific', 'audio specific', 'visual ID',
                          'audio ID']))
    tsne_multi_feats(file_dics)

    print('Done')
