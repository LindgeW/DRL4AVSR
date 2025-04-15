import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


arr = np.loadtxt('fusion_ws_32_100.txt').reshape(-1, 32, 100)[:, :, 1:-2]
# arr = np.loadtxt('fusion_ws_32_100_5db.txt').reshape(-1, 32, 100)[:, :, 1:-2]
# arr = np.loadtxt('fusion_ws_32_100_0db.txt').reshape(-1, 32, 100)[:, :, 1:-2]
# arr = np.loadtxt('fusion_ws_32_100_-5db.txt').reshape(-1, 32, 100)[:, :, 1:-2]

# arr = np.loadtxt('enc_dec_clean_avg.txt').reshape(-1, 32, 26)[:, :, :-4]
# arr = np.loadtxt('enc_dec_5db_avg.txt').reshape(-1, 32, 26)[:, :, :-4]
# arr = np.loadtxt('enc_dec_0db_avg.txt').reshape(-1, 32, 26)[:, :, :-4]
# arr = np.loadtxt('enc_dec_-5db_avg.txt').reshape(-1, 32, 26)[:, :, :-4]
# arr = arr.transpose((0, 2, 1))   ## transpose

print(arr.shape)
print(dir(plt.cm))
# cmaps = ['Accent', 'viridis', 'Blues', 'YlGnBu', 'RdYlGn', 'PuBuGn']

# colors = plt.cm.get_cmap('viridis')(range(len(arr)))
# colors = ['teal', 'blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown']

# 'viridis'
# 'plasma'
# 'inferno'
# 'magma'
# 'cividis'
# 'jet'
# 'rainbow'
# 'coolwarm'
# 'seismic'
# 'terrain'
cmaps = []
for i in range(len(arr)):
    # cmaps.append(LinearSegmentedColormap.from_list('custom_cmap' + str(i), ['white', colors[i]]))
    cmaps.append(LinearSegmentedColormap.from_list('custom_cmap' + str(i), plt.cm.get_cmap('viridis_r')(np.linspace(0, 1, 256))))

f, ax = plt.subplots(figsize=(9, 5))
for i in range(len(cmaps)):
    sns.heatmap(arr[i], ax=ax, square=True, cmap=cmaps[i], fmt='.4f', cbar=False)   # viridis


# x_ticks = np.arange(0, 50, 5)  # 每隔10个刻度显示一个标签
# ax.set_xticks(x_ticks + 0.5)  # 设置刻度位置
# ax.set_xticklabels(x_ticks, rotation=0)  # 设置刻度标签
# y_ticks = np.arange(0, 18, 2)  # 每隔5个刻度显示一个标签
# ax.set_yticks(y_ticks + 0.5)  # 设置刻度位置
# ax.set_yticklabels(y_ticks)  # 设置刻度标签

# audio-visual
x_ticks = np.arange(0, 100, 10)  # 每隔10个刻度显示一个标签
ax.set_xticks(x_ticks + 0.5)  # 设置刻度位置
ax.set_xticklabels(x_ticks, rotation=0, fontsize=14)  # 设置刻度标签
ax.set_xlabel('visual frame', fontsize=16)
y_ticks = np.arange(0, 35, 5)  # 每隔5个刻度显示一个标签
ax.set_yticks(y_ticks + 0.5)  # 设置刻度位置
ax.set_yticklabels(y_ticks, rotation=0, fontsize=14)  # 设置刻度标签
ax.set_ylabel('reduced audio frame', fontsize=16)
f.tight_layout()
f.set_tight_layout(True)


# audio-char
# x_ticks = np.arange(0, 25, 5)  # 每隔10个刻度显示一个标签
# ax.set_xticks(x_ticks + 0.5)  # 设置刻度位置
# ax.set_xticklabels(x_ticks, rotation=0, fontsize=14)  # 设置刻度标签
# ax.set_xlabel('character', fontsize=14)
# y_ticks = np.arange(0, 35, 5)  # 每隔5个刻度显示一个标签
# ax.set_yticks(y_ticks + 0.5)  # 设置刻度位置
# ax.set_yticklabels(y_ticks, rotation=0, fontsize=14)  # 设置刻度标签
# ax.set_ylabel('AV-Align frame', fontsize=14)
# f.tight_layout()
# f.set_tight_layout(True)

## transpose
# x_ticks = np.arange(0, 35, 5)  # 每隔10个刻度显示一个标签
# ax.set_xticks(x_ticks + 0.5)  # 设置刻度位置
# ax.set_xticklabels(x_ticks, rotation=0, fontsize=14)  # 设置刻度标签
# ax.set_xlabel('AV-align frame', fontsize=14)
# y_ticks = np.arange(0, 25, 5)  # 每隔5个刻度显示一个标签
# ax.set_yticks(y_ticks + 0.5)  # 设置刻度位置
# ax.set_yticklabels(y_ticks, rotation=0, fontsize=14)  # 设置刻度标签
# ax.set_ylabel('character', fontsize=14)
# f.tight_layout()
# f.set_tight_layout(True)


plt.tight_layout()
plt.savefig('attn_ws_clean.pdf', bbox_inches='tight')
# plt.savefig('attn_ws_5db.pdf', bbox_inches='tight')
# plt.savefig('attn_ws_0db.pdf', bbox_inches='tight')
# plt.savefig('attn_ws_-5db.pdf', bbox_inches='tight')

# plt.savefig('enc_dec_clean_3.pdf', bbox_inches='tight')
# plt.savefig('enc_dec_5db_3.pdf', bbox_inches='tight')
# plt.savefig('enc_dec_0db_3.pdf', bbox_inches='tight')
# plt.savefig('enc_dec_-5db_3.pdf', bbox_inches='tight')
plt.show()
plt.close()