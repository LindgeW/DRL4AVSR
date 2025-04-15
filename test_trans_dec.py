import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer

# 创建一个 TransformerDecoder
decoder_layer = TransformerDecoderLayer(d_model=512, nhead=4)
transformer_decoder = TransformerDecoder(decoder_layer, num_layers=3)

# 定义一个 hook 函数来捕获 cross attention 的 softmax 值
def hook(module, input, output):
    attn_weights = module.multihead_attn.attn_weights
    # 注意：attn_weights 可能需要进一步处理，例如应用 softmax 函数
    # 保存或处理注意力权重
    print(attn_weights)

# 注册 hook 到 TransformerDecoder 的第一个层
transformer_decoder.layers[0].register_forward_hook(hook)

# 假设你有输入数据和状态
input_data = torch.randn(10, 32, 512)  # (seq_len, batch_size, d_model)
memory = torch.randn(10, 32, 512)  # (seq_len, batch_size, d_model)
state = None

# 前向传播
output = transformer_decoder(input_data, memory)

print(transformer_decoder)