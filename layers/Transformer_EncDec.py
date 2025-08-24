import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x

##############STAR Module###############
class STAR(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        """
        STar Aggregate-Redistribute Module
        """

        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape  # input [32,11,128]

        # set FFN
        combined_mean = F.gelu(self.gen1(input)) # (B,D,L)-->(B,D,L)
        combined_mean = self.gen2(combined_mean) # (B,D,L)-->(B,D,L_core)

        # stochastic pooling
        if self.training:
            ratio = F.softmax(combined_mean, dim=1) # 在通道方向上执行softmax,为随机池化生成一个概率权重: (B,D,L_core)-->(B,D,L_core)
            ratio = ratio.permute(0, 2, 1) # (B,D,L_core)--permute->(B,L_core,D)
            ratio = ratio.reshape(-1, channels) # 转换为2维, 便于进行采样: (B,L_core,D)--reshape-->(B*L_core,D)
            indices = torch.multinomial(ratio, 1) # 从多项分布ratio的每一行中抽取一个样本,返回值是采样得到的类别的索引: (B*L_core,1); 输入如果是一维张量,它表示每个类别的概率;如果是二维张量,每行表示一个概率分布
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1) # (B*L_core,1)--view--> (B,L_core,1)--permute-->(B,1,L_core)
            combined_mean = torch.gather(combined_mean, 1, indices) # 根据索引indices在D方向上选择对应的通道元素(理解为:选择重要的通道信息): (B,D,L_core)--gather-->(B,1,L_core)    # gather函数不了解的看这个:https://zhuanlan.zhihu.com/p/661293803
            combined_mean = combined_mean.repeat(1, channels, 1) # 复制D份,将随机选择的core表示应用到所有通道上: (B,1,L_core)--repeat-->(B,D,L_core)
        else:
            weight = F.softmax(combined_mean, dim=1) # 处于非训练模式时, 首先通过softmax生成一个权重分布:(B,D,L_core)-->(B,D,L_core)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1) # 直接在D方向上进行加权求和, 然后复制D份: (B,D,L_core)--sum-->(B,1,L_core)--repeat-->(B,D,L_core)

        # mlp fusion
        combined_mean_cat = torch.cat([input, combined_mean], -1) # (B,D,L)--cat--(B,D,L_core)==(B,D,L+L_core)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat)) # (B,D,L+L_core)-->(B,D,L)
        combined_mean_cat = self.gen4(combined_mean_cat) # (B,D,L)-->(B,D,L)
        output = combined_mean_cat

        print("STAR attention is called!")

        return output
    

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model,channels, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = STAR(d_series=d_model, d_core=channels)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        '''new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )   #自注意力操作 STAR应该在这改。
        '''
        #(B,D,L) [32,11,128]STAR需要这样输入
        new_x = self.attention(x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        #return self.norm2(x + y), attn
        return(x + y), None  # attn置None


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        #x [32,11,128]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
