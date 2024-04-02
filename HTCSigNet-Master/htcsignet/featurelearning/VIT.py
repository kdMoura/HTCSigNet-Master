## from https://github.com/lucidrains/vit-pytorch
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    # 在执行fn之前执行一个Layer Norm
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # 前馈神经网络 = 2个全连接层
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放因子

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # x: [bs, 197, 1024]   197 = 1个Cls + 196个patch  1024就是每一个patch需要转为1024长度的向量
        # self.to_qkv(x)将x向量映射到长度为1024*3
        # chunk: qkv 最后是一个元祖，tuple，长度是3，每个元素形状：[1, 197, 1024]
        # 直接用x配合一个Linear生成qkv，再切分为3块
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # 再把qkv分别拆分开来
        # q: [1, 16, 197, 64]  k: [1, 16, 197, 64]  v: [1, 16, 197, 64]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # q * k转置 除以根号d_k
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # softmax得到每个token对于其他token的attention系数
        attn = self.attend(dots)
        # * v  [1, 16, 197, 64]
        out = torch.matmul(attn, v)
        # [1, 197, 1024]
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):  # 堆叠多个Encoder  depth个
            self.layers.append(nn.ModuleList([
                # 每个encoder = Attention(Multi-Head Attention) + FeedForward(MLP)
                # PreNorm：指在fn(Attention/FeedForward)之前执行一个Layer Norm
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)  # 224*224
        patch_height, patch_width = pair(patch_size)  # 16 * 16

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)  # 得到多少个token  14x14=196
        patch_dim = channels * patch_height * patch_width  # 3x16x16 = 768  patch展平后的维度
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),  # 把所有的patch拉平->768维
            nn.Linear(patch_dim, dim),  # 映射到encoder需要的维度768->1024
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 生成所有token和Cls的位置编码
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 生成Cls的初始化参数
        self.dropout = nn.Dropout(emb_dropout)  # embedding后面一般会接的一个Dropout

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)  # encoder

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(  # CLS多分类输出部分
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # img: [1, 3, 224, 224] x = [1, 196, 1024]
        # 生成每张图片的Patch Embedding
        # 图片的每一个通道切分为Token +  将3个channel的所有Token拉直，拉到一个1维，长度为768的向量 + 接一个线性层映射到encoder需要的维度768->1024
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape  # b = 1   n = 196

        # 为每张图片生成一个Cls符号 [1, 1, 1024]
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # [1, 197, 1024]   将每张图片的Cls符号和Patch Embedding进行拼接
        x = torch.cat((cls_tokens, x), dim=1)
        # 初始化位置编码 再和(Cls和Patch Embedding)对应位置相加
        x += self.pos_embedding[:, :(n + 1)]
        # embedding后接一个Dropout
        x = self.dropout(x)

        # 将最终的Embedding输入Encoder  x: [1, 197, 1024]  -> [1, 197, 1024]
        x = self.transformer(x)

        # self.pool = 'cls' 所以取第一个输出直接进行多分类 [1, 1024]
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)  # 恒等映射 [1, 1024]

        # Cls Head 多分类 [1, cls_num]
        return self.mlp_head(x)
        #return x


if __name__ == '__main__':
    v = ViT(
        image_size=224,  # 输入图像的大小
        patch_size=16,  # 每个token/patch的大小16x16
        num_classes=1000,  # 多分类
        dim=1024,  # encoder规定的输入的维度
        depth=6,  # Encoder的个数
        heads=16,  # 多头注意力机制的head个数
        mlp_dim=2048,  # mlp的维度
        dropout=0.1,  #
        emb_dropout=0.1  # embedding一半会接一个dropout
    )
    img = torch.randn(1, 3, 224, 224)
    preds = v(img)  # (1, 1000)
    print(preds.shape)