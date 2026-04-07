import math
from einops import rearrange

import torch
import torch.nn as nn
#https://github.com/Boeun-Kim/MoST/blob/main/model/transformer.py
class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=150):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class PositionalEmbeddingNew(nn.Module):

    def __init__(self, d_model, max_len=150, id_len = 5, part_len = 55):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        self.id_len = id_len
        self.part_len = part_len

    def encode_id(self, id):
        return self.pe[:, id:id+1]
    
    def encode_part(self, x):
        return self.pe[:, self.id_len:self.id_len+x.size(1)]
    
    def encode_frame(self, x):
        return self.pe[:, self.id_len+self.part_len:self.id_len+self.part_len+x.size(1)]
    
class Attention(nn.Module):
    def __init__(self, dim_emb, num_heads=8, qkv_bias=False, attn_do_rate=0.1, proj_do_rate=0.1):
        super().__init__()
        self.dim_emb = dim_emb
        self.num_heads = num_heads
        dim_each_head = dim_emb // num_heads
        self.scale = dim_each_head ** -0.5

        self.qkv = nn.Linear(dim_emb, dim_emb * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_do_rate)
        self.proj = nn.Linear(dim_emb, dim_emb)  
        self.proj_dropout = nn.Dropout(proj_do_rate)

    def forward(self, x, mask=None):

        B, N, C = x.shape  

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x

class CrossAttention(nn.Module):
    def __init__(self, dim_emb, num_heads=8, qkv_bias=False, attn_do_rate=0., proj_do_rate=0.):
        super().__init__()
        self.dim_emb = dim_emb
        self.num_heads = num_heads
        dim_each_head = dim_emb // num_heads
        self.scale = dim_each_head ** -0.5

        self.q = nn.Linear(dim_emb, dim_emb, bias=qkv_bias)
        self.k = nn.Linear(dim_emb, dim_emb, bias=qkv_bias)
        self.v = nn.Linear(dim_emb, dim_emb, bias=qkv_bias)
        
        self.attn_dropout = nn.Dropout(attn_do_rate)
        self.proj = nn.Linear(dim_emb, dim_emb)  
        self.proj_dropout = nn.Dropout(proj_do_rate)

    def forward(self, x, y, z, mask=None):

        B, N, C = x.shape
        _, N2, _ = y.shape

        q = self.q(x)  
        k = self.k(y) 
        v = self.v(z)

        q = q.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k = k.reshape(B, N2, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = v.reshape(B, N2, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x

class CrossAttentionNew(nn.Module):
    def __init__(self, dim_emb, context_dim, num_heads=8, qkv_bias=False, attn_do_rate=0., proj_do_rate=0.):
        super().__init__()
        self.dim_emb = dim_emb
        self.num_heads = num_heads
        dim_each_head = dim_emb // num_heads
        self.scale = dim_each_head ** -0.5

        self.q = nn.Linear(dim_emb, dim_emb, bias=qkv_bias)
        self.k = nn.Linear(context_dim, dim_emb, bias=qkv_bias)
        self.v = nn.Linear(context_dim, dim_emb, bias=qkv_bias)
        
        self.attn_dropout = nn.Dropout(attn_do_rate)
        self.proj = nn.Linear(dim_emb, dim_emb)  
        self.proj_dropout = nn.Dropout(proj_do_rate)

    def forward(self, x, context, mask=None):

        B, N, C = x.shape

        q = self.q(x)  
        k = self.k(context) 
        v = self.v(context)

        q = q.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k = k.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = v.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x
        
class FeedForward(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, do_rate=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(do_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, is_cont_enc = False, num_part=6, num_frame=64, dim_emb=48, 
                num_heads=8, ff_expand=1.0, qkv_bias=False, attn_do_rate=0.1, proj_do_rate=0.1):
        super().__init__()

        # for learnable positional embedding
        self.positional_emb = nn.Parameter(torch.zeros(1, num_frame, num_part, dim_emb))

        self.norm1_sp = nn.LayerNorm(dim_emb)
        self.norm1_tm = nn.LayerNorm(dim_emb*num_part)

        self.attention_sp = Attention(dim_emb, num_heads, qkv_bias, attn_do_rate, proj_do_rate)
        self.attention_tm = Attention(dim_emb*num_part, num_heads, qkv_bias, attn_do_rate, proj_do_rate)
  
        self.norm2 = nn.LayerNorm(dim_emb*num_part)
        self.feedforward = FeedForward(in_features=dim_emb*num_part, hidden_features=int(dim_emb*num_part*ff_expand), 
                                        out_features=dim_emb*num_part, do_rate=proj_do_rate)

        self.InstanceNorm = nn.InstanceNorm2d(dim_emb, affine=True) 


    def forward(self, x, mask=None, last_block=False):

        b, f, p, c = x.shape

        if last_block:
            x = rearrange(x, 'b f p c -> b c f p')
            x = self.InstanceNorm(x)
            x = rearrange(x, 'b c f p -> b f p c')

        ## part-MHA
        x_sp = rearrange(x, 'b f p c -> (b f) p c', )
        pos_emb = self.positional_emb.repeat(b, 1,1,1)
        pos_emb = rearrange(pos_emb, 'b f p c -> (b f) p c', b=b, f=f)
        x_sp = x_sp + pos_emb

        x_sp = x_sp + self.attention_sp(self.norm1_sp(x_sp), mask=None)
  
        ## temporal-MHA
        x_tm = rearrange(x_sp, '(b f) p c -> b f (p c)', b=b, f=f)
        pos_emb = rearrange(pos_emb, '(b f) p c -> b f (p c)', b=b, f=f)
        x_tm = x_tm + pos_emb

        x_tm = x_tm + self.attention_tm(self.norm1_tm(x_tm), mask=mask)
        x_out = x_tm

        x_out = x_out + self.feedforward(self.norm2(x_out))
        x_out = rearrange(x_out, 'b f (p c)  -> b f p c', p=p)

        return x_out

class TransformerEncoderNew(nn.Module):
    def __init__(self, context_dim=48, num_person = 5, num_part=6, num_frame=64, dim_emb=48, 
                num_heads=8, ff_expand=1.0, qkv_bias=False, attn_do_rate=0.1, proj_do_rate=0.1):
        super().__init__()

        # for learnable positional embedding
        #self.all_emb = PositionalEmbeddingNew(dim_emb, num_person + num_part + num_frame, id_len=num_person, part_len=num_part)
        
        self.pos_emb = nn.Parameter(torch.zeros(1, num_frame, num_part, dim_emb))

        self.norm1_sp = nn.LayerNorm(dim_emb)
        self.norm2_sp = nn.LayerNorm(dim_emb)
        self.norm1_tm = nn.LayerNorm(dim_emb*num_part)
        self.norm2_tm = nn.LayerNorm(dim_emb*num_part)

        self.attention_sp = Attention(dim_emb, num_heads, qkv_bias, attn_do_rate, proj_do_rate)
        #self.cross_attention_sp = CrossAttentionNew(dim_emb, context_dim, num_heads, attn_do_rate, proj_do_rate)
        
        
        self.attention_tm = Attention(dim_emb*num_part, num_heads, qkv_bias, attn_do_rate, proj_do_rate)
        #self.cross_attention_tm = CrossAttentionNew(dim_emb*num_part, context_dim, num_heads, attn_do_rate, proj_do_rate)
        
        self.norm2 = nn.LayerNorm(dim_emb*num_part)
        self.feedforward = FeedForward(in_features=dim_emb*num_part, hidden_features=int(dim_emb*num_part*ff_expand), 
                                        out_features=dim_emb*num_part, do_rate=proj_do_rate)


    def forward(self, x, id = 0, context=None, mask=None, last_block=False):

        b, f, p, c = x.shape
        
        # identity emb
        # positional_emb: 1, p, f, p, d
        id_pos_emb = self.pos_emb
        
        ## part-MHA
        x_sp = rearrange(x, 'b f p c -> (b f) p c', )
        #pos_emb = self.all_emb.encode_part(x_sp).repeat(b*f, 1, 1)
        pos_emb = id_pos_emb.repeat(b, 1,1,1)
        pos_emb = rearrange(pos_emb, 'b f p c -> (b f) p c', b=b, f=f)
        x_sp = x_sp + pos_emb

        x_sp = x_sp + self.attention_sp(self.norm1_sp(x_sp), mask=None)
        
        ## temporal-MHA
        x_tm = rearrange(x_sp, '(b f) p c -> b f (p c)', b=b, f=f)
        #pos_emb = self.all_emb.encode_frame(x_tm).repeat(b, 1, 1)
        #print('pos_emb', pos_emb.shape, 'x_tm', x_tm.shape)
        pos_emb = rearrange(pos_emb, '(b f) p c -> b f (p c)', b=b, f=f)
        x_tm = x_tm + pos_emb

        x_tm = x_tm + self.attention_tm(self.norm1_tm(x_tm), mask=mask)
        #x_tm = x_tm + self.cross_attention_tm(self.norm2_tm(x_tm), context)
        x_out = x_tm

        x_out = x_out + self.feedforward(self.norm2(x_out))
        x_out = rearrange(x_out, 'b f (p c)  -> b f p c', p=p)

        return x_out

class AdaIN(nn.Module): 
    def __init__(self, sty_dim, out_dim):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(out_dim, affine=False)
        self.fc = nn.Linear(sty_dim, out_dim * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class TransformerModulator(nn.Module):
    def __init__(self, dim, num_heads, num_part):
        super().__init__()
        self.num_part = num_part
        self.pos_encoder = PositionalEmbedding(dim, num_part)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim*num_part)

        self.linear = nn.Linear(dim*num_part, dim*num_part)
        self.cross_attention = CrossAttention(dim, num_heads, attn_do_rate=0.1, proj_do_rate=0.1)
        self.feedforward = FeedForward(in_features=dim*num_part, out_features=dim*num_part, do_rate=0.1)

    def forward(self, query_feat, key_feat, value_feat, block_idx):

        if block_idx == 0:
            pos_emb = self.pos_encoder(query_feat)
            query_feat = query_feat + pos_emb
            key_feat = key_feat + pos_emb
            value_feat = value_feat + pos_emb

        attended_value = self.cross_attention(self.norm1(query_feat), self.norm2(key_feat), self.norm3(value_feat), mask=None)
        attended_value = rearrange(attended_value, 'b p c -> b (p c)')
        value_feat = rearrange(value_feat, 'b p c -> b (p c)')

        attended_value = self.linear(value_feat) + attended_value
        attended_value = attended_value + self.feedforward(self.norm4(attended_value))

        return attended_value

class TransformerModulatorNew(nn.Module):
    def __init__(self, dim, num_heads, num_part, num_frame):
        super().__init__()
        self.num_part = num_part
        self.pos_encoder = PositionalEmbedding(dim, num_part)
        self.frame_encoder = PositionalEmbedding(dim, num_frame)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim*num_part)

        self.linear = nn.Linear(dim*num_part, dim*num_part)
        self.cross_attention = CrossAttention(dim, num_heads, attn_do_rate=0.1, proj_do_rate=0.1)
        self.feedforward = FeedForward(in_features=dim*num_part, out_features=dim*num_part, do_rate=0.1)

    def forward(self, query_feat, key_feat, value_feat, block_idx):
        
        B, T, P, D = query_feat.shape
        # query_feat: BK, T, 55, 16
        # key_feat: BK, 4T, 55, 16
        # value_feat: BK, 4T, 55, 16
        if block_idx == 0:
            query_feat = rearrange(query_feat, 'b f p c -> (b f) p c', )
            pos_emb = self.pos_encoder(query_feat)
            query_feat = query_feat + pos_emb
            query_feat = query_feat.reshape(B*T, -1, D)
            
            key_feat = rearrange(key_feat, 'b f p c -> (b f) p c', )
            key_feat = key_feat + pos_emb
            key_feat = key_feat.reshape(B, 4, T, -1, D).permute(0, 2, 1, 3, 4)
            key_feat = rearrange(key_feat, 'b f h p c -> (b f) (h p) c', )
            
            value_feat = rearrange(value_feat, 'b f p c -> (b f) p c', )
            value_feat = value_feat + pos_emb
            value_feat = value_feat.reshape(B, 4, T, -1, D).permute(0, 2, 1, 3, 4)
            value_feat = rearrange(value_feat, 'b f h p c -> (b f) (h p) c', )
        
        attended_value = self.cross_attention(self.norm1(query_feat), self.norm2(key_feat), self.norm3(value_feat), mask=None)
        attended_value = rearrange(attended_value, 'b p c -> b (p c)')
        #value_feat = rearrange(value_feat, 'b p c -> b (p c)')

        #attended_value = self.linear(value_feat) + attended_value
        attended_value = attended_value + self.feedforward(self.norm4(attended_value))
        attended_value = attended_value.reshape(B, T, P, D)
        
        return attended_value
    
class TransformerDecoder(nn.Module):
    def __init__(self, num_part=6, num_frame=64, dim_emb=48, 
                num_heads=8, ff_expand=1.0, qkv_bias=False, attn_do_rate=0.1, proj_do_rate=0.1):
        super().__init__()
        self.num_part=num_part

        # For learnable positional embedding
        self.positional_emb = nn.Parameter(torch.zeros(1, num_frame, num_part, dim_emb))

        # For fixed positional embedding (no use)
        self.tm_pos_encoder = PositionalEmbedding(num_part*dim_emb, num_frame)
        self.sp_pos_encoder = PositionalEmbedding(dim_emb, num_part)

        self.attention_sp = Attention(dim_emb, num_heads, qkv_bias, attn_do_rate, proj_do_rate)
        self.attention_tm = Attention(dim_emb*num_part, num_heads, qkv_bias, attn_do_rate, proj_do_rate)

        self.norm1_sp = nn.LayerNorm(dim_emb)
        self.norm1_tm = nn.LayerNorm(dim_emb*num_part)
        self.norm2_sp = nn.LayerNorm(dim_emb)
        self.norm2_tm = nn.LayerNorm(dim_emb*num_part)

        self.norm4 = nn.LayerNorm(dim_emb*num_part)
        self.feedforward = FeedForward(in_features=dim_emb*num_part, hidden_features=int(dim_emb*num_part*ff_expand), 
                                        out_features=dim_emb*num_part, do_rate=proj_do_rate)

        self.AdaIN = AdaIN(dim_emb*num_part, dim_emb)


    def forward(self, cnt_dynamics, modulated_sty_feat, cnt_mask=None):

        b, f, p, c = cnt_dynamics.shape
        
        cnt = rearrange(cnt_dynamics, 'b f p c  -> b c f p', )
        cnt = self.AdaIN(cnt, modulated_sty_feat)
            
        ## spatial-MHA
        cnt_sp = rearrange(cnt, 'b c f p -> (b f) p c')
        pos_emb = self.positional_emb.repeat(b, 1,1,1)
        pos_emb = rearrange(pos_emb, 'b f p c -> (b f) p c', b=b, f=f)
        cnt_sp = cnt_sp + pos_emb

        cnt_sp = cnt_sp + self.attention_sp(self.norm1_sp(cnt_sp), mask=None)
  
        ## temporal-MHA
        cnt_tm = rearrange(cnt_sp, '(b f) p c -> b f (p c)', b=b, f=f)
        pos_emb = rearrange(pos_emb, '(b f) p c -> b f (p c)', b=b, f=f)
        cnt_tm = cnt_tm + pos_emb

        out = cnt_tm + self.attention_tm(self.norm1_tm(cnt_tm), mask=cnt_mask)

        out = out + self.feedforward(self.norm4(out))
        out = rearrange(out, 'b f (p c)  -> b f p c', p=p)

        return out
    