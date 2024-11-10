import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import create_model
from transformers import GPT2LMHeadModel


class Attention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0, 'embedding dimension by be divisible by number of heads'
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.scale = self.head_size ** -0.5
        
        self.register_buffer('mask',torch.tril(torch.ones(1, 1, self.seq_len,self.seq_len)))
        
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)
        
        
    def forward(self, x):
        b,t,c = x.shape

        q = self.q_proj(x)  # (batch_size, seq_len, embed_dim)
        k = self.k_proj(x)  # (batch_size, seq_len, embed_dim)
        v = self.v_proj(x)  # (batch_size, seq_len, embed_dim)
        
        q = q.view(b, t, self.n_heads, self.head_size).transpose(1, 2)  # (batch_size, n_heads, seq_len, head_size)
        k = k.view(b, t, self.n_heads, self.head_size).transpose(1, 2)  # (batch_size, n_heads, seq_len, head_size)
        v = v.view(b, t, self.n_heads, self.head_size).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        attn_scores = attn_scores.masked_fill(self.mask == 0, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)  # (batch_size, n_heads, seq_len, seq_len)
        
        attn_probs = self.attn_dropout(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)  # (batch_size, n_heads, seq_len, head_size)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, self.embed_dim)
        
        out = self.proj(attn_output)
        out = self.resid_dropout(out)
        
        return out
    

class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0, 'Embedding dimension must be divisible by the number of heads'
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)  # Query projection
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)  # Key projection
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)  # Value projection

        # Scaling factor for attention
        self.scale = self.head_size ** -0.5
        
        # Output projection for the result of attention
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        # Dropout layers for attention and residual connections
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)
        
        # Apply weight initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, q, k, v):
        """
        Args:
        q: Tensor of shape (batch_size, seq_len_q, embed_dim) - Queries from the decoder
        k: Tensor of shape (batch_size, seq_len_kv, embed_dim) - Keys from the encoder
        v: Tensor of shape (batch_size, seq_len_kv, embed_dim) - Values from the encoder
        
        Returns:
        out: Tensor of shape (batch_size, seq_len_q, embed_dim) - Cross-attention output
        """
        b_q, t_q, _ = q.shape
        b_kv, t_kv, _ = k.shape
        
        q = self.q_proj(q)  # (batch_size, seq_len_q, embed_dim)
        k = self.k_proj(k)  # (batch_size, seq_len_kv, embed_dim)
        v = self.v_proj(v)  # (batch_size, seq_len_kv, embed_dim)
        
        q = q.view(b_q, t_q, self.n_heads, self.head_size).transpose(1, 2)  # (batch_size, n_heads, seq_len_q, head_size)
        k = k.view(b_kv, t_kv, self.n_heads, self.head_size).transpose(1, 2)  # (batch_size, n_heads, seq_len_kv, head_size)
        v = v.view(b_kv, t_kv, self.n_heads, self.head_size).transpose(1, 2)  # (batch_size, n_heads, seq_len_kv, head_size)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch_size, n_heads, seq_len_q, seq_len_kv)
        
        attn_probs = F.softmax(attn_scores, dim=-1)  # (batch_size, n_heads, seq_len_q, seq_len_kv)
        
        attn_probs = self.attn_dropout(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)  # (batch_size, n_heads, seq_len_q, head_size)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(b_q, t_q, self.embed_dim)
        
        out = self.proj(attn_output)        
        out = self.resid_dropout(out)
        
        return out
    

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.mlp_ratio = config.mlp_ratio
        self.mlp_dropout = config.mlp_dropout
        
        self.fc = nn.Linear(self.embed_dim, self.embed_dim * self.mlp_ratio)
        self.proj = nn.Linear(self.embed_dim * self.mlp_ratio, self.embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(self.mlp_dropout)
        
    def forward(self,x):
        x = self.fc(x)
        x = self.act(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x
    


class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.ln_1 = nn.LayerNorm(self.embed_dim)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(self.embed_dim)
        self.mlp = MLP(config)
        self.ln_3 = nn.LayerNorm(self.embed_dim)
        self.cross_attn = CrossAttention(config)
        
    def forward(self,x,enc_out):
        x = x+self.attn(self.ln_1(x))
        x = x+self.cross_attn(self.ln_2(x), enc_out, enc_out)
        x = x+self.mlp(self.ln_3(x))
        return x
    


class CaptioningModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        self.config = config
        
        vit = create_model('vit_base_patch16_224',pretrained=True,num_classes=0)
        self.patch_embed = vit.patch_embed
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = vit.cls_token
        embed_len = num_patches + vit.num_prefix_tokens
        self.pos_embed = vit.pos_embed
        self.pos_drop = nn.Dropout(p=0.)
        
        self.blocks = nn.ModuleList([vit.blocks[i] for i in range(config.depth)])
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.embed_dim),
            wpe = nn.Embedding(config.seq_len,config.embed_dim),
            drop = nn.Dropout(config.emb_dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.depth)]),
            ln_f = nn.LayerNorm(config.embed_dim)
        ))
        self.lm_head = nn.Linear(config.embed_dim,config.vocab_size,bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        
    def _pos_embed(self,x):
        pos_embed = self.pos_embed
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + pos_embed
        return self.pos_drop(x)
    
    def pretrained_layers_trainable(self,trainable=False):
        layers = [
            self.cls_token, self.patch_embed, self.pos_embed, self.blocks,
            self.transformer.wte, self.transformer.wpe,
            self.transformer.ln_f, self.lm_head
        ]
        gpt_layers = [[
            self.transformer.h[i].ln_1,self.transformer.h[i].ln_2,
            self.transformer.h[i].attn,self.transformer.h[i].mlp
        ] for i in range(self.config.depth)]
        for l in gpt_layers:
            layers.extend(l)
        
        for layer in layers:
            if not isinstance(layer,nn.Parameter):
                for p in layer.parameters():
                    p.requires_grad = trainable
            else:
                layer.requires_grad = trainable
                
        total_frozen_params = sum([p.numel() for p in self.parameters() if not p.requires_grad])
        print(f'{total_frozen_params=}')
        
    def unfreeze_gpt_layers(self,):
        gpt_layers = [[
            self.transformer.h[i].ln_1,self.transformer.h[i].ln_2,
            self.transformer.h[i].attn,self.transformer.h[i].mlp
        ] for i in range(self.config.depth)]
        flatten = []
        for l in gpt_layers:
            flatten.extend(l)
            
        for layer in flatten:
            if not isinstance(layer,nn.Parameter):
                for p in layer.parameters():
                    p.requires_grad = True
            else:
                layer.requires_grad = True
        
    @classmethod    
    def from_pretrained(self,config):
        model = CaptioningModel(config)
        sd = model.state_dict()
        keys = sd.keys()
        ignore_matches = ['blocks.','cross_attn.','ln_3','cls_token','pos_embed','patch_embed.','.attn.mask']
        vit_keys = [key for key in keys if any(match in key for match in ignore_matches)]
        gpt_keys = [key for key in keys if key not in vit_keys]
        
        gpt2_small = GPT2LMHeadModel.from_pretrained('gpt2')
        sd_hf = gpt2_small.state_dict()
        hf_keys = sd_hf.keys()
        hf_keys = [k for k in hf_keys if not k.endswith('.attn.masked_bias')]
        hf_keys = [k for k in hf_keys if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        for k in hf_keys:
            if any(match in k for match in ignore_matches):
                continue
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
            
        model.load_state_dict(sd)
        
        return model
    
    def forward(self,image,input_ids,labels=None):
        image = self.patch_embed(image)
        image = self._pos_embed(image)
        
        token_embeddings = self.transformer.wte(input_ids) # batch x seq_len
        pos_embs = torch.arange(0,input_ids.size(1)).to(input_ids.device)
        positional_embeddings = self.transformer.wpe(pos_embs)
        input_ids = self.transformer.drop(token_embeddings+positional_embeddings)
        
        for i in range(self.config.depth):
            image = self.blocks[i](image)
            input_ids = self.transformer.h[i](input_ids, image)
        
        input_ids = self.transformer.ln_f(input_ids)

        lm_logits = self.lm_head(input_ids)
        
        if labels is not None:
            # Calculate loss
            loss = F.cross_entropy(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            return loss
        
        return lm_logits
    
    def generate(self, image, sequence, max_tokens=50, temperature=1.0, deterministic=False):
            """
            Generate captions for a given image by predicting tokens.

            Args:
            image: The input image to caption.
            sequence: The initial sequence (e.g., start token) to generate the caption.
            max_tokens: The maximum number of tokens to generate.
            temperature: A value to control the randomness of generation.
            deterministic: If True, uses greedy search (picks the most likely token).
            
            Returns:
            The generated sequence.
            """
            # Initialize the sequence with the provided input
            input_ids = sequence
            
            # Process image and create image embeddings
            image = self.patch_embed(image)
            image = self._pos_embed(image)
            
            # Add initial embedding tokens (e.g., <start>)
            token_embeddings = self.transformer.wte(input_ids)
            pos_embs = torch.arange(0, input_ids.size(1)).to(input_ids.device)
            positional_embeddings = self.transformer.wpe(pos_embs)
            input_ids = self.transformer.drop(token_embeddings + positional_embeddings)
            
            for _ in range(max_tokens):
                # Process the image through the vision transformer blocks
                for i in range(self.config.depth):
                    image = self.blocks[i](image)
                    input_ids = self.transformer.h[i](input_ids, image)
                    
                input_ids = self.transformer.ln_f(input_ids)
                
                # Get logits for the next token
                logits = self.lm_head(input_ids[:, -1, :])  # Take the last token's logits
                
                # Apply temperature scaling
                logits = logits / temperature
                
                # If deterministic, pick the most likely token
                if deterministic:
                    next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
                else:
                    # Sample from the logits
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)  # Sample the next token
                
                # Append the predicted token to the sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # If the <end> token is generated, stop
                if next_token.item() == self.config.vocab_size - 1:  # Assuming <end> token is the last token in vocab
                    break
            
            return input_ids