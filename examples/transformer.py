# XLA_FLAGS=--xla_force_host_platform_device_count=8 python -m examples.transformer
import copy
from dataclasses import dataclass, field

from shardlib.shardtypes import f32, pytree_dataclass, typed_shard_map
from shardlib import shardtypes
shardtypes.register_with_typeguard()
import shardlib.shardops as shardops
from jax.sharding import Mesh
from jax.experimental import mesh_utils
import jax
import jax.numpy as jnp
import einops

# `d` is data parallel axis, `t` is tensor parallel axis
MESH = Mesh(mesh_utils.create_device_mesh([4, 2], jax.devices()), ('d', 't'))

@pytree_dataclass
class MultiHeadAttention:
    qkv: f32['num_heads/t d_model 3head_dim/d']

@pytree_dataclass
class MLP:
    up: f32['d_model hidden/t/d']
    down: f32['hidden d_model/t/d']

@pytree_dataclass 
class RMSNorm:
    gain: f32['']
    eps: f32['']

@pytree_dataclass
class TransformerBlock:
    norm1: RMSNorm
    attention: MultiHeadAttention
    norm2: RMSNorm
    mlp: MLP

@dataclass
class ModelArgs:
    batch: int = 8
    seq: int = 12
    d_model: int = 64
    num_heads: int = 4
    hidden: int = field(init=False)
    head_dim: int = field(init=False)

    def __post_init__(self):
        self.hidden = 4*self.d_model
        self.head_dim = self.d_model // self.num_heads

with MESH:
    def rms_norm_forward(x: f32[b'batch seq d_model'], w: RMSNorm) -> f32[b'batch seq d_model']:
        return w.gain * x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + w.eps)

    @typed_shard_map
    def attention_forward(
            x: f32[b'batch/d seq d_model'], w: MultiHeadAttention
        ) -> f32[b'batch/d seq d_model/t']:
        print("inside attention", x)
        w_qkv = shardops.all_gather(
            'num_heads/t d_model 3head_dim/d -> num_heads/t d_model 3head_dim', 
            w.qkv
        )
        QKV = shardops.einsum_unreduced(
            'batch/d seq d_model, num_heads/t d_model 3head_dim -> batch/d num_heads/t seq 3head_dim', 
            x, w_qkv
        )
        Q, K, V = jnp.split(QKV, 3, axis=-1)

        logits = shardops.einsum_unreduced(
            'batch/d num_heads/t seq1 head_dim, batch/d num_heads/t seq2 head_dim -> batch/d num_heads/t seq1 seq2',
            Q, K
        )
        attn_scores = jax.nn.softmax(jnp.tril(logits))

        unflattened_out = shardops.einsum_unreduced(
            'batch/d num_heads/t seq1 seq2, batch/d num_heads/t seq2 head_dim -> batch/d num_heads/t seq1 head_dim',
            attn_scores, V
        )

        return einops.rearrange(unflattened_out, 'b nh s hd -> b s (nh hd)')

    @typed_shard_map
    def mlp_forward(x: f32[b'batch/d seq d_model'], w: MLP) -> f32[b'batch/d seq d_model/t']:
        w_up = shardops.all_gather('d_model hidden/t/d -> d_model hidden/t', w.up)
        hidden_preact = shardops.einsum_unreduced(
            'batch/d seq d_model, d_model hidden/t -> batch/d seq hidden/t',
            x, w_up
        )
        hidden_act = jax.nn.relu(hidden_preact)

        hidden_act = shardops.all_gather('batch/d seq hidden/t -> batch/d seq hidden', hidden_act)
        w_down = shardops.all_gather('hidden d_model/t/d -> hidden d_model/t', w.down)
        out = shardops.einsum_unreduced(
            'batch/d seq hidden, hidden d_model/t -> batch/d seq d_model/t', 
            hidden_act, w_down
        )

        return out
    
    def transformer_block_forward(
            x: f32[b'batch seq d_model'], 
            w: TransformerBlock
        ) -> f32[b'batch seq d_model']:

        x = x + attention_forward(rms_norm_forward(x, w.norm1), w.attention)
        x = x + mlp_forward(rms_norm_forward(x, w.norm2), w.mlp)

        return x


    # init dummy weights and do forward pass
    cfg = ModelArgs()
    w_norm1 = RMSNorm(
        gain=jnp.array(1, dtype=jnp.float32), 
        eps=jnp.array(1e-5, dtype=jnp.float32)
    )
    w_mha = MultiHeadAttention(
        qkv=jnp.zeros((cfg.num_heads, cfg.d_model, 3*cfg.head_dim), dtype=jnp.float32),
    )
    w_norm2 = copy.deepcopy(w_norm1)
    w_mlp = MLP(
        up=jnp.zeros((cfg.d_model, cfg.hidden), dtype=jnp.float32), 
        down=jnp.zeros((cfg.hidden, cfg.d_model), dtype=jnp.float32)
    )
    w = TransformerBlock(
        norm1=w_norm1,
        attention=w_mha,
        norm2=w_norm2,
        mlp=w_mlp,
    )

    y = transformer_block_forward(jnp.zeros((cfg.batch, cfg.seq, cfg.d_model), dtype=jnp.float32), w)

    assert (y.shape==(cfg.batch, cfg.seq, cfg.d_model))
    print("great success!")
