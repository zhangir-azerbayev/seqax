# XLA_FLAGS=--xla_force_host_platform_device_count=8 python -m examples.transformer
import copy
from dataclasses import dataclass, field

from shardlib.shardtypes import f32, make_shardings, pytree_dataclass, typed_shard_map, typechecked
from shardlib import shardtypes
shardtypes.register_with_typeguard()
import shardlib.shardops as shardops
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
import jax
import jax.numpy as jnp
import einops
from typing import Any

# `d` is data parallel axis, `t` is tensor parallel axis
MESH = Mesh(mesh_utils.create_device_mesh([4, 2], jax.devices()), ('d', 't'))

@dataclass
class ModelArgs:
    batch: int = 8
    layers: int = 4
    seq: int = 12
    d_model: int = 64
    num_heads: int = 4
    hidden: int = field(init=False)
    head_dim: int = field(init=False)

    def __post_init__(self):
        self.hidden = 4*self.d_model
        self.head_dim = self.d_model // self.num_heads

@pytree_dataclass 
class TransformerBlock:
    ln1: f32['d_model/d']
    ln2: f32['d_model/d']

    qkv: f32['num_heads/t d_model 3head_dim/d']

    mlp_up: f32['d_model hidden/t/d']
    mlp_down: f32['hidden d_model/t/d']

@pytree_dataclass 
class Transformer(TransformerBlock):
    ln1: f32['layers d_model/d']
    ln2: f32['layers d_model/d']

    qkv: f32['layers num_heads/t d_model 3head_dim/d']

    mlp_up: f32['layers d_model hidden/t/d']
    mlp_down: f32['layers hidden d_model/t/d']
    

with MESH:
    @jax.jit
    @typechecked
    def rms_norm(x: f32[b'batch/d seq d_model']) -> f32[b'batch/d seq d_model']:
        return  x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-5)


    @typed_shard_map
    def transformer_block_forward(x: f32[b'batch/d seq d_model/t'], w: TransformerBlock) -> f32[b'batch/d seq d_model/t']:
        # first RMSnorm
        w_ln1 = shardops.all_gather('d_model/d -> d_model', w.ln1)
        gx = shardops.all_gather('batch/d seq d_model/t -> batch/d seq d_model', x)
        normed_x = w_ln1 * rms_norm(gx)

        # attention layer
        w_qkv = shardops.all_gather('num_heads/t d_model 3head_dim/d -> num_heads/t d_model 3head_dim', w.qkv)
        QKV = shardops.einsum_unreduced(
            'batch/d seq d_model, num_heads/t d_model 3head_dim -> batch/d num_heads/t seq 3head_dim', 
            normed_x, w_qkv
        )
        Q, K, V = jnp.split(QKV, 3, axis=-1)
        logits = shardops.einsum_unreduced(
            'batch/d num_heads/t seq head_dim, batch/d num_heads/t seq_ head_dim -> batch/d num_heads/t seq seq_',
            Q, K
        )
        attn_scores = jax.nn.softmax(jnp.tril(logits))

        unflattened_attn_out = shardops.einsum_unreduced(
            'batch/d num_heads/t seq seq_, batch/d num_heads/t seq_ head_dim -> batch/d num_heads/t seq head_dim',
            attn_scores, V
        )
        attn_out = einops.rearrange(unflattened_attn_out, 'b nh s hd -> b s (nh hd)')

        # residual connection
        x1 = x + attn_out

        # second RMSnorm
        w_ln2 = shardops.all_gather('d_model/d -> d_model', w.ln2)
        gx1 = shardops.all_gather('batch/d seq d_model/t -> batch/d seq d_model', x1)
        normed_x1 = w_ln2 * rms_norm(gx1)

        # MLP block
        w_up = shardops.all_gather('d_model hidden/t/d -> d_model hidden/t', w.mlp_up)
        hidden_preact = shardops.einsum_unreduced(
            'batch/d seq d_model, d_model hidden/t -> batch/d seq hidden/t',
            normed_x1, w_up
        )
        hidden_act = jax.nn.relu(hidden_preact)

        hidden_act = shardops.all_gather('batch/d seq hidden/t -> batch/d seq hidden', hidden_act)
        w_down = shardops.all_gather('hidden d_model/t/d -> hidden d_model/t', w.mlp_down)
        mlp_out = shardops.einsum_unreduced(
            'batch/d seq hidden, hidden d_model/t -> batch/d seq d_model/t', 
            hidden_act, w_down
        )

        # residual connection
        x2 = x1 + mlp_out

        return x2

    @typechecked
    def transformer_forward(x: f32[b'batch/d seq d_model/t'], w: Transformer) -> f32[b'batch/d seq d_model/t']:
        scan_fn = lambda c, a: transformer_block_forward(c, a), ()
        y, _ = jax.lax.scan(scan_fn, x, w)
        return y

    cfg = ModelArgs()

    key = jax.random.key(42)
    x = jax.random.normal(key, (cfg.batch, cfg.seq, cfg.d_model))
    x = jax.device_put(x, make_shardings(f32['batch/d seq d_model/t']))

    w = Transformer(
        ln1=jnp.ones((cfg.layers, cfg.d_model)),
        ln2=jnp.ones((cfg.layers, cfg.d_model)),
        qkv=jnp.zeros((cfg.layers, cfg.num_heads, cfg.d_model, 3*cfg.head_dim)),
        mlp_up=jnp.zeros((cfg.layers, cfg.d_model, cfg.hidden)),
        mlp_down=jnp.zeros((cfg.layers, cfg.hidden, cfg.d_model))
    )
    block_w = TransformerBlock(
        ln1=jnp.ones((cfg.d_model)),
        ln2=jnp.ones((cfg.d_model)),
        qkv=jnp.zeros((cfg.num_heads, cfg.d_model, 3*cfg.head_dim)),
        mlp_up=jnp.zeros((cfg.d_model, cfg.hidden)),
        mlp_down=jnp.zeros((cfg.hidden, cfg.d_model))
    )

    block_w = jax.tree.map(jax.device_put, block_w, make_shardings(TransformerBlock))

    print("###x:", x.shape, x.sharding)
    y = transformer_block_forward(x, block_w)
    print("###y: ", y.shape, y.sharding)
