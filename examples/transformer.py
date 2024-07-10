# XLA_FLAGS=--xla_force_host_platform_device_count=24 python -m examples.transformer
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
from typing import Any, Tuple

# `d` is data parallel axis, `t` is tensor parallel axis
MESH = Mesh(mesh_utils.create_device_mesh([3, 4, 2], jax.devices()), ('p', 'd', 't'))

@dataclass
class ModelArgs:
    num_microbatches: int = 3
    batch: int = 8
    layers: int = 6
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
class TransformerStage(TransformerBlock):
    ln1: f32['layers_per_stage d_model/d']
    ln2: f32['layers_per_stage d_model/d']

    qkv: f32['layers_per_stage num_heads/t d_model 3head_dim/d']

    mlp_up: f32['layers_per_stage d_model hidden/t/d']
    mlp_down: f32['layers_per_stage hidden d_model/t/d']

@pytree_dataclass 
class Transformer(TransformerStage):
    ln1: f32['layers/p d_model/d']
    ln2: f32['layers/p d_model/d']

    qkv: f32['layers/p num_heads/t d_model 3head_dim/d']

    mlp_up: f32['layers/p d_model hidden/t/d']
    mlp_down: f32['layers/p hidden d_model/t/d']
    

with MESH:
    @jax.jit
    @typechecked
    def rms_norm(x: f32[b'batch/d seq d_model']) -> f32[b'batch/d seq d_model']:
        return  x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-5)


    @typechecked
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
    def transformer_stage_forward(x: f32[b'batch/d seq d_model/t'], w: TransformerStage) -> f32[b'batch/d seq d_model/t']:
        scan_fn = lambda c, a: (transformer_block_forward(c, a), ())
        y, _ = jax.lax.scan(scan_fn, x, w)
        return y

    @typed_shard_map
    def pipeline_step(
        carries: f32[b'num_stages/p batch/d seq d_model/t'], 
        input: f32[b'batch/d seq d_model/t'], # trust XLA to only materialize this on device 0
        w: Transformer
    ) -> Tuple[f32[b'num_stages/p batch/d seq d_model/t'], f32[b'num_stages/p batch/d seq d_model/t']]:
        num_stages = jax.lax.psum(1, 'p')
        stage_index = jax.lax.axis_index('p')

        carries = jnp.where(
            stage_index==0,
            jnp.expand_dims(input, axis=0),
            carries
        )
        
        pp_stage = jax.vmap(transformer_stage_forward, (0, None), 0)

        stage_outputs = pp_stage(carries, w)

        # trust XLA to not do the ppermute in last pipeline stage
        perm = [(i, (i+1)%num_stages) for i in range(num_stages)]
        new_carries = jax.lax.ppermute(stage_outputs, 'p', perm=perm)

        return new_carries, stage_outputs

    @typechecked
    def transformer_forward(
        x: f32[b'num_microbatches batch/d seq d_model/t'],
        w: Transformer
    ) -> f32[b'num_microbatches batch/d seq d_model/t']: 
        num_microbatches = x.shape[0]
        num_stages = MESH.shape['p']

        init_carries = jnp.zeros((num_stages, *x.shape[1:]))
        padded_x = jnp.concatenate(
            (x, jnp.zeros((num_stages-1, *x.shape[1:]))),
            axis=0
        )

        scan_fn = lambda carries, input: pipeline_step(carries, input, w)

        _, padded_outputs = jax.lax.scan(scan_fn, init_carries, padded_x)

        outputs = padded_outputs[-num_microbatches:, -1]

        return outputs

    cfg = ModelArgs()

    key = jax.random.key(42)
    x = jax.random.normal(key, (cfg.num_microbatches, cfg.batch, cfg.seq, cfg.d_model))
    x = jax.device_put(x, make_shardings(f32['num_microbatches batch/d seq d_model/t']))

    w = Transformer(
        ln1=jnp.ones((cfg.layers, cfg.d_model)),
        ln2=jnp.ones((cfg.layers, cfg.d_model)),
        qkv=jnp.zeros((cfg.layers, cfg.num_heads, cfg.d_model, 3*cfg.head_dim)),
        mlp_up=jnp.zeros((cfg.layers, cfg.d_model, cfg.hidden)),
        mlp_down=jnp.zeros((cfg.layers, cfg.hidden, cfg.d_model))
    )
    w = jax.tree.map(jax.device_put, w, make_shardings(Transformer))

    # block_w = TransformerBlock(
    #     ln1=jnp.ones((cfg.d_model)),
    #     ln2=jnp.ones((cfg.d_model)),
    #     qkv=jnp.zeros((cfg.num_heads, cfg.d_model, 3*cfg.head_dim)),
    #     mlp_up=jnp.zeros((cfg.d_model, cfg.hidden)),
    #     mlp_down=jnp.zeros((cfg.hidden, cfg.d_model))
    # )
    # block_w = jax.tree.map(jax.device_put, block_w, make_shardings(TransformerBlock))

    print("###x:", x.shape, x.sharding)
    y = transformer_forward(x, w)
    print("###y: ", y.shape, y.sharding)
