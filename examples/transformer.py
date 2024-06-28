# XLA_FLAGS=--xla_force_host_platform_device_count=8 python -m examples/shardlib_transformer
from shardlib.shardtypes import bf16, bool_, f32, pytree_dataclass, typed_shard_map, u32, make_shardings
from shardlib import shardtypes
shardtypes.register_with_typeguard()
import shardlib.shardops as shardops
from jax.sharding import Mesh
from jax.experimental import mesh_utils
import jax
import jax.numpy as jnp
import einops

# make distributed later
MESH = Mesh(mesh_utils.create_device_mesh([8], jax.devices()), ('d'))

@pytree_dataclass
class MultiHeadAttention:
    q: f32['num_heads d_model head_dim']
    k: f32['num_heads d_model head_dim']
    v: f32['num_heads d_model head_dim']

@pytree_dataclass
class MLP:
    up: f32['d_model hidden']
    down: f32['hidden d_model']

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

batch = 8
seq = 12
d_model = 64
hidden = 4*d_model
head_dim = 16
num_heads = 4

with MESH:
    def rms_norm_forward(x: f32[b'batch/d seq d_model'], w: RMSNorm) -> f32[b'batch/d seq d_model']:
        return w.gain * x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + w.eps)


    def attention_forward(x: f32[b'batch/d seq d_model'], w: MultiHeadAttention) -> f32[b'batch/d seq d_model']:
        Q = shardops.einsum_unreduced(
            'batch/d seq d_model, num_heads d_model head_dim -> batch/d num_heads seq head_dim', 
            x, w.q
            )
        K = shardops.einsum_unreduced(
            'batch/d seq d_model, num_heads d_model head_dim -> batch/d num_heads seq head_dim', 
            x, w.k
            )
        V = shardops.einsum_unreduced(
            'batch/d seq d_model, num_heads d_model head_dim -> batch/d num_heads seq head_dim', 
            x, w.v
            )
        logits = shardops.einsum_unreduced(
            'batch/d num_heads seq1 head_dim, batch/d num_heads seq2 head_dim -> batch/d num_heads seq1 seq2',
            Q, K
            )
        weights = jax.nn.softmax(jnp.tril(logits))

        unflattened_out = shardops.einsum_unreduced(
            'batch/d num_heads seq1 seq2, batch/d num_heads seq2 head_dim -> batch/d num_heads seq1 head_dim',
            weights, V
        )

        return einops.rearrange(unflattened_out, 'b nh s hd -> b s (nh hd)')

    def mlp_forward(x: f32[b'batch/d seq d_model'], w: MLP) -> f32[b'batchd/ seq d_model']:
        hidden_preact = shardops.einsum_unreduced(
            'batch/d seq d_model, d_model hidden -> batch/d seq hidden',
            x, w.up
        )
        hidden = jax.nn.relu(hidden_preact)

        out = shardops.einsum_unreduced('batch/d seq hidden, hidden d_model -> batch/d seq d_model', hidden, w.down)

        return out
    
    # def rms_norm_forward(x: f32[b'batch/d seq d_model'], w: RMSNorm) -> 

    @typed_shard_map
    def transformer_block_forward(x: f32[b'batch/d seq d_model'], w: TransformerBlock) -> f32[b'batch/d seq d_model']:
        x = x + rms_norm_forward(attention_forward(x, w.attention), w.norm1)
        x = x + rms_norm_forward(mlp_forward(x, w.mlp), w.norm2)

        return x

    # init dummy weights
    w_norm1 = RMSNorm(gain=jnp.array(1, dtype=jnp.float32), eps=jnp.array(1e-5, dtype=jnp.float32))
    w_mha = MultiHeadAttention(
        q=jnp.zeros((num_heads, d_model, head_dim), dtype=jnp.float32),
        k=jnp.zeros((num_heads, d_model, head_dim), dtype=jnp.float32),
        v=jnp.zeros((num_heads, d_model, head_dim), dtype=jnp.float32),
    )
    w_norm2 = RMSNorm(gain=jnp.array(1, dtype=jnp.float32), eps=jnp.array(1e-5, dtype=jnp.float32))
    w_mlp = MLP(
        up=jnp.zeros((d_model, hidden), dtype=jnp.float32), 
        down=jnp.zeros((hidden, d_model), dtype=jnp.float32)
        )
    w = TransformerBlock(
        norm1=w_norm1,
        attention=w_mha,
        norm2=w_norm2,
        mlp=w_mlp,
    )

    x1 = transformer_block_forward(jnp.zeros((batch, seq, d_model), dtype=jnp.float32), w)

    assert (x1.shape==(batch, seq, d_model))
    print("great success!")