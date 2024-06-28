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
import copy

# Inside function bodies, the following abbrvs. for named dims are used
# b: batch
# s: seq
# d: d_model 
# h: hidden 
# nh: num_heads
# hd: head_dim

MESH = Mesh(mesh_utils.create_device_mesh([4, 2], jax.devices()), ('d', 't'))

@pytree_dataclass
class Weights:
    w: f32['hidden1 hidden2/t']

with MESH:
    @typed_shard_map
    def tp_layer(x: f32[b'batch/d hidden1'], w: f32[b'hidden1 hidden2/t']) -> f32[b'batch/d hidden2/t']:
        out = shardops.einsum_unreduced(
            'b/d h1, h1 h2/t -> b/d h2/t',
            x, w
        )
        
        return jax.nn.relu(out)
    
    key = jax.random.key(42)
    _, sbkey1, sbkey2 = jax.random.split(key, 3)

    w = Weights(w=jax.random.normal(sbkey1, (32, 16), dtype=jnp.float32))
    
    out = tp_layer(
        jax.random.normal(sbkey2, (8, 32), dtype=jnp.float32), 
        w.w
    )
    print('success!')