# XLA_FLAGS=--xla_force_host_platform_device_count=8 python -m shardlib_example
from shardlib.shardtypes import bf16, bool_, f32, pytree_dataclass, typed_shard_map, u32, make_shardings
from shardlib import shardtypes
shardtypes.register_with_typeguard()
import shardlib.shardops as shardops
from jax.sharding import Mesh
from jax.experimental import mesh_utils
import jax
import jax.numpy as jnp

MESH = Mesh(mesh_utils.create_device_mesh([8], jax.devices()), ('d'))

@pytree_dataclass
class Weights:
    w1: f32['in hidden1/d']
    w2: f32['hidden1 hidden2/d']
    w3: f32['hidden2/d']

with MESH:
    w = Weights(
        w1=jnp.zeros((8, 8), dtype=jnp.float32),
        w2=jnp.zeros((8, 8), dtype=jnp.float32),
        w3=jnp.zeros((8,), dtype=jnp.float32),
    )
    w = jax.tree.map(jax.device_put, w, make_shardings(Weights))

    @typed_shard_map
    def forward_pass(x: f32[b'batch/d in'], w: Weights) -> f32[b'batch/d']:
        # print("input array:")
        # print(x)
        # the core idea of sharded data parallel is that weights are gathered
        # just prior to their use
        w1 = shardops.all_gather('in hidden1/d -> in hidden1', w.w1)

        y = jax.nn.relu(shardops.einsum_unreduced('batch/d in, in hidden1 -> batch/d hidden1', x, w1))
        w2 = shardops.all_gather('hidden1 hidden2/d -> hidden1 hidden2', w.w2)
        z = jax.nn.relu(shardops.einsum_unreduced('batch/d hidden1, hidden1 hidden2 -> batch/d hidden2', y, w2))
        w3 = shardops.all_gather('hidden2/d -> hidden2', w.w3)
        return shardops.einsum_unreduced('batch/d hidden2, hidden2 -> batch/d', z, w3)
    
    x = forward_pass(jnp.zeros((32, 8), dtype=jnp.float32), w)
    assert (x.shape==(32,))


