from shardlib.shardtypes import bf16, bool_, f32, pytree_dataclass, typed_shard_map, u32, make_shardings
from shardlib import shardtypes
shardtypes.register_with_typeguard()
import shardlib.shardops as shardops
from jax.sharding import Mesh
from jax.experimental import mesh_utils
import jax
import jax.numpy as jnp
import einops
from typing import List
from dataclasses import asdict

MESH = Mesh(mesh_utils.create_device_mesh([4], jax.devices()), ('p'))

@pytree_dataclass
class MLP:
    up: f32['d_model hidden']
    down: f32['hidden d_model']

@pytree_dataclass
class MLPBlocks:
    up: f32['num_layers/p d_model hidden']
    down: f32['num_layers/p hidden d_model']


def ffn_block(
        input: f32[b'batch d_model'],
        up: f32[b'd_model hidden'],
        down: f32[b'hidden d_model']
    ) -> f32['batch d_model']:
    hidden_preact = shardops.einsum_unreduced(
        'batch d_model, d_model hidden -> batch hidden', 
        input, up
    )

    hidden_act = jax.nn.relu(hidden_preact)

    out = shardops.einsum_unreduced(
        'batch hidden, hidden d_model -> batch d_model',
        hidden_act, down
    )

    return out

with MESH:
    num_stages = 4

    key = jax.random.key(42)
    input = jax.random.normal(key, (2, 8))
    init_carry = jnp.stack([input, *[jnp.zeros_like(input) for _ in range(num_stages-1)]])
    init_carry = jax.device_put(init_carry, make_shardings(f32['num_layers/p batch d_model']))

    blocks = MLPBlocks(
        up=jnp.stack([jnp.eye(8, 16) for _ in range(num_stages)]),
        down=jnp.stack([jnp.eye(16, 8) for _ in range(num_stages)]),
    )
    blocks = jax.tree.map(jax.device_put, blocks, make_shardings(MLPBlocks)) 

    pipeline_step = jax.vmap(ffn_block, (0, 0, 0), 0)

    carry = pipeline_step(init_carry, blocks.up, blocks.down)

    print(init_carry, init_carry.sharding)
    print(blocks.up, blocks.up.sharding)
    print(blocks.down, blocks.down.sharding)
    print(carry, carry.shape, carry.sharding)
