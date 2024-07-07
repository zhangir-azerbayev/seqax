# XLA_FLAGS=--xla_force_host_platform_device_count=4 python -m examples.pp_basic
from dataclasses import dataclass, field

from shardlib.shardtypes import f32, make_shardings, pytree_dataclass, typed_shard_map
from shardlib import shardtypes
shardtypes.register_with_typeguard()
import shardlib.shardops as shardops
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
import jax
import jax.numpy as jnp
import einops
from jax.experimental.shard_map import shard_map
import functools as ft
from typing import Tuple

# 'p' is pipeline parallel axis
MESH = Mesh(mesh_utils.create_device_mesh([4], jax.devices()), ('p'))

# next: add microbatches

@dataclass 
class ModelArgs:
    batch: int = 2
    num_layers: int = 4 # check this works with size of 'p' axis
    d_model: int = 8
    hidden: int = field(init=False)

    def __post_init__(self):
        self.hidden = 2*self.d_model

@pytree_dataclass
class MLP:
    up: f32['d_model hidden']
    down: f32['hidden d_model']

@pytree_dataclass
class MLPBlocks:
    up: f32['num_layers/p d_model hidden']
    down: f32['num_layers/p hidden d_model']

with MESH:
    def ffn_block(
        input: f32[b'batch d_model'],
        w: MLP
    ) -> f32['batch d_model']:
        hidden_preact = shardops.einsum_unreduced(
            'batch d_model, d_model hidden -> batch hidden', 
            input, w.up
        )

        hidden_act = jax.nn.relu(hidden_preact)

        out = shardops.einsum_unreduced(
            'batch hidden, hidden d_model -> batch d_model',
            hidden_act, w.down
        )

        return out

    @typed_shard_map
    def pipeline_step(
        carries: f32[b'num_layers/p batch d_model'],
        weights: MLPBlocks,
    ) -> Tuple[f32[b'num_layers/p batch d_model'], f32[b'num_layers/p batch d_model']]:
        num_stages = jax.lax.psum(1, 'p')

        pp_block = jax.vmap(ffn_block, (0, 0), 0)

        outputs = pp_block(carries, weights)

        perm = [(i, (i+1)%num_stages) for i in range(num_stages)]
        new_carries = jax.lax.ppermute(outputs, 'p', perm=perm)

        return new_carries, outputs

    @jax.jit
    def execute_pipeline(
            pipeline_input: f32[b'num_layers/p batch d_model'],
            weights: MLPBlocks
    ) -> f32['num_layers/p batch d_model']: 
        num_stages = pipeline_input.shape[0]
        
        scan_fn = lambda carries, _: pipeline_step(carries, weights)
        
        _, outputs = jax.lax.scan(scan_fn, pipeline_input, None, length=num_stages)
        # print("carry after pipeline:\n", _, _.sharding)

        last_stage_outputs = outputs[-1]

        return last_stage_outputs


    cfg = ModelArgs()
    num_stages = len(jax.devices())
    key = jax.random.key(42)

    input = jax.random.normal(key, (cfg.batch, cfg.d_model))
    init_carry = jnp.stack([input, *[jnp.zeros_like(input) for _ in range(num_stages-1)]])
    init_carry = jax.device_put(init_carry, make_shardings(f32['num_layers/p batch d_model']))

    weights = MLPBlocks(
        up=jnp.stack([jnp.eye(cfg.d_model, cfg.hidden) for _ in range(num_stages)]),
        down=jnp.stack([jnp.eye(cfg.hidden, cfg.d_model) for _ in range(num_stages)]),
    )
    weights = jax.tree.map(jax.device_put, weights, make_shardings(MLPBlocks)) 


    print("pipeline input:\n", init_carry, init_carry.sharding)
    output = execute_pipeline(init_carry, weights)
    print("pipeline output:\n", output, output.sharding)
