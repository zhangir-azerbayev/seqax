# XLA_FLAGS=--xla_force_host_platform_device_count=8 python -m examples.transformer
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

    def pipeline_step(
        carries: f32[b'num_layers/p batch d_model'],
        weights: MLPBlocks,
    ) -> f32[b'num_layers/p batch d_model']:

        pipeline_step_fn = jax.vmap(ffn_block, (0, 0), 0)

        stage_output = pipeline_step_fn(carries, weights)

        return stage_output

    @typed_shard_map
    def pipeline_step_with_permute(
        carries: f32[b'num_layers/p batch d_model'],
        weights: MLPBlocks,
    ) -> f32[b'num_layers/p batch d_model']:
        stage_output = pipeline_step(carries, weights)

        num_stages = jax.lax.psum(1, 'p')
        perm = [(i, (i+1)%num_stages) for i in range(num_stages)]
        stage_output = jax.lax.ppermute(stage_output, 'p', perm=perm)

        return stage_output

    def execute_pipeline(
            pipeline_input: f32[b'num_layers/p batch d_model'],
            weights: MLPBlocks
    ) -> f32[b'num_layers/p batch d_model']:
        num_stages = pipeline_input.shape[0]
        print("pipeline input :", pipeline_input, pipeline_input.sharding)
        
        scan_fn = lambda carry, _: (pipeline_step_with_permute(carry, weights), None)
        
        carry, _ = jax.lax.scan(scan_fn, pipeline_input, None, length=num_stages-1)

        pipeline_output = pipeline_step(carry, weights)

        print("pipeline output: ", pipeline_output, pipeline_output.sharding)

        return pipeline_output


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


    output = execute_pipeline(init_carry, weights)
