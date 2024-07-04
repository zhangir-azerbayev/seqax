# XLA_FLAGS=--xla_force_host_platform_device_count=8 python -m examples.transformer
from collections.abc import Callable
import copy
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
from typing import Tuple
import functools as ft

# 'p' is pipeline parallel axis
MESH = Mesh(mesh_utils.create_device_mesh([4], jax.devices()), ('p'))

@dataclass 
class ModelArgs:
    batch: int = 2
    num_layers: int = 4 # check this works with size of 'p' axis
    d_model: int = 16

@pytree_dataclass
class PipelinedLinearLayers:
    layers: f32['num_layers/p d_model d_model']

with MESH:
    @typed_shard_map
    def pipeline_step(
        carries: f32[b'num_layers/p batch d_model'],
        weights: PipelinedLinearLayers,
        ) -> f32[b'num_layers/p batch d_model']:
        num_stages = jax.lax.psum(1, 'p')

        stage_preact = shardops.einsum_unreduced(
            'num_layers/p batch d_model1, num_layers/p d_model1 d_model2 -> num_layers/p batch d_model2',
            carries, weights.layers
        )

        stage_output = jax.nn.relu(stage_preact)

        new_carries = jax.lax.ppermute(
            stage_output,
            'p',
            perm=[(i, (i+1)%num_stages) for i in range(num_stages)]
        )

        return new_carries

    cfg = ModelArgs()
    key = jax.random.key(42)

    key, sbkey = jax.random.split(key)
    layers = PipelinedLinearLayers(
        layers=jax.random.normal(sbkey, (cfg.num_layers, cfg.d_model, cfg.d_model))
    )
    layers = jax.tree.map(jax.device_put, layers, make_shardings(PipelinedLinearLayers))

    key, sbkey = jax.random.split(key)
    carry = jax.random.normal(sbkey, (cfg.num_layers, cfg.batch, cfg.d_model))

    stage_output = pipeline_step(carry, layers)
