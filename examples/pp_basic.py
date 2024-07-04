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
    d_model: int = 16

@pytree_dataclass
class PipelinedLinearLayers:
    layers: f32['num_layers/p d_model d_model']

with MESH:
    def pipeline_step(
        carries: f32[b'num_layers/p batch d_model'],
        weights: PipelinedLinearLayers,
    ) -> f32[b'num_layers/p batch d_model']:

        stage_preact = shardops.einsum_unreduced(
            'num_layers/p batch d_model1, num_layers/p d_model1 d_model2 -> num_layers/p batch d_model2',
            carries, weights.layers
        )

        stage_output = jax.nn.relu(stage_preact)

        return stage_output

    @typed_shard_map
    def pipeline_step_with_permute(
        carries: f32[b'num_layers/p batch d_model'],
        weights: PipelinedLinearLayers,
    ) -> f32[b'num_layers/p batch d_model']:
        stage_output = pipeline_step(carries, weights)

        num_stages = jax.lax.psum(1, 'p')
        perm = [(i, (i+1)%num_stages) for i in range(num_stages)]
        stage_output = jax.lax.ppermute(stage_output, 'p', perm=perm)

        return stage_output

    def execute_pipeline(
            pipeline_input: f32[b'num_layers/p batch d_model'],
            weights: PipelinedLinearLayers
    ) -> f32[b'num_layers/p batch d_model']:
        num_stages = pipeline_input.shape[0]
        print("pipeline input :", pipeline_input, pipeline_input.sharding)
        
        scan_fn = lambda carry, _: (pipeline_step_with_permute(carry, weights), None)
        
        carry, _ = jax.lax.scan(scan_fn, pipeline_input, None, length=num_stages-1)

        pipeline_output = pipeline_step(carry, weights)

        print("pipeline output: ", pipeline_output, pipeline_output.sharding)

        return pipeline_output


    cfg = ModelArgs()
    key = jax.random.key(42)

    key, sbkey = jax.random.split(key)
    weights = PipelinedLinearLayers(
        layers=jnp.stack([jnp.identity(cfg.d_model) for _ in range(cfg.num_layers)])
    )
    weights = jax.tree.map(jax.device_put, weights, make_shardings(PipelinedLinearLayers))

    input = jax.random.normal(sbkey, (cfg.batch, cfg.d_model))
    pipeline_input = jnp.stack(
        [input, *[jnp.zeros_like(input) for _ in range(3)]]
    )
    pipeline_input = jax.device_put(pipeline_input, make_shardings(f32['num_layers/p batch d_model']))

    print(make_shardings)

    output = execute_pipeline(pipeline_input, weights)
