from collections.abc import Callable
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import optax  # type: ignore[import-untyped]
from optax import GradientTransformationExtraArgs, OptState, Params

from gs.loss_function import l1_loss
from gs.projection import project
from gs.rasterization import rasterize


class DataLogger:
    def __init__(self, save_path: Path) -> None:
        self.count = 0
        self.save_path = save_path

        self.save_path.mkdir(parents=True, exist_ok=True)

    def __call__(self, image: jax.Array) -> None:
        plt.imsave(self.save_path / f"output_{self.count:05d}.png", image.clip(0, 1))
        self.count += 1


def make_updater(
    consts: dict[str, int | float | jax.Array],
    optimizer: GradientTransformationExtraArgs,
    callback: Callable = lambda _: None,
    *,
    jit: bool = True,
) -> Callable:
    render_fn = make_render(consts, jit=jit)

    def loss_fn(
        params: dict[str, jax.Array], view: dict[str, jax.Array], target: jax.Array
    ) -> jax.Array:
        output = render_fn(params, view)
        jax.debug.callback(callback, output)
        loss = l1_loss(output, target)
        return loss

    compute_loss_and_grad = jax.value_and_grad(loss_fn)

    def update(
        params: Params,
        view: dict[str, jax.Array],
        target: jax.Array,
        opt_state: OptState,
    ) -> tuple[Params, dict[str, jax.Array], OptState, jax.Array]:
        loss, grads = compute_loss_and_grad(params, view, target)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, grads, opt_state, loss

    return jax.jit(update, donate_argnames=("params",)) if jit else update


def make_render(consts: dict[str, int | float | jax.Array], *, jit: bool = True) -> Callable:
    def render(params: dict[str, jax.Array], view: dict[str, jax.Array]) -> jax.Array:
        projected_gaussians = project(params, **view, consts=consts)

        return rasterize(projected_gaussians, consts)

    return jax.jit(render) if jit else render
