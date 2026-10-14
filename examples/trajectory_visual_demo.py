"""Actual DDIM denoising GIF and a separately labelled A* navigation reference.

python examples/trajectory_visual_demo.py --run demo --output demo/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDIMScheduler

from diffusion_lab import ResearchDiffusionPipeline
from diffusion_lab.environments import WallNavigation
from diffusion_lab.visualization.animation import plot_rainbow_trajectories, save_trajectory_gif
from diffusion_lab.visualization.plots import save_figure


def make_demo(run="demo", output="demo/figures", *, num_paths=16, seed=127):
    torch.set_num_threads(4)
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pipeline = ResearchDiffusionPipeline.from_pretrained(
        run / "pipeline", local_files_only=True, low_cpu_mem_usage=False
    ).to("cpu")
    if len(pipeline.model.sample_shape) != 1:
        raise ValueError("The denoising demo expects a point-distribution model")
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    classes = int(pipeline.model.config.num_classes)
    conditions = torch.arange(num_paths) % classes if classes else None
    generated = pipeline(
        batch_size=num_paths,
        conditions=conditions,
        num_inference_steps=100,
        guidance_scale=1.5,
        generator=torch.Generator().manual_seed(seed),
        record_history=True,
    )
    history = generated.history.numpy()
    paths = history.transpose(1, 0, 2)  # Actual noise → sample history, no interpolation.
    denoising_title = "DDIM denoising · noise → sample\n16 particles · 100 steps · CFG 1.5"
    if num_paths != 16:
        denoising_title = denoising_title.replace("16 particles", f"{num_paths} particles")
    save_trajectory_gif(
        paths,
        output / "denoising_trajectories.gif",
        fps=12,
        max_frames=60,
        title=denoising_title,
        start_label="Noise x_T",
        end_label="Sample x_0",
    )
    figure, _ = plot_rainbow_trajectories(
        paths, title=denoising_title, start_label="Noise x_T", end_label="Sample x_0"
    )
    save_figure(figure, output / "denoising_trajectories")
    plt.close(figure)

    scene = WallNavigation(dim=2, path_length=64, grid_resolution=41)
    start, goal = np.array([-0.85, -0.65]), np.array([0.85, 0.65])
    navigation_path = scene.plan(start, goal)
    navigation_title = "PLANNER REFERENCE · A → B around walls\nA* route; not learned navigation"
    save_trajectory_gif(
        navigation_path,
        output / "navigation_trajectory.gif",
        fps=12,
        max_frames=60,
        walls=scene.walls,
        title=navigation_title,
    )
    figure, _ = plot_rainbow_trajectories(
        navigation_path, walls=scene.walls, title=navigation_title
    )
    save_figure(figure, output / "navigation_trajectory")
    plt.close(figure)

    np.savez_compressed(
        output / "trajectory_data.npz",
        history=history,
        paths=paths,
        conditions=conditions.numpy() if conditions is not None else np.empty(0),
        timesteps=generated.timesteps.numpy(),
        navigation_path=navigation_path,
        navigation_start=start,
        navigation_goal=goal,
        walls=scene.walls,
    )
    metadata = {
        "seed": seed,
        "num_particles": num_paths,
        "denoising": {
            "source": str((run / "pipeline").resolve()),
            "kind": "actual DDIM noise-to-sample history",
            "steps": 100,
            "guidance_scale": 1.5,
            "history_shape": list(history.shape),
            "condition_ids": conditions.tolist() if conditions is not None else None,
        },
        "navigation": {
            "kind": "PLANNER REFERENCE, not learned navigation",
            "planner": "A*",
            "start": start.tolist(),
            "goal": goal.tolist(),
            "walls": scene.walls.tolist(),
            "path_length": scene.path_length,
            "grid_resolution": scene.grid_resolution,
            "margin": scene.margin,
            "collision_free": bool(scene.collision_free(navigation_path)),
        },
        "color": "turbo_r: red start → orange → yellow → green → blue → purple finish; original normalized waypoint/time index in [0,1], fixed across frames",
        "animation": {"fps": 12, "max_frames": 60, "actual_original_waypoints": True},
    }
    (output / "trajectory_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="demo")
    parser.add_argument("--output", default="demo/figures")
    parser.add_argument("--num-paths", type=int, default=16)
    args = parser.parse_args()
    metadata = make_demo(args.run, args.output, num_paths=args.num_paths)
    print(f"Saved actual DDIM trails and A* reference to {Path(args.output).resolve()}")
    print("Planner route collision-free:", metadata["navigation"]["collision_free"])


if __name__ == "__main__":
    main()
