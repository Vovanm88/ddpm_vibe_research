"""Small executable example: register a nonlinear continuous encoder and loss."""

import torch
from torch import nn

from diffusion_lab import (
    DiffusionTrainer,
    PointDenoiser,
    ResearchDiffusionPipeline,
    TrainerConfig,
    make_scheduler,
)
from diffusion_lab.conditioning import register_conditioner
from diffusion_lab.environments import SampleDataset
from diffusion_lab.losses import register_loss


@register_conditioner("fourier_context")
class FourierContext(nn.Module):
    def __init__(self, hidden_dim, num_classes, condition_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(3 * condition_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, conditions):
        value = conditions.to(self.projection[0].weight.dtype)
        return self.projection(
            torch.cat((value, (torch.pi * value).sin(), (torch.pi * value).cos()), dim=-1)
        )


@register_loss("charbonnier")
def charbonnier(prediction, target):
    return ((prediction - target).square() + 1e-6).sqrt().flatten(1).mean(1)


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    context = torch.linspace(-1, 1, 256)[:, None]
    samples = torch.cat((context, context.square()), dim=1) + 0.02 * torch.randn(256, 2)
    dataset = SampleDataset(samples, context)
    model = PointDenoiser(
        sample_shape=dataset.sample_shape,
        hidden_dim=32,
        num_layers=2,
        conditioner="fourier_context",
        condition_dim=1,
    )
    pipeline = ResearchDiffusionPipeline(
        model, make_scheduler("ddim", num_train_timesteps=100, prediction_type="v_prediction")
    )
    trainer = DiffusionTrainer(
        pipeline, TrainerConfig(epochs=3, batch_size=64, loss="charbonnier", ema_decay=0.9)
    )
    trainer.fit(dataset)
    with trainer.ema_weights():
        generated = pipeline(
            batch_size=8,
            conditions=torch.full((8, 1), 0.5),
            num_inference_steps=20,
            generator=torch.Generator().manual_seed(42),
        ).samples
    print("Final training loss:", trainer.history[-1]["loss"])
    print("Samples conditioned on c=.5:\n", generated)


if __name__ == "__main__":
    main()
