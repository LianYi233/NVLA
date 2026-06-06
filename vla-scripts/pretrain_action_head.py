import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import draccus
import torch
import tqdm
from accelerate import PartialState
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from prismatic.vla.datasets import RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class PretrainActionPriorConfig:
    data_root_dir: Path = Path("data/libero")
    dataset_name: str = "libero_object_no_noops"
    run_root_dir: Path = Path("runs/action_prior")

    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_steps: int = 50000
    save_freq: int = 5000
    num_steps_before_decay: int = 50000
    shuffle_buffer_size: int = 100_000

    hidden_dim: int = 4096
    use_pro_version: bool = True

    noise_std: float = 0.01
    lambda_smooth: float = 0.01
    image_aug: bool = False


class ActionOnlyBatchTransform:
    """
    Minimal transform for action-only pretraining.
    RLDSDataset will call this on each RLDS sample.
    """
    def __call__(self, rlds_batch: Dict) -> Dict:
        return {
            "actions": torch.tensor(rlds_batch["action"], dtype=torch.float32),
            "dataset_name": rlds_batch["dataset_name"],
        }


def action_collator(batch):
    actions = torch.stack([item["actions"] for item in batch], dim=0)  # [B, H, A]
    dataset_names = [item["dataset_name"] for item in batch]
    return {
        "actions": actions,
        "dataset_name": dataset_names,
    }


@draccus.wrap()
def main(cfg: PretrainActionPriorConfig):
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)

    run_name = (
        f"action-prior+{cfg.dataset_name}"
        f"+b{cfg.batch_size}"
        f"+lr{cfg.learning_rate}"
        f"+h{cfg.hidden_dim}"
    )
    run_dir = cfg.run_root_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Pretraining action prior on {cfg.dataset_name}\n"
        f"chunk_len={NUM_ACTIONS_CHUNK}, action_dim={ACTION_DIM}"
    )

    batch_transform = ActionOnlyBatchTransform()

    dataset = RLDSDataset(
        data_root_dir=cfg.data_root_dir,
        data_mix=cfg.dataset_name,
        batch_transform=batch_transform,
        resize_resolution=(224, 224),   # required by RLDSDataset, but unused here
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        train=True,
        image_aug=cfg.image_aug,
    )

    if distributed_state.is_main_process:
        save_dataset_statistics(dataset.dataset_statistics, run_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=action_collator,
        num_workers=0,
    )

    # Instantiate the full action head and train ALL of it
    action_head = L1RegressionActionHead(
        input_dim=cfg.hidden_dim,
        hidden_dim=cfg.hidden_dim,
        action_dim=ACTION_DIM,
        use_pro_version=cfg.use_pro_version,
        action_prior_num_layers=2,
        action_prior_hidden_dim=cfg.hidden_dim,
    ).to(device_id)

    trainable_params = [p for p in action_head.parameters() if p.requires_grad]
    print(f"# trainable params in full action_head: {sum(p.numel() for p in trainable_params)}")

    optimizer = AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],
        gamma=0.1,
    )

    action_head.train()
    step = 0

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as pbar:
        while step < cfg.max_steps:
            for batch in dataloader:
                gt_actions = batch["actions"].to(device_id)   # [B, H, A]

                out = action_head.pretrain_action_head(
                    gt_actions,
                    noise_std=cfg.noise_std,
                    lambda_smooth=cfg.lambda_smooth,
                )

                loss = out["loss"]
                recon_loss = out["recon_loss"]
                smooth_loss = out["smooth_loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                if distributed_state.is_main_process and step % 50 == 0:
                    print(
                        f"[step {step}] "
                        f"loss={loss.item():.6f}, "
                        f"recon={recon_loss.item():.6f}, "
                        f"smooth={smooth_loss.item():.6f}"
                    )

                if distributed_state.is_main_process and step > 0 and step % cfg.save_freq == 0:
                    ckpt_path = run_dir / f"action_head--{step}_checkpoint.pt"
                    torch.save(action_head.state_dict(), ckpt_path)
                    print(f"Saved action_head checkpoint to {ckpt_path}")

                step += 1
                pbar.update(1)

                if step >= cfg.max_steps:
                    break

    if distributed_state.is_main_process:
        final_ckpt = run_dir / "action_head--final_checkpoint.pt"
        torch.save(action_head.state_dict(), final_ckpt)
        print(f"Saved final action_head checkpoint to {final_ckpt}")

if __name__ == "__main__":
    main()