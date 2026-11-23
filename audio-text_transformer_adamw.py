#!/usr/bin/env python3
from __future__ import annotations
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple
import sys

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import random
import json
from collections import Counter

import argparse
from pathlib import Path
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train an Audio-Text Matching Transformer",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--dataset-root", default=Path("/data/COMS30017"), type=Path)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--num-heads", default=4, type=int, help="Number of Heads")
parser.add_argument("--batch-size", default=128, type=int, help="Number of pairs per mini-batch")
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--val-frequency", default=2, type=int)
parser.add_argument("--log-frequency", default=10, type=int)
parser.add_argument("--print-frequency", default=10, type=int)
parser.add_argument("-j", "--worker-count", default=0, type=int,
                    help="Data loader workers; 0 avoids copying large feature dictionaries")
parser.add_argument("--seed", default=42, type=int, help="Seed number for randomisation")

#added this so multiple experiments could be run and logged to different direcctories for easier comparison
parser.add_argument("--model-name", type=str, default="baseline", help="Name of the model experiment")

# Optimiser choice (added for the AdamW extension)
parser.add_argument("--optimizer", default="adam", choices=["adam", "adamw"],
                    help="adam = baseline, adamw = decoupled weight decay")
parser.add_argument("--weight-decay", default=0.01, type=float,
                    help="Weight decay (only used by adamw; not applied to biases/LayerNorm)")

# Adjustable parametres for training
MODEL_DIM = 256
TRANSFORMER_MLP_DIM = 512
DROPOUT = 0.1
NEGATIVES_PER_POSITIVE = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float) -> Optimizer:
    """Adam (baseline) or AdamW.

    AdamW applies weight decay directly to the weights (decoupled from the
    gradient/Adam statistics). Following common practice, 1-D parameters
    (biases and LayerNorm gains) are excluded from decay.
    """
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def forward_flops_per_sample(model: nn.Module, audio, text):
    """Hardware-independent compute estimate: forward FLOPs for ONE pair.

    Run in train mode on purpose: in eval mode PyTorch may use a fused
    TransformerEncoderLayer kernel that the FLOP counter cannot see.
    Training cost is roughly 3x forward (forward + backward).
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
        model.train()
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            model(audio[:1].to(DEVICE), text[:1].to(DEVICE))
        return int(counter.get_total_flops())
    except Exception as e:  # old torch version etc.
        print(f"FLOP counting unavailable: {e}")
        return None


class EnergyMeter:
    """GPU energy via NVML's cumulative counter (Volta or newer). Returns None
    if unavailable (CPU run, no pynvml, old GPU). NOTE: only covers the GPU,
    not CPU/RAM, and assumes torch's device index matches NVML's."""

    def __init__(self):
        self.handle = None
        if torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                self._nvml = pynvml
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)  # test call
            except Exception:
                self.handle = None

    def read_joules(self):
        if self.handle is None:
            return None
        return self._nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0  # mJ -> J


def save_training_curves(history, model_name):
    """
    Save training/validation loss and accuracy curves.

    Args:
        history: Dictionary containing epoch-level metrics.
        model_name: Experiment directory name, e.g. baseline.
    """

    figure_dir = Path(model_name) / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["train_epochs"],
        history["train_loss"],
        label="Training Loss",
        marker="o"
    )

    plt.plot(
        history["val_epochs"],
        history["val_loss"],
        label="Validation Loss",
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        figure_dir / "training_validation_loss.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # -------------------------
    # 2. Training vs validation accuracy
    # -------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(
        history["train_epochs"],
        [x * 100 for x in history["train_accuracy"]],
        label="Training Accuracy",
        marker="o"
    )

    plt.plot(
        history["val_epochs"],
        [x * 100 for x in history["val_accuracy"]],
        label="Validation Accuracy",
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        figure_dir / "training_validation_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Training figures saved to: {figure_dir}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)

def load_features(root: Path, split: str):
    audio = torch.load(root / f"{split}_audio_features.pt", map_location="cpu", weights_only=False)
    text = torch.load(root / f"{split}_text_features.pt", map_location="cpu", weights_only=False)
    labels = torch.load(root / f"{split}_labels.pt", map_location="cpu", weights_only=False)
    if not all(isinstance(x, dict) for x in (audio, text, labels)):
        raise TypeError("Expected audio, text and labels to be dictionaries keyed by split ID")
    return audio, text, labels


def as_audio_tensor(value):
    value = torch.as_tensor(value, dtype=torch.float32).squeeze()
    if value.ndim != 1:
        raise ValueError(f"Expected one audio feature vector; got {tuple(value.shape)}")
    return value


def as_text_tensor(value):
    value = torch.as_tensor(value, dtype=torch.float32)

    if value.ndim == 3 and value.shape[0] == 1:
        value = value.squeeze(0)
     
    
    if value.ndim != 2:
        raise ValueError(f"Expected [tokens, features] text tensor; got {tuple(value.shape)}")
    return value


class AudioTextDataset(Dataset):
    def __init__(self, audio, text, labels, training=False):
        self.audio = audio
        self.text = text
        self.training = training
        self.keys = list(audio.keys() & text.keys())
        if not self.keys:
            raise ValueError("No shared audio/text split IDs")
        if training:
            if len(self.keys) < 2:
                raise ValueError("Need at least two IDs for negative sampling")
        else:
            # ASSUMPTION TO VERIFY: a scalar 0/1 label for each same-key pair.
            # If val_labels.pt instead stores explicit cross-ID pairs, replace
            # this branch with a parser for that actual format.
            self.keys = [key for key in self.keys if key in labels]
            self.labels = labels
            for key in self.keys:
                label = torch.as_tensor(labels[key])
                if label.numel() != 1 or label.item() not in (0, 1):
                    raise ValueError(
                        "Validation labels are not scalar 0/1 values keyed by ID. "
                        "Inspect val_labels.pt and adapt AudioTextDataset accordingly."
                    )

    def __len__(self):
        return len(self.keys) * (1 + NEGATIVES_PER_POSITIVE) if self.training else len(self.keys)

    def __getitem__(self, index):
        if self.training:
            positive = index % (1 + NEGATIVES_PER_POSITIVE) == 0
            audio_key = self.keys[index // (1 + NEGATIVES_PER_POSITIVE)]
            if positive:
                text_key = audio_key
            else:
                # Choose a different text ID without a rejection loop.
                audio_index = index // (1 + NEGATIVES_PER_POSITIVE)
                other = torch.randint(len(self.keys) - 1, (1,)).item()
                text_key = self.keys[other + (other >= audio_index)]
            label = int(positive)
        else:
            audio_key = text_key = self.keys[index]
            label = int(torch.as_tensor(self.labels[audio_key]).item())
        return (as_audio_tensor(self.audio[audio_key]),
                as_text_tensor(self.text[text_key]),
                torch.tensor(label, dtype=torch.long))


def main(args):
    set_seed(args.seed)
    
    train_audio, train_text, train_labels = load_features(args.dataset_root, "train")
    val_audio, val_text, val_labels = load_features(args.dataset_root, "val")
    train_dataset = AudioTextDataset(train_audio, train_text, train_labels, training=True)
    val_dataset = AudioTextDataset(val_audio, val_text, val_labels)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                              pin_memory=True, num_workers=args.worker_count)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size,
                            pin_memory=True, num_workers=args.worker_count,
                            worker_init_fn=lambda worker_id: np.random.seed(torch.initial_seed() % (2**32))
                           )
    
    print("Training dataset size:", len(train_dataset))
    print("Training batches:", len(train_loader))

    audio, text, labels = next(iter(train_loader))

    print("Audio batch shape:", audio.shape)
    print("Text batch shape:", text.shape)
    print("Labels shape:", labels.shape)

    first_audio, first_text, _ = train_dataset[0]
    model = AudioTextTransformer(audio_dim=first_audio.shape[-1],
                                 text_dim=first_text.shape[-1],
                                 d_model=MODEL_DIM, num_heads=args.num_heads,
                                 mlp_dim=TRANSFORMER_MLP_DIM, dropout=DROPOUT)
    criterion = nn.CrossEntropyLoss()  # Existing classifier outputs two logits.
    optimizer = build_optimizer(model, args.optimizer, args.learning_rate, args.weight_decay)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(str(log_dir), flush_secs=5)
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer,
                      summary_writer, DEVICE, model_name=args.model_name)
    fwd_flops = forward_flops_per_sample(trainer.model, audio, text)
    trainer.train(args.epochs, args.val_frequency,
                  print_frequency=args.print_frequency,
                  log_frequency=args.log_frequency)
    
    # Each experiment saves into its own folder so runs don't overwrite each other.
    out_dir = Path(args.model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.model.state_dict(), out_dir / "transformer_weights.pth")

    # Efficiency metrics, read back by the notebook for the sustainability comparison.
    h = trainer.history
    energy = [e for e in h["epoch_energy_j"] if e is not None]
    metrics = {
        "model_name": args.model_name,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay if args.optimizer == "adamw" else 0.0,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "trainable_params": count_parameters(trainer.model),
        "samples_per_epoch": len(train_dataset),
        "fwd_flops_per_sample": fwd_flops,
        "peak_gpu_mem_mb": (torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
        "total_train_seconds": float(sum(h["epoch_train_time"])),
        "total_energy_j": float(sum(energy)) if len(energy) == len(h["epoch_energy_j"]) and energy else None,
        "history": h,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved weights + metrics to {out_dir}/")

    summary_writer.close()

class AudioTextTransformer(nn.Module):
    def __init__(
        self,
        audio_dim,
        text_dim,
        d_model=256,
        num_heads=4,
        mlp_dim=512,
        dropout=0.1
    ):
        super().__init__()
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)            
        )
        
        self.transformer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = num_heads,
            dim_feedforward = mlp_dim,
            dropout = dropout,
            activation = 'gelu',
            batch_first = True,
            norm_first = False
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(128, 2) 
        )
    
    def forward(self, audio_input, text_input):
        audio = self.audio_projection(audio_input)
        text = self.text_projection(text_input)
        audio = audio.unsqueeze(1)
        
        x = torch.cat((audio, text), dim=1)
        x = self.transformer(x)
        
        #Mean pooling (GAP) over the sequence dimension
        x = x.mean(dim=1)
        output = self.classifier(x)
        return output
    


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
        model_name: str = "baseline"
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0
        self.model_name = model_name
        
        self.history = {
            "train_epochs": [],
            "train_loss": [],
            "train_accuracy": [],
            "val_epochs": [],
            "val_loss": [],
            "val_accuracy": [],
            # efficiency tracking (training only, validation time excluded)
            "epoch_train_time": [],
            "epoch_energy_j": [],
        }
        self.energy_meter = EnergyMeter()

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        print(f"Number of trainable parameters: {count_parameters(self.model):,}")
        
        for epoch in range(start_epoch, epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            self.model.train()
            self._sync()
            epoch_start = time.perf_counter()
            energy_start = self.energy_meter.read_joules()
            data_load_start_time = time.time()
            for audio, text, labels in self.train_loader:
                audio = audio.to(self.device)
                text = text.to(self.device)
                labels = labels.to(self.device)
                
                
                data_load_end_time = time.time()


                logits = self.model(audio, text)

                #TASK 4 //done

                loss = self.criterion(logits, labels)
                
                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()
                
                batch_size = labels.size(0)

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)
                
                #accumulate epoch-level metrics
                epoch_loss += loss.item() * batch_size
                epoch_correct += (preds == labels).sum().item()
                epoch_samples += batch_size

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()
                
            self._sync()
            self.history["epoch_train_time"].append(time.perf_counter() - epoch_start)
            energy_end = self.energy_meter.read_joules()
            self.history["epoch_energy_j"].append(
                None if energy_start is None or energy_end is None else energy_end - energy_start
            )

            average_train_loss = epoch_loss / epoch_samples
            epoch_train_accuracy = epoch_correct / epoch_samples

            self.history["train_epochs"].append(epoch + 1)
            self.history["train_loss"].append(average_train_loss)
            self.history["train_accuracy"].append(epoch_train_accuracy)

            print(
                f"Epoch [{epoch + 1}], "
                f"Training Loss: {average_train_loss:.5f}, "
                f"Training Accuracy: {epoch_train_accuracy * 100:.2f}%"
            )
            
            self.summary_writer.add_scalar("epoch", epoch, self.step)
            
            #checkpointing the model every 5 epochs
            if (epoch + 1) % 5 == 0:
                save_dir = Path(self.model_name) / "weights"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"epoch_{epoch + 1}.pth"
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }, save_path)
                print(f"Checkpoint saved: {save_path}")
                
            if ((epoch + 1) % val_frequency) == 0:
                val_loss, val_accuracy = self.validate()

                self.history["val_epochs"].append(epoch + 1)
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_accuracy)
                        
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()
                
        save_training_curves(self.history, self.model_name)

    def _sync(self):
        # make GPU timing accurate
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for audio, text, labels in self.val_loader:
                audio = audio.to(self.device)
                text = text.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(audio, text)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
                "accuracy",
                {"val": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"val": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")
        return average_loss, accuracy



def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = f'AudioText_bs={args.batch_size}_lr={args.learning_rate}_run_'
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())
