import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from datasets import IterableDataset
from dataclasses import asdict
import os
from transformer import Transformer
from lr_scheduler import CustomScheduler
from utils import process_batch, prepare_tokenizer
from config import *

tokenizer = TokenizerConfig.load_tokenizer()
tokenizer = prepare_tokenizer(tokenizer)

train_data = MyDatasets.load_train().shuffle(seed=TrainConfig.seed, buffer_size=TrainConfig.buffer_size)
valid_data = MyDatasets.load_valid()

model_config = asdict(
    ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        padding_idx=tokenizer.token_to_id(TokenizerConfig.pad_token),
    )
)

criterion = CrossEntropyLoss(
    label_smoothing=TrainConfig.label_smoothing,
    ignore_index=model_config['padding_idx'],
)


# Use all available GPUs 
if TrainConfig.device == 'cuda' and torch.cuda.device_count() > 1:
    data_parallel = True
else:
    data_parallel = False


if TrainConfig.resume:
    checkpoint = torch.load(f'{TrainConfig.checkpoint_file}')
    model_config = checkpoint['model_config']
    model = Transformer(**model_config)
    model.load_state_dict(checkpoint['model_state'])

    if data_parallel:
        model = nn.DataParallel(model)

    model.to(TrainConfig.device)
    model = torch.compile(model)

    optimizer = Adam(
        model.parameters(),
        betas=OptimizerConfig.adam_betas,
        eps=OptimizerConfig.adam_eps,
        fused=True,
    )

    lr_scheduler = CustomScheduler(
        optimizer,
        model_config['d_model'],
        OptimizerConfig.warmup_steps
    )
    
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    best_val_loss = checkpoint['val_loss']
    epoch = checkpoint['epoch']


else:
    torch.manual_seed(TrainConfig.seed)
    model = Transformer(**model_config)

    if data_parallel:
        model = nn.DataParallel(model)

    model.to(TrainConfig.device)
    model = torch.compile(model)

    optimizer = Adam(
        model.parameters(),
        betas=OptimizerConfig.adam_betas,
        eps=OptimizerConfig.adam_eps,
        fused=True,
    )

    lr_scheduler = CustomScheduler(
        optimizer,
        model_config['d_model'],
        OptimizerConfig.warmup_steps
    )

    best_val_loss = float('inf')
    epoch = 0

    # create save directory if it does not exist
    if not os.path.exists(TrainConfig.save_directory):
        os.makedirs(TrainConfig.save_directory)


def compute_batch_loss(batch: dict[str, list], model, criterion) -> Tensor:
    source_token_ids, source_masks, target_token_ids, target_masks = process_batch(
        batch, tokenizer, TrainConfig.device
    )

    target_in = target_token_ids[:, :-1]
    target_out = target_token_ids[:, 1:]

    logits = model(source_token_ids, target_in, source_masks, target_masks)

    # logits: (B, T, vocab_size) -> (B * T, vocab_size)
    loss = criterion(logits.view(-1, logits.shape[-1]), target_out.reshape(-1))
    
    return loss


def calc_val_loss(model, val_dataset: IterableDataset, criterion) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_dataset.iter(TrainConfig.batch_size):
            batch_loss = compute_batch_loss(batch, model, criterion)
            total_loss += batch_loss.item()
            n_batches += 1

    return total_loss / n_batches


def save_checkpoint(file: str) -> None:
    model_state = model.module.state_dict() if data_parallel else model.state_dict()

    checkpoint = {
        'model_config': model_config,
        'model_state': model_state,
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'val_loss': best_val_loss,
        'epoch': epoch,
    }

    torch.save(checkpoint, file)


writer = SummaryWriter()
step = lr_scheduler._step_count
best_step = lr_scheduler._step_count  # used for early stopping
stop = False
train_data.set_epoch(epoch)
train_data_iterator = train_data.iter(TrainConfig.batch_size)
torch.set_float32_matmul_precision("medium")

while not stop:
    total_loss = 0.0
    for grad_accum_step in range(TrainConfig.grad_accum_steps):
        model.train()
        try:
            batch = next(train_data_iterator)
        except StopIteration:
            epoch += 1
            # Shuffle data every epoch
            train_data.set_epoch(epoch)
            train_data_iterator = train_data.iter(TrainConfig.batch_size)

            if TrainConfig.save_every_epoch:
                save_checkpoint(f"{TrainConfig.save_directory}/epoch-{epoch}.pt")

        with torch.autocast(device_type=TrainConfig.device, dtype=torch.bfloat16):
            loss = compute_batch_loss(batch, model, criterion)
        loss = loss / TrainConfig.grad_accum_steps
        total_loss += loss.item()
        loss.backward()

    grad_norm = clip_grad_norm_(model.parameters(), TrainConfig.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    lr_scheduler.step()
    step += 1


    if step % TrainConfig.log_interval == 0:
        writer.add_scalar("Training loss", total_loss, step)
        writer.add_scalar("Learning rate", optimizer.param_groups[0]["lr"], step)
        writer.add_scalar("Gradient norm", grad_norm, step)

    if step % TrainConfig.eval_interval == 0:
        val_loss = calc_val_loss(model, valid_data, criterion)
        writer.add_scalar("Validation loss", val_loss, step)
        writer.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_step = step
            save_checkpoint(f"{TrainConfig.save_directory}/best_model.pt")

        elif (
            step - best_step >= TrainConfig.early_stop_thresh
            and TrainConfig.early_stopping
        ):
            stop = True
            print(f"Early stopped training at step {step}.")
            print(f"Training loss: {total_loss:.3f}")
            print(f"Best validation loss: {best_val_loss:.3f}")
            break

    if TrainConfig.save_interval > 0 and step % TrainConfig.save_interval == 0:
        save_checkpoint(f"{TrainConfig.save_directory}/step-{int(step/1000)}k.pt")

    if step > TrainConfig.max_steps:
        stop = True
        print("Finished training.")
        print(f"Training loss: {total_loss:.3f}")
        print(f"Best validation loss: {best_val_loss:.3f}")
        break

save_checkpoint(f"{TrainConfig.save_directory}/step-{int(step/1000)}k.pt")
