import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from datasets import IterableDataset
from dataclasses import asdict
import os

from transformer import Transformer
from lr_scheduler import CustomScheduler
from utils import process_batch, prepare_tokenizer
from config import *

tokenizer = TokenizerConfig.load_tokenizer()
tokenizer = prepare_tokenizer(tokenizer)

train_data = MyDatasets.load_train()
train_data = train_data.shuffle(buffer_size=TrainConfig.buffer_size, seed=TrainConfig.seed)
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

    optimizer = Adam(
        model.parameters(),
        betas=OptimizerConfig.adam_betas,
        eps=OptimizerConfig.adam_eps,
    )

    lr_scheduler = CustomScheduler(
        optimizer,
        model_config['d_model'],
        OptimizerConfig.warmup_steps
    )
    
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    train_data.load_state_dict(checkpoint['data_state'])

    best_val_loss = checkpoint['val_loss']
    epoch = checkpoint['epoch']


else:
    model = Transformer(**model_config)

    if data_parallel:
        model = nn.DataParallel(model)

    model.to(TrainConfig.device)

    optimizer = Adam(
        model.parameters(),
        betas=OptimizerConfig.adam_betas,
        eps=OptimizerConfig.adam_eps,
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

    source_token_ids, source_masks, target_token_ids, target_masks = (
        process_batch(batch, tokenizer, TrainConfig.device)
    )

    logits = model(source_token_ids, target_token_ids[:, :-1], source_masks, target_masks)

    # logits: (B, T, vocab_size) -> (B * T, vocab_size)
    loss = criterion(
        logits.view(-1, logits.shape[-1]), 
        target_token_ids[:, 1:].contiguous().view(-1)
    )
    return loss


def calc_val_loss(model, val_dataset: IterableDataset, criterion) -> float:

    model.eval()
    total_loss = 0
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
        'data_state': train_data.state_dict()
    }

    torch.save(checkpoint, file)


writer = SummaryWriter()
step = lr_scheduler._step_count
best_step = lr_scheduler._step_count  # used for early stopping
stop = False


while not stop:

    for batch in train_data.iter(TrainConfig.batch_size):

        model.train()

        loss = compute_batch_loss(batch, model, criterion)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        step += 1

        if step % TrainConfig.log_interval == 0:
            writer.add_scalar('Training loss', loss.item(), step)
            writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], step)

        if step % TrainConfig.eval_interval == 0:
            val_loss = calc_val_loss(model, valid_data, criterion)
            writer.add_scalar('Validation loss', val_loss, step)
            writer.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                save_checkpoint(f'{TrainConfig.save_directory}/best_model.pt')
            
            elif step - best_step >= TrainConfig.early_stop_thresh and TrainConfig.early_stopping:
                stop = True
                print(f"Early stopped training at step {step}.")
                print(f"Training loss: {loss.item() :.3f}")
                print(f"Best validation loss: {best_val_loss :.3f}")
                break
        
        if TrainConfig.save_interval > 0 and step % TrainConfig.save_interval == 0:
            save_checkpoint(f'{TrainConfig.save_directory}/step-{int(step/1000)}k.pt')

        if step > TrainConfig.max_steps:
            stop = True
            print("Finished training.")
            print(f"Training loss: {loss.item() :.3f}")
            print(f"Best validation loss: {best_val_loss :.3f}")
            break
    

    if not stop:
        epoch += 1

        # Shuffle data every epoch
        train_data.set_epoch(epoch)

        if TrainConfig.save_every_epoch:
            save_checkpoint(f'{TrainConfig.save_directory}/epoch-{epoch}.pt')


save_checkpoint(f'{TrainConfig.save_directory}/step-{int(step/1000)}k.pt')
