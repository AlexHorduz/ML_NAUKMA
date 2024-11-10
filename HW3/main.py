import click
from types import SimpleNamespace

import torch
from transformers import GPT2TokenizerFast


from model import CaptioningModel
from utils import visualize_samples, plot_metrics, plot_sample_predictions


model_config = SimpleNamespace(
    vocab_size = 50_257,
    embed_dim = 768,
    num_heads = 12,
    seq_len = 1024,
    depth = 12,
    attention_dropout = 0.1,
    residual_dropout = 0.1,
    mlp_ratio = 4,
    mlp_dropout = 0.1,
    emb_dropout = 0.1,
)


def collate_fn(batch):
    images, input_ids, attention_masks = zip(*batch)
    
    # Stack images (convert list of images to a tensor)
    images = torch.stack(images, dim=0)
    
    # Pad input_ids and attention_masks to the max sequence length in the batch
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    return images, input_ids, attention_masks

@click.command()
@click.option('--data_folder', type=str, default='./data')
@click.option('--bs', type=int, default=32)
@click.option('--device', type=str, default='cuda')
@click.option('--n_epochs', type=int, default=10)
@click.option('--lr', type=float, default=1e-4)
def main(data_folder, bs, device, n_epochs, lr):
    
    # Create datasets 
    train_dataset = None # TODO
    val_dataset = None # TODO

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=bs, shuffle=False, collate_fn=collate_fn)
    
    # Visualize a few images and captions
    visualize_samples(train_dataloader) # TODO

    # Create model
    model = CaptioningModel(config) # TODO 
    model.pretrained_layers_trainable(trainable=False)
    print(f'trainable parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

    # Create tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create optimizer and scheduler
    optimizer = None # TODO
    scheduler = None # TODO

    # Define training loop 
    # TODO
    train_losses = list()
    train_perplexities = list()

    # Define validation loop
    # TODO
    val_losses = list()
    val_perplexities = list()
    val_bleus = list()


    # Plot metrics
    plot_metrics(train_losses, val_losses) # TODO plot loss and perplexity for train and val. Additionally, measure BLEU score for validation set.

    # Plot sample predictions
    plot_sample_predictions(model, tokenizer, val_dataloader) # TODO

if __name__ == '__main__':
    main()