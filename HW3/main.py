import click
from types import SimpleNamespace
import torch
from transformers import GPT2TokenizerFast
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from transformers import get_linear_schedule_with_warmup
from PIL import Image
from torchvision import transforms
import pandas as pd
import os
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from model import CaptioningModel
from utils import visualize_samples, plot_metrics, plot_sample_predictions

# Model configuration
model_config = SimpleNamespace(
    vocab_size=50_257,
    embed_dim=768,
    num_heads=12,
    seq_len=1024,
    depth=12,
    attention_dropout=0.1,
    residual_dropout=0.1,
    mlp_ratio=4,
    mlp_dropout=0.1,
    emb_dropout=0.1,
)


# Custom dataset for image captioning
class ImageCaptioningDataset(Dataset):
    def __init__(self, csv_file, images_folder, tokenizer, transform=None, max_seq_len=1024):
        """
        Args:
            csv_file (str): Path to the CSV file with image_name and caption.
            images_folder (str): Directory with all the images.
            tokenizer (GPT2TokenizerFast): Tokenizer for captioning.
            transform (callable, optional): Optional transform to be applied on an image.
            max_seq_len (int): Maximum sequence length for captions.
        """
        self.csv_file = csv_file
        self.images_folder = images_folder
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len

        # Load the CSV file with image_name and comment columns
        self.df = pd.read_csv(csv_file, delimiter="|")
        self.df = self.df[['image_name', 'comment']]  # Filter for necessary columns
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_name = self.df.iloc[idx, 0]
        caption = self.df.iloc[idx, 1]

        caption = caption + "<|endoftext|>"
        
        # Load the image
        image_path = os.path.join(self.images_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        
        # Apply any transformations to the image
        if self.transform:
            image = self.transform(image)
        
        # Tokenize the caption
        encoding = self.tokenizer(caption, truncation=True, padding="max_length", max_length=self.max_seq_len, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze(0)  # Remove batch dimension
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        
        return image, input_ids, attention_mask


# Collate function to handle padding and batching
def collate_fn(batch):
    images, input_ids, attention_masks = zip(*batch)
    
    # Stack images (convert list of images to a tensor)
    images = torch.stack(images, dim=0)
    
    # Pad input_ids and attention_masks to the max sequence length in the batch
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    return images, input_ids, attention_masks


# Function to load datasets and dataloaders
def get_dataloaders(data_folder, batch_size, tokenizer, transform=None, max_seq_len=1024):
    # Load CSV files
    train_csv = os.path.join(data_folder, 'labels.csv')
    train_dataset = ImageCaptioningDataset(csv_file=train_csv, images_folder=os.path.join(data_folder, 'flickr30k_images'),
                                           tokenizer=tokenizer, transform=transform, max_seq_len=max_seq_len)
    
    # Split the dataset into train and validation sets (e.g., 80% train, 20% validation)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_dataloader, val_dataloader


# Function to create optimizer and scheduler
def get_optimizer_and_scheduler(model, lr, train_dataloader, n_epochs):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Set up a linear learning rate scheduler with warm-up
    total_steps = len(train_dataloader) * n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    return optimizer, scheduler


# Training loop
def train_loop(model, train_dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for images, input_ids, attention_masks in train_dataloader:
        images, input_ids, attention_masks = images.to(device), input_ids.to(device), attention_masks.to(device)
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images, input_ids)
        
        # Compute loss (ignoring padding tokens in the target)
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), input_ids.view(-1), ignore_index=0)
        total_loss += loss.item()
        
        # Backward pass and optimizer step
        loss.backward()

        optimizer.step()
        scheduler.step()
        print(f"\tTrain loss: {loss.item()}")
    
    avg_loss = total_loss / len(train_dataloader)
    return avg_loss


# Validation loop
def val_loop(model, val_dataloader, device):
    model.eval()
    total_loss = 0
    predictions = []
    references = []
    
    with torch.no_grad():
        for images, input_ids, attention_masks in val_dataloader:
            images, input_ids, attention_masks = images.to(device), input_ids.to(device), attention_masks.to(device)

            outputs = model(images, input_ids)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), input_ids.view(-1), ignore_index=0)
            total_loss += loss.item()

            # Get predictions (greedy decoding)
            pred_ids = outputs.argmax(dim=-1)
            
            # Collect predictions and references
            for pred, ref in zip(pred_ids.cpu().numpy(), input_ids.cpu().numpy()):
                predictions.append([str(p) for p in pred])  # Convert to list of strings (tokens)
                references.append([[str(r) for r in ref]])  # Wrap reference in a list (for multiple references)

            print(f"Val loss: {loss.item()}")
    
    avg_loss = total_loss / len(val_dataloader)

    # Calculate BLEU score
    bleu_score = corpus_bleu(references, predictions)
    return avg_loss, bleu_score


@click.command()
@click.option('--data_folder', type=str, default='./dataset')
@click.option('--bs', type=int, default=4)
@click.option('--device', type=str, default='cpu')
@click.option('--n_epochs', type=int, default=2)
@click.option('--lr', type=float, default=1e-4)
def main(data_folder, bs, device, n_epochs, lr):
    # Create tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Get the dataloaders
    train_dataloader, val_dataloader = get_dataloaders(data_folder, bs, tokenizer, transform)
    
    # Visualize some samples
    visualize_samples(train_dataloader, tokenizer)

    # Create model
    model = CaptioningModel(model_config).to(device)

    # Set pretrained layers to be non-trainable
    model.pretrained_layers_trainable(trainable=False)
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # Set up optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(model, lr, train_dataloader, n_epochs)

    train_losses = []
    val_losses = []
    val_bleus = []
    train_perplexities = []  # To track training perplexities
    val_perplexities = []    # To track validation perplexities

    # Training loop
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        
        # Train for one epoch
        train_loss = train_loop(model, train_dataloader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss}")
        train_losses.append(train_loss)

        # Calculate Perplexity for Train Loss
        train_perplexity = torch.exp(torch.tensor(train_loss))  # Convert float loss to tensor and calculate perplexity
        print(f"Train Perplexity: {train_perplexity.item()}")
        train_perplexities.append(train_perplexity.item())
        
        # Validation loop
        val_loss, bleu_score = val_loop(model, val_dataloader, device)
        print(f"Validation Loss: {val_loss}, BLEU Score: {bleu_score}")
        val_losses.append(val_loss)
        val_bleus.append(bleu_score)

        # Calculate Perplexity for Validation Loss
        val_perplexity = torch.exp(torch.tensor(val_loss))  # Convert float loss to tensor and calculate perplexity
        print(f"Validation Perplexity: {val_perplexity.item()}")
        val_perplexities.append(val_perplexity.item())

    # Plot metrics including perplexity
    plot_metrics(train_losses, val_losses, train_perplexities, val_perplexities)  # Plot train and validation losses and perplexity
    plot_sample_predictions(model, tokenizer, val_dataloader)  # Plot sample predictions



if __name__ == '__main__':
    main()
