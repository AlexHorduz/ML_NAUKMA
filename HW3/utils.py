import matplotlib.pyplot as plt
import torch


def visualize_samples(dataloader, tokenizer, num_samples=4):
    """
    Visualize some samples from the dataset (images and captions).
    
    Args:
        dataloader (DataLoader): DataLoader for the dataset.
        tokenizer (GPT2TokenizerFast): The tokenizer used to decode the captions.
        num_samples (int): Number of samples to visualize.
    """
    images, input_ids, attention_masks = next(iter(dataloader))
    
    # Display the first few images and their captions
    fig, ax = plt.subplots(1, num_samples, figsize=(16, 4))
    for i in range(num_samples):
        ax[i].imshow(images[i].permute(1, 2, 0).cpu().numpy())  # Convert from CHW to HWC
        ax[i].axis('off')
        caption = input_ids[i]
        caption = caption[caption != 0]  # Remove padding
        decoded_caption = tokenizer.decode(caption, skip_special_tokens=True)
        ax[i].set_title(decoded_caption)
    
    plt.show()



def plot_metrics(train_losses, val_losses, train_perplexities=None, val_perplexities=None, title="Training and Validation Metrics"):
    """
    Plot the training and validation losses, as well as perplexities if provided.
    
    Args:
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
        train_perplexities (list, optional): List of training perplexities for each epoch.
        val_perplexities (list, optional): List of validation perplexities for each epoch.
        title (str): Title for the plot.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, label="Train Loss", color="blue")
    plt.plot(epochs, val_losses, label="Validation Loss", color="red")


    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    if train_perplexities and val_perplexities:
        plt.plot(epochs, train_perplexities, label="Train Perplexity", color="green", linestyle='--')
        plt.plot(epochs, val_perplexities, label="Validation Perplexity", color="orange", linestyle='--')

        plt.xlabel("Epochs")
        plt.ylabel("Perplexity")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()


def plot_sample_predictions(model, tokenizer, dataloader, num_samples=4):
    """
    Plot sample predictions from the model on the validation set.
    
    Args:
        model (nn.Module): The trained model.
        tokenizer (GPT2TokenizerFast): The tokenizer used for encoding captions.
        dataloader (DataLoader): DataLoader for the validation set.
        num_samples (int): Number of samples to visualize.
    """
    model.eval()
    
    images, input_ids, attention_masks = next(iter(dataloader))
    images, input_ids, attention_masks = images, input_ids, attention_masks  # No need for cuda()

    # Generate predictions
    with torch.no_grad():
        outputs = model(images, input_ids)
    
    # Adjust num_samples to the batch size if it's smaller
    num_samples = min(num_samples, images.size(0))  # Ensure we don't try to access more samples than are available

    # Display the first few samples
    fig, ax = plt.subplots(1, num_samples, figsize=(16, 4))
    
    # Ensure ax is iterable, even if there's only one sample
    if num_samples == 1:
        ax = [ax]
    
    for i in range(num_samples):
        ax[i].imshow(images[i].permute(1, 2, 0).cpu().numpy())  # Convert from CHW to HWC (no CUDA)
        ax[i].axis('off')
        
        # Decode the input caption
        input_caption = input_ids[i]
        input_caption = input_caption[input_caption != 0]  # Remove padding
        decoded_input_caption = tokenizer.decode(input_caption, skip_special_tokens=True)
        
        # Get the prediction (greedy decoding)
        pred_ids = outputs[i].argmax(dim=-1)  # Get the predicted sequence
        pred_caption = pred_ids[pred_ids != 0]  # Remove padding
        decoded_pred_caption = tokenizer.decode(pred_caption, skip_special_tokens=True)

        ax[i].set_title(f"True: {decoded_input_caption}\nPred: {decoded_pred_caption}")

    plt.show()



