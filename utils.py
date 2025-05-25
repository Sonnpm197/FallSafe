import numpy as np
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        """
        EarlyStopping class to monitor validation loss during training and stop early.

        Parameters:
        - patience: The number of epochs with no improvement after which training will be stopped.
        - delta: Minimum change to qualify as an improvement. If the change is less than delta, it's not considered an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False
        self.best_weights = None  # To store the model weights when the best performance is achieved

    def __call__(self, val_loss, model):
        """
        Call the EarlyStopping instance at the end of each epoch.

        Parameters:
        - val_loss: The validation loss to monitor.
        - model: The model to save the weights when the best validation loss is achieved.

        Returns:
        - self.early_stop: Whether early stopping should be triggered.
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict()  # Save the best weights
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def restore_best_weights(self, model):
        """
        Restore the model weights from the best epoch.

        Parameters:
        - model: The model whose weights will be restored to the best observed weights.
        """
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

# Learning rate scheduler function
def scheduler(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * np.exp(0.1 * (10 - epoch))

def plot_training_results(train_losses, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # Plot Training Loss vs Epochs
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Validation Loss vs Epochs
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Validation Accuracy vs Epochs
    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def is_none_or_empty(x):
    return x is None or len(x) == 0