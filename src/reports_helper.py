import pandas as pd
import matplotlib.pyplot as plt


def plot_keras_training_history_csv(csv_file, output_file):
    """Plot training and validation accuracy and loss from the CSV history file, and save the plot to a file."""

    # Read the CSV file
    history_df = pd.read_csv(csv_file)

    # Create the figure
    plt.figure(figsize=(12, 6))

    # Plot training vs validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(
        history_df["epoch"], history_df["train_accuracy"], label="Training Accuracy"
    )
    plt.plot(
        history_df["epoch"], history_df["val_accuracy"], label="Validation Accuracy"
    )
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Plot training vs validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Training Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(output_file)

    # Close the plot to free memory
    plt.close()

    print(f"Plot saved to {output_file}")
