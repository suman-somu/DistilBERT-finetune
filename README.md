
---

# Fine-Tuning and Inference with Hugging Face Transformers

This project demonstrates how to fine-tune a pre-trained DistilBERT model on a sentiment analysis task using the IMDb dataset, and how to run inference on new data using the fine-tuned model. The project uses the Hugging Face `transformers` library for model fine-tuning and inference.

## Project Structure
```
├── finetuned_distilbert/     # Directory for saving the fine-tuned model
├── logs/                     # Directory for saving training logs
├── results/                  # Directory for saving training results
├── main.py                   # Main script for fine-tuning the model
├── inference.py              # Script for running inference with the fine-tuned model
├── .gitignore                # File for excluding unnecessary files from Git
└── README.md                 # Project documentation
```

## Prerequisites

- Python 3.6+
- `transformers` library
- `datasets` library
- PyTorch
- CUDA-enabled GPU (optional but recommended for faster training)

You can install the dependencies using the following command:

```bash
pip install torch transformers datasets
```

## Steps to Run the Project

### 1. Fine-Tune the Model

Use the `main.py` script to fine-tune the DistilBERT model on the IMDb dataset. The script fine-tunes the model and saves it under the `./finetuned_distilbert` directory.

```bash
python main.py
```

### 2. Save the Fine-Tuned Model

The fine-tuned model is saved automatically at the end of training in the `finetuned_distilbert/` directory. Ensure this directory is not included in version control (handled via `.gitignore`).

### 3. Run Inference

Once the model is fine-tuned, use the `inference.py` script to run inference on new input texts. The script loads the fine-tuned model and tokenizer and prints the predicted sentiment (positive/negative) for each input.

```bash
python inference.py
```

### Example Output

```
Input: This movie was fantastic! I loved every minute of it.
Predicted Sentiment: Positive

Input: I didn't like this movie. It was really boring and poorly made.
Predicted Sentiment: Negative
```

## Project Files

### 1. `main.py`

- **Description**: This script fine-tunes a pre-trained DistilBERT model on the IMDb dataset. It tokenizes the dataset, configures the model and trainer, and saves the fine-tuned model at the end of training.

### 2. `inference.py`

- **Description**: This script loads the fine-tuned model and runs inference on new text inputs. It predicts whether the sentiment is positive or negative.

### 3. `.gitignore`

- **Description**: This file ensures that unnecessary files like caches, logs, virtual environments, and model files are not added to version control.

## Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---