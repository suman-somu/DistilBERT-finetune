
---

# Fine-Tuning and Inference with Hugging Face Transformers
|![image](https://github.com/user-attachments/assets/91df679f-eae9-4a31-a854-2df9f10df2a2) | ![image](https://github.com/user-attachments/assets/ebc8491f-bec8-4eed-a612-c676d74e1487) |
|------------|------------|


This project demonstrates how to fine-tune a pre-trained DistilBERT model on a sentiment analysis task using the IMDb dataset, and how to run inference on new data using the fine-tuned model. The project uses the Hugging Face `transformers` library for model fine-tuning and inference.

## Project Structure
```
├── results/                  # Directory for saving finetuned model
├── finetune.py               # Main script for fine-tuning the model
├── inference.py              # running inference with the fine-tuned model
├── infer_base.py             # running inference with the base model
├── .gitignore                
└── README.md                 
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

Use the `finetune.py` script to fine-tune the DistilBERT model on the IMDb dataset. The script fine-tunes the model and saves it under the `./results` directory.

```bash
python finetune.py
```

### 2. Save the Fine-Tuned Model

The fine-tuned model is saved automatically at the end of training in the `results/` directory.

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

### 1. `finetune.py`

- **Description**: This script fine-tunes a pre-trained DistilBERT model on the IMDb dataset. It tokenizes the dataset, configures the model and trainer, and saves the fine-tuned model at the end of training.

### 2. `inference.py`

- **Description**: This script loads the fine-tuned model and runs inference on sample text inputs. 


### 3. `infer_base.py`

- **Description**: This script loads the base DistilBERT model and runs inference on sample text inputs. 

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
