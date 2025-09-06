# LayoutLM Fine-Tuning Pipeline

Welcome to the **LayoutLM Fine-Tuning Pipeline** project! This repository serves as a continuation of my previous work on document extraction and LabelStudio preparation. Here, I focus on adapting annotated data from LabelStudio to fine-tune a LayoutLM model for named entity recognition (NER) on document layouts, specifically targeting financial documents like balance sheets. As my first attempt at fine-tuning a LayoutLM model, this project emphasizes understanding the model's input/output requirements, data preprocessing challenges, and end-to-end ML workflows—key skills for roles in AI and data science.

Developed to showcase hands-on experience with transformer-based models, this pipeline demonstrates proficiency in data adaptation, model training, and inference, while highlighting problem-solving in handling structured document data.

## Overview

Fine-tuning models like LayoutLM is essential for tasks involving document understanding, where spatial layout (e.g., bounding boxes) plays a crucial role alongside text. This project takes LabelStudio annotations as input, preprocesses them into LayoutLM-compatible formats, trains the model on a single document format (as an initial test), saves the fine-tuned model, and performs inference on test data. It addresses real-world challenges such as token segmentation for long sequences, bbox normalization, and label fusion from OCR and annotations.

This "first-try" approach allowed me to gain deep insights into LayoutLM's architecture, including how it processes tokenized text with spatial embeddings, and how to optimize for custom labels in NER tasks.

## Features

- **Data Preprocessing**: Scripts to fuse LabelStudio labels with OCR data, normalize bounding boxes, and tokenize inputs for LayoutLM.
- **Sequence Segmentation**: Handles long documents by splitting into segments of max length (512 tokens).
- **Model Fine-Tuning**: Fine-tunes LayoutLM-base for token classification on custom labels (e.g., accounts, balances, headers).
- **Inference and Testing**: Loads the saved model to generate predictions on new data.
- **Modular Structure**: Organized for easy extension to multi-format training or advanced fine-tuning.

## Project Structure

After cleanup, the repository includes only essential files:

```
LayoutLM_Fine-Tuning_Pipeline/
├── data_preparation/
│   ├── json_segmenter.py          # Splits long sequences into segments
│   ├── ls_ocr_fusion.py           # Fuses LS labels with OCR tokens
│   ├── prepare_layoutlm_input.py  # Prepares training data for LayoutLM
│   ├── prepare_test_input.py      # Prepares test data from LS JSON
├── testing/
│   ├── test_layoutlm.py           # Runs inference on test data
├── training/
│   ├── train_layoutlm.py          # Fine-tunes and saves the model
├── .gitignore
├── README.md
├── requirements.txt               # List of dependencies (create this if not present)
```
## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/aychachouchene/LAYOUTLM_FINETUNING_PIPELINE.git
   cd LAYOUTLM_FINETUNING_PIPELINE
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  
   ```

3. **Install Dependencies**:
```bash
   pip install -r requirements.txt
```

4. **Verify Setup**:
   Run a preparation script with sample data (provide your own JSON inputs):
   ```bash
   python data_preparation/prepare_layoutlm_input.py
   ```

## Usage

### Workflow
1. **Prepare Training Data**:
   - Fuse LS and OCR: `python data_preparation/ls_ocr_fusion.py`
   - Format for LayoutLM: `python data_preparation/prepare_layoutlm_input.py`
   - Segment long sequences: `python data_preparation/json_segmenter.py`

2. **Train the Model**:
   ```bash
   python training/train_layoutlm.py
   ```
   - Prompts for JSON path; trains for 2 epochs (configurable).

3. **Prepare Test Data**:
   ```bash
   python data_preparation/prepare_test_input.py
   ```

4. **Run Inference**:
   ```bash
   python testing/test_layoutlm.py
   ```
   - Outputs predictions in JSON.


## Skills Demonstrated
This project, as my initial foray into fine-tuning LayoutLM, highlights the following skills:
- **Machine Learning & Transformers**: Fine-tuning LayoutLM for NER, understanding multimodal inputs (text + bboxes), and managing token classification outputs.
- **Data Preprocessing**: Custom scripts for bbox normalization, tokenization with Hugging Face's LayoutLMTokenizer, sequence splitting, and label/OCR fusion—showing expertise in handling structured data.
- **Python Programming**: Advanced use of libraries like Torch, Transformers, JSON handling, and tqdm for progress tracking; modular code design for scalability.
- **Problem-Solving**: Addressing challenges like long-sequence truncation, label mapping, and device-agnostic training (CPU/GPU).
- **ML Workflow**: End-to-end pipeline from data adaptation to model saving and testing, emphasizing iterative learning and debugging in a real-world scenario.

## Contributing
Contributions welcome! Fork, branch, commit, and PR. Ensure PEP 8 compliance.

## License
MIT License—see [LICENSE](LICENSE) for details.

## Contact
For questions or collaboration, reach out at [aycha.chouchene@etudiant-enit.utm.tn](mailto:aycha.chouchene@etudiant-enit.utm.tn)
 or [choucheneaycha03@gmail.com](mailto:choucheneaycha03@gmail.com).