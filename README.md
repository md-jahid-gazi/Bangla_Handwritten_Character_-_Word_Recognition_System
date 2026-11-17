# Bangla_Handwritten_Character_&_Word_Recognition_System

**Skills for Industry Competitiveness and Innovation Program (SICIP) | BRAC University**

A Bangla handwritten character and word recognition project using the BanglaLekha-Isolated dataset. The assignment requires a drawing interface that predicts a complete Bangla word and displays the result as Bengali Unicode text.

## Student Information

- **Name:** Md. Jahid Gazi
- **Student ID:** 1000128068
- **Institution:** BRAC University
- **Program:** SICIP

## 1. Project Objective

Build a system that accepts handwritten Bangla input and predicts the complete word. The expected examples are:

| Handwritten input | Expected output |
|---|---|
| জল | জল |

The application should display the joined predicted word, not only separate character predictions.

## 2. Dataset

- **Dataset:** BanglaLekha-Isolated
- **Source:** [Mendeley Data – BanglaLekha-Isolated](https://data.mendeley.com/datasets/hf6sf8zrkc/2)
- **Expected extracted path:** `data/BanglaLekha-Isolated/Images/`

Example structure:

```text
BanglaLekha-Isolated/
└── Images/
    ├── 1/
    ├── 2/
    ├── 3/
    └── ...
```

BanglaLekha-Isolated is an isolated-character dataset. Numeric folder names represent dataset class IDs; they must not be assumed to be Bengali Unicode characters. A verified class-ID-to-Unicode mapping is required to convert predictions into Bengali text.

## 3. Project Components

| File / folder | Purpose |
|---|---|
| `train.py` | Train the model and record experiments with MLflow |
| `app.py` | Streamlit drawing interface and prediction |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Docker image build instructions |
| `labels.json` | Model output index and dataset class metadata, if used |
| `unicode_map.json` | Verified dataset class ID to Bengali Unicode mapping, if used |
| `models/` | Generated trained model files |
| `screenshots/` | Application and MLflow evidence |
| `docs/` | Additional project documentation, if maintained |

## 4. Approach

The training pipeline reads the dataset images, prepares image samples and labels, trains a neural-network model, evaluates it, and records experiment information in MLflow.

The Streamlit application provides a drawing area where a user can write a Bangla word and request a prediction. The main result should be displayed as joined Bengali Unicode text.

**Important technical distinction:** A model trained only on isolated-character images is a character classifier. Joining its character predictions is a character-sequence baseline; it is not automatically a fully trained word-level OCR model. Genuine word-level recognition requires complete word images paired with verified Unicode word transcriptions and word-level evaluation.

## 5. Installation

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux / WSL:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Dataset Setup

Extract the dataset so the images are available under:

```text
data/BanglaLekha-Isolated/Images/
```

Keep the original class-folder IDs unchanged. Inspect the class folders and verify the mapping from each dataset class ID to its Bengali Unicode label. Do not guess mappings from numeric folder names or image filenames.

## 7. Model Training and MLflow

Run training:

```bash
python train.py
```

The training script should train the model, evaluate it, save the model and corresponding label metadata, and log the experiment to MLflow. Confirm the actual output paths and metrics from the submitted code.

Start the MLflow interface:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Open:

```text
http://localhost:5000
```

**Assignment requirement:** Complete at least two MLflow training runs and capture a screenshot of the experiment page showing both runs.

Record actual experiment settings and results:

| Field | Run 1 | Run 2 |
|---|---|---|
| Run name |  |  |
| Dataset / sample count |  |  |
| Model architecture |  |  |
| Epochs |  |  |
| Batch size |  |  |
| Learning rate |  |  |
| Training metrics |  |  |
| Validation/test metrics |  |  |
| Notes |  |  |

Fill this table using real MLflow results. Do not invent values.

## 8. Run the Streamlit Application

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, normally:

```text
http://localhost:8501
```

Expected user flow:

1. Open the Streamlit application.
2. Draw one complete Bangla word.
3. Select the prediction action.
4. View the predicted word in Bengali Unicode.

The assignment requires whole-word output. Character-by-character diagnostic details may be shown as additional information, but they should not replace the joined word.

## 9. Docker

Build the image:

```bash
docker build -t bangla-handwritten-character-word-recognition-system:0.1 .
```

Run the container:

```bash
docker run -p 8501:8501 bangla-handwritten-character-word-recognition-system:0.1

**Docker image name:** `bangla-handwritten-character-word-recognition-system:0.1`. Docker image names must use lowercase characters and cannot contain spaces or `&`, so the image name uses a Docker-compatible version of the project title.
```

Open:

```text
http://localhost:8501
```

The container should start Streamlit with a host binding accessible outside the container, commonly:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

Do not claim Docker support is verified until the image has been built and the running container has been tested.

## 10.
Screenshots

Save genuine screenshots in the `screenshots/` directory:

- `streamlit_app_1.png` — Streamlit page displaying a predicted Bangla word.
- `mlflow_experiment_1.png` — MLflow experiment page showing at least two training runs.

## 11. Suggested Repository Structure

```text
Bangla_Handwritten_Character_&_Word_Recognition_System/
├── train.py
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
├── labels.json
├── unicode_map.json
├── github_link.txt
├── models/
│   └── model.keras
├── screenshots/
│   ├── streamlit_app.png
│   └── mlflow_experiment.png
└── docs/
```

Adjust optional files to match the actual implementation. Do not commit datasets, virtual environments, cache files, or large generated artifacts unnecessarily.

## 12. Evaluation Notes and Limitations

### Character classification versus word recognition

Isolated-character test accuracy measures classification of isolated character samples. It does not establish complete-word recognition accuracy.

Bengali handwriting contains connected strokes, vowel signs, phala, conjuncts, and other dependent marks. Simple vertical whitespace segmentation can split a glyph, merge adjacent glyphs, or separate a mark from its base character.

A Unicode mapping converts a predicted class ID into text; it cannot correct a wrong classification or faulty segmentation.

### Recommended future improvements

1. Verify every dataset class-to-Unicode mapping.
2. Obtain or create complete handwritten word images with Unicode transcriptions.
3. Train a sequence-recognition model using word images and transcriptions.
4. Improve Bengali-aware segmentation if using a character-based method.
5. Evaluate using word exact-match accuracy and character error rate (CER).
6. Test with handwriting from writers not represented in training data.
7. Keep preprocessing consistent between training and inference.
8. Record reproducible experiment settings in MLflow.

For genuine word-level evaluation, use a separate test set of complete word images and verified Unicode transcriptions. Report word exact-match accuracy and character error rate (CER).
