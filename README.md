# A Predictive System for Mental Health Condition Analysis Using Natural Language Processing and Machine Learning

This project demonstrates an AI-powered system for analyzing patient journal entries to provide insights into emotional states and potential diagnostic support for mental health conditions. It combines advanced Natural Language Processing (NLP) techniques with machine learning to identify patterns in text and emotional trajectories over time.

## Table of Contents

- [Project Overview](https://colab.research.google.com/drive/1WzmDctJ7rOlVMQKOhDW4ta0B3wCXOUw7#project-overview)
- [Features](https://colab.research.google.com/drive/1WzmDctJ7rOlVMQKOhDW4ta0B3wCXOUw7#features)
- [Dataset](https://colab.research.google.com/drive/1WzmDctJ7rOlVMQKOhDW4ta0B3wCXOUw7#dataset)
- [Models Used](https://colab.research.google.com/drive/1WzmDctJ7rOlVMQKOhDW4ta0B3wCXOUw7#models-used)
- [Project Structure](https://colab.research.google.com/drive/1WzmDctJ7rOlVMQKOhDW4ta0B3wCXOUw7#project-structure)
- [Setup and Installation](https://colab.research.google.com/drive/1WzmDctJ7rOlVMQKOhDW4ta0B3wCXOUw7#setup-and-installation)
- [Usage](https://colab.research.google.com/drive/1WzmDctJ7rOlVMQKOhDW4ta0B3wCXOUw7#usage)
- [Key Insights](https://colab.research.google.com/drive/1WzmDctJ7rOlVMQKOhDW4ta0B3wCXOUw7#key-insights)
  
## Project Overview

The goal of this project is to develop a tool that can assist mental health professionals by automating the preliminary analysis of patient journal entries. By detecting fine-grained emotions, mapping them to broader Ekman categories, extracting text-based features, and employing machine learning models, the system aims to flag potential concerns and provide an objective perspective on a patient's emotional state and diagnostic trends.

## Features

- **Text Preprocessing**: Cleans raw journal entries by removing links, numbers, and irrelevant special characters.
- **Emotion Detection**: Utilizes a pre-trained RoBERTa model (`SamLowe/roberta-base-go_emotions`) to extract 28 fine-grained emotions from text.
- **Ekman Emotion Mapping**: Aggregates fine-grained emotions into 7 universal Ekman categories (Anger, Disgust, Fear, Joy, Sadness, Surprise, Neutral).
- **Emotional Feature Engineering**: Calculates statistical features (mean, volatility, slope) for Ekman emotions over several days to capture emotional trends.
- **TF-IDF Vectorization**: Converts journal text into numerical representations using Term Frequency-Inverse Document Frequency (TF-IDF).
- **Machine Learning Models**:
  - Random Forest Classifier trained on emotional features.
  - Random Forest Classifier trained on TF-IDF features.
  - **Hybrid Random Forest Model** combining both emotional and TF-IDF features for enhanced predictive power.
- **Hyperparameter Tuning**: Uses `GridSearchCV` with `StratifiedKFold` to optimize model performance.
- **Model Evaluation**: Comprehensive evaluation using accuracy, classification reports, and confusion matrices for unoptimized and optimized models on validation and test sets.
- **Data Leakage Check**: Ensures no accidental overlap between training and testing datasets.
- **Interactive Visualizations**:
  - Plotly-based emotional drift plots showing daily dominant emotions.
  - Matplotlib radar charts for fine-grained and Ekman emotion profiles.
  - TF-IDF word frequency plots for each diagnosis.
- **Model Interpretability (SHAP)**: Provides explanations for model predictions, including technical waterfall plots and clinician-friendly summaries of feature importance.
- **Streamlit Web Application**: An interactive user interface (`app.py`) for entering journal entries day-by-day and visualizing the analysis results in real-time.
- **Model Persistence**: Saves and loads trained models and transformers using `joblib` for efficient deployment and reuse.

## Dataset

The project uses a synthetic dataset, `synthetic_dataset_300.csv`, containing anonymized patient journal entries over 7 days, along with a `diagnosis` field.

## Models Used

- **RoBERTa-base-go\_emotions**: For fine-grained emotion classification.
- **Random Forest Classifier**: For multi-class classification of mental health diagnoses.
- **TF-IDF Vectorizer**: For text feature extraction.

## Project Structure

- `Colab_code.ipynb`: The main Jupyter Notebook containing all the code for data loading, preprocessing, model training, evaluation, visualization, and Streamlit app generation.
- `emotion_ai_utils.py`: A Python script containing helper functions extracted from the notebook, adapted for use with the Streamlit application.
- `app.py`: The Streamlit web application script.
- `synthetic_dataset_300.csv`: The synthetic patient journal dataset.
- `artifacts/`: Directory for storing saved models, vectorizers, and other essential artifacts (created dynamically).
- `roberta_go_emotions_local/`: Local copy of the downloaded RoBERTa model (created dynamically).

## Setup and Installation

To set up and run this project locally or in Google Colab, follow these steps:

1. **Clone the repository**:
   ```
   git clone https://github.com/your-username/emotional-journal-analysis.git
   cd emotional-journal-analysis

   ```
2. **Create a virtual environment (recommended)**:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   ```
3. **Install dependencies**: The project relies on several Python libraries. You can install them using pip:
   ```
   pip install pandas numpy torch transformers scikit-learn plotly matplotlib seaborn joblib scipy tqdm pyngrok streamlit shap

   ```
4. **Download the Dataset**: Ensure `synthetic_dataset_300.csv` is in the project root directory. If not, you might need to download it separately.
5. **Hugging Face Token (Optional, for first run)**: The RoBERTa model will be downloaded automatically. If you encounter issues with Hugging Face rate limits, you might need to login using a Hugging Face token.
6. **Ngrok Authentication Token (for Streamlit app)**:
   - Create an account on [ngrok.com](https://www.google.com/url?q=https%3A%2F%2Fngrok.com%2F).
   - Obtain your authentication token.
   - In Google Colab, add your `NGROK_AUTH_TOKEN` to Colab secrets (under the 🔑 icon in the left panel). This is essential for exposing the Streamlit app to a public URL.

## Usage

### Running the Jupyter Notebook (`Colab_code.ipynb`)

Open `Colab_code.ipynb` in Google Colab or your preferred Jupyter environment. Run all cells sequentially to:

- Load and preprocess data.
- Train and evaluate the emotional, TF-IDF, and hybrid models.
- Perform visualizations and model interpretability analyses.
- Generate the `emotion_ai_utils.py` and `app.py` files.
- Save all necessary model artifacts to the `artifacts/` directory.

### Running the Streamlit Application

After running the notebook and saving the artifacts, you can launch the Streamlit app. The notebook includes a cell that automates this process using `pyngrok`:

```
# In Colab_code.ipynb, look for the cell containing:
# !pip install pyngrok
# !pkill -f streamlit
# !streamlit run app.py &>/content/logs.txt &
# ... and run it.

```

Once the Streamlit app is running, `pyngrok` will provide a public URL. Open this URL in your web browser to interact with the Emotional Journal application. You can enter patient names, submit daily journal entries, and view the comprehensive emotional and diagnostic analysis.

### Local Model Testing

The notebook also includes sections for manually testing the individual and combined models with new journal entries, allowing you to observe their predictions and confidence levels.

## Key Insights

- The hybrid model (combining emotional and TF-IDF features) generally outperforms individual models, demonstrating the value of multi-modal data for mental health diagnosis.
- RoBERTa's ability to capture fine-grained emotions provides rich input for emotional feature engineering.
- Visualizations like emotional drift plots and SHAP analyses offer valuable insights for clinicians, helping them understand emotional trajectories and the driving factors behind a diagnosis.
- Class imbalance is a significant challenge in mental health datasets, requiring techniques like `class_weight='balanced'` in models. scrie asta in code pentru read me github 
