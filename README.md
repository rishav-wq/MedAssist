# MedAssist - AI Medical Symptom Analyzer

MedAssist is an AI-powered medical chatbot designed to understand user-reported symptoms and provide potential related conditions, along with descriptions and general precautions. **This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.** Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## Features

*   **Symptom Recognition:** Understands natural language descriptions of medical symptoms.
*   **Disease Prediction:** Suggests potential medical conditions based on the provided symptoms.
*   **Information Provider:** Offers descriptions and general precautions for predicted conditions.
*   **Interactive Chat Interface:** User-friendly web interface for interaction.
*   **Symptom Suggestions:** Provides auto-suggestions as the user types.

## Demo Screenshot
*(Consider adding a screenshot of your chatbot in action here, for example, by placing an image named `screenshot.png` in a `docs` folder and using: `![MedAssist Demo](./docs/screenshot.png)`)*

## Requirements

*   Python 3.7+ (Python 3.8-3.11 recommended for best compatibility with recent libraries)
*   Dependencies (see `requirements.txt`):
    *   Flask
    *   PyTorch (CPU version is sufficient for running the application)
    *   NLTK (Natural Language Toolkit)
    *   NumPy
    *   Scikit-learn
    *   Pandas
    *   matplotlib (Primarily used during model training/evaluation phases, not strictly required for just running the pre-trained app)

## Setup and Installation

It is highly recommended to use a virtual environment to manage dependencies.

**1. Clone the Repository (if applicable):**
   If you have downloaded the project as a ZIP, extract it. If you are cloning:
   ```bash
   git clone <your-repository-url>
   cd med_chat_bots # Or your project's root folder name


**2. Create and Activate a Virtual Environment:**
    Linux/macOS:
    bash python3 -m venv venv source venv/bin/activate
    Windows (PowerShell/CMD):
    bash python -m venv venv venv\Scripts\activate
    (If python is not aliased to Python 3, you might need py -3 or python3 for creating the venv)   


**3. Install Dependencies using requirements.txt:**
    Ensure the requirements.txt file (provided below or created by you) is present in the root of your project. Then, in your activated virtual environment, run:
    pip install -r requirements.txt

Project Structure
A brief overview of the key files and directories:
med_chat_bots/
├── app.py                  # Main Flask application logic
├── nnet.py                 # Neural Network class definition (for NLU)
├── nltk_utils.py           # NLTK helper functions (tokenizer, Bag-of-Words)
├── requirements.txt        # Python package dependencies
├── data/
│   ├── dataset.csv             # Disease-to-Symptoms mapping (for training prediction model)
│   ├── list_of_symptoms.pickle # Canonical list of symptoms (feature order for prediction model)
│   ├── symptom_Description.csv # Descriptions for each disease
│   ├── symptom_precaution.csv  # Precautions for each disease
│   └── Symptom-severity.csv    # Symptom severity weightings
├── models/
│   ├── data.pth                # Trained NLU model (PyTorch) and associated data (vocabulary, tags)
 
├── static/
│   ├── css/
│   │   └── style.css           # Stylesheets for the web interface
│   ├── js/
│   │   └── main.js             # Frontend JavaScript for chat interaction and UI
│   └── assets/                 # Other static assets (images, fonts, etc.)
│       └── files/
│           └── ds_symptoms.txt # List of symptoms for UI auto-suggestions (display purposes)
├── templates/
│   └── index.html              # Main HTML page for the chatbot interface
├── intents.json              # Definitions of symptoms and user phrasing patterns (training data for NLU model)
├── Meddy.ipynb               # (Assumed) Jupyter Notebook for model training/data exploration
└── README.md     



How to Use
 1. The chatbot will greet you.
     Describe your symptoms one by one in the chat input field. Press Enter or click the send button.
     Example: "I have a headache"
     Example: "feeling nauseous"
 2. The chatbot will attempt to confirm its understanding of each symptom. This identified symptom is added to a list for the current session.
     Once you have listed all your symptoms, type the word "Done" (case-insensitive) as a separate message and send it.
     The chatbot will then analyze the collected symptoms and provide:
     A potential medical condition.
 3. A description of that condition.
     General precautions associated with it.
     A warning if symptoms appear severe based on a pre-defined severity score.
     You can use the "Start Over" button to clear the conversation and begin a new consultation. This will also clear any symptoms collected on the server for your session.