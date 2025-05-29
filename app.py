import json
import torch
import nltk
import pickle
import random
from datetime import datetime
import numpy as np
import pandas as pd
import logging # <--- ADD THIS LINE

from nnet import NeuralNet # <--- ADD THIS LINE (assuming nnet.py is in the same directory)
from nltk_utils import bag_of_words # Assuming nltk_utils.py is in the same directory
from flask import Flask, render_template, request, jsonify

# Initialize random seed
random.seed(datetime.now().timestamp())

# --- Configuration & Model Loading ---
DEVICE = torch.device('cpu')
NLU_MODEL_FILE = "models/data.pth"
PREDICTION_MODEL_FILE = 'models/fitted_model.pickle2'
LIST_OF_SYMPTOMS_FILE = 'data/list_of_symptoms.pickle'
DESCRIPTION_FILE = "data/symptom_Description.csv"
PRECAUTION_FILE = "data/symptom_precaution.csv"
SEVERITY_FILE = "data/Symptom-severity.csv"
UI_SYMPTOMS_FILE = "static/assets/files/ds_symptoms.txt"

# --- Helper: Name Cleaning Function ---
def clean_name(name_input):
    """Standardizes names (for symptoms, diseases) for consistent matching."""
    if isinstance(name_input, str):
        return name_input.lower().strip().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    return name_input # Return as is if not a string (e.g., for NaN)

# --- Load NLU Model (for symptom tag identification) ---
try:
    nlu_model_data = torch.load(NLU_MODEL_FILE, map_location=DEVICE)
    nlu_input_size = nlu_model_data['input_size']
    nlu_hidden_size = nlu_model_data['hidden_size']
    nlu_output_size = nlu_model_data['output_size']
    nlu_all_words = nlu_model_data['all_words']  # Vocabulary for NLU's Bag-of-Words
    # NLU tags should ideally be canonical. We clean them as a safeguard.
    nlu_identified_tags_list = [clean_name(tag) for tag in nlu_model_data['tags']]
    nlu_model_state = nlu_model_data['model_state']

    nlp_model = NeuralNet(nlu_input_size, nlu_hidden_size, nlu_output_size).to(DEVICE)
    nlp_model.load_state_dict(nlu_model_state)
    nlp_model.eval()
    print("NLU model loaded successfully.")
except Exception as e:
    print(f"Error loading NLU model from {NLU_MODEL_FILE}: {e}")
    # Add fallback or exit if critical
    nlp_model = None


# --- Load Data for Disease Prediction & Information ---
try:
    diseases_description_df = pd.read_csv(DESCRIPTION_FILE)
    diseases_description_df['Disease_Clean'] = diseases_description_df['Disease'].apply(clean_name)
    print(f"Disease descriptions loaded. Sample cleaned names: {diseases_description_df['Disease_Clean'].head().tolist()}")

    disease_precaution_df = pd.read_csv(PRECAUTION_FILE)
    disease_precaution_df['Disease_Clean'] = disease_precaution_df['Disease'].apply(clean_name)
    print(f"Disease precautions loaded. Sample cleaned names: {disease_precaution_df['Disease_Clean'].head().tolist()}")

    symptom_severity_df = pd.read_csv(SEVERITY_FILE)
    symptom_severity_df['Symptom_Clean'] = symptom_severity_df['Symptom'].apply(clean_name)
    symptom_severity_df['weight'] = pd.to_numeric(symptom_severity_df['weight'], errors='coerce')
    print(f"Symptom severity loaded. Sample cleaned symptoms: {symptom_severity_df['Symptom_Clean'].head().tolist()}")

    with open(LIST_OF_SYMPTOMS_FILE, 'rb') as f:
        # This is the canonical, ordered list of symptoms the prediction_model expects.
        # Ensure elements are cleaned to the canonical form.
        symptoms_list_for_prediction = [clean_name(s) for s in pickle.load(f)]
    print(f"Canonical symptom list for prediction loaded. Count: {len(symptoms_list_for_prediction)}. Sample: {symptoms_list_for_prediction[:5]}")

    with open(PREDICTION_MODEL_FILE, 'rb') as f:
        prediction_model = pickle.load(f)
    print("Disease prediction model loaded successfully.")

except FileNotFoundError as e:
    print(f"ERROR: Data file not found: {e}. Please check paths.")
    # Handle missing files (e.g., exit or run in a limited mode)
except Exception as e:
    print(f"Error loading data files: {e}")


# For UI suggestions - load once
ALL_AVAILABLE_SYMPTOMS_FOR_UI = []
try:
    with open(UI_SYMPTOMS_FILE, "r") as file:
        for s_line in file:
            # Clean for display, but internal logic relies on canonical forms
            ALL_AVAILABLE_SYMPTOMS_FOR_UI.append(s_line.strip().replace("'", "").replace("_", " ").replace(",\n", ""))
except FileNotFoundError:
    print(f"Warning: {UI_SYMPTOMS_FILE} not found. UI symptom suggestions might be empty.")
    ALL_AVAILABLE_SYMPTOMS_FOR_UI = ["fever", "cough", "headache"] # Basic fallback


user_symptoms_identified_canonical = set() # Stores canonical symptom tags from NLU

app = Flask(__name__)
# Configure Flask logging
app.logger.setLevel(logging.INFO) # Or DEBUG for more verbosity
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
app.logger.addHandler(handler)


# --- NLU Function: Identifies a symptom TAG from a sentence ---
def get_symptom_tag_from_sentence(sentence_text):
    if nlp_model is None:
        app.logger.error("NLU model is not loaded. Cannot process symptom.")
        return "unknown_symptom", 0.0 # Fallback

    tokenized_sentence = nltk.word_tokenize(sentence_text)
    X_bow = bag_of_words(tokenized_sentence, nlu_all_words) # Use NLU's specific vocabulary
    X_bow = X_bow.reshape(1, X_bow.shape[0])
    X_bow = torch.from_numpy(X_bow).to(DEVICE)

    with torch.no_grad(): # Important for inference
        output = nlp_model(X_bow)

    _, predicted_idx = torch.max(output, dim=1)
    # The tag from nlu_identified_tags_list is already cleaned to canonical form
    tag = nlu_identified_tags_list[predicted_idx.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted_idx.item()].item()

    return tag, prob


@app.route('/')
def index():
    user_symptoms_identified_canonical.clear() # Clear symptoms for a new session
    ui_symptoms_json = json.dumps(ALL_AVAILABLE_SYMPTOMS_FOR_UI)
    return render_template('index.html', data=ui_symptoms_json)


@app.route('/symptom', methods=['POST']) # Typically POST for sending data
def predict_symptom_route():
    try:
        data = request.get_json()
        if not data or 'sentence' not in data:
            app.logger.error("Invalid request: Missing 'sentence' in JSON payload.")
            return jsonify({"error": "Missing 'sentence' in request"}), 400
        
        sentence = data['sentence']
        app.logger.info(f"Received sentence: '{sentence}'")

        response_payload = {} # Prepare a structured response

        if sentence.replace(".", "").replace("!","").lower().strip() == "done":
            if not user_symptoms_identified_canonical:
                response_payload["response_text"] = random.choice([
                    "I can't know what disease you may have if you don't enter any symptoms :)",
                    "Meddy can't know the disease if there are no symptoms...",
                    "You first have to enter some symptoms!"
                ])
            else:
                app.logger.info(f"Processing 'done'. User symptoms collected (canonical): {user_symptoms_identified_canonical}")

                # Create the input vector for the prediction_model
                x_test_input = [1 if model_symptom in user_symptoms_identified_canonical else 0 for model_symptom in symptoms_list_for_prediction]
                
                app.logger.info(f"Symptoms matched for prediction model input (count of 1s): {sum(x_test_input)}")
                app.logger.debug(f"x_test_input vector: {x_test_input}")


                if sum(x_test_input) == 0:
                    response_payload["response_text"] = "I couldn't match your reported symptoms to the symptoms my prediction model understands. Please try describing them differently or ensure they are common medical symptoms."
                else:
                    x_test_np = np.asarray(x_test_input)
                    predicted_disease_raw = prediction_model.predict(x_test_np.reshape(1, -1))[0]
                    predicted_disease_clean = clean_name(predicted_disease_raw)
                    
                    app.logger.info(f"Predicted disease (raw): '{predicted_disease_raw}', (cleaned for lookup): '{predicted_disease_clean}'")
                    response_payload["predicted_disease_raw"] = predicted_disease_raw # Send raw for display if needed
                    response_payload["predicted_disease_clean"] = predicted_disease_clean


                    description_match = diseases_description_df[diseases_description_df['Disease_Clean'] == predicted_disease_clean]
                    precaution_match = disease_precaution_df[disease_precaution_df['Disease_Clean'] == predicted_disease_clean]

                    description_text = description_match['Description'].iloc[0] if not description_match.empty else "Description not available."
                    response_payload["description"] = description_text
                    
                    precautions_list_text = []
                    if not precaution_match.empty:
                        prec_row = precaution_match.iloc[0]
                        for i in range(1, 5):
                            col_name = f'Precaution_{i}'
                            if col_name in prec_row and pd.notna(prec_row[col_name]):
                                precautions_list_text.append(str(prec_row[col_name]).strip())
                    
                    precautions_final_text = "Precautions: " + ", ".join(precautions_list_text) if precautions_list_text else "Precautions not available."
                    response_payload["precautions"] = precautions_final_text

                    # Display raw disease name with spaces, not underscores
                    display_disease_name = str(predicted_disease_raw).replace('_', ' ')
                    response_text_html = f"It looks to me like you may have {display_disease_name}. <br><br> <i>Description: {description_text}</i> <br><br><b>{precautions_final_text}</b>"

                    # Calculate severity
                    severity_scores = []
                    for symptom_tag in user_symptoms_identified_canonical: # These are already cleaned canonical tags
                        severity_row = symptom_severity_df[symptom_severity_df['Symptom_Clean'] == symptom_tag]
                        if not severity_row.empty and 'weight' in severity_row.columns and pd.notna(severity_row['weight'].iloc[0]):
                            severity_scores.append(severity_row['weight'].iloc[0])
                            app.logger.info(f"Severity for {symptom_tag}: {severity_row['weight'].iloc[0]}")
                        else:
                            app.logger.warning(f"No (valid) severity weight found for symptom: {symptom_tag}")
                    
                    if severity_scores: # Check if list is not empty
                        mean_severity = np.mean(severity_scores)
                        max_severity = np.max(severity_scores)
                        app.logger.info(f"Severity scores: {severity_scores}, Mean: {mean_severity:.2f}, Max: {max_severity}")
                        if mean_severity > 4 or max_severity > 5:
                            response_text_html += "<br><br>Considering your symptoms appear to be of notable severity, and Meddy isn't a real doctor, you should consider talking to one. :)"
                    
                    response_payload["response_text"] = response_text_html
                
                user_symptoms_identified_canonical.clear()

        else: # Not "done", so identify symptom
            if nlp_model is None:
                 response_payload["response_text"] = "I'm having trouble with my NLU model right now. Please try again later."
            else:
                symptom_tag_identified, prob = get_symptom_tag_from_sentence(sentence)
                app.logger.info(f"NLU identified: Tag='{symptom_tag_identified}', Prob={prob:.4f}")

                if prob > 0.50: # Confidence threshold
                    display_symptom_name = symptom_tag_identified.replace('_', ' ')
                    response_payload["response_text"] = f"Hmm, I'm {(prob * 100):.2f}% sure this is {display_symptom_name}."
                    user_symptoms_identified_canonical.add(symptom_tag_identified) # Add canonical tag
                    response_payload["symptom_identified"] = display_symptom_name
                    response_payload["symptom_confidence"] = prob
                else:
                    response_payload["response_text"] = "I'm sorry, but I don't understand that. Can you please try rephrasing or describe another symptom?"
            
            app.logger.info(f"Current user symptoms: {user_symptoms_identified_canonical}")

        return jsonify(response_payload)

    except Exception as e:
        app.logger.error(f"Error in /symptom route: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred. Please try again."}), 500


if __name__ == '__main__':
    # Download nltk data if not present (run once)
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    
    # For WordNetLemmatizer if you switch from PorterStemmer in nltk_utils.py
    # try:
    #     nltk.data.find('corpora/wordnet')
    # except nltk.downloader.DownloadError:
    #     nltk.download('wordnet')
    # try:
    #     nltk.data.find('corpora/omw-1.4')
    # except nltk.downloader.DownloadError:
    #     nltk.download('omw-1.4')

    app.run(debug=True, host='0.0.0.0') # host='0.0.0.0' makes it accessible on your network