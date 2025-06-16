from flask import Flask, request, jsonify, render_template
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
import threading
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import io
import logging

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def clean_sequence(seq):
    return ''.join([aa for aa in seq if aa in AMINO_ACIDS])

def amino_acid_composition(seq):
    seq = seq.upper()
    length = len(seq)
    return [seq.count(aa) / length if length > 0 else 0 for aa in AMINO_ACIDS]

def extract_features(peptide):
    features = []
    peptide = peptide.upper()
    peptide = clean_sequence(peptide)
    
    if not peptide:
        return [0] * 25  # 20 amino + 5 features
    
    analysis = ProteinAnalysis(peptide)
    features.extend(amino_acid_composition(peptide))  
    features.append(analysis.charge_at_pH(7.0))
    features.append(analysis.gravy())
    features.append(analysis.molecular_weight())
    try:
        features.append(analysis.isoelectric_point())
    except:
        features.append(0)
    features.append(len(peptide))  
    return features

def build_feature_matrix(sequences):
    return np.array([extract_features(seq) for seq in sequences])

def load_data_from_fasta(amp_file, non_amp_file):
    X = []
    y = []   
    for record in SeqIO.parse(amp_file, "fasta"):
        seq = clean_sequence(str(record.seq))
        X.append(seq)
        y.append(1)
    for record in SeqIO.parse(non_amp_file, "fasta"):
        seq = str(record.seq)
        X.append(seq)
        y.append(0)
    return X, y

app = Flask(__name__, static_folder='.')
rf_model = None
svm_model = None
gb_model = None
scaler = None
models_trained = False

def train_models():
    global rf_model, svm_model, gb_model, scaler, models_trained
    try:
        print("Loading training data from FASTA files...")
        X_seqs, y = load_data_from_fasta("AMPDATASET.fasta", "NONAMP3K.fasta")
        print(f"Loaded {len(X_seqs)} peptides for training.")
        
        feature_matrix = build_feature_matrix(X_seqs)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_matrix)

        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, y, test_size=0.3, random_state=42, stratify=y)

        print("Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.4f}")

        print("Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_test)
        print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, gb_pred):.4f}")

        print("Training SVM...")
        base_svm = LinearSVC(random_state=42, max_iter=5000, dual='auto')
        svm_model = CalibratedClassifierCV(base_svm)
        svm_model.fit(X_train, y_train)
        svm_pred = svm_model.predict(X_test)
        print(f"SVM Accuracy: {accuracy_score(y_test, svm_pred):.4f}")

        models_trained = True  
        print("All models trained successfully.")

    except Exception as e:
        print(f"Error during training: {e}")
        models_trained = False 

def validate_peptide(seq):
    seq = seq.upper()
    return all(aa in AMINO_ACIDS for aa in seq) and len(seq) > 0

def predict_for_sequence(seq, model_choice):
    global rf_model, svm_model, gb_model, scaler
    seq = seq.strip().upper()
    features_unscaled = np.array(extract_features(seq)).reshape(1, -1)
    features_scaled = scaler.transform(features_unscaled)

    results = []

    if model_choice == 'random_forest' or model_choice == 'hybrid':
        if rf_model is None:
            return {'error':'Random Forest model not trained yet'}
        rf_prob = rf_model.predict_proba(features_scaled)[0, 1]
        rf_pred = rf_model.predict(features_scaled)[0]
        results.append({
            'sequence': seq,
            'model': 'Random Forest',
            'class': 'Antimicrobial' if rf_pred == 1 else 'Non-antimicrobial',
            'probability': round(rf_prob, 4),
            'likelihood_AMP': round(rf_prob, 4)
        })

    if model_choice == 'svm' or model_choice == 'hybrid':
        if svm_model is None:
            return {'error':'SVM model not trained yet'}
        svm_prob = svm_model.predict_proba(features_scaled)[0, 1]
        svm_pred = svm_model.predict(features_scaled)[0]
        results.append({
            'sequence': seq,
            'model': 'SVM',
            'class': 'Antimicrobial' if svm_pred == 1 else 'Non-antimicrobial',
            'probability': round(svm_prob, 4),
            'likelihood_AMP': round(svm_prob, 4)
        })

    if model_choice == 'gradient_boosting' or model_choice == 'hybrid':
        if gb_model is None:
            return {'error':'Gradient Boosting model not trained yet'}
        gb_prob = gb_model.predict_proba(features_scaled)[0, 1]
        gb_pred = gb_model.predict(features_scaled)[0]
        results.append({
            'sequence': seq,
            'model': 'Gradient Boosting',
            'class': 'Antimicrobial' if gb_pred == 1 else 'Non-antimicrobial',
            'probability': round(gb_prob, 4),
            'likelihood_AMP': round(gb_prob, 4)
        })

    return results

def split_protein_into_peptides(protein_seq, peptide_length=8):
    return [protein_seq[i:i + peptide_length] for i in range(0, len(protein_seq), peptide_length)]

@app.route('/predict', methods=['POST'])
def predict():
    global models_trained
    if not models_trained:
        return jsonify({'message': 'Models are still training, please wait...'}), 400

    if request.is_json:
        data = request.get_json()
        if 'sequence' not in data or 'model' not in data or 'inputType' not in data:
            return jsonify({'message': 'Invalid input: sequence, model, and inputType required'}), 400
        fasta_input=data['sequence']
        seqs = data['sequence'].strip().splitlines()
        model_choice = data['model'].strip().lower()
        input_type = data['inputType'].strip().lower()

        results = []
        for seq in seqs:
            if input_type == 'peptide':
                if validate_peptide(seq):
                    result = predict_for_sequence(seq, model_choice)
                    results.extend(result)
            elif input_type == 'protein':
                peptides = split_protein_into_peptides(seq)
                if validate_peptide(seq):
                    for peptide in peptides:
                        result = predict_for_sequence(peptide, model_choice)
                        results.extend(result)

        return jsonify(results)

    if 'file' in request.files and 'model' in request.form:
        uploaded_file = request.files['file']
        model_choice = request.form['model'].strip().lower()
        input_type = request.form.get('inputType', '').strip().lower()
        if uploaded_file.filename == '':
            return jsonify({'message': 'No selected file'}), 400

        try:
            seqs = []
            file_stream = io.StringIO(uploaded_file.stream.read().decode("utf-8"))
            for record in SeqIO.parse(file_stream, 'fasta'):
                seq_clean = clean_sequence(str(record.seq))
                if len(seq_clean) > 0:
                    seqs.append(seq_clean)

            if len(seqs) == 0:
                return jsonify({'message': 'No valid sequences found in file'}), 400

            results = []
            for seq in seqs:
                if input_type == 'peptide':
                    if validate_peptide(seq):
                        pred = predict_for_sequence(seq, model_choice)
                        results.extend(pred)
                elif input_type == 'protein':
                    peptides = split_protein_into_peptides(seq)
                    if validate_peptide(seq):
                        for peptide in peptides:
                            pred = predict_for_sequence(peptide, model_choice)
                            results.extend(pred)
            return jsonify(results)

        except Exception as e:
            return jsonify({'message': f'Error processing file: {str(e)}'}), 400

    return jsonify({'message': 'Invalid request: provide JSON data or file upload with model'}), 400

@app.route('/')
def serve_index():
    return render_template('index.html')

if __name__ == '__main__':
    threading.Thread(target=train_models).start()
    app.run(debug=True)
