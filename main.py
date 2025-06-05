import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from scipy.spatial.distance import cosine
import numpy as np
from pymongo import MongoClient
import os
import tempfile
from flask import Flask,request,jsonify
from datetime import datetime
app=Flask(__name__)

client = MongoClient(os.getenv("MONGO_URI"))

db = client["voice_auth_db"]
users = db["users"]
enrolled_embeds={}

model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device":"cpu"}
)

def load_audio(file_path, sample_rate=16000):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    try:
        waveform, orig_sample_rate = torchaudio.load(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {file_path}: {str(e)}")
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if orig_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_sample_rate, sample_rate)
        waveform = resampler(waveform)
    return waveform

def extract_embedding(audio_path):
    waveform = load_audio(audio_path)
    with torch.no_grad():
        embedding = model.encode_batch(waveform)
    return embedding.squeeze().cpu().numpy()

def authenticate(enrolled_embedding, test_embedding, threshold=0.2):
    similarity = 1 - cosine(enrolled_embedding, test_embedding)
    print(f"Similarity score: {similarity:.4f}")
    return similarity > threshold, similarity
@app.route('/enroll', methods=['POST'])
def enroll_speaker():
    if 'audio' not in request.files or 'user_id' not in request.form:
        return jsonify({"error": "Missing audio file or user_id"}), 400

    audio_file = request.files['audio']
    user_id = request.form['user_id']
    if users.find_one({"user_id": user_id}):
        return jsonify({"error": "User already enrolled"}), 400
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        embedding = extract_embedding(tmp_path).tolist()  

        users.insert_one({
            "user_id": user_id,
            "audio_path": f"voices/{user_id}.wav", 
            "embedding": embedding,  
            "enrolled_at": datetime.now()
        })

        return jsonify({
            "status": "success",
            "user_id": user_id,
            "message": "Speaker enrolled successfully"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)  

@app.route('/authenticate', methods=['POST'])
def verify_speaker():
    if 'audio' not in request.files or 'user_id' not in request.form:
        return jsonify({"error": "Missing audio file or user_id"}), 400

    audio_file = request.files['audio']
    user_id = request.form['user_id']
    user_data = users.find_one({"user_id": user_id})
    if not user_data:
        return jsonify({"error": "User not enrolled"}), 400
    enrolled_embedding = np.array(user_data["embedding"])
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        test_embedding = extract_embedding(tmp_path)
        auth_result, similarity = authenticate(enrolled_embedding, test_embedding)

        return jsonify({
            "authenticated": bool(auth_result),
            "similarity_score": float(similarity),  
            "user_id": user_id,
            "threshold": 0.5 
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)  

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)