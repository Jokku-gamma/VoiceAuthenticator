import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from scipy.spatial.distance import cosine
import numpy as np


model=EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

def load_audio(fpath,sample_rate=16000):
    wave,org_sample=torchaudio.load(fpath)
    if org_sample != sample_rate:
        resampler=torchaudio.transforms.Resample(org_sample,sample_rate)
        wave=resampler(wave)
    return wave

def extract_embedding(apath):
    wave=load_audio(apath)
    if wave.shape[0]>1:
        wave=torch.mean(wave,dim=0,keepdim=True)
    with torch.no_grad():
        embedding=model.encode_batch(wave)
    return embedding.squeeze().cpu().numpy()

def authenticate(og,test,threh=0.2):
    sim=1=cosine(og,test)
    print(f"Similarity Score :{sim:.4f}")
    return sim>threh,sim

def main():
    og=""
    test=""
    og_embed=extract_embedding(og)
    test_embed=extract_embedding(test)

    is_auth,score=authenticate(og_embed,test_embed)
    if is_auth:
        print(f"Authentication successful : {score:.4f}")

    else:
        print(f"Authenticaton failed : {score:.4f}")

if __name__=="__main__":
    main()

    