import glob
import os
import scipy
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import faiss
import json
import random
from transformers import AutoTokenizer, AutoModel, T5EncoderModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name="google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5EncoderModel.from_pretrained(model_name).to(device)
    
def process_text(text):
    text = text.lower().strip()
    report = text.replace('ekg', 'ecg')
    report = report.strip('*** ').strip(' ***').strip('***').strip('=-').strip('=')
    return report


def process_file(mat_file):
    try:
        text = scipy.io.loadmat(mat_file)["text"][0]
        return process_text(text)
    except:
        return None


def get_batch_embeddings(batch_texts, model, tokenizer):
    inputs = tokenizer(batch_texts, max_length=256, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings


def faiss_write(data_root="data/pretrain/processed_data"):
    mat_files = glob.glob(os.path.join(data_root,"*.mat"))
    print("Number of Samples: ", len(mat_files))

    train_texts = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, mat_file): mat_file for mat_file in mat_files}

        for future in tqdm(as_completed(futures), total=len(mat_files)):
            result = future.result()
            if result:
                train_texts.append(result)

    train_texts = list(set(train_texts))
    batch_size = 128
    all_embeddings = []

    for i in tqdm(range(0, len(train_texts), batch_size)):
        batch_texts = train_texts[i:i + batch_size]
        batch_embeddings = get_batch_embeddings(batch_texts, model, tokenizer)
        all_embeddings.extend(batch_embeddings)

    embeddings = np.array(all_embeddings)

    normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    train_text_embds = normalizer(embeddings)
    index = faiss.IndexFlatL2(train_text_embds.shape[1])

    index.add(train_text_embds)
    embedding_to_sample_map = {i: train_texts[i] for i in range(len(train_texts))}
    
    with open(f'data/mimic_iv_ecg_index.json', 'w') as f:
        json.dump(embedding_to_sample_map, f)
    faiss.write_index(index, f"data/mimic_iv_ecg_index.faiss")
    print("Successfully Save Files !")


def faiss_read(query_text, model, tokenizer, index, embedding_to_sample_map,  k=10):
    embedding_to_sample_map = {int(k): v for k, v in embedding_to_sample_map.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    query_inputs = tokenizer(query_text, return_tensors="pt").to(device)
    model = model.to(device)

    with torch.no_grad():
        query_embedding = model(**query_inputs).last_hidden_state.mean(dim=1).cpu().numpy()
        normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
        query_embedding = normalizer(query_embedding)

    distances, indices = index.search(-query_embedding, k)

    random_idx = random.choice(indices[0])
    random_sample = embedding_to_sample_map[random_idx]
    return random_sample


if __name__ == "__main__":
    # faiss_write("data/pretrain/processed_data") 
    faiss_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")# NOTE
    faiss_lm = T5EncoderModel.from_pretrained("google/flan-t5-small").to(device)
    faiss_index = faiss.read_index(f"data/mimic_iv_ecg_index.faiss")

    with open(f'data/mimic_iv_ecg_index.json', 'r') as f:
        faiss_embedding_to_sample_map = json.load(f)
    print(faiss_read("Atrial fibrillation. Possible anterior infarct - age undetermined. Inferior/lateral ST-T changes may be due to myocardial ischemia. Low QRS voltages in precordial leads. Abnormal ECG", faiss_lm, faiss_tokenizer, faiss_index, faiss_embedding_to_sample_map))
