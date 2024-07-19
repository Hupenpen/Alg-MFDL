from transformers import XLNetModel, XLNetTokenizer
import torch
from Bio.SeqIO import parse
import numpy as np
import pickle
import time

def get_XLNet_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using {}".format(device))
    model = XLNetModel.from_pretrained("Rostlab/prot_xlnet").to(device).eval()
    tokenizer = XLNetTokenizer.from_pretrained("Rostlab/prot_xlnet", do_lower_case=False)
    return model, tokenizer, device

def get_embeddings(model, tokenizer, device, seqs, per_residue, per_protein, max_residues=4000, max_seq_len=3000, max_batch=1):
    results = {"residue_embs": {}, "protein_embs": {}}
    seq_dict = sorted(seqs.items(), key=lambda kv: len(kv[1]), reverse=True)
    batch = []

    for seq_idx, (pdb_id, seq) in enumerate(seq_dict):
        seq = ' '.join(list(seq))
        batch.append((pdb_id, seq, len(seq)))

        n_res_batch = sum(s_len for _, _, s_len in batch) + len(seq)
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) - 1 or len(seq) > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)
            batch = []

            with torch.no_grad():
                embeddings = model(input_ids, attention_mask=attention_mask).last_hidden_state

            for batch_idx, identifier in enumerate(pdb_ids):
                emb = embeddings[batch_idx, :seq_lens[batch_idx]]
                if per_residue:
                    results["residue_embs"][identifier] = emb.detach().cpu().numpy().squeeze()
                if per_protein:
                    results["protein_embs"][identifier] = emb.mean(dim=0).detach().cpu().numpy().squeeze()

    return results

def get_seq(file_path):
    with open(file_path) as file:
        records = parse(file, "fasta")
        seqs = {record.name: str(record.seq) for record in records}
    return seqs

def save_embeddings(file_path, embeddings):
    with open(file_path, "wb") as f:
        pickle.dump(embeddings, f)

def load_embeddings(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def main(input_file, output_file):
    model, tokenizer, device = get_XLNet_model()
    seq_dict = get_seq(input_file)
    per_residue = False
    per_protein = True

    results = get_embeddings(model, tokenizer, device, seq_dict, per_residue, per_protein)
    protein_embeddings = np.array(list(results["protein_embs"].values()))

    print("Shape of embeddings:", protein_embeddings.shape)
    np.savez(output_file, protein_embeddings=protein_embeddings)

    pickle_path = output_file.replace('.npz', '.pkl')
    save_embeddings(pickle_path, results["protein_embs"])
    loaded_embeddings = load_embeddings(pickle_path)
    print("Number of proteins processed:", len(loaded_embeddings))

if __name__ == '__main__':
    input_file = 'path/to/input_fasta.fasta'  # Placeholder path
    output_file = 'path/to/output_embeddings.npz'  # Placeholder path
    main(input_file, output_file)
