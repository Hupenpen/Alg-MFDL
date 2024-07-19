from transformers import T5EncoderModel, T5Tokenizer
import torch
import numpy as np
import h5py
import time
import pickle
from Bio.SeqIO import parse

def get_T5_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using {}".format(device))
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device).eval()
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
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

def main(input_path, output_path):
    model, tokenizer, device = get_T5_model()
    per_residue = False
    per_protein = True

    seq_dict = get_seq(input_path)
    results = get_embeddings(model, tokenizer, device, seq_dict, per_residue, per_protein)

    protein_embeddings = np.array(list(results["protein_embs"].values()))
    print(protein_embeddings.shape)
    np.savez(output_path, protein_embeddings)

    pickle_path = output_path.replace('.npz', '.pkl')
    with open(pickle_path, "wb") as f:
        pickle.dump(results["protein_embs"], f)

if __name__ == '__main__':
    input_path = 'input_file_path.fasta'  # This should be replaced with the actual input file path
    output_path = 'output_file_path.npz'  # This should be replaced with the desired output file path
    main(input_path, output_path)
