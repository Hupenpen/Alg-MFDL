from transformers import AlbertModel, AlbertTokenizer
import torch
import numpy as np
from Bio.SeqIO import parse
import pickle
import time

def get_Albert_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using {}".format(device))
    model = AlbertModel.from_pretrained("Rostlab/prot_albert").to(device).eval()
    tokenizer = AlbertTokenizer.from_pretrained("Rostlab/prot_albert", do_lower_case=False)
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

def save_results(path, filename, embeddings):
    # Save embeddings as a pickle file
    with open(f"{path}/{filename}_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    # Optionally save as npz for numerical data
    embeddings_array = np.array([v for v in embeddings.values()])
    np.savez(f"{path}/{filename}_embeddings.npz", embeddings=embeddings_array)

def main(input_path, output_path, filename):
    model, tokenizer, device = get_Albert_model()
    seq_dict = get_seq(f"{input_path}/{filename}")
    results = get_embeddings(model, tokenizer, device, seq_dict, per_residue=False, per_protein=True)

    print(f"Processed {len(results['protein_embs'])} proteins.")
    save_results(output_path, filename, results['protein_embs'])

if __name__ == '__main__':
    input_path = 'path_to_fasta_files'  # Directory containing the input FASTA files
    output_path = 'path_to_save_embeddings'  # Directory where embeddings will be saved
    filename = 'example.fasta'  # The FASTA file to process
    main(input_path, output_path, filename)
