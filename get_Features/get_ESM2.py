import os
import torch
import numpy as np
import pandas as pd
import esm
import collections
from Bio.SeqIO import parse

def load_fasta_from_str(fastas, sep='\n'):
    standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
    lst_fastas = []
    lines = fastas.strip().split(sep)
    for line in lines:
        if line and line[0] == '>':
            name = line.strip()
            sequence = ''
            while lines and lines[0] != '>':
                sequence += lines.pop(0).strip()
            lst_fastas.append((name, sequence))
    return lst_fastas

def load_fasta_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return load_fasta_from_str(f.read())

def esm_embeddings(peptide_sequence_list):
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    batch_labels, batch_strs, batch_tokens = batch_converter(peptide_sequence_list)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

    embeddings_results = collections.defaultdict(list)
    for i, each_seq_rep in enumerate(sequence_representations):
        embeddings_results[i].extend(each_seq_rep.tolist())
    return pd.DataFrame(embeddings_results).T

def get_seq(path):
    file = open(path)
    records = parse(file, "fasta")
    seqs = [str(record.seq) for record in records]
    return seqs

def gen_esm_feature0(inputpath, resultpath):
    sequence_list = get_seq(inputpath)
    embeddings_results = pd.DataFrame()
    for i, seq in enumerate(sequence_list, 1):
        peptide_sequence_list = [(seq, seq)]
        one_seq_embeddings = esm_embeddings(peptide_sequence_list)
        embeddings_results = pd.concat([embeddings_results, one_seq_embeddings])
    embeddings_results.to_csv(resultpath)

def esm_embeddings1(peptide_sequence_list, initial_trim_length=20):
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    embeddings_results = []
    for name, seq in peptide_sequence_list:
        try:
            batch_labels, batch_strs, batch_tokens = batch_converter([(name, seq)])
            batch_tokens = batch_tokens.to(device)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            results = model(batch_tokens, repr_layers=[33])
            token_representations = results["representations"][33]
            seq_repr = token_representations[0, 1:batch_lens[0] - 1].mean(0)
            embeddings_results.append((name, seq_repr.cpu().tolist()))
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if 'out of memory' in str(e) and len(seq) > initial_trim_length:
                seq = seq[:-initial_trim_length]
            else:
                raise e
    return embeddings_results

def gen_esm_feature1(inputpath, resultpath):
    sequence_list = get_seq(inputpath)
    embeddings_results = []
    for i, seq in enumerate(sequence_list, 1):
        peptide_sequence_list = [(seq, seq)]
        one_seq_embeddings = esm_embeddings1(peptide_sequence_list)
        embeddings_results.extend(one_seq_embeddings)
    embeddings_df = pd.DataFrame(embeddings_results, columns=['Name', 'Embedding'])
    embeddings_df.to_csv(resultpath)

def main():
    input = 'inputpath'
    output = 'outputpath'
    gen_esm_feature1(input, output)

if __name__ == '__main__':
    main()
