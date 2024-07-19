import os
import torch
import numpy as np
import pandas as pd

def load_fasta_from_str(fastas, sep='\n'):
    standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
    lst_fastas = []
    lines = fastas.strip().split(sep)
    n_lines = len(lines)
    i = 0
    while i < n_lines:
        if lines[i] and lines[i][0] == '>':
            name = lines[i].strip()
            i += 1
            sequence = ''
            while i < n_lines and lines[i] and lines[i][0] != '>':
                sequence += lines[i].strip()
                i += 1
            assert set(sequence).issubset(standard_aa), 'The sequence must be composed of standard amino acids'
            lst_fastas.append((name, sequence))
        else:
            i += 1
    return lst_fastas

def load_fasta_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return load_fasta_from_str(f.read())

def encoding_by_dde(seqs):
    return torch.tensor(DDE(seqs), dtype=torch.float32)

def DDE(seqs):
    codons_table = {'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2, 'G': 4, 'H': 2, 'I': 3, 'K': 2,
                    'L': 6, 'M': 1, 'N': 2, 'P': 4, 'Q': 2, 'R': 6, 'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2}
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    C_N = 61
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    all_DDE_p = []
    T_m = [(codons_table[pair[0]] / C_N) * (codons_table[pair[1]] / C_N) for pair in diPeptides]

    for seq in seqs:
        N = len(seq) - 1
        D_c = [seq.count(diPeptides[i]) / N for i in range(len(diPeptides))]
        T_v = [T_m[i] * (1 - T_m[i]) / N for i in range(len(diPeptides))]
        DDE_p = [(D_c[i] - T_m[i]) / np.sqrt(T_v[i]) for i in range(len(diPeptides))]
        all_DDE_p.append(DDE_p)

    return np.array(all_DDE_p)

def main():
    seq = load_fasta_from_file("path")
    dde = encoding_by_dde([s[1] for s in seq])
    pd.DataFrame(dde).to_csv('path')
    print(dde)

if __name__ == '__main__':
    main()
