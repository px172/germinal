#!/usr/bin/env python3
"""Regression test: cdr_bias 'force' residues survive AbMPNN redesign.

cdr_bias biases only hallucination; the accepted sequence is an AbMPNN redesign
of the hallucinated backbone. redesign.get_abmpnn_sequences must therefore add
the cdr_bias FORCE positions to AbMPNN's fixed-position set, or the forced
residues drift during redesign. This test locks that in.

It drives the REAL redesign.get_abmpnn_sequences (the code path that contains the
fix) and replaces only:
  * mp.Process   -> a synchronous in-process fake (redesign forces the 'spawn'
    start method, so a normal monkeypatch would not reach a real child).
  * abmpnn_worker -> a fake that behaves like a fix_pos-RESPECTING AbMPNN (real
    ProteinMPNN/AbMPNN treats fix_pos as a hard constraint): it keeps the input
    residue at every fixed position and mutates every free position to 'W'.

No GPU, no real AbMPNN, no external config. Self-contained cdr_bias/geometry.

Run:  python tests/test_cdr_bias_honored.py       (exit 0 = pass)
  or:  pytest tests/test_cdr_bias_honored.py
"""
import os, sys, pickle, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from germinal.filters import redesign
from germinal.utils.utils import (
    resolve_cdr_bias, compute_cdr_positions, get_sequence_from_pdb,
)

# --- self-contained scFv geometry + cdr_bias (no external config) ------------
CDR_LENGTHS = [13, 10, 13, 11, 8, 9]
FW_LENGTHS = [22, 14, 37, 49, 14, 32, 10]
CDR_BIAS = {
    "H1": {0: "A", 5: "T"},
    "H2": {1: "I", 6: "G"},
    "H3": {0: "A", 12: "Y"},
    "L1": {0: "R", 10: "A", 2: "!C"},   # pos 2 is a FORBID -> must NOT be fixed
    "L2": {5: "L", 7: "S"},
    "L3": {0: "Q", 8: "T"},
}
MARKER = "W"  # free-position mutation marker; must not equal any forced residue

ONE_TO_THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
    "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
    "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
    "W": "TRP", "Y": "TYR",
}


def _write_ca_pdb(path, chains):
    serial = 1
    with open(path, "w") as f:
        for cid, seq in chains:
            for i, aa in enumerate(seq):
                f.write(
                    f"ATOM  {serial:>5}  CA  {ONE_TO_THREE[aa]} {cid}{i+1:>4}    "
                    f"{float(i):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n"
                )
                serial += 1
        f.write("END\n")


def _fake_worker(trajectory_pdb, target_chain, binder_chain,
                 residues_to_fix, run_settings, output_path):
    """fix_pos-respecting AbMPNN: keep fixed binder residues, mutate free -> W."""
    seqs = get_sequence_from_pdb(trajectory_pdb)
    H = seqs[binder_chain]
    tlen = sum(len(seqs[c]) for c in seqs if c != binder_chain)
    fixed = set()
    for tok in residues_to_fix.split(","):
        tok = tok.strip()
        if tok.startswith(binder_chain) and tok[len(binder_chain):].isdigit():
            fixed.add(int(tok[len(binder_chain):]) - 1)
    new_binder = "".join(H[p] if p in fixed else MARKER for p in range(len(H)))
    with open(output_path, "wb") as f:
        pickle.dump({"seq": ["A" * tlen + new_binder], "score": [0.0], "seqid": [1.0]}, f)


class _FakeProcess:
    def __init__(self, target=None, args=()):
        self.target, self.args, self.exitcode = target, args, 0
    def start(self):
        try:
            self.target(*self.args)
        except Exception:
            self.exitcode = 1
            raise
    def join(self):
        pass


def _run(pdb, run_settings, cdr_positions):
    orig_proc, orig_worker = redesign.mp.Process, redesign.abmpnn_worker
    redesign.mp.Process = _FakeProcess
    redesign.abmpnn_worker = _fake_worker
    try:
        return redesign.get_abmpnn_sequences(
            pdb, run_settings, cdr_positions,
            atom_distance_cutoff=0.0, target_chain="A", binder_chain="B",
        )
    finally:
        redesign.mp.Process, redesign.abmpnn_worker = orig_proc, orig_worker


def test_cdr_bias_forced_positions_survive_redesign():
    length = sum(CDR_LENGTHS) + sum(FW_LENGTHS)
    cdr_positions = compute_cdr_positions(CDR_LENGTHS, FW_LENGTHS)
    resolved = resolve_cdr_bias(CDR_BIAS, CDR_LENGTHS, FW_LENGTHS)
    forced = [(p, a) for (p, a, m) in resolved if m == "force"]
    forbid = [(p, a) for (p, a, m) in resolved if m == "forbid"]
    assert forced, "test setup produced no forced positions"
    assert MARKER not in {a for _, a in forced}, "marker collides with a forced residue"

    # synthetic binder that already carries the forced residues (hallucination out)
    binder = ["A"] * length
    for p, a in forced:
        binder[p] = a
    binder = "".join(binder)

    tmp = tempfile.mkdtemp()
    pdb = os.path.join(tmp, "traj.pdb")
    _write_ca_pdb(pdb, [("A", "ACDEFGHIKL"), ("B", binder)])

    base = dict(cdr_lengths=CDR_LENGTHS, fw_lengths=FW_LENGTHS,
                cdr_positions=cdr_positions, max_mpnn_sequences=4,
                mpnn_fix_interface=False, backbone_noise=0.0, model_path="x",
                mpnn_weights="x", sampling_temp=0.1, num_seqs=1, omit_AAs="C")

    # WITH cdr_bias: every forced residue must be preserved
    out = _run(pdb, dict(base, cdr_bias=CDR_BIAS), cdr_positions)[0]["seq"]
    lost = [(p, a, out[p]) for p, a in forced if out[p] != a]
    assert not lost, f"forced positions NOT preserved with cdr_bias: {lost}"

    # forbid positions must NOT be pinned (only force is) -> free -> redesigned
    for p, _a in forbid:
        assert out[p] == MARKER, f"forbid position {p} was pinned but should be free"

    # WITHOUT cdr_bias: the same forced positions must drift (proves the fix acts)
    out2 = _run(pdb, dict(base), cdr_positions)[0]["seq"]
    drifted = [p for p, _a in forced if out2[p] == MARKER]
    assert len(drifted) == len(forced), (
        f"expected all {len(forced)} forced positions to drift without the fix, "
        f"got {len(drifted)}"
    )
    return len(forced), len(forbid)


if __name__ == "__main__":
    nf, nb = test_cdr_bias_forced_positions_survive_redesign()
    print(f"PASS: {nf} forced residues preserved through AbMPNN with the fix, "
          f"lost without it; {nb} forbid position(s) correctly left free.")
