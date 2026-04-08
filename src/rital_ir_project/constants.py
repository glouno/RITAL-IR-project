"""Static metadata for the HiRAG paper and the datasets it uses."""

from __future__ import annotations

from dataclasses import dataclass


ULTRADOMAIN_BASE_URL = "https://huggingface.co/datasets/TommyChien/UltraDomain/resolve/main"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    role: str
    note: str


MAIN_PAPER_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="mix",
        url=f"{ULTRADOMAIN_BASE_URL}/mix.jsonl",
        role="main",
        note="UltraDomain subset used in the main HiRAG evaluation.",
    ),
    DatasetSpec(
        name="cs",
        url=f"{ULTRADOMAIN_BASE_URL}/cs.jsonl",
        role="main",
        note="Computer science subset used in the main HiRAG evaluation.",
    ),
    DatasetSpec(
        name="legal",
        url=f"{ULTRADOMAIN_BASE_URL}/legal.jsonl",
        role="main",
        note="Legal subset used in the main HiRAG evaluation.",
    ),
    DatasetSpec(
        name="agriculture",
        url=f"{ULTRADOMAIN_BASE_URL}/agriculture.jsonl",
        role="main",
        note="Agriculture subset used in the main HiRAG evaluation.",
    ),
)


OPTIONAL_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="hotpotqa",
        url="https://hotpotqa.github.io/",
        role="optional",
        note="Appendix-style EM/F1 evaluation uses 1,000 validation queries.",
    ),
    DatasetSpec(
        name="2wikimultihopqa",
        url="https://github.com/Alab-NII/2wikimultihop",
        role="optional",
        note="Appendix-style EM/F1 evaluation uses 1,000 validation queries.",
    ),
)


ALL_DATASETS = {spec.name: spec for spec in (*MAIN_PAPER_DATASETS, *OPTIONAL_DATASETS)}


PLAN_TEXT = """HiRAG reproduction plan

1. Download the four UltraDomain subsets used in the paper: mix, cs, legal, agriculture.
2. Build a flat entity graph from chunked documents.
3. Add higher graph layers by clustering entity embeddings with GMM and generating summary entities.
4. Stop adding layers when cluster sparsity stops improving.
5. Detect graph communities on the final graph.
6. At query time, retrieve:
   - top-k local entities
   - connected community reports
   - shortest-path bridge relations between community key entities
7. Compare at least:
   - HiRAG
   - NaiveRAG
   - HiRAG without HiIndex
   - HiRAG without bridge knowledge
8. If time permits, extend to HotpotQA and 2WikiMultiHopQA for EM/F1 evaluation.
"""
