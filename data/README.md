# Frozen source snapshots

The benchmark builder reads only `data/raw/` by default and verifies the hashes
recorded in each output manifest. The local snapshots came from these public
dataset files:

| Dataset split | Source | JSONL SHA-256 |
|---|---|---|
| MATH test | `qwedsacf/competition_math`, commit `d9afe06952835e34b5a148b90043bc04aa09e519`, `test-00000-of-00001-8381d31b2d187522.parquet` | `dd85b4558749f22fe67e535c4c9d8745241def75b5edc05063fd8aa5effd6b23` |
| BoolQ validation | `google/boolq`, `data/validation-00000-of-00001.parquet` | `e8fb84fbf510b022e963cddf3a3aded04151afa0ea0ef1cc1bf22f260ddd2344` |
| OpenBookQA test | `allenai/openbookqa`, `main/test-00000-of-00001.parquet` | `1e448ea67d38e7c1c95bbc961b803170c6976e12cc34596c550bfbdd302a3226` |
| ARC-Challenge test | `allenai/ai2_arc`, `ARC-Challenge/test-00000-of-00001.parquet` | `86ccaace4cd159b5c02b6b3339ebe8a7022a54f1b488e433c060b3bc38ac1f63` |
| CommonsenseQA validation | `tau/commonsense_qa`, `data/validation-00000-of-00001.parquet` | `4e83a600ff0c8ab5f0bbeec89fe45fa35b6e4dd04cb36124c7acb141d233fc14` |

The MATH parquet snapshot has SHA-256
`79f372afea6bedd226750eab23ba54dddede047670446d85178e3e5d0627c191`.
Redistribution in a public artifact must follow each upstream dataset's license;
the hashes and filenames remain sufficient to verify separately downloaded data.
