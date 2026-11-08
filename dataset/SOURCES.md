# SOURCES (facts only, for the bibliography)

Licences were read from the dataset cards on 2026-11-06. Each row of the JSON splits carries its own `source` field (public dataset name with licence, `template`, or `synthetic:generated-qa-v1`), which says which licence applies to that row.

| Name | URL | Authors / publisher | Licence | Used for | Rows in dataset/ |
|---|---|---|---|---|---|
| Bitext Retail Banking LLM Chatbot Training Dataset | https://huggingface.co/datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset | Bitext Innovations International (2024) | CDLA-Sharing-1.0 | banking_qa | 2610 |
| Alpaca-Cleaned | https://huggingface.co/datasets/yahma/alpaca-cleaned | cleaning project gururise/AlpacaDataCleaned (https://github.com/gururise/AlpacaDataCleaned), published on the Hugging Face hub by yahma | CC-BY-4.0 on the dataset card; upstream Stanford Alpaca is CC BY-NC 4.0 | general | 1300 |
| Stanford Alpaca (upstream of Alpaca-Cleaned) | https://github.com/tatsu-lab/stanford_alpaca | Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P. and Hashimoto, T.B. (2023) | data CC BY-NC 4.0 | none directly | 0 |
| Concept Q&A (banking and fraud-screening definitions) | written for this project, `preprocess_data/concepts.py` | project work | project work | banking_qa | 250 |
| Bank Account Fraud (BAF) suite, Base | https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 | Jesus, S., Pombal, J., Alves, D., Cruz, A., Saleiro, P., Ribeiro, R.P., Gama, J. and Bizarro, P. (2022), NeurIPS Datasets and Benchmarks | CC BY-NC-SA 4.0 | explain_score contexts (scored by the team LR), dataset_fact numbers (EDA tables) | see build_manifest.json |
| Team classifier and model card | https://github.com/KennyDao1324/ai4cyber-fraud (commit d0fc662) | the team | team repo | model scores, threshold, model facts | n/a |
| Template text (questions, answers, policy) | `preprocess_data/templates_*.py`, `facts.py` | project work | project work | explain_score, dataset_fact, policy | n/a |

## Retrieval

- Bitext: Hugging Face `main` revision 3e3621092fc6baaf7f53ceb6f091c60ae99acb67 (last modified 2024-07-15), downloaded 2026-11-06; raw CSV sha256 f306958595d35f3ab50b6f2c8c1fcf1ba63f9c54e4c993c9905f2856082fe67a; 25,545 rows.
- Alpaca-Cleaned: Hugging Face `main` revision 12567cabf869d7c92e573c7c783905fc160e9639 (last modified 2023-04-10), downloaded 2026-11-06; raw JSON sha256 bd844b8247a0f543804b6ce0882b0aaec4bbf5e8d66167df6213a0f1e4fe878b; 51,760 rows.
- Author names above come from the dataset cards; Huy to confirm them against the sources before the bibliography is final.

## Licence facts

- Stanford Alpaca data (Taori et al. 2023) is released under CC BY-NC 4.0 (non-commercial). Its outputs were generated with OpenAI text-davinci-003, and OpenAI's terms restrict using model output to develop competing models. Alpaca-Cleaned relabels the cleaned data as CC-BY-4.0. Use here: non-commercial coursework, research only.
- Bitext (CDLA-Sharing-1.0) and BAF (CC BY-NC-SA 4.0) are share-alike: data derived from them that is shared must keep the same terms. The JSON splits therefore mix rows under several licences; the per-row `source` field shows which one applies. BAF is also non-commercial.
- The Bitext and Alpaca rows are filtered copies (see Filters below); the BAF-derived numbers are statistics and model scores, not BAF rows.

## Filters (Bitext)

- Raw 25,545 rows; without offensive-language tag W 21,816; short, placeholder-free 5,898; without protected / decision words, profanity, URLs or code 5,735; without answers that ask for account numbers, passwords, PINs, names or other personal data 3,991; after de-duplication 3,991.
- Rows taken: 2,610 across train, val and test (tier 1 intents close to account opening, security and fraud handling first, then fees, branches and ATMs, then loans and mortgages).

## Evaluated and not used

- Josephgflowers/Finance-Instruct-500k (Apache-2.0): about 518k rows; fraud-related questions mostly refer to a missing text or contain extraction noise.
- gbharti/finance-alpaca (MIT label): informal forum answers, upstream licence unclear.
- databricks/databricks-dolly-15k (CC-BY-SA-3.0, human-written): not chosen because Alpaca-Cleaned was available under a CC-BY label; this is a preference, not a licence ruling (Bitext and BAF are share-alike as well).
- PolyAI/banking77 (CC-BY-4.0): intent labels only, no answers.
