---
configs:
- config_name: inputs
  data_files:
  - split: train
    path: ["runs/*.jsonl"]
- config_name: truths
  data_files:
  - split: train
    path: ["trec-leaberboard.txt"]

tira_configs:
  resolve_inputs_to: "."
  resolve_truths_to: "."
  baseline:
    link: https://github.com/trec-auto-judge/auto-judge-code/tree/main/trec25/judges/naive
    command: /naive-baseline.py --rag-responses $inputDataset --output $outputDir/trec-leaderboard.txt
    format:
      name: ["trec-eval-leaderboard"]
  input_format:
    name: "trec-rag-runs"
  truth_format:
    name: "trec-eval-leaderboard"
  evaluator:
    image: ghcr.io/trec-auto-judge/auto-judge-code/cli:0.0.1
    command: trec-auto-judge evaluate --input ${inputRun}/trec-leaderboard.txt --aggregate --output ${outputDir}/evaluation.prototext
ir_dataset:
  directory: "../example-irds-corpus"
---
