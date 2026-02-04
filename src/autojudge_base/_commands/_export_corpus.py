import click
from pathlib import Path
from ..io import load_hf_dataset_config_or_none
import gzip


@click.argument("corpus-directory", type=Path)
def export_corpus(corpus_directory: Path) -> int:
    """Export a corpus into a pre-defined directory so that everything is self-contained."""
    import ir_datasets
    from tira.ir_datasets_loader import IrDatasetsLoader
    # todo create a shared method

    files_to_remove = ["documents.jsonl", "queries.jsonl", "queries.xml", "metadata.json"]
    for f in files_to_remove:
        if (corpus_directory / f).is_file():
            (corpus_directory / f).unlink()

    irds_id = load_hf_dataset_config_or_none(corpus_directory / "README.md", ["ir_dataset"])["ir_dataset"]["ir_datasets_id"]
    ds = ir_datasets.load(irds_id)
    irds_loader = IrDatasetsLoader()
    irds_loader.load_dataset_for_fullrank(
        irds_id,
        corpus_directory,
        output_dataset_truth_path=None,
        skip_qrels=True
    )

    docs_file = corpus_directory / "documents.jsonl"
    docs = (docs_file).read_text()
    docs_file.unlink()
    with gzip.open(corpus_directory / "corpus.jsonl.gz", "wt") as f:
        f.write(docs)


    return 0