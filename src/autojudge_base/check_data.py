from pathlib import Path

import click

from autojudge_base.request import Request, load_requests_from_file, load_requests_from_irds


def print_request_from_path(request_path: Path):

    # Load from JSONL file
    requests: list[Request] = load_requests_from_file(request_path)

    # Access request fields
    for req in requests:
        print(req.request_id)         # Unique topic identifier
        print(req.title)              # The query/question (required)
        print(req.problem_statement)  # Detailed description of the information need
        print(req.background)         # User background/context for personalization


def print_reports(reports_path: Path):
    from autojudge_base.report import load_report

    reports = load_report(reports_path)

    for report in reports:
        # Get sentences with citations in unified format (does not modify report)
        for sentence in report.get_sentences_with_citations():
            text = sentence.text                    # The sentence text
            citations = sentence.citations or []    # Doc IDs ordered by priority

            # Get the cited document content
            for doc_id in citations:
                if report.documents and doc_id in report.documents:
                    doc = report.documents[doc_id]
                    print(f"Citation: {doc.title} - {doc.text[:100]}...")

        for doc in (report.documents or {}).values():
            print(f"Document: {doc.id}: \n -----  {doc.get_text()} \n ---- \n")


@click.group()
def main():
    """Inspect AutoJudge request and report data files."""


@main.command()
@click.argument("request_path", type=click.Path(exists=True, path_type=Path))
def requests(request_path: Path):
    """Print request fields from a JSONL file."""
    print_request_from_path(request_path)


@main.command()
@click.argument("reports_path", type=click.Path(exists=True, path_type=Path))
def reports(reports_path: Path):
    """Print report sentences, citations, and documents."""
    print_reports(reports_path)


if __name__ == "__main__":
    main()

