from pathlib import Path

# Test data paths - these are optional and used only by integration tests
# that require external datasets
TEST_DATA = Path(__file__).parent / "data"
TREC_25_DATA = Path(__file__).parent.parent / "trec25" / "datasets"
TREC_26_DATA = Path(__file__).parent.parent / "trec26" / "datasets"