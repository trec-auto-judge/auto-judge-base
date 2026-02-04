import unittest
from trec_auto_judge.io import load_runs_failsave
from pathlib import Path
from trec_auto_judge.report import *
from . import TREC_25_DATA
from tempfile import TemporaryDirectory
from glob import glob
import json



class TestSimplisticAccessToRuns(unittest.TestCase):
    
    #  The test cases need to be adjusted to meet the Report format. 
    
    def _assert_reports_metadata(self, actual, expected_metadata):
        self.assertEqual(len(expected_metadata), len(actual))
        for idx, expected_m in enumerate(expected_metadata):
            m = actual[idx].metadata
            self.assertEqual(expected_m["team_id"], m.team_id)
            self.assertEqual(expected_m["run_id"], m.run_id)
            self.assertEqual(str(expected_m["topic_id"]), str(m.topic_id))
            self.assertEqual(str(expected_m["narrative_id"]), str(m.narrative_id))

    
    #  Here an example:  `test_new_report_format`
    def test_new_report_format(self):
        expected_metadata = [
            {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_01', 'topic_id': '28', 'narrative_id': '28'},
            {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_02', 'topic_id': '28', 'narrative_id': '28'},
            {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_02_citations', 'topic_id': '28', 'narrative_id': '28'}
        ]
        expected_paths = [
            'my_best_run_01',
            'my_best_run_02',
            'my_best_run_02_with_citations'
        ]
        print("TREC_25_DATA / spot-check-dataset", TREC_25_DATA / "spot-check-dataset" /"runs")
        actual = load_runs_failsave(TREC_25_DATA / "spot-check-dataset" / "runs")
        print("actual reports", actual)
        self.assertEqual(3, len(actual))

        self.assertEqual(len(expected_metadata), len(actual))
        self._assert_reports_metadata(actual=actual, expected_metadata=expected_metadata)

        # LD: this information is not exposed
        # self.assertEqual(expected_paths, [Path(i["path"]).name for i in actual])    
    
        
    def _assert_topic_unified(self, reports):
        """After unification, topic_id and narrative_id must refer to same id."""
        for r in reports:
            self.assertIsNotNone(r.metadata.topic_id)
            self.assertIsNotNone(r.metadata.narrative_id)
            self.assertEqual(str(r.metadata.topic_id), str(r.metadata.narrative_id))


    def _mk_reports(self, *, topic_style: str, narrative_as_int: bool = False):
        def md(run_id: str):
            if topic_style == "topic_id":
                return ReportMetaData(
                    team_id="my_fantastic_team",
                    run_id=run_id,
                    topic_id="28",
                )
            if topic_style == "narrative_id":
                return ReportMetaData(
                    team_id="my_fantastic_team",
                    run_id=run_id,
                    narrative_id=(28 if narrative_as_int else "28"),
                )
            if topic_style == "both":
                return ReportMetaData(
                    team_id="my_fantastic_team",
                    run_id=run_id,
                    topic_id="28",
                    narrative_id=(28 if narrative_as_int else "28"),
                )
            raise ValueError(f"unknown topic_style: {topic_style}")

        return [
            Report(metadata=md("my_best_run_01"), responses=[]),
            Report(metadata=md("my_best_run_02"), responses=[]),
            Report(metadata=md("my_best_run_02_citations"), responses=[]),
        ]

    def test_non_existing_repo_yields_no_runs(self):
        actual = load_runs_failsave(Path("/this-directory/does-not-exist"))
        self.assertEqual([], actual)

    def test_trec25_spot_check_runs(self):
        expected_metadata = [
            {"team_id": "my_fantastic_team", "run_id": "my_best_run_01", "topic_id": "28", "narrative_id": "28"},
            {"team_id": "my_fantastic_team", "run_id": "my_best_run_02", "topic_id": "28", "narrative_id": "28"},
            {"team_id": "my_fantastic_team", "run_id": "my_best_run_02_citations", "topic_id": "28", "narrative_id": "28"},
        ]

        actual = load_runs_failsave(TREC_25_DATA / "spot-check-dataset" / "runs")
        self.assertEqual(3, len(actual))
        self._assert_reports_metadata(actual, expected_metadata)
        self._assert_topic_unified(actual)

    # --- object-based tests ---

    def test_reports_topic_specified_by_topic_id(self):
        reports = self._mk_reports(topic_style="topic_id")
        self._assert_topic_unified(reports)

    def test_reports_topic_specified_by_narrative_id_str(self):
        reports = self._mk_reports(topic_style="narrative_id", narrative_as_int=False)
        self._assert_topic_unified(reports)

    def test_reports_topic_specified_by_narrative_id_int(self):
        reports = self._mk_reports(topic_style="narrative_id", narrative_as_int=True)
        self._assert_topic_unified(reports)

    def test_reports_topic_specified_by_both_fields(self):
        reports = self._mk_reports(topic_style="both", narrative_as_int=False)
        self._assert_topic_unified(reports)

    def test_reports_topic_specified_by_both_fields_narrative_int(self):
        reports = self._mk_reports(topic_style="both", narrative_as_int=True)
        self._assert_topic_unified(reports)

    def test_reports_inconsistent_topic_and_narrative_raises(self):
        with self.assertRaises(ValueError):
            ReportMetaData(
                team_id="my_fantastic_team",
                run_id="my_best_run_01",
                topic_id="28",
                narrative_id="29",  # inconsistent
            )
    
    # def test_non_existing_repo_yields_no_runs(self):
    #     expected = []
    #     actual = load_runs_failsave(Path("/this-directory/does-not-exist"))
    #     self.assertEqual(expected, actual)

    # def test_trec25_spot_check_runs(self):
    #     expected_metadata = [
    #         {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_01', 'topic_id': '28', 'narrative_id': '28'},
    #         {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_02', 'topic_id': '28', 'narrative_id': '28'},
    #         {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_02_citations', 'topic_id': '28', 'narrative_id': '28'}
    #     ]
    #     expected_paths = [
    #         'my_best_run_01',
    #         'my_best_run_02',
    #         'my_best_run_02_with_citations'
    #     ]
    #     actual = load_runs_failsave(TREC_25_DATA / "spot-check-dataset")
    #     self.assertEqual(3, len(actual))
    #     self.assertEqual(expected_metadata, [i["metadata"] for i in actual])
    #     self.assertEqual(expected_paths, [Path(i["path"]).name for i in actual])

    # def test_trec25_spot_check_runs_with_narrative(self):
    #     expected_metadata = [
    #         {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_01', 'topic_id': '28', 'narrative_id': '28'},
    #         {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_02', 'topic_id': '28', 'narrative_id': '28'},
    #         {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_02_citations', 'topic_id': '28', 'narrative_id': '28'}
    #     ]
    #     expected_paths = [
    #         'my_best_run_01',
    #         'my_best_run_02',
    #         'my_best_run_02_with_citations'
    #     ]

    #     with TemporaryDirectory() as d:
    #         for f in glob(str(TREC_25_DATA / "spot-check-dataset" / "runs") + "/*" ):
    #             f = Path(f)
    #             txt = []
    #             for l in f.read_text().split("\n"):
    #                 if not l:
    #                     continue
    #                 l = json.loads(l)
    #                 l["metadata"]["narrative_id"] = l["metadata"]["topic_id"]
    #                 del l["metadata"]["topic_id"]
    #                 txt.append(json.dumps(l))
    #             (Path(d) / f.name).write_text("\n".join(txt))

    #         actual = load_runs_failsave(Path(d))
    #         self.assertEqual(3, len(actual))
    #         self.assertEqual(expected_metadata, [i["metadata"] for i in actual])
    #         self.assertEqual(expected_paths, [Path(i["path"]).name for i in actual])

    # def test_trec25_spot_check_runs_with_narrative_and_topic_id(self):
    #     expected_metadata = [
    #         {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_01', 'topic_id': '28', 'narrative_id': '28'},
    #         {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_02', 'topic_id': '28', 'narrative_id': '28'},
    #         {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_02_citations', 'topic_id': '28', 'narrative_id': '28'}
    #     ]
    #     expected_paths = [
    #         'my_best_run_01',
    #         'my_best_run_02',
    #         'my_best_run_02_with_citations'
    #     ]

    #     with TemporaryDirectory() as d:
    #         for f in glob(str(TREC_25_DATA / "spot-check-dataset" / "runs") + "/*" ):
    #             f = Path(f)
    #             txt = []
    #             for l in f.read_text().split("\n"):
    #                 if not l:
    #                     continue
    #                 l = json.loads(l)
    #                 l["metadata"]["narrative_id"] = l["metadata"]["topic_id"]
    #                 txt.append(json.dumps(l))
    #             (Path(d) / f.name).write_text("\n".join(txt))

    #         actual = load_runs_failsave(Path(d))
    #         self.assertEqual(3, len(actual))
    #         self.assertEqual(expected_metadata, [i["metadata"] for i in actual])
    #         self.assertEqual(expected_paths, [Path(i["path"]).name for i in actual])

    # def test_trec25_spot_check_runs_with_narrative_and_topic_id_as_int(self):
    #     expected_metadata = [
    #         {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_01', 'topic_id': '28', 'narrative_id': '28'},
    #         {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_02', 'topic_id': '28', 'narrative_id': '28'},
    #         {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_02_citations', 'topic_id': '28', 'narrative_id': '28'}
    #     ]
    #     expected_paths = [
    #         'my_best_run_01',
    #         'my_best_run_02',
    #         'my_best_run_02_with_citations'
    #     ]

    #     with TemporaryDirectory() as d:
    #         for f in glob(str(TREC_25_DATA / "spot-check-dataset" / "runs") + "/*" ):
    #             f = Path(f)
    #             txt = []
    #             for l in f.read_text().split("\n"):
    #                 if not l:
    #                     continue
    #                 l = json.loads(l)
    #                 l["metadata"]["topic_id"] = int(l["metadata"]["topic_id"])
    #                 l["metadata"]["narrative_id"] = l["metadata"]["topic_id"]
    #                 txt.append(json.dumps(l))
    #             (Path(d) / f.name).write_text("\n".join(txt))

    #         actual = load_runs_failsave(Path(d))
    #         self.assertEqual(3, len(actual))
    #         self.assertEqual(expected_metadata, [i["metadata"] for i in actual])
    #         self.assertEqual(expected_paths, [Path(i["path"]).name for i in actual])

    # def test_trec25_spot_check_runs_with_qa_id_as_int(self):
    #     expected_metadata = [
    #         {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_01', 'topic_id': '28', 'narrative_id': '28'},
    #         {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_02', 'topic_id': '28', 'narrative_id': '28'},
    #         {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_02_citations', 'topic_id': '28', 'narrative_id': '28'}
    #     ]
    #     expected_paths = [
    #         'my_best_run_01',
    #         'my_best_run_02',
    #         'my_best_run_02_with_citations'
    #     ]

    #     with TemporaryDirectory() as d:
    #         for f in glob(str(TREC_25_DATA / "spot-check-dataset" / "runs") + "/*" ):
    #             f = Path(f)
    #             txt = []
    #             for l in f.read_text().split("\n"):
    #                 if not l:
    #                     continue
    #                 l = json.loads(l)
    #                 l["metadata"]["qa_id"] = int(l["metadata"]["topic_id"])
    #                 del l["metadata"]["topic_id"]
    #                 txt.append(json.dumps(l))
    #             (Path(d) / f.name).write_text("\n".join(txt))

    #         actual = load_runs_failsave(Path(d))
    #         self.assertEqual(3, len(actual))
    #         self.assertEqual(expected_metadata, [i["metadata"] for i in actual])
    #         self.assertEqual(expected_paths, [Path(i["path"]).name for i in actual])

    # def test_trec25_spot_check_runs_with_inconsistent_narrative_and_topic_id(self):
    #     with TemporaryDirectory() as d:
    #         for f in glob(str(TREC_25_DATA / "spot-check-dataset" / "runs") + "/*" ):
    #             f = Path(f)
    #             txt = []
    #             for l in f.read_text().split("\n"):
    #                 if not l:
    #                     continue
    #                 l = json.loads(l)
    #                 l["metadata"]["narrative_id"] = l["metadata"]["topic_id"] + "1"
    #                 txt.append(json.dumps(l))
    #             (Path(d) / f.name).write_text("\n".join(txt))

    #         with self.assertRaises(ValueError):
    #             load_runs_failsave(Path(d))

    # def test_trec25_spot_check_runs_with_duplicate_narrative_and_topic_id(self):
    #     with TemporaryDirectory() as d:
    #         for f in glob(str(TREC_25_DATA / "spot-check-dataset" / "runs") + "/*" ):
    #             f = Path(f)
    #             (Path(d) / f.name).write_text(f.read_text())
    #             (Path(d) / (f.name + "-duplicate")).write_text(f.read_text())

    #         with self.assertRaises(ValueError):
    #             load_runs_failsave(Path(d))