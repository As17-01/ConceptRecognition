import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from omegaconf import OmegaConf

spec = importlib.util.spec_from_file_location(
    "distribute_corpus", Path(__file__).resolve().parents[1]
    / "scripts/distribute_corpus/distribute_corpus.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

TOPICS = [
    {"id": "topic_01", "name": "Внимание", "description": "Работа с вниманием"},
    {"id": "topic_02", "name": "Flow", "description": "Непрерывное движение"},
]


def response(value):
    return SimpleNamespace(status="completed", output_text=json.dumps(value))


class DistributionTests(unittest.TestCase):
    def test_multiple_topics_unmatched_resume_and_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            src, dst = root / "source", root / "output"
            for name in ["class_a", "class_b"]:
                folder = src / name
                folder.mkdir(parents=True)
                (folder / "summary.md").write_text(name)
                (folder / "transcript.md").write_text("Full transcript")
                (folder / "extras").mkdir()
                (folder / "extras" / "notes.md").write_text("Notes")
            digest = root / "digest.txt"
            digest.write_text("Corpus topics")
            cfg = OmegaConf.create(dict(src=str(src), dst=str(dst), digest_src=str(digest),
                                        model="mock", topics_max_tokens=4096, classify_max_tokens=2048))
            client = Mock()
            client.responses.create.side_effect = [
                response({"topics": TOPICS}),
                response({"assignments": [{"topic_id": t["id"], "reason": "Practiced", "role": "primary", "evidence": "class_a"}
                                           for t in TOPICS], "unmatched_reason": ""}),
                response({"assignments": [], "unmatched_reason": "No matching topic"}),
            ]
            self.assertEqual(module.distribute(cfg, client), (2, 0))
            for topic in ["Внимание", "Flow"]:
                self.assertEqual((dst / topic / "class_a" / "extras" / "notes.md").read_text(), "Notes")
            self.assertTrue((dst / "_unmatched" / "class_b" / "summary.md").exists())
            self.assertTrue((src / "class_a" / "transcript.md").exists())
            payload = json.loads(client.responses.create.call_args.kwargs["input"])
            self.assertEqual(payload["corpus_digest"], "Corpus topics")
            self.assertEqual(payload["class_summary"], "class_b")
            taxonomy_payload = json.loads(client.responses.create.call_args_list[0].kwargs["input"])
            self.assertEqual(taxonomy_payload["class_summaries"],
                             {"class_a": "class_a", "class_b": "class_b"})
            self.assertEqual(module.distribute(cfg, client), (2, 0))
            self.assertEqual(client.responses.create.call_count, 3)
            (src / "class_a" / "summary.md").write_text("Changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                module.distribute(cfg, client)

    def test_invalid_model_output_is_rejected(self):
        with self.assertRaises(ValueError):
            module.validate_topics({"topics": [{**TOPICS[0], "name": "../escape"}]})
        with self.assertRaises(ValueError):
            module.validate_assignment({"assignments": [{"topic_id": "unknown", "reason": "x"}],
                                        "unmatched_reason": ""}, TOPICS, "summary")
        with self.assertRaisesRegex(ValueError, "exact excerpt"):
            module.validate_assignment({"assignments": [{"topic_id": "topic_01",
                "role": "primary", "reason": "Explained", "evidence": "invented quote"}],
                "unmatched_reason": ""}, TOPICS, "Actual summary")
        client = Mock()
        client.responses.create.return_value = SimpleNamespace(status="incomplete", output_text="{}")
        with self.assertRaisesRegex(ValueError, "did not complete"):
            module.request_json(client, "mock", 10, "instructions", {})


if __name__ == "__main__":
    unittest.main()
