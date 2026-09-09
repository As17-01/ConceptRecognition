import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('distributor', Path(__file__).resolve().parents[1] / 'scripts/distribute_corpus/distribute_corpus.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class EvidenceTests(unittest.TestCase):
    topics = [{'id': 'topic_01'}]

    def result(self, evidence):
        return {'assignments': [{'topic_id': 'topic_01', 'role': 'primary',
                                'reason': 'Developed', 'evidence': evidence}], 'unmatched_reason': ''}

    def test_formatting(self):
        summary = 'Разбирали *flow* — «мягкое движение»\nи опору.'
        m.validate_assignment(self.result('flow - "мягкое движение" и опору.'), self.topics, summary)

    def test_paraphrase_rejected(self):
        with self.assertRaises(ValueError):
            m.validate_assignment(self.result('Развивали движение'), self.topics, 'Разбирали flow')

    def test_correction(self):
        cfg = SimpleNamespace(model='mock', classify_max_tokens=2048)
        with patch.object(m, 'request_json', side_effect=[self.result('wrong'), {'assignments': [{'topic_id': 'topic_01', 'role': 'primary', 'reason': 'Developed', 'evidence_id': 'p1'}], 'unmatched_reason': ''}]) as call:
            result = m.classify_with_retry(None, cfg, self.topics, 'digest', 'Разбирали flow')
            self.assertEqual(result['assignments'][0]['evidence'], 'Разбирали flow')
            self.assertEqual(call.call_count, 2)

    def test_retry_limit(self):
        cfg = SimpleNamespace(model='mock', classify_max_tokens=2048)
        with patch.object(m, 'request_json', return_value=self.result('wrong')) as call:
            with self.assertRaises(ValueError):
                m.classify_with_retry(None, cfg, self.topics, 'digest', 'Разбирали flow')
            self.assertEqual(call.call_count, 3)


if __name__ == '__main__':
    unittest.main()
