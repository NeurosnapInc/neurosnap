"""Offline checks for safe, repeatable reference publishing."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("publish_docs", Path(__file__).parents[1] / "scripts/publish_docs.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PublishTest(unittest.TestCase):
    def test_publish_is_non_destructive_and_rejects_invalid_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "html", root / "published"
            source.mkdir()
            target.mkdir()
            (target / "authored-guide.md").write_text("retain me")
            (source / "_static").mkdir()
            (source / "_static" / "test.css").write_text("body {}")
            (source / "index.html").write_text("<article>SDK</article>")
            (source / "objects.inv").write_bytes(b"inventory")
            manifest = {"version": "2026.11.05", "pages": [{"name": "index.html", "title": "SDK", "html": "<p>Reference</p>"}]}
            (source / "neurosnap-reference.json").write_text(json.dumps(manifest))
            first = module.publish(source, target)
            self.assertEqual(first, module.publish(source, target))
            (source / "_static" / "test.css").write_text("body {color: navy}")
            second = module.publish(source, target)
            self.assertNotEqual(first, second)
            self.assertEqual(json.loads((target / "current.json").read_text())["release"], second)
            self.assertEqual((target / "authored-guide.md").read_text(), "retain me")
            self.assertTrue((target / "releases" / first / "index.html").is_file())
            (source / "index.html").unlink()
            with self.assertRaises(ValueError):
                module.publish(source, target)
            self.assertEqual(json.loads((target / "current.json").read_text())["release"], second)

    def test_rejects_overlapping_source_and_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                module.publish(tmp, Path(tmp) / "published")


if __name__ == "__main__":
    unittest.main()
