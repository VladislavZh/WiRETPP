from __future__ import annotations

import re
import unittest
from pathlib import Path


class DocumentationTest(unittest.TestCase):
    def test_mathematics_uses_markdown_math_delimiters(self) -> None:
        source = Path("docs/MATHEMATICS_SOURCE.md").read_text(encoding="utf-8")
        for unsupported in (r"\(", r"\)", r"\[", r"\]"):
            self.assertNotIn(unsupported, source)
        self.assertEqual(source.count("$$") % 2, 0)

    def test_rendered_mathematics_references_every_svg(self) -> None:
        rendered = Path("docs/MATHEMATICS.md").read_text(encoding="utf-8")
        links = re.findall(r"assets/math/equation-\d{3}\.svg", rendered)
        assets = sorted(Path("docs/assets/math").glob("equation-*.svg"))
        self.assertEqual(len(links), len(assets))
        self.assertEqual(len(links), len(set(links)))
        for link in links:
            svg = Path("docs") / link
            self.assertTrue(svg.is_file(), svg)
            self.assertIn("currentColor", svg.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
