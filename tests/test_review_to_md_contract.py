import unittest

from review_to_md import (
    _apply_update_needed_headers_and_sort,
    _build_stage3_snippet_evidence_summary,
    _extract_atomic_requirements,
    _filter_docs_for_file_name,
    _format_atomic_requirements_markdown,
    render_markdown_pdf_bytes,
)


class ReviewToMdContractTests(unittest.TestCase):
    def test_extract_atomic_requirements_prefers_requirement_cues(self) -> None:
        directive_text = (
            "The site shall maintain a written emergency response plan that is reviewed annually. "
            "General background information appears here. "
            "The program must include a process to track corrective actions to closure."
        )

        requirements = _extract_atomic_requirements(directive_text, max_items=5)

        self.assertGreaterEqual(len(requirements), 2)
        self.assertTrue(any("shall maintain" in req.lower() for req in requirements))
        self.assertTrue(any("must include" in req.lower() for req in requirements))

    def test_stage3_sections_sorted_by_update_needed(self) -> None:
        markdown = "\n".join(
            [
                "## Stage 3 Compliance Updates",
                "",
                "### file-b.md",
                "- Update needed: No",
                "",
                "### file-a.md",
                "- Update needed: Yes",
                "",
                "### file-c.md",
                "- Update needed: Unclear",
            ]
        )

        sorted_markdown = _apply_update_needed_headers_and_sort(markdown)

        yes_index = sorted_markdown.find("### file-a.md")
        unclear_index = sorted_markdown.find("### file-c.md")
        no_index = sorted_markdown.find("### file-b.md")

        self.assertTrue(yes_index < unclear_index < no_index)
        self.assertIn("Update needed: Yes", sorted_markdown)

    def test_stage3_header_uses_markdown_formatted_update_value(self) -> None:
        markdown = "\n".join(
            [
                "### file-a.md — Update needed: Unclear",
                "#### Update evaluation",
                "- Update needed: **Yes**",
                "- Evidence confidence: **High**",
            ]
        )

        sorted_markdown = _apply_update_needed_headers_and_sort(markdown)

        self.assertIn(
            "### file-a.md — Update needed: Yes — Evidence confidence: High",
            sorted_markdown,
        )

    def test_stage3_header_uses_non_bulleted_evidence_confidence(self) -> None:
        markdown = "\n".join(
            [
                "### file-a.md",
                "#### Update evaluation",
                "- Update needed: Yes",
                "Evidence confidence: Medium",
            ]
        )

        sorted_markdown = _apply_update_needed_headers_and_sort(markdown)

        self.assertIn(
            "### file-a.md — Update needed: Yes — Evidence confidence: Medium",
            sorted_markdown,
        )

    def test_stage3_header_defaults_unknown_confidence(self) -> None:
        markdown = "\n".join(
            [
                "### file-a.md",
                "#### Update evaluation",
                "- Update needed: No",
            ]
        )

        sorted_markdown = _apply_update_needed_headers_and_sort(markdown)

        self.assertIn(
            "### file-a.md — Update needed: No — Evidence confidence: Unknown",
            sorted_markdown,
        )

    def test_atomic_requirements_markdown_has_expected_sections(self) -> None:
        output = _format_atomic_requirements_markdown(
            directive_name="directive.pdf",
            requirements=["The program shall establish controls."],
        )

        self.assertIn("## Atomic Requirements Review", output)
        self.assertIn("## Extracted Requirements", output)
        self.assertIn("## Review Guidance", output)

    def test_stage3_snippet_summary_keeps_files_with_missing_snippets(self) -> None:
        summary = _build_stage3_snippet_evidence_summary(
            [
                {
                    "file_name": "file-a.md",
                    "documents": [{}],
                }
            ]
        )

        self.assertIn("## Stage 3 snippet evidence by file", summary)
        self.assertIn("- file-a.md", summary)
        self.assertIn("No snippet text extracted", summary)

    def test_stage3_snippet_summary_empty_input_message(self) -> None:
        summary = _build_stage3_snippet_evidence_summary([])
        self.assertEqual(summary, "No Stage 3 snippet evidence available.")

    def test_filter_docs_for_file_name_matches_normalized_variants(self) -> None:
        documents = [
            {"metadata_storage_name": "411_2 original.pdf", "content": "alpha"},
            {"metadata_storage_name": "other_file.pdf", "content": "beta"},
        ]

        filtered = _filter_docs_for_file_name(documents, "411_2original.pdf")

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["metadata_storage_name"], "411_2 original.pdf")

    def test_render_markdown_pdf_bytes_non_empty_for_markdown(self) -> None:
        markdown = "\n".join(
            [
                "# Stage 3 Compliance Updates",
                "",
                "- Update needed: Yes",
                "1. First numbered point",
                "",
                "```",
                "sample_code = True",
                "```",
            ]
        )

        pdf_bytes = render_markdown_pdf_bytes(markdown)

        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 0)

    def test_render_markdown_pdf_bytes_non_empty_for_table_markdown(self) -> None:
        markdown = "\n".join(
            [
                "## Existing text snippets vs suggested edits",
                "",
                "| Existing NETL text snippet | Suggested edit |",
                "|---|---|",
                "| Existing sentence | Updated sentence aligned to DOE O 458.1 |",
            ]
        )

        pdf_bytes = render_markdown_pdf_bytes(markdown)

        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 0)


if __name__ == "__main__":
    unittest.main()
