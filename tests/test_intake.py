"""Unit tests for src/intake.py.

Pure logic only: fixtures on disk, tmp_path, no network, no model calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.intake import (
    DiffError,
    ReviewContext,
    intake,
    load_ruleset,
    read_diff,
    split_diff,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = PROJECT_ROOT / "examples"
RULESETS = PROJECT_ROOT / "data" / "rulesets"

FAKE_KEY_LINE = 'GEMINI_API_KEY = "sk-test-DO-NOT-USE-1234567890abcdef"'
SQL_LINE = '"SELECT id, name FROM users WHERE name = \'" + username'


def read_example(name: str) -> str:
    return (EXAMPLES / name).read_text(encoding="utf-8")


def chunk_by_header(chunks: list[str], header: str) -> str:
    matches = [c for c in chunks if c.splitlines()[0].startswith(header)]
    assert len(matches) == 1, f"expected exactly one chunk for {header!r}"
    return matches[0]


class TestSplitDiffChunkCounts:
    def test_two_file_fixture_splits_into_exactly_two_chunks(self):
        chunks = split_diff(read_example("clean_two_file.diff"))
        assert len(chunks) == 2
        assert all(c.startswith("diff --git ") for c in chunks)

    def test_three_file_fixture_splits_into_exactly_three_chunks(self):
        chunks = split_diff(read_example("three_file_issue.diff"))
        assert len(chunks) == 3
        assert all(c.startswith("diff --git ") for c in chunks)


class TestSplitDiffChunkBoundaries:
    def test_each_chunk_holds_exactly_one_git_header(self):
        for name in ("clean_two_file.diff", "three_file_issue.diff"):
            chunks = split_diff(read_example(name))
            for chunk in chunks:
                headers = [ln for ln in chunk.splitlines() if ln.startswith("diff --git ")]
                assert len(headers) == 1

    def test_three_file_chunks_keep_their_own_content_no_bleed(self):
        chunks = split_diff(read_example("three_file_issue.diff"))
        settings_chunk = chunk_by_header(chunks, "diff --git a/app/settings.py")
        store_chunk = chunk_by_header(chunks, "diff --git a/app/store.py")
        pricing_chunk = chunk_by_header(chunks, "diff --git a/app/pricing.py")

        assert FAKE_KEY_LINE in settings_chunk
        assert SQL_LINE in store_chunk
        assert "def total_with_tax" in pricing_chunk

        for other in (store_chunk, pricing_chunk):
            assert "GEMINI_API_KEY" not in other
        for other in (settings_chunk, pricing_chunk):
            assert "username" not in other
        for other in (settings_chunk, store_chunk):
            assert "total_with_tax" not in other

    def test_inline_two_file_split_with_metadata_lines(self):
        text = (
            "diff --git a/one.py b/one.py\n"
            "index 1111111..2222222 100644\n"
            "--- a/one.py\n"
            "+++ b/one.py\n"
            "@@ -1,2 +1,3 @@\n"
            " line one\n"
            "+added in one\n"
            "diff --git a/two.py b/two.py\n"
            "new file mode 100644\n"
            "index 0000000..3333333\n"
            "--- /dev/null\n"
            "+++ b/two.py\n"
            "@@ -0,0 +1,1 @@\n"
            "+first line of two\n"
        )
        one, two = split_diff(text)
        assert one.startswith("diff --git a/one.py b/one.py")
        assert "--- a/one.py" in one and "+added in one" in one
        assert "two.py" not in one
        assert two.startswith("diff --git a/two.py b/two.py")
        assert "new file mode 100644" in two and "+first line of two" in two
        assert "added in one" not in two

    def test_rename_and_delete_metadata_stay_with_their_chunk(self):
        text = (
            "diff --git a/old_name.py b/new_name.py\n"
            "similarity index 92%\n"
            "rename from old_name.py\n"
            "rename to new_name.py\n"
            "--- a/old_name.py\n"
            "+++ b/new_name.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-value = 1\n"
            "+value = 2\n"
            "diff --git a/gone.py b/gone.py\n"
            "deleted file mode 100644\n"
            "index 4444444..0000000\n"
            "--- a/gone.py\n"
            "+++ /dev/null\n"
            "@@ -1,1 +0,0 @@\n"
            "-bye = True\n"
        )
        renamed, deleted = split_diff(text)
        assert "similarity index 92%" in renamed
        assert "rename to new_name.py" in renamed
        assert "deleted file mode 100644" in deleted
        assert "-bye = True" in deleted
        assert "gone.py" not in renamed

    def test_added_line_looking_like_git_header_does_not_split(self):
        text = (
            "diff --git a/notes.txt b/notes.txt\n"
            "--- a/notes.txt\n"
            "+++ b/notes.txt\n"
            "@@ -1,1 +1,2 @@\n"
            " hello\n"
            "+diff --git a/fake b/fake\n"
        )
        chunks = split_diff(text)
        assert len(chunks) == 1
        assert "+diff --git a/fake b/fake" in chunks[0]

    def test_preamble_before_first_header_is_dropped(self):
        text = "Some chat preamble about the change.\n" + read_example("clean_two_file.diff")
        chunks = split_diff(text)
        assert len(chunks) == 2
        assert all(c.startswith("diff --git ") for c in chunks)
        assert "preamble" not in chunks[0]

    def test_text_without_git_header_is_malformed(self):
        assert split_diff("") == []
        assert split_diff("hello world this is not a diff") == []

    def test_plain_unified_diff_without_git_header_is_malformed(self):
        text = "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n-a\n+b\n"
        assert split_diff(text) == []


class TestIntake:
    async def test_empty_string_raises_friendly_diff_error(self):
        with pytest.raises(DiffError) as excinfo:
            await intake("")
        assert "paste or point me at a unified diff" in excinfo.value.message

    async def test_whitespace_only_raises_friendly_diff_error(self):
        with pytest.raises(DiffError) as excinfo:
            await intake(" \n\t\r\n ")
        assert "paste or point me at a unified diff" in excinfo.value.message

    async def test_malformed_fixture_raises_friendly_diff_error(self):
        with pytest.raises(DiffError) as excinfo:
            await intake(read_example("malformed.diff"))
        message = excinfo.value.message
        assert "diff --git" in message
        assert "Traceback" not in message

    async def test_git_header_without_hunks_raises_friendly_diff_error(self):
        with pytest.raises(DiffError) as excinfo:
            await intake("diff --git a/x.py b/x.py\nindex 111..222 100644\n")
        assert "nothing to review" in excinfo.value.message.lower()

    async def test_returns_two_chunks_for_clean_fixture(self):
        chunks = await intake(read_example("clean_two_file.diff"))
        assert len(chunks) == 2
        assert all(c.startswith("diff --git ") for c in chunks)

    async def test_returns_three_chunks_for_issue_fixture(self):
        chunks = await intake(read_example("three_file_issue.diff"))
        assert len(chunks) == 3
        assert FAKE_KEY_LINE in " ".join(chunks)


class TestReadDiff:
    async def test_reads_existing_diff_file(self, tmp_path):
        diff_path = tmp_path / "change.diff"
        diff_path.write_text(read_example("clean_two_file.diff"), encoding="utf-8")
        text = await read_diff(diff_path)
        assert text.startswith("diff --git a/app/greeter.py")

    async def test_accepts_str_and_path_alike(self, tmp_path):
        diff_path = tmp_path / "change.diff"
        diff_path.write_text(read_example("clean_two_file.diff"), encoding="utf-8")
        assert await read_diff(str(diff_path)) == await read_diff(diff_path)

    async def test_missing_file_raises_friendly_diff_error(self, tmp_path):
        with pytest.raises(DiffError) as excinfo:
            await read_diff(tmp_path / "no_such.diff")
        message = excinfo.value.message
        assert "couldn't find a diff" in message
        assert "Traceback" not in message

    async def test_directory_instead_of_file_raises_friendly_diff_error(self, tmp_path):
        with pytest.raises(DiffError):
            await read_diff(tmp_path)

    async def test_non_utf8_content_raises_friendly_diff_error(self, tmp_path):
        diff_path = tmp_path / "binary-ish.diff"
        diff_path.write_bytes(b"diff --git a/x b/x\n\xff\xfe\x00bad\n")
        with pytest.raises(DiffError) as excinfo:
            await read_diff(diff_path)
        assert "UTF-8" in excinfo.value.message


class TestReviewContext:
    def test_matches_pdf_shape_with_default_strictness(self):
        ctx = ReviewContext(repo="demo", language="python", ruleset_id="default")
        assert ctx.repo == "demo"
        assert ctx.language == "python"
        assert ctx.ruleset_id == "default"
        assert ctx.strictness == "normal"

    def test_strictness_is_overridable(self):
        ctx = ReviewContext(repo="demo", language="python", ruleset_id="strict", strictness="strict")
        assert ctx.strictness == "strict"


class TestLoadRuleset:
    def test_default_ruleset_happy_path_returns_formatted_text(self):
        text = load_ruleset("default")
        assert text.startswith('Ruleset "Default Desk Rules"')
        assert "SEC-01" in text and "TEST-01" in text and "STYLE-01" in text
        for area in ("security", "tests", "style"):
            assert f"({area})" in text
        assert "Error:" not in text

    def test_strict_ruleset_happy_path_includes_reporting_rules(self):
        text = load_ruleset("strict")
        assert "Strict Desk Rules" in text
        assert "REPORT-01" in text
        assert "Error:" not in text

    def test_ruleset_files_hold_six_to_ten_rules_each(self):
        for ruleset_id in ("default", "strict"):
            data = json.loads((RULESETS / f"{ruleset_id}.json").read_text(encoding="utf-8"))
            assert data["ruleset_id"] == ruleset_id
            assert 6 <= len(data["rules"]) <= 10

    def test_missing_ruleset_file_returns_sentence_not_raise(self):
        text = load_ruleset("does_not_exist")
        assert text.startswith("Error: ruleset 'does_not_exist' could not be loaded")
        assert "Continue using built-in judgement and say so." in text

    def test_bad_json_returns_sentence(self, tmp_path):
        rulesets = tmp_path / "data" / "rulesets"
        rulesets.mkdir(parents=True)
        (rulesets / "broken.json").write_text("{not json", encoding="utf-8")
        text = load_ruleset("broken", base_dir=tmp_path)
        assert text.startswith("Error: ruleset 'broken' could not be loaded")
        assert "JSON" in text

    def test_wrong_shape_returns_sentence(self, tmp_path):
        rulesets = tmp_path / "data" / "rulesets"
        rulesets.mkdir(parents=True)
        (rulesets / "shapeless.json").write_text('{"name": "No rules here"}', encoding="utf-8")
        text = load_ruleset("shapeless", base_dir=tmp_path)
        assert text.startswith("Error: ruleset 'shapeless' could not be loaded")
        assert "no rules" in text

    def test_custom_base_dir_happy_path(self, tmp_path):
        rulesets = tmp_path / "data" / "rulesets"
        rulesets.mkdir(parents=True)
        (rulesets / "tiny.json").write_text(
            json.dumps(
                {
                    "ruleset_id": "tiny",
                    "name": "Tiny Rules",
                    "rules": [{"id": "T-01", "area": "style", "rule": "Keep functions short."}],
                }
            ),
            encoding="utf-8",
        )
        text = load_ruleset("tiny", base_dir=tmp_path)
        assert 'Ruleset "Tiny Rules"' in text
        assert "[T-01] (style) Keep functions short." in text

    def test_ruleset_id_with_path_separators_returns_sentence_not_raise(self):
        text = load_ruleset("../secrets")
        assert text.startswith("Error: ruleset '../secrets' could not be loaded")


class TestFixtureSanity:
    def test_issue_fixture_plants_exactly_the_fake_key(self):
        text = read_example("three_file_issue.diff")
        assert FAKE_KEY_LINE in text
        assert text.count("DO-NOT-USE") == 1

    def test_clean_fixture_contains_no_secret_literals(self):
        text = read_example("clean_two_file.diff")
        assert "API_KEY" not in text
        assert "SECRET" not in text
        assert "sk-" not in text

    def test_fixture_chunk_counts(self):
        assert len(split_diff(read_example("clean_two_file.diff"))) == 2
        assert len(split_diff(read_example("three_file_issue.diff"))) == 3
