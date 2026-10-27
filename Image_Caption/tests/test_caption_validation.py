"""
Tests for caption validation and cleaning logic.

These tests import model-layer helpers and verify string processing
without loading the actual BLIP model.
"""

from __future__ import annotations

import pytest

# We import the private helper directly for white-box testing.
from model import _clean_caption


class TestCleanCaption:
    """Verify caption post-processing: whitespace normalisation, prompt
    stripping, capitalisation, and sentence-ending punctuation."""

    def test_basic_cleaning(self):
        assert _clean_caption("a dog sitting on a bench") == "A dog sitting on a bench."

    def test_already_capitalised(self):
        assert _clean_caption("A cat on the roof.") == "A cat on the roof."

    def test_extra_whitespace(self):
        assert _clean_caption("  a   dog   playing  ") == "A dog playing."

    def test_prompt_stripping(self):
        result = _clean_caption(
            "a detailed description of a sunset over mountains",
            prompt="a detailed description of",
        )
        assert result == "A sunset over mountains."

    def test_prompt_stripping_with_colon(self):
        result = _clean_caption(
            "a detailed description of: a boat on a lake",
            prompt="a detailed description of",
        )
        assert result == "A boat on a lake."

    def test_empty_string(self):
        assert _clean_caption("") == ""

    def test_whitespace_only(self):
        assert _clean_caption("   ") == ""

    def test_trailing_period_not_duplicated(self):
        result = _clean_caption("a dog.")
        assert result == "A dog."
        assert not result.endswith("..")

    def test_single_word(self):
        assert _clean_caption("landscape") == "Landscape."

    def test_prompt_is_entire_text(self):
        result = _clean_caption(
            "a detailed description of",
            prompt="a detailed description of",
        )
        assert result == ""
