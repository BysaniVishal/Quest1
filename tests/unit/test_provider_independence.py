"""Structural check for design.docx section 12's central architectural
claim: provider-specific logic is isolated entirely to MediaResolver.
Everything downstream must be unable to special-case a provider because it
is never even told which one it's operating on. No network involved --
complements (but does not replace) the real cross-provider validation in
tests/e2e/test_real_media.py."""

import inspect

import pytest

from dialogue_frame_finder.frame_mapping import locate_frame
from dialogue_frame_finder.onset import resolve_dialogue_onset
from dialogue_frame_finder.search import search_dialogue

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("fn", [search_dialogue, resolve_dialogue_onset, locate_frame])
def test_core_pipeline_functions_take_no_provider_specific_arguments(fn):
    params = inspect.signature(fn).parameters
    assert not any(
        keyword in p.lower() for p in params for keyword in ("provider", "okru", "youtube")
    )
