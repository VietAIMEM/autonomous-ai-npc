"""Shared test support for the PHASE 34 dispatch-correlation protocol.

The brain now tags every accepted dispatch with a unique ``dispatch_id`` and
only lets a result complete the exact dispatch that produced it. Unity echoes
that id with the result; these offline tests simulate Unity, so they must echo
it too. :func:`confirm` does exactly that: it copies the id from the action the
brain is currently awaiting onto a hand-built result, so the tests exercise the
real matching path instead of bypassing it.
"""

from __future__ import annotations

from dataclasses import replace

from backend.brain.models import ActionResult


def confirm(owner, result):
    """Tag ``result`` with the dispatch id ``owner`` is awaiting.

    ``owner`` is the ``AutonomousBrain`` a test is about to tick. A ``None``
    result, a result that already carries an id, or one whose action does not
    match the pending action is returned unchanged.
    """
    if result is None:
        return None
    pending = owner.state.pending_action
    if (
        isinstance(result, ActionResult)
        and result.dispatch_id is None
        and pending is not None
        and pending.action == result.action
    ):
        return replace(result, dispatch_id=pending.dispatch_id)
    return result
