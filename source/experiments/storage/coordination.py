"""Coordinate shared fit writers without changing scientific identity.

The nonblocking locks in this module are used only for fit and prediction
publication. Contention is returned to the session scheduler so a GPU worker
can run another session or exit while the owning invocation finishes.
"""

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class SharedArtifactBusy(RuntimeError):
    """Report an active fit or prediction writer to the scheduler."""

    def __init__(self, fit_id: str, phase: str, lock_path: Path) -> None:
        self.fit_id = fit_id
        self.phase = phase
        self.lock_path = lock_path.resolve()
        super().__init__(
            f"Active shared {phase} writer for fit {fit_id}."
        )


@contextmanager
def shared_writer_lock(
    path: Path,
    fit_id: str,
    phase: str,
) -> Iterator[None]:
    """Acquire one shared-artifact lock or report immediate contention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SharedArtifactBusy(fit_id, phase, path) from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def shared_writer_available(path: Path) -> bool:
    """Return whether a deferred lock can be acquired without waiting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        try:
            return True
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
