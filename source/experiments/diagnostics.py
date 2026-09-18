"""Observe completed components outside training and collect render failures.

The child process only reads completed histories and writes component-local
plots. No callbacks or model source are changed; no metric JSON is copied.
Failures are returned to orchestration without invalidating scientific work.
"""

from contextlib import contextmanager
import logging
import multiprocessing
from pathlib import Path
from typing import Callable, Iterator

LOGGER = logging.getLogger(__name__)
POLL_SECONDS = 1.0


def render_safely(
    failures: list[str],
    label: str,
    function: Callable,
    *args: object,
    **kwargs: object,
) -> None:
    """Collect a rendering error while allowing independent work to continue."""
    try:
        function(*args, **kwargs)
    except Exception as error:
        LOGGER.exception("Rendering failed: %s", label)
        failures.append(f"{label}: {type(error).__name__}: {error}")


def _watch_components(
    run: Path,
    style: dict,
    stop: object,
    result: object,
) -> None:
    """Render each published component once; return errors over a pipe."""
    from .history import render_component

    seen = set()
    failures = []
    while True:
        finished = stop.is_set()
        for completion in sorted((run / "components").rglob("complete.json")):
            if completion not in seen:
                render_safely(
                    failures,
                    str(completion.parent.resolve()),
                    render_component,
                    completion.parent,
                    style,
                )
                seen.add(completion)
        if finished:
            result.send(failures)
            result.close()
            return
        stop.wait(POLL_SECONDS)


@contextmanager
def history_monitor(
    run: Path,
    style: dict,
    failures: list[str],
    enabled: bool = True,
) -> Iterator[None]:
    """Observe completions in a spawned CPU process, never in batch callbacks.

    Parameters
    ----------
    run : Path
        Owning fit directory, possibly not created yet.
    style : dict
        Shared presentation settings.
    failures : list of str
        Invocation-owned error collector.
    enabled : bool, optional
        Whether to start rendering; default is True.
    """
    if not enabled:
        yield
        return
    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(
        target=_watch_components,
        args=(run, style, stop, writer),
    )
    try:
        process.start()
    except Exception as error:
        reader.close()
        writer.close()
        failures.append(f"Cannot start history renderer: {error}")
        yield
        return
    writer.close()
    try:
        yield
    finally:
        stop.set()
        try:
            failures.extend(reader.recv())
        except EOFError:
            failures.append(f"History rendering worker failed: {run.resolve()}")
        finally:
            reader.close()
            process.join()
        if process.exitcode:
            failures.append(f"History worker exited with {process.exitcode}.")
