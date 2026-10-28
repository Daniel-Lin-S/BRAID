"""Name Multi-session training worker logs from immutable job metadata.

Input is a launch directory, job identity, and plan specification. Output is
one worker log under jobs/<kind>/<session>/ for target jobs, or jobs/<kind>/
for source jobs. Full fit IDs retain unambiguous links to plan and artifacts.
"""

from pathlib import Path


JOB_DIRECTORY = "jobs"
PERCENT_SCALE = 100


def job_label(key: str, spec: dict) -> str:
    """Describe a fit for lifecycle messages and its log filename.

    Parameters
    ----------
    key : str
        Full immutable fit identifier.
    spec : dict
        Plan job specification, including target split or source settings.

    Returns
    -------
    str
        Readable label with the full fit identifier appended.
    """
    kind = spec["kind"]
    seed = spec["seed"]
    if kind == "source":
        detail = (
            f"sources{spec['source_count']}_seed{seed}_n{spec['latent']}_"
            f"{spec['objective']}_{spec['sharing']}"
        )
    else:
        fraction = spec["fraction"] * PERCENT_SCALE
        percentage = round(fraction)
        if abs(fraction - percentage) > 1e-8:
            raise ValueError(
                f"Expected whole-percent label budget, got {fraction}."
            )
        detail = f"fold{spec['fold']}_labels{percentage}pct_seed{seed}"
        if kind != "preprocessing":
            detail += f"_n{spec['latent']}"
        if kind == "adaptation":
            detail += (
                f"_sources{spec['source_count']}_{spec['objective']}_"
                f"{spec['sharing']}"
            )
    return f"{detail}__{key}"


def job_log_path(launch: Path, key: str, spec: dict) -> Path:
    """Return the worker log path under its job type and target session.

    Parameters
    ----------
    launch : Path
        Launch log directory.
    key : str
        Full immutable fit identifier.
    spec : dict
        Plan job specification.

    Returns
    -------
    Path
        Nested log path. Its parent directory is not created here.
    """
    path = launch / JOB_DIRECTORY / spec["kind"]
    if spec["kind"] != "source":
        path /= spec["session"]
    return path / f"{job_label(key, spec)}.log"


def source_session_log_path(
    launch: Path, source_count: int, session: str,
    component: str, size: int,
) -> Path:
    """Nest one source-session component log under its recording."""
    if source_count <= 0 or size <= 0 or not session or not component:
        raise ValueError("Source session log identity is incomplete.")
    if "/" in session or ".." in session:
        raise ValueError(f"Invalid source session log name: {session}.")
    return (
        launch / JOB_DIRECTORY / "source_n3" / session
        / f"sources{source_count}_n3{size}__{component}.log"
    )
