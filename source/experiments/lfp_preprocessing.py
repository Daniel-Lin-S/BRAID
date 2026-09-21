"""Run ordered, provenance-bearing preprocessing on broadband signals.

Input sources expose time-first broadband voltage samples without loading a
whole recording. Configured step plugins transform that source in order. The
final output contains finite LFP voltages on the requested model time grid.
"""

from dataclasses import dataclass
from fractions import Fraction
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy.signal import resample_poly

from .cache import file_digest

DEFAULT_CHANNEL_BLOCK_SIZE = 8
DEFAULT_SAMPLE_BLOCK_SIZE = 1_000_000
MAXIMUM_RESAMPLE_DENOMINATOR = 1_000_000
RATE_TOLERANCE = 1e-8


class SignalSource(Protocol):
    """Time-first, channel-addressable signal source."""

    shape: tuple[int, int]
    timestamps: np.ndarray
    channel_ids: list[str]

    def read(self, rows: slice, columns: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class PipelineContext:
    """Immutable timing and memory limits supplied to preprocessing steps.

    Parameters
    ----------
    target_timestamps : ndarray, shape (T,)
        Exact model sample grid in seconds.
    target_sampling_rate_hz : float
        Requested model sampling rate.
    source_sampling_rate_hz : float
        Effective broadband sampling rate inferred from synchronized timing.
    channel_block_size : int, optional
        Maximum channels resampled together; default is 8.
    sample_block_size : int, optional
        Maximum native samples read while preparing a global transform;
        default is 1,000,000.
    """

    target_timestamps: np.ndarray
    target_sampling_rate_hz: float
    source_sampling_rate_hz: float
    channel_block_size: int = DEFAULT_CHANNEL_BLOCK_SIZE
    sample_block_size: int = DEFAULT_SAMPLE_BLOCK_SIZE


@dataclass
class PipelineResult:
    """Final aligned voltages and resolved preprocessing provenance."""

    values: np.ndarray
    timestamps: np.ndarray
    steps: list[dict[str, Any]]
    resampling: dict[str, Any]


class PreprocessingStep(Protocol):
    """Contract implemented by ordered LFP preprocessing plugins."""

    name: str
    input_domain: str
    output_domain: str
    parameters: dict[str, Any]
    plugin_reference: str
    implementation_sha256: str

    def apply(
        self, source: SignalSource, context: PipelineContext
    ) -> SignalSource: ...

    def provenance(self) -> dict[str, Any]: ...


class ReferencedSource:
    """Apply a precomputed common reference during bounded source reads."""

    def __init__(self, source: SignalSource, reference: np.ndarray) -> None:
        if reference.shape != (source.shape[0],):
            raise ValueError(
                "Expected one common-reference value per native sample, "
                f"got {reference.shape}."
            )
        self.source = source
        self.reference = reference
        self.shape = source.shape
        self.timestamps = source.timestamps
        self.channel_ids = source.channel_ids

    def read(self, rows: slice, columns: np.ndarray) -> np.ndarray:
        """Read referenced voltages for one row and channel selection."""
        values = self.source.read(rows, columns)
        reference = self.reference[rows, None]
        result = values - reference
        if not np.isfinite(result).all():
            raise ValueError(
                "Common-average referencing produced nonfinite values."
            )
        return result


class ArraySource:
    """Expose an in-memory aligned array through the source protocol."""

    def __init__(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        channel_ids: list[str],
        resampling: dict[str, Any],
    ) -> None:
        self.values = values
        self.timestamps = timestamps
        self.channel_ids = channel_ids
        self.resampling = resampling
        self.shape = values.shape

    def read(self, rows: slice, columns: np.ndarray) -> np.ndarray:
        """Read one aligned array selection."""
        return self.values[rows][:, columns]


def _unknown_parameters(
    name: str, parameters: dict[str, Any], supported: set[str]
) -> None:
    unknown = set(parameters) - supported
    if unknown:
        raise ValueError(f"Unsupported {name} parameters: {sorted(unknown)}.")


class CommonAverageReference:
    """Subtract a configurable native-rate channel average."""

    name = "common_average_reference"
    input_domain = "native"
    output_domain = "native"

    def __init__(self, parameters: dict[str, Any]) -> None:
        _unknown_parameters(self.name, parameters, {"channel_ids"})
        selected = parameters.get("channel_ids")
        if selected is not None and (
            not isinstance(selected, list)
            or not selected
            or len(selected) != len(set(selected))
            or any(
                not isinstance(value, str) or not value for value in selected
            )
        ):
            raise ValueError(
                "common_average_reference.channel_ids must be null or a "
                "nonempty unique list."
            )
        self.parameters = {"channel_ids": selected}

    def apply(
        self, source: SignalSource, context: PipelineContext
    ) -> SignalSource:
        """Compute the reference once and return a lazy referenced source."""
        selected = self.parameters["channel_ids"]
        reference_ids = source.channel_ids if selected is None else selected
        missing = set(reference_ids) - set(source.channel_ids)
        if missing:
            raise ValueError(
                "Common-average reference channels are absent: "
                f"{sorted(missing)}."
            )
        columns = np.array(
            [source.channel_ids.index(value) for value in reference_ids],
            dtype=int,
        )
        if not len(columns):
            raise ValueError("Common-average referencing selected no channels.")
        reference = np.empty(source.shape[0], dtype=np.float32)
        for start in range(0, source.shape[0], context.sample_block_size):
            stop = min(start + context.sample_block_size, source.shape[0])
            values = source.read(slice(start, stop), columns)
            reference[start:stop] = np.mean(
                values, axis=1, dtype=np.float64
            ).astype(np.float32)
        if not np.isfinite(reference).all():
            raise ValueError(
                "Common-average reference contains nonfinite values."
            )
        return ReferencedSource(source, reference)

    def provenance(self) -> dict[str, Any]:
        """Return the resolved scientific settings for this step."""
        return {"name": self.name, "parameters": self.parameters}


class PolyphaseResample:
    """Apply SciPy polyphase anti-alias filtering and exact-grid alignment."""

    name = "polyphase_resample"
    input_domain = "native"
    output_domain = "aligned"

    def __init__(self, parameters: dict[str, Any]) -> None:
        supported = {"anti_alias_lowpass_hz", "window", "beta"}
        _unknown_parameters(self.name, parameters, supported)
        if set(parameters) != supported:
            missing = supported - set(parameters)
            raise ValueError(
                f"Missing polyphase_resample parameters: {sorted(missing)}."
            )
        cutoff = parameters["anti_alias_lowpass_hz"]
        beta = parameters["beta"]
        window = parameters["window"]
        if (
            isinstance(cutoff, bool)
            or not isinstance(cutoff, (int, float))
            or cutoff <= 0
        ):
            raise ValueError("anti_alias_lowpass_hz must be positive.")
        if window != "kaiser":
            raise ValueError(f"Expected window='kaiser', got {window!r}.")
        if (
            isinstance(beta, bool)
            or not isinstance(beta, (int, float))
            or beta <= 0
        ):
            raise ValueError("Kaiser beta must be positive.")
        self.parameters = {
            "anti_alias_lowpass_hz": float(cutoff),
            "window": window,
            "beta": float(beta),
        }

    def apply(
        self, source: SignalSource, context: PipelineContext
    ) -> SignalSource:
        """Resample channel blocks and interpolate onto the requested grid."""
        expected = context.target_sampling_rate_hz / 2
        cutoff = self.parameters["anti_alias_lowpass_hz"]
        if not np.isclose(cutoff, expected, rtol=0, atol=RATE_TOLERANCE):
            raise ValueError(
                "scipy.signal.resample_poly uses the output Nyquist cutoff; "
                f"expected {expected} Hz, got {cutoff} Hz."
            )
        ratio = Fraction(
            context.target_sampling_rate_hz / context.source_sampling_rate_hz
        ).limit_denominator(MAXIMUM_RESAMPLE_DENOMINATOR)
        up, down = ratio.numerator, ratio.denominator
        effective_rate = context.source_sampling_rate_hz * up / down
        output = np.empty(
            (len(context.target_timestamps), source.shape[1]),
            dtype=np.float32,
        )
        rows = slice(0, source.shape[0])
        window = (self.parameters["window"], self.parameters["beta"])
        for start in range(0, source.shape[1], context.channel_block_size):
            stop = min(start + context.channel_block_size, source.shape[1])
            columns = np.arange(start, stop)
            values = source.read(rows, columns)
            sampled = resample_poly(
                values,
                up,
                down,
                axis=0,
                window=window,
                padtype="constant",
            )
            sampled_t = source.timestamps[0] + (
                np.arange(len(sampled), dtype=float) / effective_rate
            )
            if (
                context.target_timestamps[0] < sampled_t[0]
                or context.target_timestamps[-1] > sampled_t[-1]
            ):
                raise ValueError(
                    "Resampled broadband does not cover the model time grid."
                )
            for offset in range(sampled.shape[1]):
                output[:, start + offset] = np.interp(
                    context.target_timestamps,
                    sampled_t,
                    sampled[:, offset],
                )
        if not np.isfinite(output).all():
            raise ValueError("Polyphase resampling produced nonfinite values.")
        metadata = dict(
            library="scipy.signal.resample_poly",
            up=up,
            down=down,
            source_sampling_rate_hz=context.source_sampling_rate_hz,
            effective_output_sampling_rate_hz=effective_rate,
            target_sampling_rate_hz=context.target_sampling_rate_hz,
            **self.parameters,
        )
        return ArraySource(
            output,
            context.target_timestamps.copy(),
            list(source.channel_ids),
            metadata,
        )

    def provenance(self) -> dict[str, Any]:
        """Return the resolved scientific settings for this step."""
        return {"name": self.name, "parameters": self.parameters}


def _step(configuration: dict[str, Any]) -> PreprocessingStep:
    """Instantiate one validated preprocessing plugin configuration."""
    if not isinstance(configuration, dict):
        raise TypeError("Each preprocessing step must be a mapping.")
    if set(configuration) != {"plugin", "parameters"}:
        raise ValueError(
            "Each preprocessing step must contain plugin and parameters."
        )
    reference = configuration["plugin"]
    parameters = configuration["parameters"]
    if not isinstance(reference, str) or ":" not in reference:
        raise ValueError(
            "Preprocessing plugin must have the form module:class."
        )
    if not isinstance(parameters, dict):
        raise TypeError("Preprocessing step parameters must be a mapping.")
    module, name = reference.split(":", 1)
    imported = import_module(module)
    implementation = getattr(imported, name)
    result = implementation(parameters)
    for attribute in (
        "name",
        "input_domain",
        "output_domain",
        "apply",
        "provenance",
    ):
        if not hasattr(result, attribute):
            raise TypeError(
                f"Preprocessing plugin {reference!r} lacks {attribute}."
            )
    if not isinstance(result.name, str) or not result.name:
        raise TypeError(
            f"Preprocessing plugin {reference!r} has an invalid name."
        )
    domains = {"native", "aligned"}
    if (
        result.input_domain not in domains
        or result.output_domain not in domains
    ):
        raise ValueError(
            f"Preprocessing plugin {reference!r} has invalid domains "
            f"{result.input_domain!r} -> {result.output_domain!r}."
        )
    module_path = getattr(imported, "__file__", None)
    if not module_path or not Path(module_path).is_file():
        raise ValueError(
            f"Preprocessing plugin has no source file: {reference!r}."
        )
    result.plugin_reference = reference
    result.implementation_sha256 = file_digest(Path(module_path))
    return result


def _validate_source(source: SignalSource, step: PreprocessingStep) -> None:
    """Validate the structural and finite output contract of one step."""
    shape = getattr(source, "shape", None)
    timestamps = getattr(source, "timestamps", None)
    channel_ids = getattr(source, "channel_ids", None)
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(type(value) is not int or value < 1 for value in shape)
    ):
        raise TypeError(
            f"Step {step.name} returned an invalid signal shape: {shape}."
        )
    if (
        not isinstance(timestamps, np.ndarray)
        or timestamps.shape != (shape[0],)
        or not np.isfinite(timestamps).all()
        or np.any(np.diff(timestamps) <= 0)
    ):
        raise ValueError(
            f"Step {step.name} returned invalid signal timestamps."
        )
    if (
        not isinstance(channel_ids, list)
        or len(channel_ids) != shape[1]
        or len(channel_ids) != len(set(channel_ids))
        or any(not isinstance(value, str) or not value for value in channel_ids)
    ):
        raise ValueError(
            f"Step {step.name} returned invalid signal channel IDs."
        )
    if step.output_domain == "aligned":
        values = getattr(source, "values", None)
        if not isinstance(values, np.ndarray) or values.shape != shape:
            raise TypeError(
                f"Aligned step {step.name} returned invalid values."
            )
        if not np.isfinite(values).all():
            raise ValueError(
                f"Aligned step {step.name} returned nonfinite values."
            )


def preprocessing_steps(settings: dict[str, Any]) -> list[PreprocessingStep]:
    """Resolve and validate an ordered preprocessing pipeline."""
    if set(settings) != {"steps"} or not isinstance(settings["steps"], list):
        raise ValueError("preprocessing must contain a steps list.")
    if not settings["steps"]:
        raise ValueError("LFP preprocessing requires at least one step.")
    steps = [_step(configuration) for configuration in settings["steps"]]
    names = [step.name for step in steps]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate preprocessing step names: {names}.")
    domain = "native"
    for step in steps:
        if step.input_domain != domain:
            raise ValueError(
                f"Step {step.name} expects {step.input_domain}, got {domain}."
            )
        domain = step.output_domain
    if domain != "aligned":
        raise ValueError(
            "LFP preprocessing must end with an aligned signal step."
        )
    return steps


def preprocessing_identity(settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Return ordered parameters and implementation hashes for all steps."""
    return [
        {
            **step.provenance(),
            "plugin": step.plugin_reference,
            "implementation_sha256": step.implementation_sha256,
        }
        for step in preprocessing_steps(settings)
    ]


def run_preprocessing(
    source: SignalSource,
    settings: dict[str, Any],
    context: PipelineContext,
) -> PipelineResult:
    """Execute configured steps and return finite aligned LFP voltages."""
    current = source
    steps = preprocessing_steps(settings)
    provenance = []
    for step in steps:
        current = step.apply(current, context)
        _validate_source(current, step)
        provenance.append(
            {
                **step.provenance(),
                "plugin": step.plugin_reference,
                "implementation_sha256": step.implementation_sha256,
            }
        )
    values = getattr(current, "values", None)
    resampling = getattr(current, "resampling", None)
    if not isinstance(values, np.ndarray) or not isinstance(resampling, dict):
        raise TypeError("Final preprocessing step did not return aligned data.")
    if current.timestamps.shape != context.target_timestamps.shape:
        raise ValueError(
            "Expected final timestamps with shape "
            f"{context.target_timestamps.shape}, got "
            f"{current.timestamps.shape}."
        )
    if not np.array_equal(current.timestamps, context.target_timestamps):
        raise ValueError(
            "Final preprocessing timestamps do not match the target grid."
        )
    return PipelineResult(
        values,
        current.timestamps,
        provenance,
        resampling,
    )
