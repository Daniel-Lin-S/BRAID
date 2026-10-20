"""Version fitting and inference separately from analysis source files.

Inputs are explicit numerical module versions. Outputs are deterministic
signatures for new BRAID fits and inference; analysis retains a code digest.
Legacy source hashes remain eligible only through certified transitions.
"""

from importlib.metadata import version
from pathlib import Path
import platform

from experiments.storage.cache import file_digest, fingerprint
from experiments.storage.scientific_modules import (
    MODULE_VERSIONS, fit_module_digest,
)

SOURCE = Path(__file__).resolve().parents[2]
SIGNATURE_SCHEMA = 2
LEGACY_FITTING_CANONICAL = (
    "ad69a9feeccabcca1b324c2cec1cc391d4324d4b7e78e7fe3119493a01c29c5c"
)
LEGACY_FITTING_POST_TRANSFORMER = (
    "7202fb249cc2e4fb23a6ee4cd77c1e25fd471aeec09b8a8d38df12a61a770654"
)
# Only certified numerical-equivalence pairs permit immutable bundle reuse.
FITTING_REUSE = frozenset({
    (
        "c42f4c7d43da5675729ce75e66f738dd9a9b58ea0f6b53c5b667efa0d085619b",
        "06a2d229cc8dba85e03a45fab93d01d26f25b4382a14938ae923c361d32ce06b",
    ),
    (
        "46b13713e9f884b4948786483f7b875b54493e87772e91a8e098c6bc97d6670c",
        "c42f4c7d43da5675729ce75e66f738dd9a9b58ea0f6b53c5b667efa0d085619b",
    ),
    (
        "ad69a9feeccabcca1b324c2cec1cc391d4324d4b7e78e7fe3119493a01c29c5c",
        "46b13713e9f884b4948786483f7b875b54493e87772e91a8e098c6bc97d6670c",
    ),
    (
        "e457f919edcfc3a49358b5ca1d617bc5ee3aae5fbb3330c3382f338954446f4f",
        "338ef3f4781672973338c3fa4ef934d2400a43b99d85d98d81b8be0ee23530b9",
    ),
    (
        "338ef3f4781672973338c3fa4ef934d2400a43b99d85d98d81b8be0ee23530b9",
        "042c67c49a52ec50c90bca7dee26b046cd996ef50b04f27a4403bb852467624d",
    ),
    (
        "042c67c49a52ec50c90bca7dee26b046cd996ef50b04f27a4403bb852467624d",
        "814f85ac160e622e4fc46e4244ca5688c1748431dc4957b15ed4f96faa8399c7",
    ),
    (
        "814f85ac160e622e4fc46e4244ca5688c1748431dc4957b15ed4f96faa8399c7",
        "c5dbd575ed067cd581cc14e9e8e2dfe4f7581c189b40c65eb4bb5c5ac97b5184",
    ),
    (
        "c5dbd575ed067cd581cc14e9e8e2dfe4f7581c189b40c65eb4bb5c5ac97b5184",
        "ad69a9feeccabcca1b324c2cec1cc391d4324d4b7e78e7fe3119493a01c29c5c",
    ),
})
INFERENCE_REUSE = frozenset({
    # The explicit inference version retains the reviewed legacy path.
    (
        "ea27a870989f7ef6ee5d9baeacfda88dcf279ab6a12dd72e992108ecba817d87",
        "dc7cd9c8cbe1c400d0783c0e77328dce594dac66e431c16f277312d2f47713dc",
    ),
    (
        "aa186cf5db457a9904ffe73bed7af941344b5c48a381f0b2a75a5cc57342a062",
        "c3629ba0b1feb4510eabbffaede2ac0e9365fd27e16cebb2dd20e613cfd5a7f3",
    ),
    (
        "5f21a16941d35bfc41ea3da3610e3c5b3f1e7de36533548e1c3fadda936435a7",
        "aa186cf5db457a9904ffe73bed7af941344b5c48a381f0b2a75a5cc57342a062",
    ),
    (
        "ea27a870989f7ef6ee5d9baeacfda88dcf279ab6a12dd72e992108ecba817d87",
        "5f21a16941d35bfc41ea3da3610e3c5b3f1e7de36533548e1c3fadda936435a7",
    ),
    (
        "5de680b8a69abb410057b53c555c1601eeb91feb83b5abfba03da1c450cb88a5",
        "22ddcba8335dd9dae6106b7757cd21559ef7862a6e7e0397c101a909b20f91af",
    ),
    (
        "22ddcba8335dd9dae6106b7757cd21559ef7862a6e7e0397c101a909b20f91af",
        "93dae4befe7f5e019312b38e868224c3dd3c219a808c585e25e5e2635cf22c0f",
    ),
    (
        "93dae4befe7f5e019312b38e868224c3dd3c219a808c585e25e5e2635cf22c0f",
        "d19d145d1d9596783bda90d19a3cd57d7037e9d019c322c94efdb5bfaac34e1f",
    ),
    (
        "d19d145d1d9596783bda90d19a3cd57d7037e9d019c322c94efdb5bfaac34e1f",
        "c70143ee942256d00f0b552d817271f442eeb8d6098b4ac34abda65d39be7339",
    ),
    (
        "c70143ee942256d00f0b552d817271f442eeb8d6098b4ac34abda65d39be7339",
        "b1e58a3b0d274961e5710ec8366f46d08cd02c32cbf5a94618c1413d72b13c84",
    ),
    (
        "b1e58a3b0d274961e5710ec8366f46d08cd02c32cbf5a94618c1413d72b13c84",
        "ea27a870989f7ef6ee5d9baeacfda88dcf279ab6a12dd72e992108ecba817d87",
    ),
})


def scientific_versions() -> dict:
    """Read numerical dependency versions without initializing a model."""
    return dict(
        python=platform.python_version(),
        packages={
            name: version(name)
            for name in (
                "tensorflow", "tf-keras", "numpy", "scipy", "h5py",
                "scikit-learn", "PyYAML",
            )
        },
    )


def implementation_signatures() -> dict[str, str]:
    """Return versioned runtime signatures and separate analysis digest."""
    fitting = dict(schema=SIGNATURE_SCHEMA, modules=MODULE_VERSIONS)
    inference = dict(
        schema=SIGNATURE_SCHEMA,
        version=MODULE_VERSIONS["braid.inference"],
    )
    source_audit = {
        str(path.relative_to(SOURCE)): file_digest(path)
        for path in sorted(SOURCE.rglob("*.py"))
    }
    analysis = {
        name: file_digest(SOURCE / "experiments" / name)
        for name in (
            "reports/evaluation.py", "datasets/populations.py",
        )
    }
    return dict(
        fitting_implementation=fingerprint(fitting),
        fitting_source_audit=source_audit,
        inference_implementation=fingerprint(inference),
        analysis_implementation=fingerprint(analysis),
    )


def _compatible_transition(
    recorded: str | None,
    expected: str | None,
    transitions: frozenset[tuple[str, str]],
) -> bool:
    """Return whether certified forward transitions connect fingerprints."""
    if recorded == expected:
        return True
    if recorded is None or expected is None:
        return False
    pending = [recorded]
    visited = {recorded}
    while pending:
        current = pending.pop()
        for predecessor, successor in transitions:
            if predecessor != current or successor in visited:
                continue
            if successor == expected:
                return True
            visited.add(successor)
            pending.append(successor)
    return False


def legacy_fitting_signatures() -> frozenset[str]:
    """Return reviewed raw-code signatures from existing BRAID fits."""
    signatures = {
        recorded for recorded, _ in FITTING_REUSE
        if _compatible_transition(
            recorded, LEGACY_FITTING_CANONICAL, FITTING_REUSE
        )
    }
    signatures.update({
        LEGACY_FITTING_CANONICAL, LEGACY_FITTING_POST_TRANSFORMER,
    })
    return frozenset(signatures)


def semantic_legacy_targets() -> frozenset[str]:
    """Return new MLP and LSTM fit signatures certified for legacy reuse."""
    result = set()
    for architecture in ("multilayer_perceptron", "lstm"):
        model = dict(model=dict(stage_1=dict(
            state_transition=dict(architecture=architecture)
        )))
        result.add(fit_module_digest("generic", model, (1,)))
    return frozenset(result)


def compatible_fitting(recorded: str | None, expected: str | None) -> bool:
    """Return whether a completed fit is numerically reusable."""
    if (
        expected in semantic_legacy_targets()
        and recorded in legacy_fitting_signatures()
    ):
        return True
    return _compatible_transition(recorded, expected, FITTING_REUSE)


def fitting_predecessors(expected: str | None) -> tuple[str, ...]:
    """Return certified predecessor fingerprints for one implementation."""
    predecessors = {
        recorded for recorded, _ in FITTING_REUSE
        if recorded != expected
        and _compatible_transition(recorded, expected, FITTING_REUSE)
    }
    if expected in semantic_legacy_targets():
        predecessors.update(legacy_fitting_signatures())
    return tuple(sorted(predecessors))


def compatible_inference(recorded: str | None, expected: str | None) -> bool:
    """Accept only exact or explicitly certified numerical implementations.

    Parameters
    ----------
    recorded, expected : str or None
        Inference fingerprints from the saved bundle and current request.

    Returns
    -------
    bool
        Whether numerical provenance permits reuse without rewriting.
    """
    return _compatible_transition(recorded, expected, INFERENCE_REUSE)
