"""Load YAML configurations for BRAID fitting."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Union

import yaml

BRAID_PACKAGE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIRECTORY = (
    BRAID_PACKAGE_DIRECTORY.parents[1] / "assets" / "config"
)
DIMENSIONS_KEY = "dimensions"
MODEL_KEY = "model"
TRAINING_KEY = "training"
FORECAST_KEY = "forecast"
PREPROCESSING_KEY = "preprocessing"
NONE_VALUE = "none"
STAGE_NAMES = ("stage_1", "stage_2")
DECODER_NAMES = ("neural_decoder", "behaviour_decoder")
REQUIRED_DIMENSIONS = {
    "behaviour_relevant_neural_state_size": "n1",
    "behaviour_irrelevant_neural_state_size": "n2",
}
OPTIONAL_DIMENSIONS = {
    "non_neural_behaviour_state_size": "n3",
    "preprocessing_state_size": "n_pre",
}
TRAINING_NAMES = {
    "maximum_epochs": "epochs",
    "training_batch_size": "batch_size",
    "initialisation_attempts": "init_attempts",
    "maximum_refit_attempts": "max_attempts",
    "early_stopping_patience": "early_stopping_patience",
    "early_stopping_measure": "early_stopping_measure",
    "recurrent_early_stopping_start_epoch": "start_from_epoch_rnn",
    "decoder_early_stopping_start_epoch": "start_from_epoch_reg",
    "create_validation_from_training": "create_val_from_training",
    "validation_set_ratio": "validation_set_ratio",
    "optimiser": "optimizer_name",
    "optimiser_arguments": "optimizer_args",
    "learning_rate_schedule": "lr_scheduler_name",
    "learning_rate_schedule_arguments": "lr_scheduler_args",
    "trainable_components": "trainableParams",
    "clear_tensorflow_graph": "clear_graph",
    "verbose": "verbose",
    "save_training_logs": "save_logs",
    "skip_prediction_calculation": "skip_predictions",
}
MODEL_NAMES = {
    "allow_stage_2_behaviour_contribution": "allow_nonzero_Cz2",
    "include_direct_neural_to_behaviour_mapping": "has_Dyz",
    "include_input_feedthrough": "has_UFT",
    "include_input_in_decoders": "has_UFT_reg",
    "neural_decoder_uses_full_state": "model1_Cy_Full",
    "behaviour_decoder_uses_full_state": "model2_Cz_Full",
    "skip_neural_decoder": "skip_Cy",
}
FORECAST_NAMES = {
    "enable_forward_prediction": "enable_forward_pred",
    "steps_ahead": "steps_ahead",
    "steps_ahead_loss_weights": "steps_ahead_loss_weights",
    "use_observed_input_in_dynamics": "observable_U_in_Kfw",
    "use_observed_input_in_decoder": "observable_U_in_Cfw",
}
PREPROCESSING_NAMES = {
    "remove_flat_dimensions": "remove_flat_dims",
    "reuse_existing_models": "use_existing_prep_models",
    "standardise_inputs": "zscore_inputs",
    "standardise_each_dimension": "zscore_per_dim",
}
MLP_NAMES = {
    "bias_initialiser": "bias_initializer",
    "bias_regulariser": "bias_regularizer_name",
    "bias_regulariser_arguments": "bias_regularizer_args",
    "kernel_initialiser": "kernel_initializer",
    "kernel_regulariser": "kernel_regularizer_name",
    "kernel_regulariser_arguments": "kernel_regularizer_args",
}
MLP_SETTINGS = set(MLP_NAMES) | {
    "activation",
    "dropout_rate",
    "output_activation",
    "use_bias",
}
LSTM_NAMES = {
    "activation": "activation",
    "recurrent_activation": "recurrent_activation",
    "use_bias": "use_bias",
    "kernel_initialiser": "kernel_initializer",
    "recurrent_initialiser": "recurrent_initializer",
    "bias_initialiser": "bias_initializer",
    "unit_forget_bias": "unit_forget_bias",
    "dropout_rate": "dropout",
    "recurrent_dropout_rate": "recurrent_dropout",
}
LSTM_COMPONENTS = {
    "kernel_regulariser": ("regularizers", "kernel_regularizer"),
    "recurrent_regulariser": ("regularizers", "recurrent_regularizer"),
    "bias_regulariser": ("regularizers", "bias_regularizer"),
    "kernel_constraint": ("constraints", "kernel_constraint"),
    "recurrent_constraint": ("constraints", "recurrent_constraint"),
    "bias_constraint": ("constraints", "bias_constraint"),
}
LSTM_SETTINGS = set(LSTM_NAMES) | set(LSTM_COMPONENTS)
LSTM_SETTINGS |= {
    f"{name}_arguments" for name in LSTM_COMPONENTS
}
STAGE_ARGUMENTS = {
    "stage_1": {
        "state_transition": "A1_args",
        "input_mapping": "K1_args",
        "neural_decoder": "Cy1_args",
        "behaviour_decoder": "Cz1_args",
    },
    "stage_2": {
        "state_transition": "A2_args",
        "input_mapping": "K2_args",
        "neural_decoder": "Cy2_args",
        "behaviour_decoder": "Cz2_args",
    },
}
DEFAULT_BASE_ARGUMENTS = {
    "allow_nonzero_Cz2": True,
    "early_stopping_measure": "loss",
    "early_stopping_patience": 3,
    "enable_forward_pred": True,
    "has_Dyz": False,
    "has_UFT": True,
    "has_UFT_reg": True,
    "init_attempts": 1,
    "max_attempts": 10,
    "model1_Cy_Full": False,
    "model2_Cz_Full": True,
    "optimizer_name": "Adam",
    "remove_flat_dims": True,
    "save_logs": True,
    "skip_Cy": False,
    "skip_predictions": False,
    "verbose": True,
    "zscore_inputs": True,
    "zscore_per_dim": False,
}


def get_default_config_path(config_name: str) -> Path:
    """Return an absolute path to a repository configuration file."""
    path = DEFAULT_CONFIG_DIRECTORY / config_name
    if not path.is_file():
        raise FileNotFoundError(
            f"BRAID configuration file does not exist: {path}."
        )
    return path


def load_braid_fit_arguments(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load fit-ready BRAIDModel.fit keyword arguments from YAML."""
    configuration_path = Path(config_path)
    configuration = _load_yaml_mapping(configuration_path)
    if configuration_path.resolve() != get_default_config_path(
        "default.yaml"
    ).resolve():
        configuration = _deep_merge_mappings(
            _load_yaml_mapping(get_default_config_path("default.yaml")),
            configuration,
        )
    model_settings = _require_mapping(
        configuration.get(MODEL_KEY, {}),
        MODEL_KEY,
    )
    arguments = dict(DEFAULT_BASE_ARGUMENTS)
    arguments.update(_map_settings(
        configuration.get(TRAINING_KEY, {}),
        TRAINING_KEY,
        TRAINING_NAMES,
    ))
    arguments.update(_load_model_arguments(model_settings))
    arguments.update(_map_settings(
        configuration.get(FORECAST_KEY, {}),
        FORECAST_KEY,
        FORECAST_NAMES,
    ))
    arguments.update(_map_settings(
        configuration.get(PREPROCESSING_KEY, {}),
        PREPROCESSING_KEY,
        PREPROCESSING_NAMES,
    ))
    _validate_forecast_arguments(arguments)
    return {
        **_load_dimensions(
            model_settings.get(DIMENSIONS_KEY, {}),
            f"{MODEL_KEY}.{DIMENSIONS_KEY}",
        ),
        "args_base": arguments,
    }


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    """Read and validate a YAML top-level mapping."""
    if not path.is_file():
        raise FileNotFoundError(
            f"BRAID configuration file does not exist: {path}."
        )
    with path.open(encoding="utf-8") as config_file:
        configuration = _replace_none_values(yaml.safe_load(config_file))
    if not isinstance(configuration, Mapping) or not configuration:
        raise ValueError("BRAID configuration must be a non-empty mapping.")
    _reject_unknown_keys(
        configuration,
        {MODEL_KEY, TRAINING_KEY, FORECAST_KEY, PREPROCESSING_KEY},
        "configuration",
    )
    return dict(configuration)


def _replace_none_values(value: Any) -> Any:
    """Convert the YAML string none to Python None recursively."""
    if isinstance(value, Mapping):
        return {key: _replace_none_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_none_values(item) for item in value]
    return None if value == NONE_VALUE else value


def _deep_merge_mappings(
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return nested configuration defaults updated by explicit overrides."""
    merged = copy.deepcopy(dict(defaults))
    for key, value in overrides.items():
        default_value = merged.get(key)
        if isinstance(default_value, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_mappings(default_value, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_dimensions(settings: Any, location: str) -> Dict[str, int | None]:
    """Validate YAML dimensions and derive BRAIDModel.fit state arguments."""
    settings = _require_mapping(settings, location)
    _reject_unknown_keys(
        settings,
        set(REQUIRED_DIMENSIONS) | set(OPTIONAL_DIMENSIONS),
        location,
    )
    n1 = _require_positive_integer(
        settings.get("behaviour_relevant_neural_state_size"),
        f"{location}.behaviour_relevant_neural_state_size",
    )
    n2 = _require_nonnegative_integer(
        settings.get("behaviour_irrelevant_neural_state_size"),
        f"{location}.behaviour_irrelevant_neural_state_size",
    )
    n3 = _optional_positive_integer(
        settings.get("non_neural_behaviour_state_size"),
        f"{location}.non_neural_behaviour_state_size",
    )
    n_pre = _optional_positive_integer(
        settings.get("preprocessing_state_size"),
        f"{location}.preprocessing_state_size",
    )
    return {
        "nx": n1 + n2 + (n3 or 0),
        "n1": n1,
        "n3": n3,
        "n_pre": n_pre,
    }


def _map_settings(
    settings: Any,
    location: str,
    names: Mapping[str, str],
) -> Dict[str, Any]:
    """Validate public setting names and translate them to fit arguments."""
    settings = _require_mapping(settings, location)
    _reject_unknown_keys(settings, set(names), location)
    return {names[key]: value for key, value in settings.items()}


def _load_model_arguments(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Translate model structure and stage mappings to fit arguments."""
    _reject_unknown_keys(
        settings,
        set(MODEL_NAMES) | {DIMENSIONS_KEY, *STAGE_NAMES},
        MODEL_KEY,
    )
    if "stage_1" not in settings:
        raise ValueError("model.stage_1 must define the stage mappings.")
    arguments = {
        MODEL_NAMES[key]: value
        for key, value in settings.items()
        if key in MODEL_NAMES
    }
    stage_1 = _require_mapping(settings["stage_1"], "model.stage_1")
    stage_2 = copy.deepcopy(settings.get("stage_2", stage_1))
    arguments.update(_load_stage_arguments("stage_1", stage_1))
    arguments.update(_load_stage_arguments("stage_2", stage_2))
    return arguments


def _load_stage_arguments(stage_name: str, settings: Any) -> Dict[str, Any]:
    """Translate one stage's mapping definitions to fit arguments."""
    location = f"{MODEL_KEY}.{stage_name}"
    settings = _require_mapping(settings, location)
    allowed = {
        "combine_state_and_input_mapping",
        "state_transition",
        "input_mapping",
        *DECODER_NAMES,
    }
    _reject_unknown_keys(settings, allowed, location)
    required = {"state_transition", *DECODER_NAMES}
    if required - set(settings):
        raise ValueError(
            f"{location} must define state_transition, neural_decoder, and "
            "behaviour_decoder."
        )
    combine = settings.get("combine_state_and_input_mapping", False)
    if not isinstance(combine, bool):
        raise ValueError(
            f"{location}.combine_state_and_input_mapping must be Boolean."
        )
    if combine and "input_mapping" in settings:
        raise ValueError(
            f"{location}.input_mapping is invalid when "
            "combine_state_and_input_mapping is true."
        )
    if not combine and "input_mapping" not in settings:
        raise ValueError(
            f"{location}.input_mapping is required when "
            "combine_state_and_input_mapping is false."
        )
    names = STAGE_ARGUMENTS[stage_name]
    transition = _load_transition_arguments(
        settings["state_transition"],
        f"{location}.state_transition",
    )
    arguments = {
        names["state_transition"]: transition["mapping"],
        f"{stage_name}_state_transition_architecture":
            transition["architecture"],
        f"{stage_name}_lstm_arguments": transition["lstm_arguments"],
        names["neural_decoder"]: _load_mlp_arguments(
            settings["neural_decoder"],
            f"{location}.neural_decoder",
        ),
        names["behaviour_decoder"]: _load_mlp_arguments(
            settings["behaviour_decoder"],
            f"{location}.behaviour_decoder",
        ),
    }
    input_arguments = {"unifiedAK": combine}
    if not combine:
        input_arguments.update(_load_mlp_arguments(
            settings["input_mapping"],
            f"{location}.input_mapping",
        ))
    arguments[names["input_mapping"]] = input_arguments
    return arguments


def _load_transition_arguments(settings: Any, location: str) -> Dict[str, Any]:
    """Load an MLP or LSTM state-transition definition."""
    settings = _require_mapping(settings, location)
    architecture = settings.get("architecture")
    if architecture == "multilayer_perceptron":
        _reject_unknown_keys(
            settings,
            {"architecture"} | MLP_SETTINGS | {"hidden_size", "depth"},
            location,
        )
        mapping = {
            key: value for key, value in settings.items()
            if key != "architecture"
        }
        return {
            "architecture": architecture,
            "mapping": _load_mlp_arguments(mapping, location),
            "lstm_arguments": None,
        }
    if architecture == "lstm":
        _reject_unknown_keys(
            settings,
            {"architecture"} | LSTM_SETTINGS,
            location,
        )
        lstm_settings = {
            key: value for key, value in settings.items()
            if key != "architecture"
        }
        return {
            "architecture": architecture,
            "mapping": {},
            "lstm_arguments": _load_lstm_arguments(
                lstm_settings,
                location,
            ),
        }
    raise ValueError(
        f"{location}.architecture must be multilayer_perceptron or lstm."
    )


def _load_mlp_arguments(settings: Any, location: str) -> Dict[str, Any]:
    """Translate an MLP definition to RegressionModel arguments."""
    settings = _require_mapping(settings, location)
    _reject_unknown_keys(
        settings,
        MLP_SETTINGS | {"hidden_size", "depth"},
        location,
    )
    arguments = {
        MLP_NAMES.get(key, key): value
        for key, value in settings.items()
        if key not in {"hidden_size", "depth"} and value is not None
    }
    depth = settings.get("depth")
    hidden_size = settings.get("hidden_size")
    if (depth is None) != (hidden_size is None):
        raise ValueError(f"{location} requires both hidden_size and depth.")
    if depth is not None:
        arguments["units"] = [_require_positive_integer(
            hidden_size,
            f"{location}.hidden_size",
        )] * _require_positive_integer(depth, f"{location}.depth")
    return arguments


def _load_lstm_arguments(
    settings: Mapping[str, Any],
    location: str,
) -> Dict[str, Any]:
    """Translate YAML LSTM settings to TensorFlow constructor arguments."""
    arguments = {
        LSTM_NAMES[key]: value
        for key, value in settings.items()
        if key in LSTM_NAMES and value is not None
    }
    for name, (module_name, argument_name) in LSTM_COMPONENTS.items():
        component = _load_tensorflow_component(
            module_name,
            settings.get(name),
            settings.get(f"{name}_arguments"),
            f"{location}.{name}",
        )
        if component is not None:
            arguments[argument_name] = component
    return arguments


def _load_tensorflow_component(
    module_name: str,
    name: Any,
    arguments: Any,
    location: str,
) -> Any:
    """Create one TensorFlow regulariser or constraint from YAML settings."""
    if name is None:
        if arguments not in (None, {}):
            raise ValueError(f"{location}_arguments requires {location}.")
        return None
    if not isinstance(name, str):
        raise ValueError(f"{location} must be a string or none.")
    if arguments is None:
        return name
    if not isinstance(arguments, Mapping):
        raise ValueError(f"{location}_arguments must be a mapping or none.")
    try:
        import tensorflow as tf
        constructor = getattr(getattr(tf.keras, module_name), name)
    except AttributeError as error:
        raise ValueError(
            f"Unsupported TensorFlow {module_name[:-1]}: {name}."
        ) from error
    return constructor(**arguments)


def _validate_forecast_arguments(arguments: Mapping[str, Any]) -> None:
    """Validate forecast horizons and their optional loss weights."""
    horizons = arguments.get("steps_ahead")
    if horizons is None:
        return
    if not isinstance(horizons, list) or not horizons:
        raise ValueError("forecast.steps_ahead must be a non-empty list.")
    if any(not isinstance(value, int) or value < 1 for value in horizons):
        raise ValueError(
            "forecast.steps_ahead must contain positive integers."
        )
    weights = arguments.get("steps_ahead_loss_weights")
    if weights is not None and len(weights) != len(horizons):
        raise ValueError(
            "forecast.steps_ahead_loss_weights must match steps_ahead."
        )
    if any(value > 1 for value in horizons):
        if not arguments.get("enable_forward_pred", False):
            raise ValueError(
                "forecast.enable_forward_prediction must be true for "
                "horizons greater than one."
            )


def _require_mapping(value: Any, location: str) -> Dict[str, Any]:
    """Return value as a mapping or raise a clear configuration error."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping.")
    return dict(value)


def _reject_unknown_keys(
    settings: Mapping[str, Any],
    allowed_keys: set[str],
    location: str,
) -> None:
    """Reject settings outside the public YAML schema."""
    if set(settings) - allowed_keys:
        raise ValueError(
            f"Unsupported setting in {location}. Use the documented YAML "
            "configuration names."
        )


def _require_positive_integer(value: Any, location: str) -> int:
    """Return a strictly positive integer setting."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{location} must be a positive integer.")
    return value


def _optional_positive_integer(value: Any, location: str) -> int | None:
    """Return an optional strictly positive integer setting."""
    if value is None:
        return None
    return _require_positive_integer(value, location)


def _require_nonnegative_integer(value: Any, location: str) -> int:
    """Return a nonnegative integer setting."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{location} must be a nonnegative integer.")
    return value
