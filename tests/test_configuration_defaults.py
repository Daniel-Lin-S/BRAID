"""Regression tests for default BRAID fitting configuration."""

import unittest

from BRAID.config import get_default_config_path, load_braid_fit_arguments


class ConfigurationDefaultsTest(unittest.TestCase):
    """Validate compatibility defaults for public BRAID configurations."""

    def test_behaviour_decoder_is_not_full_state_by_default(self) -> None:
        """Keep the initial BRAID tutorial decoder configuration."""
        arguments = load_braid_fit_arguments(
            get_default_config_path(
                "behaviour_decoder_mlp_1x64.yaml"
            )
        )

        self.assertFalse(arguments["args_base"]["model2_Cz_Full"])


if __name__ == "__main__":
    unittest.main()
