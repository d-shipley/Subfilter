from loguru import logger

logger.disable("subfilter")

# User configurable global options DEFAULT values
global_config = {
    'test_level': 0,                                 # int, [1, 2] Case ID to select variable lists for testing.
    }
