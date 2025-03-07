
from typing import Dict
from llama_stack.distribution.datatypes import Api, ProviderSpec
from .config import ilabPostTrainingConfig


async def get_provider_impl(
    config: ilabPostTrainingConfig,
    deps: Dict[Api, ProviderSpec],
):
    from .post_training import ilabPostTrainingImpl

    impl = ilabPostTrainingImpl(
        config,
        deps[Api.datasetio],
        deps[Api.datasets],
    )
    return impl