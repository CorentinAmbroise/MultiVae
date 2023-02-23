from pydantic.dataclasses import dataclass
from pythae.config import BaseConfig


@dataclass
class AutoConfig(BaseConfig):
    @classmethod
    def from_json_file(cls, json_path):
        """Creates a :class:`~multivae.config.BaseMultiVAEConfig` instance from a JSON config file. It
        builds automatically the correct config for any `pythae.models`.

        Args:
            json_path (str): The path to the json file containing all the parameters

        Returns:
            :class:`BaseMultiVAEConfig`: The created instance
        """

        config_dict = cls._dict_from_json(json_path)
        config_name = config_dict.pop("name")

        if config_name == "BaseMultiVAEConfig":
            from ..base import BaseMultiVAEConfig

            model_config = BaseMultiVAEConfig.from_json_file(json_path)

        elif config_name == "JMVAEConfig":
            from ..jmvae import JMVAEConfig

            model_config = JMVAEConfig.from_json_file(json_path)

        else:
            raise NameError(
                "Cannot reload automatically the model configuration... "
                f"The model name in the `model_config.json may be corrupted. Got `{config_name}`"
            )

        return model_config