from typing import Optional, Sequence, Union

from tensorflow import keras

from zookeeper.core.field import ComponentField, Field  # type: ignore
from zookeeper.tf.dataset import Dataset
from zookeeper.tf.preprocessing import Preprocessing


class Experiment:
    """A wrapper around a Keras experiment.

    Subclasses must implement their training loop in `run`.
    """

    # Nested components
    dataset: Dataset = ComponentField()
    preprocessing: Preprocessing = ComponentField()
    model: keras.models.Model = ComponentField()
    experiment_name: str = Field()
    lab_blocks: Sequence[bool] = Field()    
    resume_from: Optional[str] = Field()


    # Parameters
    epochs: int = Field()
    batch_size: int = Field()
    loss: Optional[
        Union[Sequence[Union[keras.losses.Loss, str, None]], keras.losses.Loss, str]
    ] = Field()
    optimizer: Union[keras.optimizers.Optimizer, str] = Field()
