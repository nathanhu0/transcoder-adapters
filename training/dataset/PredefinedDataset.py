from enum import Enum
from functools import partial
from typing import Literal
from torch.utils.data import Dataset, DataLoader
from torch import Generator as TorchGenerator

from .collate import collate_fn
from .types import DatasetItem, SizedDataset

from .gemma.config import FineWebLMSysMixedConfig

from .datasetspecific_config import DatasetSpecificConfig, DatasetType
from .openthoughts.config import OpenThoughtsConfig


class LengthExcessionBehavior(Enum):
    TRUNCATE = "truncate"
    ERROR = "error"


class PredefinedDataset:
    """
    Specifies how a dataset should be loaded and processed for training
    """

    # Type for loaded datasets
    DatasetSplits = Literal["train", "val"]
    LoadedDatasets = dict[DatasetSplits, SizedDataset[DatasetItem]]
    Dataloaders = dict[DatasetSplits, DataLoader[DatasetItem]]

    def __init__(
        self,
        dataset_type: DatasetType,
        *,
        tokenizer,
        length_excession_behavior: LengthExcessionBehavior = LengthExcessionBehavior.ERROR,
        loss_on_prompt: bool = False,
        dataset_specific_config: DatasetSpecificConfig | None = None,
        batch_size: int = 1,
        dataloader_seed: int = 81,
    ):
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.length_excession_behavior = length_excession_behavior
        self.loss_on_prompt = loss_on_prompt
        self.dataset_specific_config = dataset_specific_config
        if dataset_specific_config is not None:
            assert dataset_specific_config.dataset_type == dataset_type, (
                f"Config type mismatch: config is for {dataset_specific_config.dataset_type}, "
                f"but dataset_type is {dataset_type}"
            )

        self._loaded_datasets: PredefinedDataset.LoadedDatasets | None = None
        """
        Cache for loaded datasets. Stores training split in "train" and validation split in "val" (if applicable). Initialized to None, and populated on first call to load_dataset().
        """

        self.batch_size = batch_size
        self.dataloader_seed = dataloader_seed

    def _load_dataset(self):
        """
        Loads the dataset according to the type and configuration parameters specified in the constructor. Caches the loaded dataset for future calls; there's no need to call this method more than once per instance.
        """
        if self._loaded_datasets is None:
            self._loaded_datasets = self._make_dataset()
        return self._loaded_datasets

    def _make_dataset(self) -> LoadedDatasets:
        print(f"Loading training dataset of type {self.dataset_type} with config:", self.dataset_specific_config)

        match self.dataset_type:
            case DatasetType.OPEN_THOUGHTS:
                from training.dataset.openthoughts.open_thoughts import (
                    OpenThoughtsDataset,
                )

                assert isinstance(self.dataset_specific_config, OpenThoughtsConfig)
                datasets: PredefinedDataset.LoadedDatasets = {
                    "train": OpenThoughtsDataset(
                        data_path=self.dataset_specific_config.data_path,
                        tokenizer=self.tokenizer,
                        max_length=self.dataset_specific_config.max_seq_length,
                        format=self.dataset_specific_config.data_format,
                        truncate=self.length_excession_behavior
                        == LengthExcessionBehavior.TRUNCATE,
                        loss_on_prompt=self.loss_on_prompt,
                    )
                }
                if self.dataset_specific_config.val_data_path is not None:
                    datasets["val"] = OpenThoughtsDataset(
                        data_path=self.dataset_specific_config.val_data_path,
                        tokenizer=self.tokenizer,
                        max_length=self.dataset_specific_config.max_seq_length,
                        format=self.dataset_specific_config.data_format,
                        truncate=self.length_excession_behavior
                        == LengthExcessionBehavior.TRUNCATE,
                        loss_on_prompt=self.loss_on_prompt,
                    )
                return datasets

            case DatasetType.FINEWEB_LMYSYSCHAT_MIXED:
                from training.dataset.gemma.mixed import MixedDataset
                from training.dataset.gemma.fineweb import FineWebDataset
                from training.dataset.gemma.lmsys_chat import LMSYSChatDataset

                assert isinstance(self.dataset_specific_config, FineWebLMSysMixedConfig)
                pretraining_dataset = FineWebDataset(
                    data_path=self.dataset_specific_config.pretraining_datapath,
                    tokenizer=self.tokenizer,
                    max_length=self.dataset_specific_config.pretraining_max_seq_length,
                    truncate=self.length_excession_behavior
                    == LengthExcessionBehavior.TRUNCATE,
                )
                chat_dataset = LMSYSChatDataset(
                    data_path=self.dataset_specific_config.chat_conversations_datapath,
                    tokenizer=self.tokenizer,
                    max_length=self.dataset_specific_config.chat_max_seq_length if self.dataset_specific_config.chat_max_seq_length != "pretraining_max_seq_length" else self.dataset_specific_config.pretraining_max_seq_length,
                    truncate=self.length_excession_behavior
                    == LengthExcessionBehavior.TRUNCATE,
                )
                mixed = MixedDataset(
                    datasets=(pretraining_dataset, chat_dataset), weights=(0.5, 0.5)
                )
                return {
                    "train": mixed,
                }
            case _:
                raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

    def _make_dataloader(self) -> Dataloaders:
        """
        Creates new dataloaders for each split in the dataset.
        """
        collate_with_tokenizer = partial(collate_fn, tokenizer=self.tokenizer)

        generator = TorchGenerator()
        generator.manual_seed(self.dataloader_seed)

        assert self._loaded_datasets is not None, (
            "Datasets must be loaded before creating dataloaders"
        )
        assert "train" in self._loaded_datasets, (
            "Training split ('train') is required in loaded datasets"
        )
        dataloaders: PredefinedDataset.Dataloaders = {
            "train": DataLoader(
                self._loaded_datasets["train"], # pyright: ignore[reportArgumentType]
                batch_size=self.batch_size,  # used to be micro_batch_size instead of using the normal batch_size: there were two seperate parameters. This was because we were trying gradient accumulation, however we decided it wasn't worth it so in all cases we set micro_batch_size = batch_size. So, I'm removing micro_batch_size and just using batch_size directly.
                shuffle=True,
                collate_fn=collate_with_tokenizer,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                generator=generator,
            )
        }

        if "val" in self._loaded_datasets:
            dataloaders["val"] = DataLoader(
                self._loaded_datasets["val"], # pyright: ignore[reportArgumentType]
                batch_size=1,
                shuffle=False,
                collate_fn=collate_with_tokenizer,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
            )

        assert dataloaders.keys() == self._loaded_datasets.keys(), (
            "Note: Dataloader did not generate loaders for all dataset splits"
        )
        return dataloaders

    def load_datasets_and_dataloaders(self) -> tuple[LoadedDatasets, Dataloaders]:
        """
        Loads the dataset (if not already loaded) and creates dataloaders for each split. Returns a tuple of (loaded_datasets [all splits of the loaded dataset], dataloaders [one data loader for each split]).
        """
        datasets = self._load_dataset()
        dataloaders = self._make_dataloader()
        return datasets, dataloaders
