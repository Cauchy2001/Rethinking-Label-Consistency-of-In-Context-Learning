'''Random Retriever'''

from openicl import DatasetReader
from openicl.icl_retriever import BaseRetriever
from openicl.utils.logging import get_logger
from typing import List, Union, Optional
from tqdm import trange
import numpy as np
from accelerate import Accelerator

logger = get_logger(__name__)


class RandomRetriever(BaseRetriever):


    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 ice_num: Optional[int] = 1,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 seed: Optional[int] = 43,
                 accelerator: Optional[Accelerator] = None
                 ) -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token, ice_num, index_split,
                         test_split, accelerator)
        self.seed = seed

    def retrieve(self):
        np.random.seed(self.seed)
        num_idx = len(self.index_ds)
        rtr_idx_list = []
        logger.info("Retrieving data for test set...")
        for _ in trange(len(self.test_ds), disable=not self.is_main_process):
            idx_list = np.random.choice(num_idx, self.ice_num, replace=False).tolist()
            rtr_idx_list.append(idx_list)
        return rtr_idx_list
