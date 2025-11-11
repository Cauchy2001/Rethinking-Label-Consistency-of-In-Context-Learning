"""Zeroshot Retriever"""

from datasets import Dataset, DatasetDict
from typing import List, Union, Optional, Tuple, Dict
from openicl import DatasetReader, PromptTemplate
from openicl.icl_retriever import BaseRetriever
from openicl.utils.check_type import _check_str
from accelerate import Accelerator


class ZeroRetriever(BaseRetriever):


    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_eos_token: Optional[str] = '',
                 prompt_eos_token: Optional[str] = '',
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 accelerator: Optional[Accelerator] = None
                 ) -> None:
        super().__init__(dataset_reader, '', ice_eos_token, prompt_eos_token, 0, index_split, test_split, accelerator)

    def retrieve(self) -> List[List]:
        rtr_idx_list = [[] for _ in range(len(self.test_ds))]
        return rtr_idx_list
