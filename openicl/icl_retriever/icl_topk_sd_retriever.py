"""Topk-SD Retriever"""

from sklearn.preprocessing import normalize
from openicl import DatasetReader
from openicl.icl_dataset_reader import DatasetEncoder
from openicl.icl_retriever import BaseRetriever
from openicl.utils.collators import DataCollatorWithPaddingAndCuda
from openicl.utils.logging import get_logger
import torch
from torch.utils.data import DataLoader
from typing import Optional
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import tqdm
import faiss
import copy
import math
import numpy as np
from accelerate import Accelerator
import random

logger = get_logger(__name__)



def calc(a, b, lamda):
    res = lamda * a + ( 1 - lamda ) * b
    res = res / np.linalg.norm(res, ord=2)
    return res
    

class TopkSDRetriever(BaseRetriever):

    model = None

    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 sentence_transformers_model_name: Optional[str] = 'all-mpnet-base-v2',
                 ice_num: Optional[int] = 1,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 tokenizer_name: Optional[str] = 'gpt2-xl',
                 batch_size: Optional[int] = 1,
                 accelerator: Optional[Accelerator] = None,
                 lamda : Optional[float] = 0.7
                 ) -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token, ice_num, index_split,
                         test_split, accelerator)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name
        gen_datalist = self.dataset_reader.generate_input_field_corpus(self.test_ds)
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.pad_token = '[PAD]'
        self.tokenizer.padding_side = "right"
        
        

        self.encode_dataset = DatasetEncoder(gen_datalist, tokenizer=self.tokenizer)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=self.device)
        self.dataloader = DataLoader(self.encode_dataset, batch_size=self.batch_size, collate_fn=co)

        self.model = SentenceTransformer(sentence_transformers_model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.lamda = lamda
        
        self.index = self.create_index()
        
    

    def create_index(self):
        self.select_datalist = self.dataset_reader.generate_input_field_corpus(self.index_ds)
        self.select_labellist = self.dataset_reader.generate_output_field_corpus(self.index_ds)
        encode_datalist = DatasetEncoder(self.select_datalist, tokenizer=self.tokenizer)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=self.device)
        dataloader = DataLoader(encode_datalist, batch_size=self.batch_size, collate_fn=co)
        index = faiss.IndexIDMap(faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension()))
        res_list = self.forward(dataloader, process_bar=True, information="Creating index for index set...")
        id_list = np.array([res['metadata']['id'] for res in res_list])
        self.embed_list = np.stack([res['embed'] for res in res_list])
        label_len = len(set(self.select_labellist))
        self.label_points = np.zeros((label_len,self.embed_list.shape[1]))
        for label in range(label_len):
            indices = [j for j, l in enumerate(self.select_labellist) if int(l) == label]
            self.label_points[label] = np.mean(self.embed_list[indices], axis=0)
            self.label_points[label] = self.label_points[label] / np.linalg.norm(self.label_points[label], ord=2)
            for idx in indices:
                self.embed_list[idx] = calc( self.embed_list[idx], self.label_points[label], self.lamda)
        self.base_point = normalize((np.mean(self.label_points, axis=0)).reshape(1, -1), norm='l2').flatten()
        index.add_with_ids(self.embed_list, id_list)
        return index

    def knn_search(self, ice_num):
    
        label_len = len(set(self.select_labellist))
        res_list = self.forward(self.dataloader, process_bar=True, information="Embedding test set...")
        rtr_idx_list = [[] for _ in range(len(res_list))]
        logger.info("Retrieving data for test set...")
        for entry in tqdm.tqdm(res_list, disable=not self.is_main_process):
            idx = entry['metadata']['id']
            embed = entry['embed']
            embed = calc( embed, self.base_point, self.lamda)
            embed = np.expand_dims(embed, axis=0).astype(np.float32)
            near_ids = self.index.search(embed, ice_num)[1][0].tolist()
            rtr_idx_list[idx] = near_ids
        return rtr_idx_list

    def forward(self, dataloader, process_bar=False, information=''):
        res_list = []
        _dataloader = copy.deepcopy(dataloader)
        if process_bar:
            logger.info(information)
            _dataloader = tqdm.tqdm(_dataloader, disable=not self.is_main_process)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})  
        self.tokenizer.pad_token = '[PAD]'
        for _, entry in enumerate(_dataloader):
            with torch.no_grad():
                metadata = entry.pop("metadata")
                raw_text = self.tokenizer.batch_decode(entry['input_ids'], skip_special_tokens=True, verbose=False)
                res = self.model.encode(raw_text, show_progress_bar=False)
            res_list.extend([{"embed": r, "metadata": m} for r, m in zip(res, metadata)])
        return res_list

    def retrieve(self):
        return self.knn_search(self.ice_num)
