"""MDL Retriever"""

from openicl import DatasetReader, PromptTemplate
from openicl.icl_retriever.icl_topk_retriever import TopkRetriever
from openicl.utils.calculate import entropy
from openicl.utils.logging import get_logger
from typing import List, Union, Optional, Tuple
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import tqdm
import torch
import torch.nn as nn
import numpy as np
from accelerate import Accelerator
from torch.nn import DataParallel



logger = get_logger(__name__)


class MDLRetriever(TopkRetriever):

    metric_model = None

    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 sentence_transformers_model_name: Optional[str] = 'all-mpnet-base-v2',
                 ice_num: Optional[int] = 1,
                 candidate_num: Optional[int] = 1,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 tokenizer_name: Optional[str] = 'gpt2-xl',
                 ce_model_name: Optional[str] = 'gpt2-xl',
                 model_tokenizer_name: Optional[str] = 'llama2',
                 batch_size: Optional[int] = 1,
                 select_time: Optional[int] = 5,
                 accelerator: Optional[Accelerator] = None,
                 ice_template: Optional[PromptTemplate] = None,
                 prompt_template: Optional[PromptTemplate] = None,
                 labels: Optional[List] = None,
                 seed: Optional[int] = 1
                 ) -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token,
                         sentence_transformers_model_name, ice_num, index_split, test_split, tokenizer_name, batch_size,
                         accelerator)
        self.ce_model_name = ce_model_name
        self.candidate_num = candidate_num
        self.select_time = select_time
        self.ice_template = ice_template
        self.prompt_template = prompt_template
        self.labels = labels
        self.seed = seed

        
        self.model_tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_name)
        self.model_tokenizer.pad_token = self.model_tokenizer.eos_token
        self.model_tokenizer.pad_token_id = self.model_tokenizer.eos_token_id
        self.model_tokenizer.padding_side = "right"

    def topk_search(self):
        np.random.seed(self.seed)
        res_list = self.forward(self.dataloader)
        rtr_idx_list = [[] for _ in range(len(res_list))]

        logger.info("Retrieving data for test set...")
        print('-------------------')
        
        for entry in tqdm.tqdm(res_list, disable=not self.is_main_process):
            # print('--------------')
            idx = entry['metadata']['id']

            embed = np.expand_dims(entry['embed'], axis=0)
            near_ids = self.index.search(embed, min(self.candidate_num, len(self.index_ds)))[1][0].tolist()
            candidates = []
            mdl_scores = []
            for j in range(self.select_time):
                if j == 0:
                    rand_idx_list = near_ids[:self.ice_num]
                else:
                    rand_idx_list = np.random.choice(near_ids, self.ice_num, replace=False)
                    rand_idx_list = [int(i) for i in rand_idx_list]
                candidates.append(rand_idx_list)

                ice = self.generate_ice(rand_idx_list, ice_template=self.ice_template)
                mask_length = len(self.tokenizer(ice + self.ice_eos_token, verbose=False)['input_ids'])
                if self.labels is None:
                    labels = self.get_labels(self.ice_template, self.prompt_template)
                else:
                    labels = self.labels
                prompt_list = []
                loss_list = []
                for label in labels:
                    prompt = self.generate_label_prompt(idx, ice, label, self.ice_template, self.prompt_template)
                    prompt_list = [ prompt ]
                    
                    # print(prompt_list)
                    res = self.cal_ce(prompt_list, mask_length=mask_length)  
                    # print("loss list")
                    # print(res)
                    loss_list.append(res[0])
                # print(prompt_list)
                # print(loss_list)
                probs = np.exp(-np.array(loss_list))
                normalized_probs = probs / probs.sum(0, keepdims=True)
                neg_entropy = -entropy(normalized_probs, label_dim=0)
                mdl_scores.append(neg_entropy)

            rtr_idx_list[idx] = candidates[mdl_scores.index(max(mdl_scores))]
            rtr_idx_list[idx] = [int(i) for i in rtr_idx_list[idx]]
        del self.metric_model
        torch.cuda.empty_cache()
        return rtr_idx_list

    def retrieve(self):
        return self.topk_search()

    
    
    def cal_ce(self, input_texts: List[str], mask_length=None):
        device = torch.device("cuda:0")
        if self.metric_model is None:
            logger.info(f'Load model {self.metric_model} for calculating MDL...')
            if 'Llama' in self.ce_model_name:
                self.metric_model = LlamaForCausalLM.from_pretrained(self.ce_model_name)
            else:
                self.metric_model = AutoModelForCausalLM.from_pretrained(self.ce_model_name)
            self.metric_model.to(device)
        inputs = self.model_tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.metric_model(**inputs)

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.model_tokenizer.pad_token_id).to(device)
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        loss = loss_fct(shift_logits, shift_labels.view(-1)).view(shift_labels.size())
        if mask_length is not None:
            dim = loss.shape[-1] - mask_length
            if dim < 0:
                dim = 0
                mask = torch.cat([torch.zeros([loss.shape[0], mask_length], dtype=torch.float),
                      torch.ones([loss.shape[0], dim], dtype=torch.float)], -1)
                mask = mask.to(device)
            else:
                mask = torch.ones_like(loss)

            if mask.shape != loss.shape:
                min_length = min(mask.shape[-1], loss.shape[-1])
                mask = mask[:, :min_length]
                loss = loss[:, :min_length]

            # ִ�г˷�����
            loss = torch.mul(mask, loss)

        lens = (inputs["input_ids"] != self.model_tokenizer.pad_token_id).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= mask_length
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        
        torch.cuda.empty_cache()
        
        return ce_loss
