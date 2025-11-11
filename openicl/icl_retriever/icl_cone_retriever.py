"""MDL Retriever"""

from openicl import DatasetReader, PromptTemplate
from openicl import DatasetReader, PromptTemplate
from openicl.icl_retriever.icl_topk_retriever import TopkRetriever
from openicl.utils.calculate import entropy
from openicl.utils.logging import get_logger
from typing import List, Union, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
import torch
import numpy as np
from accelerate import Accelerator
from transformers import LlamaTokenizer, LlamaForCausalLM


logger = get_logger(__name__)


class ConERetriever(TopkRetriever):

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
                 model_tokenizer_name: Optional[str] = 'llama2',
                 ce_model_name: Optional[str] = 'gpt2-xl',
                 batch_size: Optional[int] = 1,
                 ppl_batch_size: Optional[int] = 1,
                 select_time: Optional[int] = 5,
                 accelerator: Optional[Accelerator] = None,
                 ice_template: Optional[PromptTemplate] = None,
                 basic_prompt: Optional[str] = None,
                 prompt_template: Optional[PromptTemplate] = None,
                 labels: Optional[List] = None,
                 seed: Optional[int] = 1
                 ) -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token,
                         sentence_transformers_model_name, ice_num, index_split, test_split, tokenizer_name, batch_size,
                         accelerator)
        print("--1--")
        self.ce_model_name = ce_model_name
        self.candidate_num = candidate_num
        self.select_time = select_time
        self.ice_template = ice_template
        self.prompt_template = prompt_template
        self.labels = labels
        self.seed = seed
        self.ppl_batch_size = 1
        self.basic_prompt = basic_prompt
        self.model_tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_name)
        print("--2--")
            
        self.model_tokenizer.pad_token = self.model_tokenizer.eos_token
        self.model_tokenizer.pad_token_id = self.model_tokenizer.eos_token_id
        self.model_tokenizer.padding_side = "right"

    def topk_search(self):
        np.random.seed(self.seed)
        res_list = self.forward(self.dataloader)
        rtr_idx_list = [[] for _ in range(len(res_list))]

        # key word is the word in the template to predict the label
        key_word = self.ice_template.template[0].split('</text>')[-1].split()[0]
        
        logger.info("Retrieving data for test set...")
        for entry in tqdm.tqdm(res_list, disable=not self.is_main_process):
            idx = entry['metadata']['id']

            # get embedding and the near_ids
            embed = np.expand_dims(entry['embed'], axis=0)
            near_ids = self.index.search(embed, min(self.candidate_num, len(self.index_ds)))[1][0].tolist()
            candidates = []
            mdl_scores = []

            prompts = []
            mask_lengths = []
            test_lengths = []

            for j in range(self.candidate_num):
                rand_idx_list = [near_ids[j]]
                candidates.append(rand_idx_list)

                # ice is the in-context demonstrations
                ice = self.generate_ice(rand_idx_list, ice_template=self.ice_template)
                
                # ice_eos_token has been added in ice
                mask_length = len(self.model_tokenizer(ice, verbose=False)['input_ids'])

                if self.labels is None:
                    labels = self.get_labels(self.ice_template, self.prompt_template)
                else:
                    labels = self.labels

                prompt = self.generate_label_prompt(idx, ice, labels[0], self.ice_template, self.prompt_template)

                if self.basic_prompt:
                    prompt = self.basic_prompt + prompt

                # consider the test pos of the ice + input + Type:
                test_pos = prompt.rindex(key_word) + len(key_word)
                test_length = len(self.model_tokenizer(prompt[:test_pos], verbose=False)['input_ids'])
                
                # get the batch of prompt, mask_length, test_length
                prompts.append(prompt)
                mask_lengths.append(mask_length)
                test_lengths.append(test_length)


            for batch_id in range(self.candidate_num // self.ppl_batch_size):
                with torch.no_grad():
                    loss_list = self.cal_ce(prompts[batch_id * self.ppl_batch_size: (batch_id + 1) * self.ppl_batch_size], mask_lengths=mask_lengths[batch_id * self.ppl_batch_size: (batch_id + 1) * self.ppl_batch_size], test_lengths=test_lengths[batch_id * self.ppl_batch_size: (batch_id + 1) * self.ppl_batch_size])
                    mdl_scores.extend(loss_list)
            
            if self.candidate_num % self.ppl_batch_size != 0:
                with torch.no_grad():
                    end_pos = self.candidate_num // self.ppl_batch_size * self.ppl_batch_size
                    loss_list = self.cal_ce(prompts[end_pos:], mask_lengths=mask_lengths[end_pos:], test_lengths=test_lengths[end_pos:])
                    mdl_scores.extend(loss_list)

            ppl_scores = list(sorted(list(enumerate(mdl_scores)), key=lambda x: x[1]))
            # get the most lower ppl demonstrations for each test input
            rtr_idx_list[idx] = [int(candidates[ppl_scores[i][0]][0]) for i in range(self.ice_num)]
            torch.cuda.empty_cache()
        return rtr_idx_list

    def retrieve(self):
        return self.topk_search()

    def cal_ce(self, input_texts: List[str], mask_lengths=None, test_lengths=None):
        device = torch.device("cuda:1")
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

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.model_tokenizer.pad_token_id)
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        loss = loss_fct(shift_logits, shift_labels.view(-1)).view(shift_labels.size())
        if mask_lengths is not None and test_lengths is not None:
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_lengths[i], test_lengths[i]):
                    mask[i][j] = 1
            loss = loss * mask

        ce_loss = torch.sum(loss, 1)
        return ce_loss
