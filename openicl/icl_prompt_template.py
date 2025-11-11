"""Prompt Template"""

from typing import Dict, Optional, Union, Hashable
from .utils.check_type import _check_type_list, _check_dict


class PromptTemplate:


    def __init__(self,
                 template: Union[Dict, str],
                 column_token_map: Dict,
                 selected_column_name: Optional[str] = None,
                 selected_column_map: Optional[Dict] = None,
                 ice_token: Optional[str] = None,
                 sep_token: Optional[str] = None,
                 ) -> None:
        self.template = _check_type_list(template, [Dict, str])
        self.column_token_map = _check_dict(column_token_map)
        self.selected_column_name = _check_type_list(selected_column_name, [None, str])
        self.selected_column_map = _check_type_list(selected_column_map, [None, Dict])
        self.ice_token = _check_type_list(ice_token, [None, str])
        self.sep_token = _check_type_list(sep_token, [None, str])
        if (self.selected_column_name is not None and self.selected_column_map is None) or \
                self.selected_column_name is None and self.selected_column_map is not None:
            raise ValueError("self.selected_column_name and self.selected_column_map should be set together")
        self._check_template_legacy()

    def _check_template_legacy(self):
        if isinstance(self.template, Dict):
            # Check if token exists in values of tp_dict 
            for tp_dict_val in self.template.values():
                if not isinstance(tp_dict_val, str):
                    raise TypeError(f"dictionary of template expects a str value, but got '{tp_dict_val}'")
                if self.ice_token is not None and self.ice_token not in tp_dict_val:
                    raise LookupError(f"'{self.ice_token}' not in '{tp_dict_val}'")
        if isinstance(self.template, str):
            if self.ice_token is not None and self.ice_token not in self.template:
                raise LookupError(f"'{self.ice_token}' not in '{self.template}'")

        # Check duplicates
        if len(self.column_token_map.values()) != len(set(self.column_token_map.values())):
            raise ValueError(f"There are duplicates in self.column_token_map.values()")
        if self.ice_token is not None and self.ice_token in self.column_token_map.values():
            raise ValueError(f"There are duplicates between self.column_token_map.values() and self.ice_token")

    def generate_ice_item(self, entry: Dict, label: Hashable) -> str:

        # Select the corresponding template 
        tp = self.template[label] if isinstance(self.template, Dict) else self.template
        # Remove sep token
        if self.sep_token is not None:
            tp.replace(self.sep_token, '')
        # Remove ice_token
        if self.ice_token is not None:
            tp = tp.replace(self.ice_token, '')
        # Replace context token
        for key, token in self.column_token_map.items():
            if self.selected_column_map is not None and key == self.selected_column_name:
                tp = tp.replace(token, str(self.selected_column_map[label]))
            else:
                tp = tp.replace(token, str(entry[key]))
        # print(tp)
        return tp

    def generate_label_prompt_item(self, entry: Dict, ice: str, label: Hashable, remain_sep: Optional[bool] = False) -> str:

        if self.ice_token is None:
            raise ValueError("PromptTemplate.ice_token should be not None when generates prompt")
        # Select the corresponding template
        tp = self.template[label] if isinstance(self.template, Dict) else self.template
        # Remove sep token
        if not remain_sep and self.sep_token is not None:
            tp.replace(self.sep_token, '')
        # Insert in-context examples
        tp = tp.replace(self.ice_token, ice)
        # Replace context token
        for key, token in self.column_token_map.items():
            if self.selected_column_map is not None and key == self.selected_column_name:
                tp = tp.replace(token, str(self.selected_column_map[label]))
            else:
                tp = tp.replace(token, str(entry[key]))
        return tp


    def generate_item(self, entry: Dict, output_field: Optional[Hashable] = None,
                      output_field_replace_token: Optional[str] = '',
                      ice_field_replace_token: Optional[str] = '') -> str:

        tp = None
        if isinstance(self.template, str):
            tp = self.template
        else:
            pred_label = None
            if self.selected_column_name is not None:
                pred_label = entry[self.selected_column_name]
            if pred_label in self.template.keys():
                tp = self.template[pred_label]
            else:
                tp = self.template[list(self.template.keys())[0]]
        if self.ice_token is not None:
            tp = tp.replace(self.ice_token, ice_field_replace_token)
        # Remove sep token
        if self.sep_token is not None:
            tp.replace(self.sep_token, '')
        for key, token in self.column_token_map.items():
            if output_field is not None and key == output_field:
                tp = tp.replace(token, output_field_replace_token)
            else:
                tp = tp.replace(token, str(entry[key]))
        return tp

    def _check_prompt_template(obj) -> "PromptTemplate":
        if isinstance(obj, PromptTemplate):
            return obj
        else:
            raise TypeError(f"Expect a PromptTemplate object, but got {obj}")

    def __repr__(self):
        return f"PromptTemplate({{\n\ttemplate: {self.template},\n\tcolumn_token_map: {self.column_token_map},\n\tice_token: {self.ice_token}\n}})"
