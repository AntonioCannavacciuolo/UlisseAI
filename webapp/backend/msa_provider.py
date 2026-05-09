import os
import json
import torch
import logging
from typing import List, Dict, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

logger = logging.getLogger("ulisse_memo")

class MSAManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MSAManager, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.memory_bank = None
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._instance.is_loading = False
        return cls._instance

    def load_model(self, model_id: str = "EverMind-AI/MSA-4B", adapter_path: Optional[str] = "./ulisse-memo-adapter"):
        if self.model is not None:
            return
        
        self.is_loading = True
        logger.info(f"Loading MSA model {model_id} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            
            model_kwargs = {
                "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
                "trust_remote_code": True
            }
            
            if HAS_FLASH_ATTN and self.device == "cuda":
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2.")
            else:
                logger.warning("Flash Attention not found or CPU detected. Using standard attention.")

            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            
            if adapter_path and os.path.exists(adapter_path):
                try:
                    from peft import PeftModel
                    self.model = PeftModel.from_pretrained(self.model, adapter_path)
                    logger.info(f"LoRA adapter loaded from {adapter_path}")
                except ImportError:
                    logger.warning("PEFT not installed. Skipping LoRA adapter.")
            
            if self.device == "cpu":
                self.model = self.model.to("cpu")
                
            self.model.eval()
            logger.info("MSA model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading MSA model: {e}")
            raise e
        finally:
            self.is_loading = False

    def load_memory_bank(self, path: str = "./memory_bank.pt"):
        if os.path.exists(path):
            try:
                self.memory_bank = torch.load(path, map_location=self.device)
                logger.info(f"Memory bank loaded from {path}")
            except Exception as e:
                logger.error(f"Error loading memory bank: {e}")
        else:
            logger.warning(f"Memory bank not found at {path}. Initializing empty.")
            self.memory_bank = []

    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        if self.model is None:
            self.load_model()
        
        # apply_chat_template handles tool definitions for Qwen3/MSA format
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return self._parse_response(response_text)

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parses Qwen3 native JSON tool call format."""
        try:
            # Check if it looks like JSON tool calls
            # Qwen3 tool calls are often wrapped or just raw JSON
            if "tool_calls" in text:
                # Try to find the JSON part
                start_idx = text.find("{")
                end_idx = text.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    json_str = text[start_idx:end_idx+1]
                    parsed = json.loads(json_str)
                    if "tool_calls" in parsed:
                        return {"content": text[:start_idx].strip(), "tool_calls": parsed["tool_calls"]}
            
            return {"content": text, "tool_calls": None}
        except Exception:
            return {"content": text, "tool_calls": None}

msa_manager = MSAManager()
