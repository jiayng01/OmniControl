import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os
from huggingface_hub import login


class PromptDecomposer:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3", device=None):
        load_dotenv()
        token = os.getenv("read")
        login(token=token)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            token=True,
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def decompose(self, user_prompt: str) -> str:
        system_prompt = (
            "You are a motion synthesis assistant. Given a natural language prompt, break it down into structured motion descriptions "
            "with start pose, limb orientation, facing direction, movement direction, and notable keyframes. "
            "Use a time-aligned chunking strategy to ensure each piece is grounded in motion."
            "\nExample Input: 'A person runs forward, turns right, and waves with their right hand.'"
            "\nOutput:"
            "\n1. [Frames 0-15] Start standing upright, facing forward, arms relaxed by the side."
            "\n2. [Frames 15-40] Running forward, legs alternating, arms swinging naturally."
            "\n3. [Frames 40-50] Turns body 90 degrees to the right, left foot pivots."
            "\n4. [Frames 50-70] Starts waving with right hand, arm lifted and moves side-to-side."
            "\nNow perform the same breakdown for the following input."
        )

        full_prompt = system_prompt + f"\nInput: '{user_prompt}'\nOutput:"

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return result.split("Output:")[-1].strip()

    def enrich_prompt(self, prompt: str) -> str:
        """Conditionally apply prompt decomposition."""
        print(f"Original prompt: {prompt}", flush=True)
        refined_prompt = self.decompose(prompt)
        print(f"Refined prompt: {refined_prompt}", flush=True)
        return refined_prompt


# Usage (at inference time)
# decomposer = PromptDecomposer()
# user_text = "A person dances, then spins and bows."
# control_prompt = decomposer.enrich_prompt(user_text)
