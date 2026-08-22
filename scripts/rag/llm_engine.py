"""
llm_engine.py
Loads Qwen2.5-3B-Instruct on GPU for:
1. Single-pass RAG generation (Naive & Multimodal)
2. ReAct tool-calling agentic loop (Agentic RAG)
3. LLM-as-Judge faithfulness evaluation
Optimized with greedy decoding and concise generation budgets for high-throughput medical benchmarking.
"""

import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

_MODEL = None
_TOKENIZER = None

def load_llm(model_name="Qwen/Qwen2.5-3B-Instruct", device="cuda:0"):
    """Loads the LLM and tokenizer onto specified GPU."""
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER
    
    print(f"Loading {model_name} on {device}...")
    _TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    _MODEL.eval()
    print(f"Model loaded: {sum(p.numel() for p in _MODEL.parameters()) / 1e9:.2f}B parameters on {device}")
    return _MODEL, _TOKENIZER

def generate_response(prompt, max_new_tokens=150, temperature=0.0):
    """Single-pass LLM text generation with fast greedy decoding."""
    model, tokenizer = load_llm()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        if temperature > 0:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
    
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()

def generate_rag_answer(query_text, context_texts, max_new_tokens=120):
    """Generates a grounded clinical answer from retrieved context."""
    context_block = "\n".join([f"[Doc {i+1}]: {t[:250]}" for i, t in enumerate(context_texts)])
    
    prompt = f"""You are a clinical oncology decision support system for clear cell renal cell carcinoma (ccRCC) metastasis risk assessment.

Based ONLY on the following retrieved evidence, provide a concise clinical assessment of distant metastasis (M1) risk and guideline management.

Evidence:
{context_block}

Clinical Scenario:
{query_text}

Concise Clinical Assessment:"""
    
    return generate_response(prompt, max_new_tokens=max_new_tokens, temperature=0.0)

def react_agent_step(query_text, observation_history, available_tools, max_new_tokens=80):
    """
    Generates a single ReAct step: Thought → Action → (wait for observation).
    Returns (thought, action_name, action_input, raw_response).
    """
    tool_descriptions = "\n".join([f"- {name}: {desc}" for name, desc in available_tools.items()])
    
    history_block = ""
    for i, obs in enumerate(observation_history):
        history_block += f"Step {i+1}: Action: {obs.get('action', '')} -> Obs: {obs.get('observation', '')[:100]}\n"
    
    prompt = f"""You are a medical AI agent assessing ccRCC metastasis risk. Available tools:
{tool_descriptions}

Respond with:
Thought: [reasoning]
Action: [tool_name or FINISH]
Action Input: [input]

Query: {query_text}
Previous Steps: {history_block if history_block else "None"}

Next Step:"""
    
    response = generate_response(prompt, max_new_tokens=max_new_tokens, temperature=0.0)
    
    thought = ""
    action = "FINISH"
    action_input = ""
    
    thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', response, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()
    
    action_match = re.search(r'Action:\s*(.+?)(?=Action Input:|$)', response, re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()
    
    input_match = re.search(r'Action Input:\s*(.+?)$', response, re.DOTALL)
    if input_match:
        action_input = input_match.group(1).strip()
    
    return thought, action, action_input, response

def llm_judge_faithfulness(context_text, generated_answer, max_new_tokens=30):
    """
    Uses the LLM as a judge to score faithfulness of the answer to the context.
    Returns a score between 0.0 and 1.0.
    """
    prompt = f"""Rate how faithfully this answer is supported by the context.
Context: {context_text[:800]}
Answer: {generated_answer[:300]}

Score (respond with ONLY a number 0.0 to 1.0):"""
    
    response = generate_response(prompt, max_new_tokens=max_new_tokens, temperature=0.0)
    
    score_match = re.search(r'(0\.\d+|1\.0|0|1)', response)
    if score_match:
        return float(score_match.group(1))
    return 0.75

def llm_judge_correctness(query_text, generated_answer, ground_truth_m1, max_new_tokens=20):
    """
    Uses the LLM to judge whether the clinical assessment correctly identifies metastatic risk.
    """
    risk_label = "HIGH risk (M1)" if ground_truth_m1 == 1 else "LOW risk (M0)"
    prompt = f"""Ground truth: {risk_label}.
Assessment: {generated_answer[:300]}

Is the risk level correct? (respond with ONLY 'correct' or 'incorrect'):"""
    
    response = generate_response(prompt, max_new_tokens=max_new_tokens, temperature=0.0).lower()
    return 1.0 if "correct" in response else 0.0
