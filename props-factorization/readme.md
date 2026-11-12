# Prompted Policy Search: Reinforcement Learning through Linguistic and Numerical Reasoning in LLMs


This repo serves as the code base for Prompted Policy Search (ProPS and ProPS<sup>+</sup>). The project website is [here](https://props-llm.github.io/).

<p align="center">
<img src = "static/banner.gif" width ="800" />
</p>

## Key Takeaways
In this paper, we demonstrate that:
1. LLMs can perform numerical optimization for Reinforcement Learning (RL) tasks.
2. LLMs can incorporate semantics signals, (e.g., goals, domain knowledge, ...), leading to more informed exploraton and sample-efficient learning.
3. Our proposed ProPS outperforms all baselines on 8 out of 15 Gymnasium tasks.


# Getting Started

## Install RL Tasks

- The RL tasks are based on gymnasium. Please install according to `https://github.com/Farama-Foundation/Gymnasium`
- There are 2 customized environments in the folders `./envs/gym-maze-master` and `./envs/gym-navigation-main`. If you want to train the maze or navigation agent, please pip install the packages.

## Install the LLM APIs

We utilized the standard Google Gemini, OpenAI-compatible APIs, and Anthropic APIs. Please install the packages accordingly.

- `https://ai.google.dev/gemini-api/docs`
- `https://platform.openai.com/docs/overview`
- `https://console.groq.com/docs/overview`
- `https://docs.anthropic.com/en/release-notes/api`

### Groq Models

Groq models are supported through an OpenAI-compatible client.

- Set `GROQ_API_KEY` in your environment or `.env`
- Optionally set `GROQ_BASE_URL` (defaults to `https://api.groq.com/openai/v1`)
- In config, set `llm_model_name` to either:
	- `groq/llama-3.3-70b-versatile` (recommended explicit provider prefix)
	- `llama-3.3-70b-versatile` (also supported)

### Ollama Local Models

Ollama models are supported for local inference.

- Install and run Ollama from `https://ollama.com/`
- Pull one or more local models, for example:
	- `ollama pull llama3.2:latest`
	- `ollama pull qwen2.5:7b`
- Make sure the Python package is installed (already listed in `requirements.txt`):
	- `pip install ollama`
- In config, set `llm_model_name` to either:
	- `ollama/llama3.2:latest` (recommended explicit provider prefix)
	- `llama3.2:latest` (also supported)

Optional:
- If Ollama is running on a non-default host, set `OLLAMA_HOST` (for example `http://127.0.0.1:11434`).

Quick local test configs:
- `configs/cartpole/cartpole_lu_ollama_llama32_test.yaml`
- `configs/cartpole/cartpole_lu_ollama_phi3_test.yaml`
- `configs/cartpole/cartpole_lu_ollama_deepseek15b_test.yaml`
- `configs/cartpole/cartpole_lu_ollama_qwen25_test.yaml`

## Start Training
In order to run an experiment, please run `python main.py --config <configuration_file>`.

Examples:
- `python main.py --config configs/cartpole/cartpole_lu_ollama_llama32_test.yaml`
- `python main.py --config configs/cartpole/cartpole_lu_ollama_phi3_test.yaml`
- `python main.py --config configs/cartpole/cartpole_lu_ollama_deepseek15b_test.yaml`

### Decomposition Config (Factorized Policy)

For factorized linear policies, you can now choose decomposition mode directly in YAML:

```yaml
use_factorized_policy: true
factor_rank: 2
decomposition_type: lu   # lu | qr | svd
frozen_factor: null      # LU: L/U, QR: Q/R, SVD: U/S/Vt
```

Template recommendations:
- `lu`: `llm_si_template_name: num_optim_lu.j2`
- `qr`: `llm_si_template_name: num_optim_qr.j2`
- `svd`: `llm_si_template_name: num_optim_svd.j2`

Sample configs:
- `configs/cartpole/cartpole_lu_test.yaml`
- `configs/cartpole/cartpole_qr_test.yaml`
- `configs/cartpole/cartpole_svd_test.yaml`
