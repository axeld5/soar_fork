# Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI

<p align="center">
          🤗 <a href="https://huggingface.co/collections/julien31/soar-arc-6856d27681fce01d9af4c4a3">Hugging Face (data and model)</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="todo">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://julienp.netlify.app/posts/soar/">Blog</a>
</p>

Large Language Models (LLMs) have become incredibly powerful, but they often hit a wall when faced with truly complex reasoning tasks that require discovering a solution from scratch. Simply throwing more computing power or using a bigger model often yields diminishing returns. But what if a model could learn from its own experience, getting smarter with every attempt?

We introduce a framework called **SOAR (Self-improving Operators for Automated program Refinements)** that does just that. By creating a "virtuous cycle" of evolutionary search and learning, SOAR enables AI models to bootstrap their own capabilities and solve problems previously beyond their reach. we tested SOAR on the Abstraction and Reasoning Corpus (ARC-AGI-1), a notoriously difficult benchmark designed to challenge an AI's core reasoning abilities. We show that using SOAR with only open weight LLM, we can significantly outperforming much larger closed source LLMs.

## SOAR framework
![soar](docs/image/soar_fig.png)

1.  **Evolutionary Search (Sample & Refine):** SOAR uses an LLM to generate an initial pool of thousands of candidate programs (the "sampling" step). It then tests these programs and uses the LLM again to intelligently modify or "refine" the most promising ones based on their performance.

2.  **Learning from Hindsight:** SOAR takes all the programs generated during the search phase—including both successes and failures—and uses them as training data. The key insight is that any failed program is simply a *correct* program for a *different* task. By "relabeling" these failed attempts as correct solutions for the synthetic tasks they inadvertently solve, SOAR creates a diverse dataset to learn from.

This process creates a powerful feedback loop: the fine-tuned model becomes better at sampling and refining, which leads to a more effective search in the next iteration, which in turn generates even better training data. And unlike previous approaches that rely on human-engineered domain-specific languages or human-generated solutions, SOAR learns to synthesize programs in Python solely from its own synthesis attempts, encompassing both successes and failures.


## info install

### conda inference env
```
pip install --upgrade pip

git clone https://github.com/flowersteam/SOAR
cd SOAR
conda create --name sglang47 \
    python=3.11 \
    -y
conda activate sglang47

pip install "sglang[all]>=0.4.7"

pip install -e .
pip install -r requirements

```

### conda train env
```
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate unsloth_env

pip install unsloth
cd SOAR
pip install -e .
pip install -r requirements.txt
```

## Run SOAR
To run SOAR, please refer to execution instructions located in the experience folder. 