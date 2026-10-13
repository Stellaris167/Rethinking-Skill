<div align="center" style="max-width: 900px; margin: 0 auto; padding: 24px 0; line-height: 1.6;">
  <h1>Understanding Agent Skills: How Well Do They Transfer to Smaller Models?</h1>
  <p>Evaluation code and results for our study of agent skill transfer to smaller language models.</p>
  <p><a href="more_results.md">More results</a></p>
</div>

We will continue to update this repository with additional results.

## Repository Structure

```text
.
|-- eval.py                    # Unified evaluation entry point
|-- prepare_data.py            # Dataset preparation using fixed test splits
|-- evaluation/
|   |-- agent.py               # ReAct agent and skill, evidence, and code tools
|   |-- benchmarks.py          # HotpotQA, DAPO-Math, and BigCodeBench evaluation
|   |-- scoring.py             # Answer extraction and benchmark scoring
|   |-- model.py               # Model API calls and optional answer judge
|   |-- skills.py              # Markdown and file-structured skill loading
|   |-- data.py                # Dataset adapters and public/private test splits
|   |-- alfworld.py            # ALFWorld rollouts and evaluation
|   `-- alfworld_env/          # ALFWorld environment integration
|-- configs/
|   `-- evaluation.json        # Benchmark evaluation settings
|-- prompts/                   # Minimal prompts for the four benchmarks
|-- data/                      # Fixed test IDs and ALFWorld task list
|-- requirements.txt           # Python dependencies
|-- more_results.md            # Additional experimental results
|-- LICENSE
`-- README.md
```

## Note for Reproduction

If you find any issues with this repository, please feel free to [contact us](https://github.com/Stellaris167/Rethinking-Skill/issues). We will provide support to help you reproduce our experimental results. &#129303;
