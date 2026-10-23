# vsdt — Stiffness-aware digital-twin verification for VSSEA cobots

Code and data for the revised manuscript *"Stiffness-Aware Digital-Twin Verification of
LLM-Generated Plans for Collaborative Robots with Variable-Stiffness Series Elastic Actuators"*.

Everything in the paper is produced by the scripts below, and every number in the text comes
from `paper/numbers.tex`, which `scripts/export_numbers.py` generates from `results/`.

## Layout
```
vsdt/
  vssea.py      leaf-spring VSAM: Euler-Bernoulli/Timoshenko stiffness, plane-stress Q8 FE,
                elastica (large deflection), stress-limited capacity, modulation time/energy
  robot.py      arms: original 9-frame chain (paper_arm, analysed only) and the UR5e-class study arm
                (ur5e_class_arm, published DH + 0.15 m gripper); lumped masses, gravity, mass matrix, Jacobians
  ik.py         damped-least-squares IK (analytic or finite-difference Jacobian)
  safety.py     ISO/TS 15066 body model; elastic-joint impact model (secondary impacts);
                clamping model; operational-space projection
  contact.py    conservative lookup tables for the contact models
  geometry.py   exact capsule/box distances
  scene.py      work-cell model, operator capsules, 150-cell benchmark generator
  plan.py       JSON plan schema + parsing
  verifier.py   the digital twin: checks C0-C7, certified sampling, quantitative feedback
  planners.py   template plan (arc transit away from the operator), verifier-guided repair and speed/stiffness
                pass; modes: rigid (fixed stiff), soft (fixed soft), variable (scheduled); the rigid-joint
                robot baseline = mode rigid with VerifierConfig(rigid_K=3.1e4)
  llm/          prompts, API clients (OpenAI-compatible, Ollama, vLLM, Anthropic), runner, analysis
scripts/        exp_vssea.py (E1), exp_robot_contact.py (E2, E3), exp_planning.py (E5),
                exp_faults.py (E4), make_figures.py, export_numbers.py
tests/          unit tests against independent computations (pytest)
paper/          revised manuscript (IEEEtran)
```

## Reproduce
```bash
pip install numpy scipy matplotlib jsonschema pytest
python -c "from vsdt import contact; contact.build(); contact.build_clamp()"   # ~25 min, 2 cores
python scripts/exp_vssea.py
python scripts/exp_robot_contact.py
python scripts/exp_planning.py          # writes results/benchmark.json + e5_planning.jsonl
python scripts/exp_faults.py
python scripts/make_figures.py
python scripts/export_numbers.py        # -> paper/numbers.tex, paper/*_table.tex
pytest -q
```

## LLM experiments (E6), to be run with your own API key or a local model
```bash
# local model (Ollama, OpenAI-compatible endpoint)
python -m vsdt.llm.runner --provider ollama --model qwen2.5:32b-instruct
# hosted models
export OPENAI_API_KEY=...;    python -m vsdt.llm.runner --provider openai    --model gpt-4.1
export ANTHROPIC_API_KEY=...; python -m vsdt.llm.runner --provider anthropic --model claude-sonnet-4-5
# any OpenAI-compatible server
python -m vsdt.llm.runner --provider compatible --base-url http://host:8000/v1 --model <name>

python -m vsdt.llm.analyze results/llm_runs/*.jsonl   # -> results/llm_table.tex (copy to paper/)
```
Defaults: all 150 cells × 5 conditions × 3 seeds = 2250 episodes per model, up to 5 rounds each.
Use `--limit 10 --seeds 1` for a quick pilot. Runs are resumable: finished episodes are skipped.
`--provider mock` exercises the pipeline with a scripted test double; its output is excluded
from the analysis and must not be reported.

Model names above are examples. Use the exact model identifiers you run, and report them, their
versions and the run dates in the paper.
