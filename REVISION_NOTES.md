# Revision notes — what changed and why

These notes are for the author team. They list the problems found in the rejected manuscript, what the revision does about each one, and what you still need to do before submission.

## 1. Problems that most likely caused the rejection

| # | Problem in the rejected version | Severity |
|---|---|---|
| 1 | **The VSSEA mechanism and its numbers match Sariyildiz et al., IEEE/ASME T-Mech 2023 (doi:10.1109/TMECH.2022.3232471) without citation.** Matching items: the name "VSSEA", the M1 + VSAM + M2 architecture, the five-requirement list, ~100 Nm, π/2 rad/s, the 1:100 harmonic drive, 36 Nm/kg, the leaf springs with sliding rollers, "theoretically unbounded stiffness", and "to the best of our knowledge no existing VSA…". | Critical: reads as appropriation of published work |
| 2 | Fabricated or unverifiable references: "J. Doe" (b2); a journal that does not exist (b6, "IEEE Letters on Control Systems"); CLIP (b4) cited for LLM instruction following; wrong venues or pages (b3, b7, b8, b9); an irrelevant survey (b05) | Critical: desk-reject level |
| 3 | The two halves (actuator, LLM digital twin) were never connected. The Future Work section said so explicitly. | Major |
| 4 | "Dynamic model" section empty. Axial spring law (k = EA/L) applied to a bending leaf spring. Integral limits ∫_A^0. Claim of near-zero-to-infinite stiffness. | Major |
| 5 | The DH table (9 frames, all a_i = 0) is not a realizable collaborative arm. The elbow hangs below the work surface, and joint 2 needs up to 160 Nm with a 2.5 kg payload (actuators: 100 Nm). The text called the arm "6-DOF", "9-link" and "7 origins". | Major |
| 6 | Inconsistent results. "85 %" lead-time reduction where the table gives 96.7 %; "100 % safety" while the human baseline shows 0.5 %; "150 test cases" never described; no LLM model, prompts, trial counts, variance or statistics. | Major |
| 7 | Textbook material presented as results: stress–strain, necking and toughness figures, and a KUKA workspace comparison. Claims of "experimental validation" with no experiment. | Major |

## 2. What the revision does

- **New framing.** The paper is now about a *stiffness-aware digital-twin verifier* for LLM-generated plans. The VSSEA is modelled because the planner must choose stiffness, so the two halves are integrated. The mechanism is credited to Sariyildiz et al. and is **not** claimed as new.
- **Actuator model, verified.** It comprises:
  - Euler–Bernoulli and Timoshenko stiffness models;
  - an independent plane-stress FE model (agreement within 0.13 %);
  - a geometrically exact elastica, showing the linear model over-predicts torque by 13 % in the softest setting;
  - a bounded stiffness range of 21–3459 Nm/rad, including harmonic-drive compliance;
  - a stress-limited torque capacity as a function of stiffness;
  - modulation time and energy.
- **Arm.** The original chain is analysed and shown to be infeasible. The study arm uses the published UR5e DH parameters with VSSEA modules and a 0.15 m gripper.
- **Contact safety.** A new contact model maps joint stiffness to ISO/TS 15066 speed limits. It covers free impacts, including secondary impacts (up to +41 % force), and clamping. The model is validated against an ODE solver, and the lookup tables were checked for conservativeness.
- **Verifier.** It runs certified collision checks (conservative advancement) on the deflected arm and checks torque/stress capacity, precision, ISO contact limits for free and clamped contact, and task semantics. Every violation comes with a magnitude and a repair bound.
- **Experiments that actually ran** (E1–E5). E6, the LLM experiments, is designed and coded; you run it with your API key or a local model (see README).
- **Honest results, including the unflattering ones.**
  - Fixed soft stiffness fails every cell, because the springs cannot carry the load.
  - Verifier-guided planning solves all tier-1/2 cells and 27/45 feasible tier-3 cells, and refuses all 5 infeasible cells.
  - The elastic arm beats a rigid-joint arm only in the 7 cells where ISO contact limits bind (median 3 %, up to 31 % faster).
  - Stiffness *scheduling* shortened only 1 cycle, because at the published joint speed (π/2 rad/s) modulation costs about 1 s and the safety limits rarely bind.
  - The paper states all of this; reviewers trust honest negatives more than "100 %".
  - The 18 unsolved tier-3 cells (16 of them geometric: hands, arms, head keep-out) are exactly where E6 can show what an LLM with quantitative feedback adds.
- **References.** All have been rebuilt and checked against publisher, arXiv or university-repository pages during this revision (list in §4).
- **Claims made honest.** "100 % safety" is now stated as a property of the verification gate. Hardware claims are removed, since there is no prototype.

## 3. What you must do before submission

1. **Run E6 (the LLM experiments)** with at least one strong hosted model and one open model. Then run `python -m vsdt.llm.analyze` and copy `results/llm_table.tex` into `paper/`. Fill in the red `\todo` sentences in E6.
2. **Decide on authorship and wording around Sariyildiz et al.** If your group had any contact with that team, or the CAD was derived from their design, say so explicitly. Otherwise keep the current wording: the mechanism is "after [Sariyildiz 2023]".
3. **Replace the shim compile.** Compile `paper/main.tex` with the real IEEEtran class (e.g. on Overleaf). The `../figures/` paths assume the repository layout; flatten them if the journal wants a single folder.
4. **Check reference details.** The page numbers of Wolf et al. 2016 (2418–2430) were taken from memory, not a primary page; please confirm them. The 2023 Sariyildiz paper is cited by DOI only (volume and pages not verified).
5. **Add a code/data DOI** (GitHub + Zenodo) in the "Code and data availability" section.
6. **Strongly recommended for a Q1 venue:** a single-joint benchtop test of the leaf-spring module. It would cover the stiffness-vs-roller curve, step response and modulation time. Even one test turns "virtual prototype" into "validated model".

## 4. Reference verification log

Checked against primary or repository pages during this revision: ISO/TS 15066 Tables A.2/A.3 (values in `vsdt/safety.py`); Sariyildiz et al. T-Mech 2023 (DOI); Sariyildiz arXiv:2401.07430; Choi et al. T-RO 2011 (DOI); Kirschner et al. arXiv:2203.02706; Kambhampati et al. ICML 2024; Skreta et al. arXiv:2303.14100; AutoTAMP arXiv:2306.06531; Safety Chip arXiv:2309.09919; Text2Motion arXiv:2303.12153; Vemprala et al. arXiv:2306.17582; Deng et al. arXiv:2506.18178 (the TASE 2026 version in the old manuscript could not be confirmed, so the arXiv version is cited); AutoPLC arXiv:2412.02410; Antero et al. Robotics 13(9):137; Dong et al. JMSY 83:675–686; UR5e DH parameters (Universal Robots).
The remaining entries are classic papers whose details are standard (Pratt 1995, Hogan 1985, Vanderborght 2013, Haddadin 2009, Khatib 1987, Spong 1987, Wampler 1986, Schwarzer 2005, Cowper 1966, Tao 2019, Malik & Brem 2021, Wilson 1927, McNemar 1947, Holm 1979, etc.). A final check by you against Google Scholar is still advisable.

## 5. Suggested target journals

- *Robotics and Computer-Integrated Manufacturing* (Q1) fits LLM + digital twin + collaborative manufacturing best.
- *Journal of Manufacturing Systems* (Q1) is a similar fit.
- *IEEE Transactions on Automation Science and Engineering* (Q1) fits if the LLM results are strong.
- *IEEE/ASME Transactions on Mechatronics* would require the hardware test.
