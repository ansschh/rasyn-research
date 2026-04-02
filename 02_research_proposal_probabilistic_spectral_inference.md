# Research Proposal 2 — Amortized Bayesian Spectral Inference Across NMR and MS/MS

## One-sentence thesis

Do **not** frame forward NMR, inverse NMR, and MS/MS structure elucidation as disconnected modality-specific tasks. Frame them as one **probabilistic inverse problem** over a latent molecular graph, geometry, and spectral state, where forward simulators and posterior inference are learned jointly and uncertainty guides the next measurement.

---

## Executive summary

The spectroscopy literature is improving quickly, but the field is still asking too small a question.

The standard tasks are:
- **forward NMR prediction**: structure \(\rightarrow\) shifts/spectra,
- **inverse NMR**: spectra \(\rightarrow\) structure,
- **MS/MS de novo generation**: fragmentation spectrum \(\rightarrow\) structure,
- and more recently **multimodal chemistry models** that attempt to connect spectra, molecules, and text.

Each of these subfields now has strong landmark papers:
- **NMRNet** helped standardize and modernize forward NMR prediction with an SE(3)-aware benchmarked framework.
- **NMRexp** dramatically expanded the experimental NMR data substrate.
- the **ACS Central Science 2024** multitask NMR structure elucidation paper showed that 1D routine spectra alone can support strikingly strong search-space collapse.
- **NMRTrans** argues that experimental NMR requires different modeling choices than simulated spectra and proposes set-based peak representations.
- **DreaMS** established the foundation-model paradigm for tandem mass spectra.
- **MSNovelist** and more recent generative systems showed the power of constrained generation from spectral evidence.
- **MassSpecGym** and related benchmark efforts are trying to standardize evaluation.
- multimodal models are starting to connect spectra, structures, and text.

These are real advances. But they still usually organize around a brittle pattern:

> one modality in, one canonical structure out.

That is not how scientific spectral reasoning works.

A spectroscopist does not actually want:
- “the model’s one best SMILES.”

They want:
- a calibrated posterior over plausible structures,
- an explanation of what each modality is ruling in or out,
- a forward model that checks whether the candidate is spectrally consistent,
- and ideally a recommendation for which **next measurement** would most reduce uncertainty.

This proposal argues for a new research direction:

1. Treat NMR and MS/MS structure elucidation as one **amortized Bayesian inverse problem**.
2. Represent spectra as **unordered or weakly ordered sparse point sets**, not as forced language sequences.
3. Use learned **forward simulators** (NMR and fragmentation) inside posterior inference.
4. Build a system that outputs a **posterior over molecules** and can choose the **next most informative measurement**.

This is not a trivial extension of existing work. It reframes the task from deterministic prediction to **scientific inference under partial observation**.

---

## Why this direction matters for Rasyn

Your internal strategy documents correctly identify NMR and MS/MS as the next high-value product frontier after synthesis planning. That is right. But the best long-term move is not to train one NMR model and one MS model independently. It is to build a **cross-modal structure inference engine** that can eventually support:
- product verification,
- impurity analysis,
- structure elucidation,
- route confirmation,
- and agent-driven experimental planning.

That becomes a central analytical layer in an Integrated Chemistry Environment:
- synthesis produces a candidate,
- spectra are collected,
- the system reasons over posterior structure hypotheses,
- and the result flows back into route confidence and documentation.

---

## 1. Lay of the land: what the community has tried, and why

### 1.1 Forward NMR prediction: from handcrafted descriptors to equivariant deep learning

Forward NMR prediction is one of the most natural machine learning problems in chemistry:
- the input is a molecular structure,
- the output is a set of local chemical shifts,
- and the physics is strongly local but not purely local.

Earlier methods used:
- handcrafted atom-centered features,
- message-passing networks,
- DFT-assisted datasets,
- and supervised regression on curated databases.

Why these were tried:
- NMR shifts are tied to atomic environments,
- the task is a good fit for local structural representations,
- and forward prediction is easier than inverse structure elucidation because it is a direct map from structure to signal.

Recent work such as **NMRNet** makes two important moves:
1. use **SE(3)-aware modeling** so the representation respects three-dimensional atomic environments,
2. build a more unified benchmark so the field can compare models fairly.

That is a substantial conceptual step. The literature increasingly recognizes that:
- 3D geometry matters,
- benchmark fragmentation has slowed the field,
- and data scale is critical.

### 1.2 Data scale changed the game: NMRexp and the shift to experimental corpora

A huge historical weakness in NMR ML has been over-reliance on:
- small experimental datasets,
- or large computed-shift corpora that do not fully capture real-world messiness.

**NMRexp** matters because it massively expands curated experimental NMR data and makes experimental-scale training more realistic. This changes what is scientifically possible:
- better forward models on real data,
- better calibration,
- and more realistic inverse-model evaluation.

Why the community moved here:
- simulated or DFT-derived data are useful but distributionally different from actual lab spectra,
- experimental metadata quality and scale were previously the main bottlenecks.

### 1.3 Inverse NMR: from search + rules to deep structure generation

Inverse NMR is much harder than forward prediction because the map from spectra to structure is:
- one-to-many,
- noisy,
- incomplete,
- and combinatorial.

Older systems often used:
- formula priors,
- fragment libraries,
- rule-based heuristics,
- database search,
- or strong preprocessing pipelines.

That made sense because the search space explodes combinatorially. But it also limited generality.

The **ACS Central Science 2024** paper is important because it used a multitask ML setup with minimal preprocessing and no molecular formula prior, showing that 1D NMR alone can drastically narrow the structure space.

Why that paper’s approach made sense:
- the inverse task is too hard to learn directly from raw spectra to full molecules without decomposing it,
- so the authors first learned a substructure/assembly problem and then used multitask learning to bridge to spectra.

More recent directions such as **NMRTrans** and other experimental-data-native models argue that the community’s abstraction of spectra as sequences is misguided. Peaks are not words. They are sparse, unordered or partially ordered observations with uncertainty and overlap.

Why that approach was tried:
- because simulated spectra are too clean,
- because experimental peak sets are more naturally treated as sets or point processes,
- and because positional encoding can inject false structure into the input.

### 1.4 MS/MS: from expert fragmentation trees to spectral foundation models

Mass spectrometry followed a similar arc:
- earlier methods focused on database lookup, fragmentation trees, and expert-engineered pipelines,
- then learned spectral predictors and retrieval systems improved,
- and now **DreaMS** has established the foundation-model direction by training self-supervised spectral representations at scale.

Why DreaMS was a natural next step:
- mass spectra repositories contain huge amounts of unlabeled data,
- self-supervision fits that regime,
- and spectral representation quality is a bottleneck for many downstream tasks.

At the same time, de novo structure generation remains hard. **MSNovelist** showed the power of an intermediate representation (fingerprints / formula constraints) rather than naive direct generation. More recent work such as **FlowMS** pushes generative modeling further.

Why the field tried intermediates:
- direct spectrum-to-SMILES is too unconstrained,
- fragmentation data are partial and noisy,
- so adding structural constraints makes the inverse problem more tractable.

### 1.5 Multimodal chemistry models: promising, but still too often “fusion first, inference later”

There is a growing trend toward:
- multimodal pretraining across spectra, structure, text, and images,
- cross-modal embedding spaces,
- large chemistry language models that can reason over mixed inputs.

This is directionally good. But many multimodal systems still stop at:
- joint embeddings,
- retrieval,
- or generic text-conditioned reasoning.

The deeper missing piece is a **principled posterior over molecular hypotheses** grounded by forward spectral consistency.

---

## 2. Core diagnosis of the current literature

The main structural issue is this:

> The field often treats spectral interpretation as **sequence transduction** or **retrieval**, when it is fundamentally an **inverse problem under uncertainty**.

That leads to four recurring blind spots.

### Blind spot 1 — The wrong output object
The community often predicts one best structure or top-k list. But the scientifically correct object is a **posterior over structures**:
\[
p(m \mid s_{\text{NMR}}, s_{\text{MS}}, \ldots)
\]

### Blind spot 2 — The wrong input abstraction
Spectra are often forced into token sequences, fixed bins, or images. But peaks behave more naturally like:
- sparse measures,
- unordered sets,
- or marked point processes with intensities, uncertainties, multiplicities, and modality-specific physics.

### Blind spot 3 — Forward and inverse are not trained together enough
A strong inverse system should be able to ask:
- “If this candidate structure were true, what spectrum would I expect?”
That is a forward-model question. The literature usually trains forward and inverse models separately.

### Blind spot 4 — The task ends too early
Real spectroscopic reasoning does not end at “here is a candidate.” It continues:
- which candidate is most plausible?
- what evidence distinguishes candidates?
- what next measurement would collapse the posterior most efficiently?

That last question is almost absent from the literature, yet it is one of the highest-value scientific questions.

---

## 3. The surprising question this proposal asks

The non-obvious question is **not**:

> “How do we get higher top-1 structure accuracy from NMR or MS/MS?”

The more interesting question is:

> “Can we build a **posterior inference engine** that reasons jointly across NMR and MS/MS, explains uncertainty, and recommends the **next measurement** that would maximally reduce structural ambiguity?”

That is a much deeper question.

This changes the task from:
- *prediction* to *inference*,
- *one-shot output* to *iterative scientific reasoning*,
- *modality-specific models* to *joint posterior structure estimation*.

---

## 4. Core research hypothesis

### Primary hypothesis

A model that treats NMR and MS/MS as **partial noisy views** of a latent molecular state, and that jointly trains **forward simulation** and **posterior inference**, will outperform modality-specific deterministic generators on:
- calibrated candidate coverage,
- robustness on experimental spectra,
- and real-world utility for structure elucidation workflows.

### Secondary hypotheses

**H1 — Spectra should be modeled as sparse measures / peak sets, not only sequences or images.**  
This will improve robustness on experimental spectra.

**H2 — Joint forward + inverse training is superior to independent training.**  
The inverse model should be constrained by what the forward simulators consider spectrally plausible.

**H3 — Retrieval + generation + probabilistic re-ranking is better than any single one alone.**  
Retrieval gives good priors, generation gives novelty, posterior reranking gives scientific coherence.

**H4 — Multimodal posterior inference beats modality-specific best guesses.**  
Combining NMR and MS/MS should shrink ambiguity far more than either modality in isolation.

**H5 — An agent that can choose the next measurement by posterior entropy reduction creates more value than a better one-shot predictor alone.**

---

## 5. Mathematical and scientific lenses that can change the architecture

### 5.1 Inverse problems and Bayesian inference

This proposal explicitly frames structure elucidation as:
\[
p(m \mid x_{\text{NMR}}, x_{\text{MS}})
\propto p(x_{\text{NMR}} \mid m)\, p(x_{\text{MS}} \mid m)\, p(m)
\]

Here:
- \(m\) is the latent molecule (graph, optional stereochemistry, optional conformer distribution),
- \(x_{\text{NMR}}\) is the NMR observation,
- \(x_{\text{MS}}\) is the tandem-MS observation,
- \(p(m)\) may come from chemical priors, retrieval databases, or generative priors.

This is conceptually obvious, but the field usually approximates it with separate predictors rather than actually building the posterior.

### 5.2 Marked point processes and sparse measure representations

Spectra are not natural language. Peaks are more like points in a sparse measure:
- location (chemical shift or m/z),
- intensity,
- multiplicity or fragmentation annotation,
- uncertainty / linewidth / collision-energy context,
- modality tag.

That suggests using:
- set transformers,
- point-cloud architectures,
- marked point process likelihoods,
- sparse optimal transport for alignment.

This is especially natural for experimental spectra where ordering is not fundamental.

### 5.3 Optimal transport for spectral alignment

Candidate evaluation often reduces to:
- “how close is the predicted spectrum of this candidate to the observed spectrum?”

Instead of naive bin-wise losses or peak matching heuristics, use **optimal transport** or unbalanced OT between sparse spectral measures. This can better handle:
- small shifts,
- missing/extra peaks,
- intensity mismatch,
- and modality-specific noise.

That gives a more physically meaningful similarity than exact bin overlap.

### 5.4 Equivariant geometry and conformer-aware forward simulation

For forward NMR, 3D geometry matters. For fragmentation, geometry and bond energetics matter indirectly. This suggests:
- equivariant graph networks,
- conformer ensembles,
- uncertainty from conformer populations,
- and amortized approximations to physically informed simulators.

### 5.5 Sequential Monte Carlo / particle-based posterior inference

A posterior over molecules is combinatorial. One promising direction is to use:
- retrieval to propose initial candidates,
- generative models to expand candidate space,
- particle filtering or SMC to update weights under multiple spectral modalities,
- and forward simulators for candidate scoring.

This turns structure elucidation into a search/inference loop rather than one-shot decoding.

### 5.6 Bayesian experimental design / value of information

This is where the proposal becomes unusually strong.

Suppose the posterior over structures is broad. Instead of forcing a guess, the system should ask:
- Should we collect \(^{13}C\) if we currently only have \(^{1}H\)?
- Should we acquire HSQC or HMBC?
- Should we change collision energy in MS/MS?
- Which next measurement maximizes expected information gain?

This is classical **Bayesian experimental design**:
\[
a^\star = \arg\max_a \mathbb{E}_{x' \sim p(x' \mid a, \text{current posterior})}
\left[ \mathrm{IG}(m; x' \mid a) \right]
\]

That is almost absent in current ML-for-spectroscopy work and could be extremely high impact.

---

## 6. Proposed architecture

## 6.1 System overview

The system has four main pieces:

1. **Forward spectral world models**
   - structure \(\rightarrow\) NMR,
   - structure \(\rightarrow\) MS/MS.

2. **Candidate prior / proposal mechanism**
   - retrieval from databases,
   - generative structure proposal,
   - chemical prior model.

3. **Posterior inference engine**
   - combines modalities,
   - reweights or resamples candidate structures,
   - produces calibrated posterior over candidates.

4. **Measurement policy**
   - recommends the next most informative spectrum/setting.

### 6.2 Component A: forward NMR model

Input:
- molecular graph,
- optional 3D conformer ensemble,
- modality type (\(^{1}H\), \(^{13}C\), maybe later 2D NMR),
- solvent or experimental metadata when available.

Output:
- per-atom shift distributions,
- simulated spectra or peak-set distributions,
- uncertainty due to conformer ambiguity or model uncertainty.

Potential architecture:
- SE(3)-equivariant graph encoder,
- conformer pooling,
- atom-wise shift heads,
- differentiable spectrum constructor.

### 6.3 Component B: forward MS/MS model

Input:
- molecular graph and optional adduct/precursor metadata,
- collision energy,
- ionization mode.

Output:
- spectral peak-set distribution,
- fragment propensity or latent fragmentation tree representation.

Potential architecture:
- graph encoder + energy-conditioned decoder,
- or spectral foundation model fine-tuned for conditional simulation,
- plus explicit fragment-consistency auxiliary losses.

### 6.4 Component C: inverse candidate proposal

This should be a hybrid of:
- **retrieval** from known compounds or substructures,
- **generative proposal** for novel structures,
- and **constraint-aware editing** around high-probability candidates.

Why hybrid?
- retrieval is strong when the answer is near-known chemistry,
- generation is required for novelty,
- editing is efficient when ambiguity is local.

Potential inputs:
- NMR peak sets,
- MS/MS peak sets,
- optional approximate formula if available,
- optional prior chemistry context from synthesis step or route plan.

### 6.5 Component D: posterior reranker

For each candidate \(m_i\), compute:
\[
\log p(m_i \mid x_{\text{NMR}}, x_{\text{MS}}) \approx
\log p(m_i) +
\log p(x_{\text{NMR}} \mid m_i) +
\log p(x_{\text{MS}} \mid m_i)
\]

Use:
- learned forward simulators,
- OT-based spectral consistency scores,
- chemistry priors,
- and optional retrieval confidence.

This produces:
- posterior probability,
- candidate uncertainty,
- evidence attribution by modality.

### 6.6 Component E: next-measurement policy

Given the current posterior and available measurement actions, estimate expected posterior entropy reduction or decision value. Candidate actions:
- acquire \(^{13}C\),
- acquire HSQC,
- rerun MS/MS at different energy,
- collect a different analytical view.

This policy could initially be trained in simulation using forward models, then evaluated on held-out multi-modal datasets.

---

## 7. What is novel here, beyond “multimodal fusion”

Again, this is the key distinction.

### Novelty 1 — Posterior over molecules, not one best output
That is a scientific shift, not just an architecture tweak.

### Novelty 2 — Forward and inverse are trained as one system
The candidate should not just “look plausible to the decoder”; it should survive forward spectral scrutiny.

### Novelty 3 — Spectra are treated as sparse physical observations
This opens the door to point-process models, OT, and better experimental robustness.

### Novelty 4 — The system can ask for the next measurement
This moves from passive prediction to active scientific reasoning.

### Novelty 5 — NMR and MS/MS are unified through latent molecular inference
Most current work remains modality-specific even when “multimodal.”

---

## 8. Research questions

### RQ1
Does representing spectra as peak sets / sparse measures outperform sequence or image encodings on experimental NMR and MS/MS?

### RQ2
Does a joint forward-inverse framework improve candidate calibration and exact-structure recovery?

### RQ3
Can joint NMR + MS/MS posterior inference substantially outperform separate modality-specific models on realistic experimental data?

### RQ4
How much value does a retrieval prior provide relative to pure generation under novelty constraints?

### RQ5
Can a next-measurement policy reduce the number of experiments needed for confident structure assignment?

---

## 9. Concrete experimental program

## Phase I — Build the multimodal data substrate

### 9.1 Data sources

NMR:
- NMRexp,
- nmrshiftdb2 / NMRNet benchmark assets,
- SDBS where licensing/use is permissible,
- experimental spectra associated with public datasets.

MS/MS:
- DreaMS-compatible corpora / repository spectra,
- GNPS/MassIVE where annotations exist,
- MassSpecGym benchmark tasks,
- NIST-like subsets if available through proper access channels.

Joint or pseudo-joint:
- molecules that have both NMR and MS/MS views,
- simulated paired data where experimental pairs are sparse,
- route-generated candidate molecules from retrosynthesis outputs.

### 9.2 Representation standardization

For each spectrum, store:
- peak location,
- intensity,
- uncertainty/linewidth if available,
- measurement context,
- modality tag,
- acquisition metadata.

This is essential. If you collapse everything into images too early, you lose the structure needed for a point-process view.

## Phase II — Forward spectral world models

Train:
- NMR forward predictor on experimental and computed data,
- MS/MS forward predictor on large-scale spectral data.

Evaluate not only MAE or cosine similarity, but also:
- calibration,
- robustness on experimental noise,
- OT-based spectral agreement,
- cross-dataset transfer.

## Phase III — Inverse candidate proposal

Compare:
- direct decoder from spectra,
- retrieval-only,
- generation-only,
- retrieval + editing,
- retrieval + generation + posterior reweighting.

Metrics:
- exact structure recovery,
- top-k coverage,
- posterior calibration,
- candidate diversity,
- recovery on out-of-database molecules.

## Phase IV — Joint posterior inference

Test:
- NMR only,
- MS/MS only,
- NMR + MS/MS,
- NMR + MS/MS + synthesis-context prior.

Metrics:
- posterior entropy,
- exact recovery,
- calibrated recall at top-k,
- evidence attribution quality,
- ambiguity reduction from combining modalities.

## Phase V — Next-measurement policy

Simulate multi-step analytical workflows:
- begin with only \(^{1}H\),
- then choose the next acquisition,
- compare information-gain policies against fixed policies or greedy heuristics.

Metrics:
- average number of measurements to identify the structure,
- posterior entropy reduction,
- success under cost constraints.

---

## 10. Example model stack

### 10.1 Spectral encoders

Use modality-specific encoders that operate on sparse peak sets:
- Set Transformer,
- point-cloud transformer,
- marked point process encoder,
- or sparse attention over peak tokens with modality-aware embeddings.

### 10.2 Molecular latent space

Represent candidates as:
- 2D graphs,
- optional 3D conformers,
- latent embeddings with uncertainty,
- and maybe a structured latent for substructures/fragments.

### 10.3 Candidate generator

Hybrid:
- retrieval from embedded spectral/molecular index,
- graph diffusion or constrained autoregressive decoder,
- local edit proposals around retrieved candidates.

### 10.4 Spectral consistency scorer

Use forward models plus OT-based discrepancy:
\[
\mathrm{score}(m) =
-\lambda_1 \, \mathrm{OT}(x_{\text{NMR}}, \hat{x}_{\text{NMR}}(m))
-\lambda_2 \, \mathrm{OT}(x_{\text{MS}}, \hat{x}_{\text{MS}}(m))
+\log p(m)
\]

### 10.5 Posterior calibrator

Use:
- temperature scaling,
- conformal wrappers,
- or Bayesian reweighting with holdout calibration.

The system should output:
- posterior mass over top candidates,
- confidence that the true structure is in the set,
- and recommended next actions if confidence is insufficient.

---

## 11. Why this could work scientifically

This proposal aligns with where the literature is already pointing:
- experimental-data-native modeling,
- larger data substrates,
- better benchmarks,
- multimodal structure reasoning,
- foundation-style spectral encoders.

But it adds a critical layer the field still lacks:
- a principled posterior,
- and an action policy for what to measure next.

That makes it scientifically richer and more operationally useful than one more structure decoder.

---

## 12. Main risks and what negative results would still teach us

### Risk 1 — True multimodal paired datasets may be limited
That would motivate semi-synthetic pair generation and weakly paired training strategies.

### Risk 2 — Forward simulation errors may dominate posterior scoring
That would reveal where the forward-model fidelity bottleneck lies and perhaps motivate better physics-informed components.

### Risk 3 — Retrieval may dominate generation and make novelty hard
That is still a useful result because it clarifies how much of practical elucidation is “known chemistry plus discrimination.”

### Risk 4 — Next-measurement policy may be difficult to evaluate without live experiments
Simulation and retrospective multi-spectrum datasets can still provide a meaningful first benchmark.

### Risk 5 — Posterior calibration may be challenging across modalities
Even that negative result is valuable; it would expose a major hidden weakness in existing spectral AI systems.

---

## 13. What success would look like

### Short-term success
- A strong forward-model suite for NMR and MS/MS on experimental data.
- A posterior reranking framework that clearly improves exact recovery and calibration.
- A benchmark showing the advantage of NMR+MS posterior inference over modality-isolated decoders.

### Medium-term success
- Integrated Rasyn analytical verification:
  - proposed product structure,
  - observed spectra,
  - calibrated posterior,
  - discrepancy explanation,
  - and recommendation for next measurement.

### Long-term success
- An “analytical world model” for chemistry that supports structure elucidation, QC, impurity reasoning, and autonomous measurement planning.

---

## 14. Recommended first paper

### Working title
**Multimodal Bayesian Structure Elucidation from NMR and Tandem Mass Spectra**

### Core claim
A posterior inference system that jointly uses learned NMR and MS/MS forward models plus sparse spectral encoders produces better-calibrated and more useful structure hypotheses than deterministic modality-specific baselines.

### Minimum viable contribution
- peak-set encoder,
- forward NMR + MS consistency scoring,
- candidate retrieval/generation hybrid,
- posterior calibration benchmark.

---

## 15. Why this is a strong research bet

This direction is strong because it is not a simple “bigger multimodal model” story. It is a **change in formalism**:
- from single best answers to posteriors,
- from token prediction to physical sparse observations,
- from passive prediction to active measurement design.

That is a research identity, not just an engineering roadmap.

---

## Selected literature and reference anchors

### NMR forward and data
1. **Toward a unified benchmark and framework for deep learning-based prediction of nuclear magnetic resonance chemical shifts (NMRNet)**. Nature Computational Science (2025).  
   https://www.nature.com/articles/s43588-025-00783-z

2. **NMRexp: A database of 3.3 million experimental NMR spectra**. Scientific Data (2025).  
   https://www.nature.com/articles/s41597-025-06245-5

### Inverse NMR
3. **Accurate and efficient structure elucidation from routine one-dimensional NMR spectra using multitask machine learning**. ACS Central Science (2024).  
   https://pubs.acs.org/doi/10.1021/acscentsci.4c01132  
   arXiv version: https://arxiv.org/abs/2408.08284

4. **NMRTrans: Structure Elucidation from Experimental NMR Spectra via Set Transformers**. arXiv (2026).  
   https://arxiv.org/abs/2602.10158

5. **NMR-Solver: Automated Structure Elucidation via Large-Scale Spectral Matching and Physics-Guided Fragment Optimization**. arXiv (2025).  
   https://arxiv.org/abs/2509.00640

### MS/MS representations and de novo generation
6. **Self-supervised learning of molecular representations from millions of tandem mass spectra using DreaMS**. Nature Biotechnology (2025).  
   https://www.nature.com/articles/s41587-025-02663-3

7. **MSNovelist: de novo structure generation from mass spectra**. Nature Methods (2022).  
   https://www.nature.com/articles/s41592-022-01486-3

8. **MassSpecGym: A benchmark for the discovery and identification of molecules**. arXiv (2024).  
   https://arxiv.org/abs/2410.23326

9. **FlowMS: Flow Matching for De Novo Structure Elucidation from Mass Spectra**. arXiv (2026).  
   https://arxiv.org/abs/2603.18397

### Multimodal / benchmark context
10. **A multimodal spectroscopic dataset for molecular structure elucidation**. NeurIPS Datasets and Benchmarks (2024).  
   (Use the final dataset release / benchmark page in implementation.)

11. Recent multimodal chemistry and spectroscopy models such as **ChemDFM-X**, **SpectraLLM**, **SpectrumWorld/SpectrumLab**, and related benchmark papers should be treated as adjacent context rather than the main conceptual anchor.

### Internal context
12. `RASYN_STRATEGY_AND_MODELS.md` — internal rationale for prioritizing analytical AI after synthesis.
13. `04_CHEMINFORMATICS_GREEN_PROCESS.md` and `03_PROTOCOLS_QC_ELN_COMPUTATIONAL.md` — integration context for analytical workflows and QC.
