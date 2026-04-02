# Research Proposal 1 — Robust Synthesis Planning as a Distributional, Set-Valued Decision Process

## One-sentence thesis

Do **not** treat retrosynthesis, condition prediction, yield prediction, and feasibility as four disconnected point-prediction tasks. Treat them as one **uncertain control-and-planning problem** over reaction hypergraphs, where the object to predict is not a single answer but a **robust admissible region** of reaction choices and operating conditions.

---

## Executive summary

The synthesis-planning literature has become technically sophisticated but is still structurally misaligned with the real problem chemists care about.

The community has separately optimized:
- **single-step retrosynthesis** (what precursors might make the product?),
- **condition recommendation** (what solvent/catalyst/reagent/temperature was recorded in similar reactions?),
- **yield prediction** (how much product might form?),
- **feasibility/robustness prediction** (will the reaction work, and how stable is it to perturbation?).

Each of those subfields has made real progress. Template-free retrosynthesis moved from encoder-decoder baselines to scaled pretraining and reaction foundation models. Condition prediction moved from simple context classification to multi-label and ranking-based systems. Yield prediction moved from narrow HTE-specific models toward transfer learning and uncertainty. Feasibility work has started to use Bayesian deep learning and active learning. But the fields are still organized around **paper-sized surrogate tasks**, not around the **decision object** a working chemist actually needs.

A chemist does not want:
- one best retrosynthetic disconnection,
- one exact condition tuple,
- one scalar yield prediction,
- one binary feasibility label.

A chemist wants something more like this:

> “Given a target, what are the route families whose **chance of laboratory success** is high, whose **condition windows** are broad enough to survive real-world perturbation, whose predicted performance remains acceptable under vendor/inventory/lab shifts, and whose uncertainty is small enough to justify execution?”

That is a **distributional**, **set-valued**, **route-level** object.

This proposal argues for a new research program:

1. Learn **reaction-level robustness regions** instead of point conditions.
2. Learn **yield and feasibility fields** over the condition space, not scalar outputs at one point.
3. Treat multistep synthesis as **risk-sensitive search** over reaction hypergraphs.
4. Use **distributionally robust optimization**, **viability theory**, **causal invariance**, **conformal uncertainty**, and **active boundary learning** to make the system scientifically useful.

The proposal is ambitious but grounded. It uses the strongest current ideas in the literature, then pivots away from the community’s default evaluation lens.

---

## Why this direction matters for Rasyn

Your internal strategy documents already contain the right instinct: routes without conditions are not useful; yield needs uncertainty; feasibility matters; analytical feedback should eventually close the loop. The missing step is to turn that product instinct into a **coherent scientific program**.

This proposal would give Rasyn a research direction that is:
- **high-value** for product,
- **publishable** as methodology,
- **non-trivial** rather than incremental,
- and **aligned with a lab-facing chemistry operating system** rather than a leaderboard-only model stack.

This can become the scientific core of an “Integrated Chemistry Environment”:
- route generation,
- route ranking,
- route de-risking,
- route adaptation under inventory/supplier constraints,
- and active experimentation to reduce uncertainty where it matters most.

---

## 1. Lay of the land: what the community has tried, and why

### 1.1 Retrosynthesis: from templates to sequence models to scaled pretraining

The field’s early modern wave focused on **template-based retrosynthesis**: classify the reaction template, apply it, and rank outputs. This made sense because chemistry transformations are structured and often repetitive. Templates encode human knowledge and improve validity. But the space of reaction templates is huge and brittle; rare chemistry and novel combinations are hard to cover.

That motivated **template-free sequence generation** and graph-based methods. The rationale was straightforward:
- if reaction templates are too rigid,
- and if product-to-reactant mapping resembles translation,
- then transformers and graph-edit models might generalize better.

Recent work pushed this direction further:
- **RSGPT** scaled retrosynthesis using synthetic pretraining on more than ten billion generated datapoints, plus RLAIF and augmentation.
- **ReactionT5** showed that pretraining directly on reaction data can transfer to retrosynthesis and related tasks.
- graph-based multitask systems and edit-based models tried to restore local chemical structure biases that pure token models can miss.

Why these approaches were tried:
- **transformers** because chemistry strings behave enough like sequences to benefit from language-model scaling,
- **graph models** because chemistry is fundamentally relational,
- **scaling/pretraining** because public curated reaction datasets are too small,
- **RL or reranking** because exact string accuracy underestimates route usefulness.

The problem is that retrosynthesis papers still usually optimize **single-step top-k accuracy**, which is only weakly linked to route executability.

Key literature anchors:
- RSGPT: large-scale pretraining and RLAIF for retrosynthesis.
- ReactionT5: reaction-foundation-model pretraining across tasks.
- Syntheseus: careful re-evaluation shows SOTA rankings change under better benchmarking.
- retro-fallback: planning under uncertainty instead of pretending reaction predictions are deterministic.
- DirectMultiStep: route-level sequence generation to bypass expensive tree search.

### 1.2 Multistep planning: search is not a solved wrapper around single-step models

Multistep planners such as AiZynthFinder popularized the pipeline:
1. predict single-step disconnections,
2. search a tree or AND/OR graph,
3. terminate when building blocks are found.

This was natural because single-step models were easier to train and inspect. But the field is increasingly discovering that **search quality**, **route scoring**, **uncertainty**, and **evaluation design** matter at least as much as one-step model accuracy.

Important recent insight:
- it is possible to get high “solvability” metrics while still generating poor routes,
- and different methods invert the tradeoff between finding *some* route and finding a *good* one.

That is why newer papers focus on:
- uncertainty-aware search (retro-fallback),
- route-level direct generation (DirectMultiStep),
- and benchmark reform (Syntheseus, RetroCast-style thinking).

Why the community tried these:
- MCTS and route search emerged because synthesis is combinatorial.
- Route generators emerged because iterative search scales badly.
- Uncertainty-aware planning emerged because lab reality is stochastic and literature coverage is incomplete.

### 1.3 Condition prediction: from exact context mimicry to multimodal recommendation

The classic modern condition paper is **Gao et al. (ACS Central Science, 2018)**, which framed reaction context recommendation as prediction of catalyst, solvent, reagent, and temperature using large Reaxys data. That paper mattered because it turned “conditions” into a scalable ML problem.

Its rationale was excellent for the time:
- conditions co-occur,
- context matters,
- and the community needed a first strong benchmark.

But there is a hidden assumption: the model is graded by how well it recovers the **recorded** context. That is not the same as predicting all chemically valid contexts.

More recent papers began to respond to this:
- **two-stage multi-label + ranking** systems separate candidate generation from ranking,
- **hard negative sampling** tries to teach the model not just what is present but what is implausible,
- **ORDerly** cleaned data and benchmarked the task more honestly.

Why those approaches were tried:
- because condition prediction is **multimodal**,
- because exact-label recovery is too harsh if many valid alternatives exist,
- and because messy reaction data silently inflates scores.

### 1.4 Yield prediction: the field learned the hard way that narrow benchmarks lie

Yield prediction got early excitement because it looks like the perfect “practical chemistry ML” problem. High-throughput experimentation (HTE) datasets allowed strong in-distribution performance. But the literature has increasingly shown that:
- many yield models memorize substrate-catalyst patterns,
- split strategy dominates the apparent difficulty,
- OOD generalization is poor,
- and noisy literature yields are not clean scientific targets.

This is why recent work moved in several directions:
- strong classical baselines on HTE,
- pretrained reaction encoders such as ReactionT5,
- multi-view or graph pretraining for OOD improvement,
- active learning and coreset methods such as **RS-Coreset**,
- and more explicit calls for calibrated uncertainty.

Why these were tried:
- HTE is one of the few settings where chemistry ML has dense, systematic labels.
- Pretraining was tried because raw HTE is too small for deep models.
- Active learning was tried because experiments, not GPUs, are the expensive resource.
- Uncertainty is increasingly unavoidable because yield is noisy, conditional, and lab-dependent.

### 1.5 Feasibility and robustness: the newest, most product-relevant frontier

Feasibility is subtly different from yield:
- yield asks “how much?”
- feasibility asks “does this work at all?”
- robustness asks “does it still work if the lab, substrate batch, or condition point shifts a little?”

The recent **Nature Communications 2025** work on global reaction feasibility and robustness prediction with Bayesian deep learning is important because it explicitly takes robustness seriously and uses HTE plus uncertainty-aware learning. This direction is close to what product systems actually need.

Why this was tried:
- the field has finally acknowledged that literature is heavily biased toward success cases,
- and that “best estimated condition” is not the same as “stable reaction setup.”

### 1.6 Core diagnosis of the current synthesis literature

Across these subfields, the literature is drifting toward the right ideas, but the field still mostly predicts **point objects** while the real chemistry decision problem depends on **regions**, **risk**, **adaptation**, and **information value**.

That gap is where the opportunity lies.

---

## 2. The surprising question this proposal asks

The non-obvious question is **not**:

> “How do we improve top-1 retrosynthesis, top-10 condition match, or R² on yield?”

The non-obvious question is:

> “What if the central object in synthesis planning is not a route or a condition tuple, but a **robust viability geometry** over route families and operating windows?”

More concretely:

- A reaction step should be represented by a **set-valued admissible region** in condition space.
- Yield should be modeled as a **field** over that region, not a number at one point.
- Feasibility should be modeled as a **probability surface** under latent lab/domain variation.
- Route planning should optimize **chance-constrained success**, **CVaR**, or other risk-sensitive objectives across the whole tree.

That is a different worldview.

The community usually asks:
- “Which exact condition tuple is best?”
- “Which route has the best heuristic score?”

This proposal asks:
- “Which route remains good under perturbation?”
- “Which transformations have wide safe operating windows?”
- “Which uncertainty should be reduced next, and by which experiment?”
- “Can robustness geometry be a better organizing principle than exact-match prediction?”

---

## 3. Core research hypothesis

### Primary hypothesis

A synthesis system that explicitly models **condition manifolds**, **yield/feasibility fields**, and **route-level risk** will outperform point-prediction pipelines on the metrics that actually matter for laboratory success:
- success under perturbation,
- robustness across labs,
- and usefulness to chemists making execution decisions.

### Secondary hypotheses

**H1 — Set-valued condition prediction beats point condition prediction.**  
Predicting a calibrated **region of viable contexts** will better capture chemistry’s multimodality than exact tuple prediction.

**H2 — Yield should be modeled as a distribution over condition space, not a scalar over reactions.**  
A reaction with a broad moderate-yield plateau may be preferable to a higher but unstable peak.

**H3 — Route ranking should optimize risk-sensitive objectives.**  
Chance-constrained and CVaR-based route selection will outperform greedy expected-yield ranking.

**H4 — Causal invariance across labs/reaction sources will improve OOD transfer.**  
If the model can separate transformation-intrinsic signals from source-specific biases, robustness will improve.

**H5 — Active experimentation near robustness boundaries is much more data-efficient than random HTE.**  
The most informative experiments are those that refine the boundary between stable and unstable operating regions.

---

## 4. Mathematical and scientific lenses that can change the architecture

This is where the proposal becomes more than “another chemistry transformer.”

### 4.1 Viability theory and set-valued analysis

**Why relevant:**  
A reaction step is not best described by one condition point. It is better described by the set of conditions under which it remains successful. That is a classic setting for **set-valued dynamics** and **viability kernels**.

**How to use it:**  
For a given transformation \( T \), let the condition vector be \( c \in \mathcal{C} \) and latent lab/domain state be \( z \). Define:
\[
p_{\text{succ}}(T, c, z)
\]
and a yield field
\[
y(T, c, z).
\]

Then define a robust viability set:
\[
\mathcal{V}_\alpha(T) = \{ c \in \mathcal{C} : \inf_{z \sim \mathcal{Z}} \Pr[\text{success} \mid T,c,z] \ge \alpha \}.
\]

This changes the task:
- from “predict the best tuple”
- to “estimate the robust admissible region.”

### 4.2 Distributionally robust optimization (DRO)

**Why relevant:**  
Patent data, ORD data, and HTE data come from different distributions. Labs, purification practices, and reporting biases vary. A point-estimate model often quietly overfits a source distribution.

**How to use it:**  
Train under distributional ambiguity sets:
\[
\min_\theta \sup_{Q \in \mathcal{U}(P)} \mathbb{E}_{Q}[\ell_\theta]
\]
where \( \mathcal{U}(P) \) is a Wasserstein or f-divergence ball around the empirical training distribution.

Use DRO for:
- route scoring under supplier/inventory/lab shifts,
- source-robust yield and feasibility models,
- selecting steps whose performance is stable under plausible domain perturbation.

### 4.3 Causal representation learning / invariant prediction

**Why relevant:**  
Reaction datasets contain spurious correlations:
- certain catalysts only appear with certain substrates because of literature habits,
- purification/reporting practices correlate with source,
- certain domains dominate because they are fashionable.

**How to use it:**  
Treat source datasets or labs as environments and encourage representations that preserve causal transformation signals across environments. This can use:
- invariant risk minimization,
- domain-adversarial training,
- or explicit causal graphs over substrate class, reaction family, source, conditions, and outcomes.

Key idea: learn what is **stable across labs** rather than what is merely frequent.

### 4.4 Hypergraph search + risk-sensitive dynamic programming

Single-step retrosynthesis naturally produces a hypergraph:
- one product node can map to sets of precursor nodes,
- success probabilities compound nonlinearly,
- route alternatives interact.

This suggests:
- AND/OR graph formulations,
- stochastic shortest path problems,
- CVaR-aware planning,
- worst-path or fallback-aware optimization.

Instead of route score = sum(costs) or product(expected yields), use:
\[
J(\pi) = \lambda_1 \mathbb{E}[\text{yield}] - \lambda_2 \text{cost} - \lambda_3 \text{uncertainty} - \lambda_4 \mathrm{CVaR}_{\beta}(\text{failure})
\]
with constraints on stock, time, safety, or green chemistry.

### 4.5 Conformal prediction and selective prediction

Chemists need the system to say:
- “I know,”
- “I am uncertain,”
- or “I abstain.”

Conformal methods can provide coverage guarantees for:
- feasible condition sets,
- yield intervals,
- feasibility risk bands,
- and route ranking uncertainty.

Selective prediction is extremely important here because forcing a model to answer under high uncertainty encourages unsafe recommendations.

### 4.6 Physics-informed regularization

This proposal is not advocating expensive quantum chemistry everywhere. But light-touch physical priors could help:
- reaction class and mechanistic family embeddings,
- approximate thermodynamic or reactivity descriptors,
- pKa and solvation priors,
- catalyst-ligand compatibility priors,
- scale-sensitive heat/stoichiometry descriptors for robustness.

The goal is not “replace data with physics,” but “use low-cost physical structure to regularize the geometry of the predicted fields.”

---

## 5. Proposed architecture

## 5.1 System overview

The system has five coupled components:

1. **Transformation generator**  
   Product \(\rightarrow\) candidate reaction hyperedges / retrosynthetic steps.

2. **Condition manifold model**  
   For each transformation, predict a calibrated set or density over viable conditions.

3. **Yield-feasibility field model**  
   Predict \( y(T, c) \) and \( p_{\text{succ}}(T, c) \) with uncertainty.

4. **Risk-sensitive route planner**  
   Search the synthesis hypergraph with chance constraints / fallback-aware scoring.

5. **Boundary-focused active learner**  
   Select the next experiments that most efficiently shrink uncertainty about robustness regions.

### 5.2 Component A: transformation generator

This can start from a strong existing one-step backbone:
- a scaled sequence model,
- graph-edit model,
- or reaction-foundation encoder.

But unlike typical pipelines, this component does **not** need to perfectly rank exact reactants on its own. Its job is to generate plausible transformation families with calibrated uncertainty and diversity.

Good starting backbones:
- retrosynthesis generator inspired by RSGPT or ReactionT5,
- plus graph-based reranking or synthons/edit structure.

### 5.3 Component B: condition manifold model

Instead of predicting one tuple:
\[
(\text{solvent},\text{catalyst},\text{reagent},T,t)
\]
predict either:

**Option 1 — energy-based set model**  
Learn an energy function
\[
E_\phi(T, c)
\]
whose low-energy region approximates the viable condition set.

**Option 2 — normalizing flow / diffusion over condition space**  
Model a density
\[
p_\phi(c \mid T)
\]
over mixed discrete-continuous conditions.

**Option 3 — latent manifold + decoder**  
Embed reactions into a latent condition manifold and decode multiple viable contexts.

Key difference from standard condition prediction:
- training must recognize that multiple conditions are valid even if only one was recorded,
- loss must reward calibrated candidate coverage rather than only exact-match mimicry.

Potential practical design:
- discrete heads for solvent/catalyst/reagent vocabularies,
- continuous heads for temperature/time/concentration,
- plus a compatibility energy between heads rather than independent prediction,
- plus a set-prediction or density-estimation objective.

### 5.4 Component C: yield and feasibility field model

Input:
- reaction transformation,
- proposed condition point,
- optional cheap physical descriptors,
- source/lab metadata when available.

Output:
- predicted yield distribution,
- predicted success probability,
- epistemic uncertainty,
- optional robustness gradient with respect to condition perturbation.

Recommended formulation:
\[
\hat{y} = f_\theta(T,c), \qquad \hat{p}_{\text{succ}} = g_\theta(T,c)
\]
with uncertainty from:
- deep ensembles,
- Bayesian last layers,
- evidential regression,
- or conformal wrapping on top of calibrated residual models.

Important: the model should also estimate **local sensitivity**:
\[
\nabla_c f_\theta(T,c), \quad \nabla_c g_\theta(T,c)
\]
or a Hessian-like stability measure, because a narrow sharp optimum is less attractive than a wide plateau.

### 5.5 Component D: robust route planner

The planner receives, for each step:
- candidate transformations,
- viable condition regions,
- yield/feasibility estimates with uncertainty,
- stock/vendor constraints,
- optional safety or sustainability penalties.

Then it solves a risk-sensitive planning problem:
- maximize probability at least one route works,
- or maximize expected downstream success subject to failure-risk bounds,
- or compute a Pareto frontier over cost, speed, yield, and robustness.

The route planner should support:
- **fallback-aware portfolios of routes**,
- **route families** rather than one winner,
- and **contingency planning** if a step fails or an input reagent is unavailable.

### 5.6 Component E: active robustness learning

Once a route/step becomes important, the system selects follow-up experiments not to maximize raw yield, but to **learn the shape of the viability region**.

That means querying points:
- near the estimated boundary of success/failure,
- where uncertainty about robustness is high,
- or where route ranking would flip depending on the answer.

This is much more aligned with planning than standard Bayesian optimization over a single reaction target.

---

## 6. What is novel here, beyond “combine the tasks”

This is the crucial point.

Many papers “combine tasks.” That is not enough. This proposal is novel because it changes the **scientific object** of prediction.

### Novelty 1 — Condition windows, not condition tuples
The model should output a **region** or posterior over viable conditions.

### Novelty 2 — Robustness geometry, not just outcome prediction
A route with a large robustness basin may be better than one with a slightly higher mean yield.

### Novelty 3 — Route selection under risk, not expected score
Planning should optimize robust success, not only top-k route convenience.

### Novelty 4 — Boundary learning, not indiscriminate active learning
The most informative experiments are near the viability boundary.

### Novelty 5 — Causal/stable chemistry features across environments
This explicitly targets the lab-transfer problem that most benchmarks hide.

---

## 7. Research questions

### RQ1
Can reaction condition prediction be reframed as **set-valued estimation** with calibrated coverage of viable condition regions?

### RQ2
Does modeling a **yield-feasibility field** over condition space improve route selection versus separate point estimators?

### RQ3
Can route ranking based on **robustness-aware criteria** outperform standard search heuristics on laboratory-relevant metrics?

### RQ4
Can causal invariance across patent/literature/HTE sources improve OOD robustness?

### RQ5
Can active learning focused on **robustness boundaries** reduce experimental burden relative to random or EI-style acquisition?

---

## 8. Concrete experimental program

## Phase I — Data curation and benchmark redesign

### 8.1 Build a unified reaction-outcome dataset

Combine:
- ORD / ORDerly,
- selected HTE datasets,
- literature/patent reaction data where condition parsing is reliable,
- optional in-house execution data if available.

Need standardized representations for:
- discrete conditions,
- continuous conditions,
- source/lab metadata,
- yield,
- success/failure labels,
- purification/reporting flags where possible.

### 8.2 Create a benchmark that matches the proposal

Current benchmarks are not enough. Define four task layers:

1. **Condition-set coverage**  
   Does the predicted region contain known successful conditions?

2. **Robustness estimation**  
   Can the model distinguish narrow optima from broad safe windows?

3. **Route-level execution success**  
   Does the planner choose routes that actually survive perturbation?

4. **Data-efficiency of robustness learning**  
   How many experiments are needed to estimate viability boundaries?

### 8.3 Split strategy

You should not rely on random splits alone.

Use:
- reaction-family split,
- substrate scaffold split,
- time split,
- source/lab split,
- and perturbation split.

If possible, include synthetic perturbation tests:
- slight temperature shifts,
- solvent swaps within class,
- supplier/inventory substitutions,
- noise in concentration or scale.

## Phase II — Condition manifold modeling

Train and compare:
- exact-match tuple classifiers,
- multi-label + ranking baselines,
- latent manifold models,
- energy-based models,
- mixed discrete-continuous diffusion or flow models.

Metrics:
- top-k exact tuple match,
- marginal component match,
- coverage of held-out successful conditions,
- calibration of predicted viable sets,
- robustness-window quality.

Key ablation:
- with vs without compatibility modeling,
- with vs without source-invariance regularization,
- with vs without retrieval from precedent literature.

## Phase III — Yield/feasibility field estimation

Compare:
- reaction-only yield prediction,
- reaction + point condition prediction,
- reaction + full condition manifold context,
- robust models vs standard ERM,
- causal-invariant vs non-invariant encoders.

Metrics:
- RMSE / MAE / rank correlation for yield,
- Brier / AUROC / AUPRC for feasibility,
- calibration error,
- OOD performance under scaffold/source splits,
- local stability estimation accuracy under perturbation.

Important evaluation:
- does the model assign higher utility to broad plateaus than narrow peaks when appropriate?

## Phase IV — Route-level robust planning

Baselines:
- standard single-step + MCTS,
- AiZynthFinder-like pipeline,
- expected-yield route ranking,
- retro-fallback-style uncertainty-aware planning,
- direct route models as route candidates followed by robust reranking.

Your method:
- chance-constrained hypergraph search,
- fallback-aware route portfolio optimization,
- route selection by robust score / CVaR / admissible-window mass.

Metrics:
- route success under perturbation,
- average fallback depth needed,
- cost-adjusted successful synthesis probability,
- uncertainty-aware abstention quality,
- route family diversity.

## Phase V — Boundary-focused active learning

Simulated or HTE environment:
- start with sparse data,
- allow the learner to choose the next experiments,
- compare against random, uncertainty sampling, expected improvement, and Thompson-like baselines.

Metrics:
- volume error of the estimated viability region,
- route-ranking regret,
- number of experiments to reach target confidence.

---

## 9. A concrete model stack

Below is one practical starting point.

### 9.1 Shared reaction encoder

A reaction encoder pretrained on:
- large reaction corpora,
- masked/replaced reaction token modeling,
- contrastive product-reactant objectives,
- optional graph-text or reaction-context alignment.

Could be transformer-based, graph-based, or hybrid.

### 9.2 Mixed discrete-continuous condition model

Represent conditions as:
- categorical tokens for solvent/catalyst/reagent/base/additives,
- continuous variables for temperature, time, concentration, equivalents.

Use:
- a latent variable model \( z_c \),
- decoders for discrete and continuous heads,
- pairwise or higher-order compatibility energy,
- and a coverage-aware loss.

### 9.3 Yield-feasibility head

Use:
- heteroscedastic regression for yield,
- binary feasibility with calibrated uncertainty,
- deep ensembles or Bayesian last layers,
- optionally a local Lipschitz or sensitivity penalty to stabilize the field.

### 9.4 Planner

Implement a robust route score such as:
\[
S(\text{route}) = \log P(\text{route works}) - \lambda_c C - \lambda_u U - \lambda_s S_{\text{safety}}
\]
where
- \(P(\text{route works})\) accounts for uncertainty and fallback structure,
- \(C\) is cost,
- \(U\) is epistemic uncertainty,
- \(S_{\text{safety}}\) is optional safety or process penalty.

Also consider a route portfolio objective:
\[
\max_{\mathcal{R}} \Pr(\exists r \in \mathcal{R}: r \text{ succeeds}) - \lambda |\mathcal{R}|
\]

---

## 10. Why this could work scientifically

The proposal works with the grain of current evidence:
- retrosynthesis scaling works, but route quality needs better planning;
- condition prediction is multimodal;
- yield models need OOD robustness and uncertainty;
- feasibility demands negative data and robustness reasoning.

The proposal’s bet is that these are not separate annoyances. They are all symptoms of the same deeper issue:

> the field predicts **point actions** in a world whose chemistry is governed by **robust operating regions** and **stochastic execution**.

That is why the proposal is not just a product convenience idea. It is a change in formalization.

---

## 11. Main risks and what negative results would still teach us

### Risk 1 — Robustness regions may be too hard to estimate from literature-only data
That would imply the need for targeted HTE or in-house closed-loop experimentation.

### Risk 2 — Set-valued condition prediction may be hard to evaluate fairly
That would motivate better benchmark design and maybe human evaluation or execution-based validation.

### Risk 3 — Route-level improvement may be bottlenecked by poor step generation
Then the result still teaches that route robustness depends more on upstream candidate diversity than on downstream scoring.

### Risk 4 — Causal invariance methods may not help if environments are poorly defined
Then source/lab metadata quality is the bottleneck, which is also valuable to know.

### Risk 5 — Physics-inspired priors may add engineering burden without enough gain
Then the cleanest result may still come from data-centric robust planning rather than stronger mechanistic regularization.

A high-quality negative result here would still be publishable because the field badly needs better alignment between benchmarks and chemical decisions.

---

## 12. What success would look like

### Short-term success
- A benchmark and model suite that demonstrates condition-set coverage and robust route ranking.
- A paper showing that point-prediction pipelines underperform on perturbation-aware route selection.
- A system that can produce route families with explicit robustness/uncertainty explanations.

### Medium-term success
- Integration into Rasyn route planning.
- Better route prioritization and fewer brittle suggestions.
- The ability to recommend not just a route but a **de-risking experiment plan**.

### Long-term success
- A synthesis planning system that behaves less like an autocomplete model and more like a **decision-support system for uncertain chemistry**.

---

## 13. Recommended first paper

### Working title
**Robust Synthesis Planning via Set-Valued Condition Modeling and Risk-Sensitive Route Search**

### Core claim
A planner that models viable condition regions and route-level risk chooses more executable routes than pipelines built from point retrosynthesis, point condition recommendation, and scalar yield prediction.

### Minimum viable contribution
- unified benchmark,
- condition-region model,
- robust planner,
- execution-simulated or HTE-backed validation.

---

## 14. Why this is a strong research bet

This direction is:
- **not trivial**,
- **not just scaling**,
- **not just combining tasks**,
- and **not locked to one architecture trend**.

It asks a more scientifically faithful question:
- not “what reaction should I predict?”,
- but “what action remains good when the world is noisy?”

That is the kind of reframing that can produce a real research identity.

---

## Selected literature and reference anchors

### Core synthesis planning
1. **RSGPT: a generative transformer model for retrosynthesis planning pre-trained on ten billion datapoints**. Nature Communications (2025).  
   https://pmc.ncbi.nlm.nih.gov/articles/PMC12314115/

2. **ReactionT5: a pre-trained transformer model for accurate chemical reaction prediction with limited data**. Journal of Cheminformatics (2025).  
   https://link.springer.com/article/10.1186/s13321-025-01075-4

3. **Re-evaluating Retrosynthesis Algorithms with Syntheseus**. OpenReview / DMLR at ICLR (2024).  
   https://openreview.net/forum?id=I9huj1zxRj

4. **Retro-fallback: retrosynthetic planning in an uncertain world**. arXiv (2024 version).  
   https://arxiv.org/abs/2310.09270

5. **Direct Route Generation for Multistep Retrosynthesis**. JCIM / arXiv (2024–2025).  
   https://pubs.acs.org/doi/10.1021/acs.jcim.4c01982  
   https://arxiv.org/abs/2405.13983

### Conditions, yield, robustness
6. **Using machine learning to predict suitable conditions for organic reactions**. ACS Central Science (2018).  
   https://pubmed.ncbi.nlm.nih.gov/30555898/

7. **Enhancing chemical synthesis: a two-stage deep neural network for predicting feasible reaction conditions**. Journal of Cheminformatics (2024).  
   https://link.springer.com/article/10.1186/s13321-024-00805-4

8. **ORDerly: Data Sets and Benchmarks for Chemical Reaction Data**. Journal of Chemical Information and Modeling / PMC version (2024).  
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11094788/

9. **Towards global reaction feasibility and robustness prediction with high throughput data and Bayesian deep learning**. Nature Communications (2025).  
   https://www.nature.com/articles/s41467-025-59812-0

10. **How should the machine learning community think about reaction yield prediction?** JCIM perspective / review literature around 2024.  
   https://pmc.ncbi.nlm.nih.gov/articles/PMC10778086/

11. **RS-Coreset** (active representation learning for reaction yield prediction). Communications Chemistry / Nature portfolio (2025).  
   https://www.nature.com/articles/s42004-025-01434-0

### Internal context
12. `RASYN_STRATEGY_AND_MODELS.md` — internal ranking of model priorities and product alignment.
13. `RASYN_ICE_TOOL_MASTER_PLAN.md` — integration context for how route planning fits the broader ICE.
