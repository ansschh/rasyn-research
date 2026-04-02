# Research Proposal 3 — A Typed Probabilistic Chemistry Operating System for Multi-Chemistry Models and Agentic Systems

## One-sentence thesis

Do **not** build chemistry agents whose real state lives only in prompts, hidden chain-of-thought, or ad hoc tool calls. Build a **typed, probabilistic chemistry operating system** in which molecules, reactions, spectra, inventory, procedures, hypotheses, and tool outputs live in an explicit belief graph with provenance, verification, and safe execution semantics.

---

## Executive summary

Agentic chemistry systems are moving fast:
- tool-augmented LLMs,
- chemistry-specialized language models,
- multi-agent orchestration,
- autonomous experiment loops,
- cross-modal chemistry models.

This is exciting, and several recent systems are genuinely important:
- **ChemCrow** showed that tool-augmented LLMs can do nontrivial chemistry work more reliably than raw prompting.
- **Coscientist** demonstrated multi-agent autonomous scientific workflows.
- **LLM-RDF** pushed toward end-to-end reaction development with specialized agents.
- **ChemGraph** showed that agentic decomposition can work well in computational chemistry workflows.
- chemistry-specialized LLMs such as **ChemDFM** and more recent reasoner/multimodal variants showed that domain adaptation matters.
- the broader review literature now takes autonomous agents in chemistry seriously as a new systems layer.

But almost all of these systems still inherit a deep architectural weakness from general-purpose agentic AI:

> the real scientific state of the world is still mostly **implicit**.

It lives in:
- prompt text,
- scratchpads,
- tool outputs pasted into context windows,
- brittle memories,
- and informal human interpretation.

That is not enough for chemistry.

Chemistry is not only about “reasoning better.” It is about maintaining and transforming a highly structured, partially observed scientific state:
- molecular candidates,
- route hypotheses,
- stock availability,
- condition constraints,
- instrument outputs,
- analytical discrepancies,
- uncertainty,
- and provenance.

A chemistry agent that lacks an explicit state model will eventually fail in predictable ways:
- inconsistent assumptions,
- hallucinated tool state,
- missing provenance,
- inability to verify whether a recommendation remains supported,
- fragile long-horizon planning,
- and unsafe execution behavior.

This proposal argues for a different direction:

1. Build a **typed probabilistic belief graph** as the core state of the chemistry system.
2. Treat tools as typed operators with preconditions, postconditions, and uncertainty effects.
3. Use narrow chemistry predictors and verifiers as first-class components of the agent loop.
4. Let the language model be a **planner/proposer/interpreter**, not the sole holder of truth.
5. Evaluate the system on **workflow reliability, provenance completeness, uncertainty management, and reproducibility**, not only task completion.

This is not “another multi-agent system.” It is a research program about how to architect a chemistry-native operating system.

---

## Why this direction matters for Rasyn

Your internal documents already point toward an always-open agentic chemistry workspace inspired partly by Biomni but extended into chemistry. That is directionally correct. But if Rasyn stops at:
- an LLM plus tool calls plus some memory,
it will hit the same ceiling as many current agent systems.

The real opportunity is larger:
- a stateful chemistry environment,
- persistent working context,
- tool retrieval,
- safe execution,
- typed dataflow,
- analytical verification,
- provenance,
- and explicit uncertainty.

This proposal can become the systems-research spine of the entire product.

---

## 1. Lay of the land: what the community has tried, and why

### 1.1 Tool-augmented LLMs: strong first step, necessary but not sufficient

**ChemCrow** is one of the landmark papers because it made a simple but crucial point: chemistry gets much better when the LLM is given the right tools. That system uses multiple chemistry tools and shows that tool access mitigates hallucination and broadens capability.

Why the community went here:
- general LLMs lack accurate chemistry-specific calculators and databases,
- tool use reduces the burden on the base model,
- and many chemistry tasks are naturally decomposable into search, lookup, simulation, and reasoning.

This was the right move. But the resulting systems are often still organized as:
- question,
- reason in text,
- call tool,
- summarize result.

That can work for short tasks. It does not create a robust long-horizon scientific state.

### 1.2 Multi-agent decomposition: divide by role, improve specialization

Systems like **Coscientist**, **LLM-RDF**, and **ChemGraph** increasingly decompose workflows into roles:
- literature scout,
- planner,
- executor,
- analyzer,
- interpreter.

Why the community tried this:
- one large agent is hard to steer,
- specialization improves tool use,
- and explicit division of labor resembles human scientific teams.

Again, this was a sensible move. But multi-agent decomposition alone does not solve:
- state consistency,
- provenance,
- uncertainty tracking,
- or tool-side semantics.

You can have many agents and still have the wrong architecture if they all share state only through text.

### 1.3 Chemistry-specialized base models: domain adaptation matters

Chemistry LLMs such as **ChemDFM** and related models were trained on large chemistry corpora and instruction data. The rationale was straightforward:
- chemistry is a dense technical domain,
- general LLMs miss terminology, concepts, and common reasoning patterns,
- and domain-specific pretraining materially helps.

This is real progress. But domain-specific language fluency is not the same as a chemistry operating system.

### 1.4 Multimodal and cross-modal chemistry models

Some recent systems extend chemistry models to handle:
- molecular structures,
- spectra,
- text,
- images,
- and maybe reaction data.

That is important because scientific work is inherently multimodal. But again, the dominant pattern remains:
- encode everything,
- put it into a model or a prompt,
- ask for an answer.

The hidden system state is still under-modeled.

### 1.5 Biomni-style persistent scientific workspaces

Your internal Biomni documents highlight several important systems ideas:
- persistent execution context,
- tool retrieval,
- a kind of live computational workspace,
- and agentic orchestration over scientific tasks.

That direction is highly relevant. But your own internal analysis correctly notes weaknesses that should not be copied naively:
- unsafe raw execution,
- weak isolation,
- implicit session/thread assumptions,
- and architecture choices that do not yet provide chemistry-native typed state and provenance.

That is exactly the opening for a better design.

---

## 2. Core diagnosis of the current agentic chemistry literature

The field is currently over-focused on:
- **bigger models**,
- **more tools**,
- **more agents**.

Those help, but they do not address the deeper systems problem.

### Diagnosis 1 — Scientific state is implicit rather than explicit
A chemistry workflow has entities and relations:
- sample A,
- candidate structure B,
- route C,
- spectrum D,
- inventory record E,
- uncertainty F,
- provenance edge G.

If those do not exist as first-class typed objects, the agent has no stable substrate on which to reason.

### Diagnosis 2 — Tools are treated as text-producing oracles
Many agent systems call tools and receive strings or blobs, then let the LLM interpret the results. But chemistry tools have semantics:
- preconditions,
- output schemas,
- units,
- side effects,
- confidence,
- and relationships to the current scientific state.

Ignoring that creates failure modes.

### Diagnosis 3 — Verification is too weak
Chemistry workflows need structured checks:
- is the proposed reagent available?
- does the predicted route violate a safety constraint?
- is the proposed structure consistent with the observed spectra?
- did this step consume inventory and create a new sample artifact?
- can the result be reproduced from recorded provenance?

Many current agent loops do not encode those checks deeply enough.

### Diagnosis 4 — Long-horizon memory is not scientific memory
Persisting chat context is not the same as maintaining a stateful, queryable laboratory memory. Scientific memory should support:
- versioning,
- provenance,
- uncertainty updates,
- contradiction detection,
- and object-level retrieval.

### Diagnosis 5 — Benchmarks are too task-centric
Most evaluations ask whether the agent got the answer or completed a short task. For chemistry systems, we also care about:
- provenance completeness,
- state consistency,
- tool misuse rate,
- uncertainty calibration,
- reproducibility,
- and cost-aware scientific utility.

---

## 3. The surprising question this proposal asks

The non-obvious question is **not**:

> “How do we make a chemistry agent more capable with more tools and bigger models?”

The more interesting question is:

> “Can we design a chemistry agent more like a **typed operating system or proof assistant for scientific state**, where the model proposes actions but a structured belief graph, verifiers, and tool semantics determine what is actually believed and executed?”

That is a different architectural philosophy.

It implies:
- the model is no longer the sole memory,
- tool outputs are not just strings,
- and the system’s state transition semantics matter as much as the planner.

---

## 4. Core research hypothesis

### Primary hypothesis

A chemistry agent built around a **typed probabilistic belief graph**, plus tool contracts and verifiers, will outperform prompt-centric and multi-agent-text-only systems on:
- long-horizon workflow reliability,
- provenance and reproducibility,
- hallucination resistance,
- and scientific usefulness in real chemistry tasks.

### Secondary hypotheses

**H1 — Explicit typed state matters more than adding another generic agent role.**  
A smaller model with better state semantics may beat a larger prompt-only system.

**H2 — Tool contracts and verifiers dramatically reduce chemistry-specific hallucinations.**  
If each tool call updates a typed state with schema-checked edges and uncertainty, many failure modes become detectable.

**H3 — Persistent scientific memory should be object-centric, not transcript-centric.**  
This will improve long-horizon tasks and cross-session continuity.

**H4 — Multi-chemistry models become more useful when attached to an explicit state substrate.**  
Foundation models alone do not solve orchestration.

**H5 — Workflow evaluation should target state integrity and reproducibility, not just end answers.**

---

## 5. Mathematical and systems lenses that can change the architecture

### 5.1 Type theory and operational semantics

Chemistry entities naturally form types:
- Molecule,
- ReactionStep,
- Route,
- Sample,
- Spectrum,
- InventoryItem,
- Procedure,
- Hypothesis,
- SafetyConstraint,
- VendorQuote,
- AnalyticalObservation.

Each tool can then be specified as a typed operator:
\[
\texttt{plan\_retro}: \texttt{Molecule} \rightarrow \texttt{RouteSet}
\]
\[
\texttt{predict\_nmr}: \texttt{Molecule} \rightarrow \texttt{SpectrumDistribution}
\]
\[
\texttt{parse\_lcms}: \texttt{RawInstrumentFile} \rightarrow \texttt{AnalyticalObservation}
\]

This lets the system enforce:
- preconditions,
- postconditions,
- units,
- and state updates.

This is not just software hygiene. It is a way to stop the LLM from inventing impossible transitions.

### 5.2 Probabilistic graphical models / belief states

The chemistry workspace should maintain beliefs, not just facts. Many objects are uncertain:
- route quality,
- candidate identity,
- yield estimates,
- sample assignment,
- spectral interpretation.

So the state is not just a graph. It is a **belief graph**:
- typed nodes,
- typed edges,
- probabilities or confidence intervals,
- provenance metadata,
- timestamps and versioning.

This gives the agent a principled substrate for:
- updating beliefs,
- resolving contradictions,
- and planning under uncertainty.

### 5.3 Graph rewriting and category-theoretic thinking

Chemistry workflows are state transformations:
- a reaction consumes reagents and creates a sample,
- analysis creates evidence edges from a sample to a candidate structure,
- route planning creates hypothesis edges,
- inventory updates change resource availability.

That can be fruitfully viewed through:
- graph rewriting systems,
- compositional workflows,
- provenance-preserving morphisms,
- or category-inspired typed process composition.

The value is not philosophical elegance. It is that composition rules become explicit and machine-checkable.

### 5.4 Provenance semirings / dependency tracking

Scientific systems need to know:
- what evidence supports this belief?
- which tool outputs contributed?
- if one upstream result is invalidated, what downstream beliefs become suspect?

Provenance semiring ideas from databases and dataflow systems are relevant here. They allow compact propagation of support and dependency information.

### 5.5 POMDPs and Bayesian experimental design

An agentic chemistry workflow is a partially observable decision process:
- the true reaction state is partly hidden,
- structure identity is uncertain,
- the agent chooses experiments to reduce uncertainty and achieve goals.

This suggests planning formulations from:
- POMDPs,
- belief-space planning,
- and Bayesian experimental design.

Crucially, the belief state should live in the typed graph, not only in hidden model activations.

### 5.6 Safe execution and sandboxed computation

Your internal Biomni analysis already notes why unsafe raw execution is a problem. A chemistry OS should support:
- persistent workspace state,
- but with sandboxing,
- resource control,
- reproducible environments,
- and tool isolation.

That is not incidental. It is part of scientific trust.

---

## 6. Proposed architecture

## 6.1 Core objects

Define a typed schema for at least:

- **MoleculeCandidate**
- **ReactionTransformation**
- **RouteHypothesis**
- **ConditionRegion**
- **YieldEstimate**
- **FeasibilityEstimate**
- **Sample**
- **Spectrum**
- **AnalyticalEvidence**
- **InventoryItem**
- **VendorOffer**
- **ProcedureDraft**
- **ExecutionEvent**
- **ScientificClaim**
- **ProvenanceRecord**

Each object should carry:
- canonical ID,
- version,
- provenance links,
- uncertainty/confidence,
- source modality/tool,
- timestamps.

## 6.2 Belief graph

The system state is a graph such as:

- `RouteHypothesis` **supports** `MoleculeCandidate`
- `Spectrum` **observes** `Sample`
- `AnalyticalEvidence` **supports/refutes** `MoleculeCandidate`
- `InventoryItem` **enables** `RouteHypothesis`
- `ExecutionEvent` **creates** `Sample`
- `ProcedureDraft` **instantiates** `ReactionTransformation`

The graph must allow:
- multiple competing hypotheses,
- confidence updates,
- contradiction marking,
- branch-and-merge workflow histories.

## 6.3 Tool contracts

Each tool is registered with:
- typed inputs,
- typed outputs,
- preconditions,
- postconditions,
- confidence semantics,
- side effects on the belief graph.

Example:

```text
tool: predict_route_portfolio
input: MoleculeCandidate
output: RouteHypothesis[]
preconditions:
  - structure valid
postconditions:
  - add route hypotheses
  - attach route scores and uncertainty
  - no inventory side effects
```

```text
tool: import_nmr_spectrum
input: RawInstrumentFile, Sample
output: Spectrum, AnalyticalEvidence
preconditions:
  - sample exists
postconditions:
  - attach spectrum to sample
  - create evidence node
```

## 6.4 Role of the LLM / multi-chemistry model

The LLM or multimodal foundation model should do things like:
- propose next actions,
- interpret user intent,
- summarize graph state,
- draft procedures,
- decide which tools to call,
- generate hypotheses,
- explain contradictions.

It should **not** be the sole authority on state truth.

This is the heart of the proposal:
- the model is a controller and interpreter,
- the graph plus verifiers are the memory and truth substrate.

## 6.5 Verifier layer

Every important claim should be checkable by one or more verifiers:
- structure-spectra consistency verifier,
- route feasibility verifier,
- inventory feasibility verifier,
- unit and schema validator,
- provenance completeness checker,
- safety and compliance checker.

The belief graph should distinguish:
- proposed,
- verified,
- contradicted,
- uncertain,
- deprecated.

## 6.6 Persistent workspace and safe execution

A session should have:
- a persistent working memory,
- queryable graph state,
- notebook/code execution in a sandbox,
- retrieval over prior objects and evidence,
- and explicit branching/version control.

This is “Biomni-like” in spirit but chemistry-native and safer.

---

## 7. What is novel here, beyond “another chemistry agent”

### Novelty 1 — State is first-class and typed
Most current systems are still transcript-first. This proposal is state-first.

### Novelty 2 — Beliefs and uncertainty are explicit
Not all scientific claims are equal. The system should know what is uncertain and why.

### Novelty 3 — Tools have semantics, not just APIs
Preconditions/postconditions and side effects become part of the research design.

### Novelty 4 — Verifiers are first-class citizens
Narrow chemistry models are not side modules; they are structural checks in the agent loop.

### Novelty 5 — Evaluation targets scientific integrity
This is a systems science proposal, not just a benchmark-chasing one.

---

## 8. Research questions

### RQ1
Does explicit typed state improve long-horizon workflow reliability more than adding more agents or larger context windows?

### RQ2
How much do tool contracts and verifier layers reduce hallucination, tool misuse, and unsupported claims?

### RQ3
What is the right granularity of chemistry objects in the belief graph?

### RQ4
Can a belief-graph chemistry OS support cross-session continuity and branch-aware scientific workflows better than transcript memory?

### RQ5
How should we benchmark scientific agents when reproducibility and provenance matter as much as end-task completion?

---

## 9. Concrete experimental program

## Phase I — Design the chemistry state schema

Define:
- object ontology,
- edge ontology,
- uncertainty representation,
- provenance representation,
- versioning model.

This needs careful design work with real workflow traces.

Outputs:
- a typed schema,
- serialization rules,
- graph query primitives,
- and a set of canonical transformations.

## Phase II — Build a baseline chemistry OS runtime

Minimum components:
- belief graph store,
- tool registry with contracts,
- verifier layer,
- planner/controller interface,
- sandboxed code execution,
- persistent workspace memory.

Baseline agents to compare:
1. prompt-only LLM with tools,
2. multi-agent text-based orchestration,
3. your typed-belief-graph system.

## Phase III — Integrate narrow chemistry models as verifiers

Attach:
- route planning models,
- condition/yield/feasibility models,
- NMR/MS consistency models,
- inventory/vendor modules,
- safety/compliance modules.

Each updates or checks the graph rather than merely returning text blobs.

## Phase IV — Workflow benchmarks

Build end-to-end tasks such as:
- propose and de-risk a synthesis route,
- verify a synthesized product against spectra,
- choose reagents under inventory constraints,
- draft a procedure and update ELN objects,
- diagnose contradictions between route hypothesis and analytical evidence.

For each task, measure:
- task completion,
- state consistency,
- provenance completeness,
- unsupported-claim rate,
- tool misuse rate,
- uncertainty calibration,
- reproducibility from recorded state.

## Phase V — Long-horizon continuity and branching

Evaluate:
- can the system resume work across sessions?
- can it maintain competing hypotheses?
- can it branch, compare branches, and merge validated results?
- can it explain why a previously believed claim is now downgraded?

This is where transcript-centric agents typically fail.

---

## 10. Example benchmark ideas

### 10.1 Unsupported-claim benchmark
Give the system partial evidence and measure whether it makes claims not justified by the current graph.

### 10.2 Provenance recovery benchmark
Ask the system to reconstruct why a conclusion was reached and which objects support it.

### 10.3 Contradiction handling benchmark
Inject conflicting spectra, vendor information, or route evidence and see whether the system tracks contradiction rather than overwriting silently.

### 10.4 Tool misuse benchmark
Measure invalid tool calls, schema violations, and impossible state transitions.

### 10.5 Workflow replay benchmark
Given a saved belief graph and action trace, can the system reproduce the same decision path and outcome?

---

## 11. Example architecture stack

### 11.1 Planner
A model that maps user intent + graph snapshot \(\rightarrow\) proposed action sequence.

### 11.2 State manager
Responsible for:
- applying typed transitions,
- versioning,
- provenance propagation,
- confidence updates,
- contradiction detection.

### 11.3 Verifier ensemble
A set of task-specific checkers that approve, downgrade, or refute claims.

### 11.4 Retriever
Queries:
- graph state,
- know-how documents,
- literature,
- vendor/inventory sources,
- historical workflows.

### 11.5 Executor
Runs tool calls or sandboxed code, then returns typed artifacts.

### 11.6 Explainer
Turns the state and provenance into human-readable rationale.

---

## 12. Why this could work scientifically

This proposal is not speculative fluff because it lines up with what the literature already hints:
- tool use helps,
- specialized agents help,
- chemistry-specific models help,
- persistent workspaces help,
- multimodal models help.

The proposal’s claim is simply that these advances will plateau unless the system has a stronger scientific state model.

In other words:

> the next scaling law for chemistry agents may be **state quality and verifier quality**, not only model size.

That is a genuinely interesting systems hypothesis.

---

## 13. Main risks and what negative results would still teach us

### Risk 1 — Schema design may become too rigid
If so, the result still teaches what minimal state formalism is required and where flexibility is necessary.

### Risk 2 — Graph/state overhead may slow simple tasks
That is acceptable if the payoff appears on long-horizon or high-stakes workflows.

### Risk 3 — Users may not want to think in explicit objects
Then the UI must hide the graph while preserving the underlying state semantics.

### Risk 4 — Verifier disagreement may be hard to reconcile
That is still scientifically useful, because disagreement should surface rather than remain hidden.

### Risk 5 — Building the benchmark is expensive
True, but benchmark creation itself would be a real research contribution because current agent benchmarks miss the most important chemistry-system properties.

---

## 14. What success would look like

### Short-term success
- A typed chemistry state schema and runtime.
- A benchmark showing that graph-based state improves reliability over transcript memory.
- A paper demonstrating fewer unsupported claims and better provenance under complex workflows.

### Medium-term success
- A Rasyn workspace in which:
  - route planning,
  - spectra interpretation,
  - inventory,
  - documentation,
  - and execution traces
  all live in one persistent belief graph.

### Long-term success
- A chemistry operating system that supports humans and agents working together over weeks or months, not only minutes.

---

## 15. Recommended first paper

### Working title
**A Typed Probabilistic State Model for Reliable Agentic Chemistry Workflows**

### Core claim
Representing chemistry workflows as typed belief-graph state transitions with verifier-backed tool semantics yields more reliable and reproducible agent behavior than transcript-centric tool-using LLM baselines.

### Minimum viable contribution
- state ontology,
- runtime,
- benchmark,
- head-to-head evaluation versus prompt-centric agent baselines.

---

## 16. Why this is a strong research bet

This direction is compelling because it is:
- not another generic LLM benchmark paper,
- not another “more agents!” paper,
- and not locked to a specific foundation model.

It is a systems research identity:
- chemistry as typed state,
- tools as state transitions,
- science as uncertainty-bearing evidence accumulation.

That is exactly the kind of architecture that can make a product like Rasyn feel fundamentally different from “chat plus plugins.”

---

## Selected literature and reference anchors

### Agentic chemistry and tool use
1. **Augmenting large language models with chemistry tools (ChemCrow)**. Nature Machine Intelligence (2024).  
   https://www.nature.com/articles/s42256-024-00832-8

2. **Autonomous chemical research with large language models (Coscientist)**. Nature (2023).  
   https://www.nature.com/articles/s41586-023-06792-0

3. **An automatic end-to-end chemical synthesis development platform powered by large language models (LLM-RDF)**. Nature Communications (2024).  
   https://www.nature.com/articles/s41467-024-54457-x

4. **ChemGraph as an agentic framework for computational chemistry workflows**. Communications Chemistry (2025).  
   https://www.nature.com/articles/s42004-025-01776-9

5. **A review of large language models and autonomous agents in chemistry**. Chemical Science (2025).  
   https://pubs.rsc.org/en/content/articlehtml/2025/sc/d4sc03921a

### Chemistry-specialized foundation models
6. **ChemDFM: A Large Language Foundation Model for Chemistry**. arXiv (2024), later extended in Cell Reports Physical Science context.  
   https://arxiv.org/abs/2401.14818

7. **Developing ChemDFM as a large language foundation model for chemistry**. Cell Reports Physical Science (2025).  
   https://www.cell.com/cell-reports-physical-science/fulltext/S2666-3864%2825%2900122-5

8. **Large Language Models to Accelerate Organic Chemistry Synthesis (Chemma)**. arXiv (2025).  
   https://arxiv.org/abs/2504.18340

### Internal context
9. `BIOMNI_ARCHITECTURE_DEEP_DIVE.md` — internal analysis of Biomni’s execution model, tool retrieval, and architectural tradeoffs.
10. `BIOMNI_IMPLEMENTATION_COMPLETE.md` — implementation context for a Biomni-inspired stack and where it should be improved.
11. `RASYN_STRATEGY_AND_MODELS.md` — product/research context for the chemistry workspace direction.
12. `RASYN_ICE_TOOL_MASTER_PLAN.md` — tool and workflow surface area that motivates the need for a proper chemistry OS.
