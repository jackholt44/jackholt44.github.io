---
layout: post
title: "Lessons from using agents in regulated environments"
date: 2026-05-20
redirect_from:
  - /2026/05/20/agent-performance-was-never-the-hard-part/
  - /2026/05/20/the-agent-was-the-tractable-component/
---

As a Principal Engineer at Faculty, I spent a year building an agent harness for large enterprises in regulated domains. Our goal was production-level performance. Details are anonymised.

The hard parts were persisting and linking their outputs, decomposing them along organisation lines, keeping them running against APIs built for humans, and getting accuracy up to a human reviewer.

## Our setup

We used a lightweight, domain-agnostic data model. Every agent emits **steps**: a record of actions it actually took (e.g. "read the document", "emailed the broker"). A **run** is a single invocation, an ordered set of steps. The agent asserts whether the run succeeded; the harness does not infer it ("email server was down, run failed"). A **task** is the business outcome, such as "insurance quote complete", and it owns many runs. The run history is what a later run reads to continue the task. An insurance quote that takes three weeks of broker correspondence is one task and N runs, each picking the workflow up with new information rather than resuming a paused process.

<figure class="figure-80">
  <img src="/assets/img/data-model.png" alt="Worked example of the Task/Run/Step model: an insurance quote spanning three runs over five days">
  <figcaption>The data model in practice: one insurance quote, three runs spanning five days.</figcaption>
</figure>

The harness holds no domain knowledge: "get a quote by doing X" never enters it. Customisation sits with three roles: a builder who has the domain knowledge and writes the agent, a dashboard user who works from its outputs, and a manager who owns the production bar. The harness connects them through the task/run/step model. Domain behaviour is picked up by convention-based discovery, so it stays out of the harness.

We built first-class support for eval-driven development. In a regulated domain the production bar is near-human accuracy and a record that stands up in an audit, so roughly 75% of the engineering went into measuring agent performance rather than building agents. Once an evaluation was cheap to run, it was obvious whether a change had helped.

## Modelling the org as a DAG

A business decision is rarely one agent's output. "Offer this insurance quote" depends on several determinations, each owned by a different team, and the decision exists only once those results compose. Each team has the domain knowledge to build its own agent, and should be able to ship without understanding the others. The dependencies sit on team boundaries. That is Conway's Law: the system was going to mirror the organisation. We made the mirroring an explicit dependency, so it would not turn up later as coupling between teams.

We modelled that structure as a DAG. Each agent is a node, each cross-team dependency is an edge. The organisation's decision graph and the agent graph are the same graph.

On a successful run the harness derives a trigger from the output and records it. A downstream agent subscribes to the triggers it depends on; when one fires, it loads state from the upstream task in the shared model and continues. The edge is a row in the trigger collection, not a message bus or a new orchestration tier. Trigger state is one collection in a database we already ran: a lookup of `task X → trigger → task Y`.

The same rows are the provenance record. Every edge is a stored `task → trigger → task`, so the chain behind an output is a query. In a regulated domain you have to answer "why did the agent do this". The trigger table is that answer, available when the decision is made, rather than a reconstruction from logs afterwards.

A trigger is written only on success, so a failed upstream run leaves no edge. The downstream agent receives nothing and does not run. Upstream success is a precondition of the edge, so a bad result cannot cascade. The missing row is queryable too, which answers "why didn't the agent run".

A team could build, evaluate, and ship its agent against one contract: emit a trigger on success, subscribe to the triggers you depend on. There is no synchronous coordination with any other team. The interface they share is the trigger row.

The harness's job was to make that organisational shape explicit and cheap to build against.

<figure class="figure-60">
  <img src="/assets/img/triggers.png" alt="Agent dependency DAG with triggers as edges">
  <figcaption>Organisation dependencies modelled as a DAG: agents are nodes, triggers are edges.</figcaption>
</figure>

## Retryable runs

The internal APIs these agents called were built for people using dashboards. Machine-to-machine auth was rare, and the APIs were sized for human throughput. An agent clears in minutes a queue a team would have worked through over a day. At that rate the APIs we depended on degraded or failed, so retries were part of the design.

We modelled agents as retryable at the run level. A run succeeds or fails, a failure is classified as retryable or terminal, and retryable failures are re-enqueued. A variant of durable execution (Temporal, Azure Durable Functions).

Side effects from a failed run have already happened: an email was sent, a broker was contacted. A retry has to work out what already happened and continue past it. That is task-level idempotency of run-level concepts. Deciding what already happened is domain reasoning, so the agent does it. The harness supplies the raw record. We accepted the cost of making builders write that reconciliation. An agent that cannot be retried safely cannot run at production volume.

In exchange, the same contract works across orchestration providers. Where a provider could resume mid-run, reconciliation handed control back into the run in place. Where it could not, the idempotency check made a restart from the top safe. Operators also had a dashboard action to retry a failed run with context the agent did not have.

<figure class="figure-90">
  <img src="/assets/img/reconcile-flow.png" alt="Agent retry and reconciliation flow">
  <figcaption>Agent retry flow with state reconciliation.</figcaption>
</figure>

## Right-censored labels

In fraud and insurance the label is right-censored. At decision time you observe that a confirming event has not happened *yet*. "The user has not completed ID verification" is a different claim from "they never will". The label matures as those events resolve, so a decision recorded early sits on a less mature label than the same decision recorded later. `Response = f(query, t)`, not `f(query)`.

A human team worked a queue over a day, and labels matured during that lag before the decision was recorded. An agent clears the same queue in minutes, so every decision moves earlier in label-time. Measured accuracy drops because the eval is scoring an early decision against a reference label observed later, once it had matured.

<figure class="figure-90">
  <img src="/assets/img/eval-gap.png" alt="Label maturation gap introduced by fast agents">
  <figcaption>The agent decides before the label has matured.</figcaption>
</figure>

The textbook fix is an as-of filter that reproduces the world at T. Most internal APIs have no notion of T, and where one existed it was not built for thousands of concurrent eval runs. These are the same APIs as in the retry section, and they capped how fast we could read historical state. The full set rarely finished overnight, so it ran periodically, while a label-balanced subset weighted toward the failures that kept recurring ran daily.

We also could not snapshot the original decision to make the eval reproducible. The reviewer had fetched that data through their own tools, outside the harness.

So we instrumented what the agent actually received. Every HTTP response could be hashed and stored, and runs were ordered so a disagreement could trigger a re-run against fresh labels. The harness flagged when the API's answer had materially changed between the two calls. A disagreement then splits in two: the agent was wrong, or the upstream answer had changed.

That let us evaluate fairly against systems that fail and data that changes. It did not close the gap to human accuracy.

## Tacit knowledge and triage

The remaining gap was tacit knowledge: the judgement an experienced person accrues doing the role. On many tasks the agent was already asymptotic to human accuracy and recall. There was only so much of that we could write down. Past that, the curve flattened.

We tried injecting salient examples at inference time. It helped only when the retrieved examples were highly relevant. Below that, they hurt. These tasks are ambiguous, and the model anchored on the examples, so a marginally relevant neighbour caused regressions. The relevance bar was high, and few examples cleared it. In exploration, a usefully close neighbour existed for 2% of validation-set runs.

Raising that share means density: more examples, and a shorter distance to a useful neighbour. Once the agent was live, every confirmed failure in the rollout review collected two things: a label of correct, incorrect, or partial, and a free-text reason. That feedback went into the example store. Each confirmed failure shortened the expected distance to the nearest useful neighbour for similar cases.

<figure>
  <img src="/assets/img/tacit-asymptote.png" alt="Asymptotic agent performance and dynamic example injection">
  <figcaption>Example injection once accuracy has flattened.</figcaption>
</figure>

Density accumulates slowly, and some cases have no near neighbour. We used a cheap triage model (LightGBM) to identify the hard cases and route them to human review, instead of shipping them as autonomous. In the harness those checks are post-processing nodes downstream of the domain agent, still inside the same run and task for the dashboard user.

The domain agent runs inside higher-order nodes that enforce the production bar. That takes the builder off the hook for aggregate performance. The manager owns the threshold, and the wrapping nodes are how it is applied.

<figure class="figure-90">
  <img src="/assets/img/pareto.png" alt="Pareto segmentation of agent tasks to reach production grade">
  <figcaption>Segmenting tasks so the ones that clear the bar can run, and the rest go to review.</figcaption>
</figure>

## What stayed hard

The difficult work sat around the agent: how the organisation splits a decision, how the APIs it depends on fail, how labels mature after the fact, and how much of the job was never written down. The harness can record the work, follow those dependencies, and retry. Experience still has to be accrued. Example density shortens the gap for cases we have already seen; the rest go to a person.
