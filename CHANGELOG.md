# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Release tags have
no `v` prefix; each tag publishes the image
`4pdosc/llm-scaler-decision-gen:<version>`.

## [Unreleased]

## [0.8.0] - 2026-11-01

### Added
- CI on every pull request and push to `main`: the pytest suite, plus a gate
  for the license text and committed credentials.
- Dependabot for the pip dependencies and the GitHub Actions.
- `NOTICE`.
- DisaggregatedSet (PD-disagg) workload support. `resolve_placement` now
  checks for a `disaggregatedsets.disaggregatedset.x-k8s.io` CR named after
  the service before falling through to LWS / StatefulSet / Deployment.
  The DS bundle is a single scaling unit: `spec_replicas` reads the
  top-level `spec.slices`, and `gpus_per_replica` is the per-slice GPU
  cost summed over roles (`role.replicas × leaderWorkerTemplate.size ×
  per-pod GPUs`). CPU-only roles (e.g. a router folded into the DS)
  contribute zero. The new `Placement.members` field lists the member
  workloads for log tracing. Reading the DS requires `get/list/watch` on
  `disaggregatedsets.disaggregatedset.x-k8s.io` — RBAC is owned by the
  deployment chart.
- PD engine scaling for CRs whose `maximumDeployment.type` is `pdengine`
  (bounds in engines, not bundle replicas).
  - A pdengine row in `/decisions` carries `shape.pd` alongside
    `replicas.active` (the live `spec.slices`, which the autoscaler
    never moves). The block pairs the committed per-slice prefill
    and decode engine counts with the `liveReplicas` the DS spec is
    currently running.
  - The signal→role policy maps ttft violations to prefill growth,
    otps violations to decode growth, rejection spikes to both roles at
    the same multiplier, and a comfort shed to whichever role has the
    larger SLO-margin; bounds violations are corrected at the live
    P:D ratio (grow → ceil, shrink → floor, residual walked one
    engine at a time, ties to decode).
  - Physical engine counts come from
    `bodylog_service_replicas_ready{..., role="prefill"|"decode"}`
    once llm-openresty#26 adds the `role` label (an additional
    series; the unlabelled total stays). Every other signal —
    ttft, otps, rejection, request / rejection counts — stays
    service-level.
  - Inert until llm-openresty#26: without the `role` label the PD
    branch holds at `hold-pd-no-role-physical` instead of freezing
    blind — the per-role direction freeze can't see ramps/drains
    otherwise. Boot-seed committed values still appear on the wire.
  - Bounds violations are corrected at the live P:D ratio through
    the same-ratio walk; there is no separate "compress to (1p,1d)"
    path for small windows. A window too small to fit one engine
    per role (per-slice `max < 2`) omits the service — emitted
    as `hold-pdengine-bounds`.
  - A CR whose `type` flips between `replica` and `pdengine`
    reseeds the ledger from the placement rather than crash —
    `serviceview.reset_committed` drops the foreign-currency value,
    and the next tick boots from the workload's own shape.
  - pdengine growth is booked against the pool's *leftover* after
    replica services have been scheduled by the cross-service
    planner (which is untouched here); the PD branch never
    preempts a replica service's want. Sheds always land.

### Changed
- GitHub Actions pinned to commit SHAs.
- The README says where the released image is published.
- **Breaking:** removing `maximumDeployment` from a listed service's CR drops the
  service from `/decisions` on the next tick, instead of freezing it
  at its last committed count. Adding `maximumDeployment` back
  re-lists the service from a fresh boot seed (cooldowns restarted).
  Prior behavior kept the service listed at its last count, which
  read as "the scaler is still holding this" to consumers;
  llm-operator treats a missing row as "hold replicas, mark
  ScalingActive=False" and hands control back to the CR author.
  Replica and pdengine services behave the same way here.

### Fixed
- Per-pod GPU accounting is now the *sum* of container
  `nvidia.com/gpu` limits, not the max. PD-colocate pods (prefill +
  decode engine containers sharing one pod, each requesting its own
  slice of the node's GPUs) now bill their full footprint. Single-
  container workloads are unaffected (`max == sum`).

### Removed
- The internal GitLab pipeline (`.gitlab-ci.yml`).
- `config/manager/manager-unstable.yaml`, an internal canary deployment that
  pulled from an internal registry. `config/manager/manager.yaml` is unchanged.
- The whole `config/` directory (`rbac/rbac.yaml` and `manager/manager.yaml`).
  The ClusterRole and Deployment are owned by the deployment Helm chart;
  the in-repo copies had drifted out of sync and an `kubectl apply` against
  them would silently revert chart-managed RBAC.

## [0.7.0] - 2026-09-25

### Added
- Release workflow: a version tag builds the image for linux/amd64 and
  linux/arm64 and pushes it to Docker Hub. A prerelease tag does not move
  `latest`.

### Changed
- **Breaking:** the SLO requirement CRs are read from the API group
  `inference.modelsphere.dev` instead of `inference.x-k8s.io`, and the RBAC in
  `config/rbac/rbac.yaml` follows. The `/decisions` document keeps its
  `apiVersion` (`llmscaling.inference.x-k8s.io/v1alpha1`).
- Project URLs point at the `modelsphere` GitHub organization.

## [0.6.0] - 2026-09-21

First tagged release.

### Added
- The decision generator: every tick (60s by default) it rebuilds replica
  decisions from `LLMSLORequirement` CRs, Prometheus signals and Kubernetes
  state, shares a finite GPU pool out across services, and serves the result
  at `GET /decisions`, with `/healthz` and `/readyz`.
- `JobSLORequirement` support: queue-depth-driven scaling for job workloads,
  alongside the LLM services.
- A service whose CR has no maximum is treated as opting out and skipped.
- Apache-2.0 `LICENSE` and an English README.
- The Dockerfile's base image and PyPI index are build args that default to
  the public ones.

[Unreleased]: https://github.com/modelsphere/slo-scaler-decision-gen/compare/0.8.0...HEAD
[0.8.0]: https://github.com/modelsphere/slo-scaler-decision-gen/compare/0.7.0...0.8.0
[0.7.0]: https://github.com/modelsphere/slo-scaler-decision-gen/compare/0.6.0...0.7.0
[0.6.0]: https://github.com/modelsphere/slo-scaler-decision-gen/releases/tag/0.6.0
