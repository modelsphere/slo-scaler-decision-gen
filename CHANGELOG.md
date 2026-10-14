# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Release tags have
no `v` prefix; each tag publishes the image
`4pdosc/llm-scaler-decision-gen:<version>`.

## [Unreleased]

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

### Changed
- GitHub Actions pinned to commit SHAs.
- The README says where the released image is published.
- **Fix:** per-pod GPU accounting is now the *sum* of container
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

[Unreleased]: https://github.com/modelsphere/slo-scaler-decision-gen/compare/0.7.0...HEAD
[0.7.0]: https://github.com/modelsphere/slo-scaler-decision-gen/compare/0.6.0...0.7.0
[0.6.0]: https://github.com/modelsphere/slo-scaler-decision-gen/releases/tag/0.6.0
