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

### Changed
- GitHub Actions pinned to commit SHAs.
- The README says where the released image is published.

### Removed
- The internal GitLab pipeline (`.gitlab-ci.yml`).
- `config/manager/manager-unstable.yaml`, an internal canary deployment that
  pulled from an internal registry. `config/manager/manager.yaml` is unchanged.

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
