# slo-scaler-decision-gen

Decides how many replicas each LLM inference service should be running, and
publishes that as a list over HTTP. It does not scale anything itself — an
operator reads `GET /decisions` and acts on it.

The split is deliberate. Sizing is the part with the interesting rules (SLO
thresholds, cooldowns, a finite GPU pool to share out), and keeping it in a
process that only reads makes it something you can interrogate: point curl at it
and see exactly what it thinks, without anything moving.

```
  LLMSLORequirement CRs ─┐
  JobSLORequirement CRs ─┤
  Kubernetes workloads ──┼─▶ slo-scaler-decision-gen ─▶ GET /decisions ─▶ operator
  Prometheus signals ────┘         (60s tick)
```

## What it decides on

Every tick (60s by default) it rebuilds the whole picture and re-derives every
service's replica count from scratch — no deltas, ever. A delta has an implicit
base, and disagreeing about that base is the bug class this design refuses to
have.

The inputs are:

- **`LLMSLORequirement` / `JobSLORequirement` CRs** — the target: which
  thresholds matter for this service, and its `[min, max]` bounds. The bounds are
  the CR's authority; proposals are clamped to them and nothing else may widen
  them. A service with no maximum is treated as opting out entirely rather than
  as unbounded.
- **Live signals from Prometheus** — the evidence that a threshold is being
  violated, or has been comfortably met long enough to scale back down.
- **Kubernetes state** — what is actually running, where it is placed, and how
  much pool capacity is left. Replica counts come from the workload rather than
  from anything this process remembers, so a hand-edited Deployment is seen, not
  overwritten from a stale ledger.

Sizing rules are applied top-down, first match wins, and are a pure function of
their arguments — no clock, no state, no environment. Cooldown and comfort reach
them as plain booleans, decided elsewhere, so a rule cannot accidentally depend
on when it ran.

After every service has a proposal, a global allocator hands out the finite pool.
That stage is pure too: same inputs, same allocation.

## Running it

```bash
make dev     # editable install with test extras
make test    # 183 tests, no cluster and no Prometheus needed
make run     # against your current kubeconfig
```

Nothing but the Kubernetes and Prometheus clients is required at runtime; the
tests need neither.

In a cluster:

```bash
kubectl apply -f config/rbac/rbac.yaml
kubectl apply -f config/manager/manager.yaml
```

The manifests assume a `llm-scaler` namespace and an image you have built and
pushed yourself (`make docker`). RBAC is read-only apart from the CR watches:
this process never writes to the cluster.

## HTTP surface

| Method | Path | Purpose |
|---|---|---|
| GET | `/decisions` | the current decision list |
| GET | `/healthz` | liveness |
| GET | `/readyz` | readiness |

`/decisions` and `/readyz` both stay unready until the controller has completed a
real tick. Serving the boot placeholder as though it were data looks exactly like
"every service should have zero replicas", which is the one wrong answer that a
consumer would act on immediately.

```json
{
  "apiVersion": "llmscaling.inference.x-k8s.io/v1alpha1",
  "decisions": [
    {"namespace": "modelforge", "serviceId": "kimi", "replicas": {"active": 3}}
  ]
}
```

The wire format lives in exactly one function, so changing it is a one-file diff
that cannot pass review unnoticed.

## Configuration

All by environment variable:

| Variable | Default | |
|---|---|---|
| `PORT` | `8080` | HTTP listener |
| `TICK_SECONDS` | `60` | how often the whole picture is rebuilt |
| `PROM_URL` | `http://kube-prometheus-stack-prometheus.monitoring:9090` | where the signals come from |
| `PROM_TIMEOUT_S` | `5` | per-query timeout |
| `LOG_LEVEL` | `INFO` | applies to this process only — the Kubernetes client dumps whole HTTP bodies at DEBUG and is capped at WARNING regardless |

## Building the image

```bash
docker build -t decision-gen:latest .
```

The base image and the PyPI index are build args, so a build that cannot reach
Docker Hub or PyPI can point them elsewhere:

```bash
docker build \
  --build-arg BASE_IMAGE=my-registry/python:3.12-slim \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .
```

## Layout

| Module | |
|---|---|
| `controller.py` | the tick driver, and the only writer of the served snapshot |
| `serviceview.py` | per-service state machine |
| `triggers.py` | the sizing rules — pure |
| `planner.py` | global allocation across the pool — pure |
| `thresholds.py` | what counts as a violation, and what counts as comfortable |
| `direction.py` | reconciles the ledger against what is physically there |
| `signals.py` | the Prometheus client, and the only place PromQL lives |
| `slo_store.py` | in-memory cache of the SLO CRs, kept current by a watch |
| `k8s_state.py` | workload resolution, placement, pool capacity |
| `snapshot.py` | the `/decisions` wire format |
| `server.py` | read-only HTTP handlers over the snapshot |

The one-owner-per-concern arrangement is load-bearing: every PromQL string is in
`signals.py`, every wire-format decision is in `snapshot.py`, and the pure
modules have no way to reach a clock or the environment.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
