"""Tests for decision_gen.k8s_state — workload resolution + pool gap."""

import logging
from unittest.mock import MagicMock
from types import SimpleNamespace

import pytest

from decision_gen import k8s_state
from decision_gen.k8s_state import K8sState, Placement


POOL_H100 = "NVIDIA-H100-80GB-HBM3"
POOL_H800 = "NVIDIA-H800"
POOL_A100 = "NVIDIA-A100-SXM4-80GB"


# ---------- helpers: build fake workload payloads (dicts mimicking wire format) ----------

def _affinity_with_pool(pool, op="In"):
    return {
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [{
                    "matchExpressions": [{
                        "key": "nvidia.com/gpu.product",
                        "operator": op,
                        "values": [pool],
                    }],
                }],
            },
        },
    }


def _pod_template(gpu_limit=2, pool=POOL_H100, containers=None):
    if containers is None:
        containers = [{
            "name": "main",
            "resources": {"limits": {"nvidia.com/gpu": gpu_limit}},
        }]
    return {
        "spec": {
            "affinity": _affinity_with_pool(pool),
            "containers": containers,
        },
    }


def _lws_spec(size=2, per_pod_gpus=8, pool=POOL_H100):
    """Dict mimicking LWS wire format."""
    return {
        "spec": {
            "replicas": 1,
            "leaderWorkerTemplate": {
                "size": size,
                "leaderTemplate": _pod_template(gpu_limit=per_pod_gpus, pool=pool),
                "workerTemplate": _pod_template(gpu_limit=per_pod_gpus, pool=pool),
            },
        },
    }


class _Api404(Exception):
    def __init__(self):
        super().__init__("not found")
        self.status = 404


class _Api403(_Api404):
    """403 variant of the shared ApiException stub — same class family so the
    autouse monkeypatch on k8s_state.ApiException catches both."""

    def __init__(self):
        super().__init__()
        self.status = 403


def _fake_apis(*, lws=None, sts=None, deploy=None, ds=None, nodes=None, pods=None,
               ds_error=None):
    """Build a (apps, core, custom) triple; missing workloads raise ApiException(404).

    Both LWS and DS resolve through CustomObjectsApi; dispatch by plural so
    a test can provide one, both, or neither. `ds_error` (an exception instance)
    makes the DS lookup raise instead of returning a payload — used to cover
    the 403-fallthrough path where RBAC hasn't granted disaggregatedsets yet.
    """
    apps = MagicMock()
    if sts is None:
        apps.read_namespaced_stateful_set.side_effect = _Api404()
    else:
        apps.read_namespaced_stateful_set.return_value = sts
    if deploy is None:
        apps.read_namespaced_deployment.side_effect = _Api404()
    else:
        apps.read_namespaced_deployment.return_value = deploy

    core = MagicMock()
    core.list_node.return_value = SimpleNamespace(items=nodes or [])
    core.list_pod_for_all_namespaces.return_value = SimpleNamespace(items=pods or [])

    custom = MagicMock()
    by_plural = {}
    if lws is not None:
        by_plural[k8s_state.LWS_PLURAL] = lws
    if ds is not None:
        by_plural[k8s_state.DS_PLURAL] = ds

    def _get(group, version, namespace, plural, name):
        if plural == k8s_state.DS_PLURAL and ds_error is not None:
            raise ds_error
        if plural in by_plural:
            return by_plural[plural]
        raise _Api404()
    custom.get_namespaced_custom_object.side_effect = _get
    return apps, core, custom


# Patch the ApiException in k8s_state to use our test exception
@pytest.fixture(autouse=True)
def _patch_api_exception(monkeypatch):
    monkeypatch.setattr(k8s_state, "ApiException", _Api404)


# ---------- resolve_placement: discovery order ----------

def test_lws_found_first_beats_sts():
    """If both LWS and same-named STS exist, LWS wins (STS is a side-effect)."""
    lws = _lws_spec(size=2, per_pod_gpus=8)
    sts_dict = {"spec": {"template": _pod_template(gpu_limit=99)}}
    apps, core, custom = _fake_apis(lws=lws, sts=sts_dict)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("kimi", "kimi-k25")
    assert p is not None
    assert p.kind == "lws"
    assert p.gpus_per_replica == 16   # size=2 × per_pod=8


def test_resolve_by_service_id_when_no_alias():
    """Aliases are empty; resolution must look up the serviceId directly.
    Regression: an earlier hard-coded alias pointed at a workload that was
    renamed, 404'd everywhere, and masked a perfectly resolvable serviceId."""
    deploy = {"spec": {"template": _pod_template(gpu_limit=2)}}
    apps, core, custom = _fake_apis(deploy=deploy)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("modelforge", "fallback-modelforge-01")
    assert p is not None
    assert p.kind == "deployment"
    assert p.gpus_per_replica == 2
    # serviceId was used directly as the workload name (aliases are {}).
    call_args = apps.read_namespaced_deployment.call_args
    assert call_args.args[0] == "fallback-modelforge-01"
    assert call_args.args[1] == "modelforge"


def test_falls_through_to_sts_when_no_lws():
    sts = {"spec": {"template": _pod_template(gpu_limit=4)}}
    apps, core, custom = _fake_apis(sts=sts)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("kimi", "kimi-k25")
    assert p is not None
    assert p.kind == "statefulset"
    assert p.gpus_per_replica == 4


def test_falls_through_to_deployment():
    deploy = {"spec": {"template": _pod_template(gpu_limit=2)}}
    apps, core, custom = _fake_apis(deploy=deploy)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("kimi", "kimi-k25")
    assert p is not None
    assert p.kind == "deployment"


def test_no_workload_returns_none():
    apps, core, custom = _fake_apis()
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    assert state.resolve_placement("kimi", "missing") is None


def test_sidecar_deployment_filtered():
    """Deployment with zero GPU limits on all containers → filtered (None)."""
    deploy = {"spec": {"template": _pod_template(gpu_limit=0)}}
    apps, core, custom = _fake_apis(deploy=deploy)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("foo", "cart-api")
    assert p is None


def test_multi_pool_template_unmanageable():
    """Affinity lists two pools → we can't pick; None."""
    bad_affin = {
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [{
                    "matchExpressions": [{
                        "key": "nvidia.com/gpu.product",
                        "operator": "In",
                        "values": [POOL_H100, POOL_A100],
                    }],
                }],
            },
        },
    }
    deploy = {"spec": {"template": {
        "spec": {
            "affinity": bad_affin,
            "containers": [{"name": "m", "resources": {"limits": {"nvidia.com/gpu": 2}}}],
        },
    }}}
    apps, core, custom = _fake_apis(deploy=deploy)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    assert state.resolve_placement("kimi", "x") is None


def test_no_affinity_defaults_to_h100():
    """V0 default: no GPU-product pin anywhere → DEFAULT_GPU_POOL.
    Covers pods like minimax-h3 that carry no affinity at all."""
    deploy = {"spec": {"template": {"spec": {
        "containers": [{"name": "m", "resources": {"limits": {"nvidia.com/gpu": 4}}}],
    }}}}
    apps, core, custom = _fake_apis(deploy=deploy)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("ns", "svc")
    assert p is not None and p.pool == POOL_H100
    assert p.gpus_per_replica == 4


def test_non_gpu_affinity_does_not_block_default():
    """A pod with affinity for disk/zone/etc but no GPU pin must still
    fall into the default rule — the extractor's `None` means
    "unconstrained", not "ambiguous". Regression for the review finding:
    the tri-state distinction is what keeps these workloads manageable."""
    deploy = {"spec": {"template": {"spec": {
        "affinity": {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [{
                        "matchExpressions": [{
                            "key": "kubernetes.io/hostname",
                            "operator": "In",
                            "values": ["node-a"],
                        }],
                    }],
                },
            },
        },
        "containers": [{"name": "m", "resources": {"limits": {"nvidia.com/gpu": 2}}}],
    }}}}
    apps, core, custom = _fake_apis(deploy=deploy)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("ns", "svc")
    assert p is not None and p.pool == POOL_H100


def test_lws_leader_worker_pool_mismatch_unmanageable():
    lws = {
        "spec": {
            "replicas": 1,
            "leaderWorkerTemplate": {
                "size": 2,
                "leaderTemplate": _pod_template(gpu_limit=8, pool=POOL_H100),
                "workerTemplate": _pod_template(gpu_limit=8, pool=POOL_A100),
            },
        },
    }
    apps, core, custom = _fake_apis(lws=lws)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    assert state.resolve_placement("kimi", "kimi-k25") is None


def test_lws_gpus_per_replica_is_size_times_per_pod():
    lws = _lws_spec(size=4, per_pod_gpus=8)
    apps, core, custom = _fake_apis(lws=lws)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("kimi", "kimi-k25")
    assert p is not None
    assert p.gpus_per_replica == 4 * 8 == 32


def test_deployment_pool_from_affinity():
    deploy = {"spec": {"template": _pod_template(gpu_limit=2, pool=POOL_A100)}}
    apps, core, custom = _fake_apis(deploy=deploy)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("kimi", "kimi-k25")
    assert p is not None
    assert p.pool == POOL_A100


def test_deployment_pool_from_affinity_snake_case():
    """kubernetes-python's to_dict() emits snake_case keys for nested
    sub-objects (attribute names), not JSON/camelCase keys. The canonical
    example was modelforge-fallback-sglang: pool sat under
    node_affinity.required_during_scheduling_ignored_during_execution.
    Guard against regression — this is the shape we see in production."""
    snake_affinity = {
        "node_affinity": {
            "required_during_scheduling_ignored_during_execution": {
                "node_selector_terms": [{
                    "match_expressions": [{
                        "key": "nvidia.com/gpu.product",
                        "operator": "In",
                        "values": [POOL_H100],
                    }],
                }],
            },
        },
    }
    deploy = {"spec": {"template": {
        "spec": {
            "affinity": snake_affinity,
            "containers": [{
                "name": "main",
                "resources": {"limits": {"nvidia.com/gpu": 2}},
            }],
        },
    }}}
    apps, core, custom = _fake_apis(deploy=deploy)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("kimi", "kimi-k25")
    assert p is not None
    assert p.pool == POOL_H100
    assert p.gpus_per_replica == 2


# ---------- PD shapes (colocate Deployment + DisaggregatedSet bundle) ----------

def _colocate_pod_template(gpus_per_engine=4, pool=POOL_H100):
    """Colocate shape: prefill + decode + router in ONE pod.

    Each engine container requests `gpus_per_engine`; router requests none.
    Mirrors `sglang-qwen36-colocate` — scheduler sees a single pod that
    occupies 2×gpus_per_engine GPUs via sum-of-containers accounting.
    """
    return {
        "spec": {
            "affinity": _affinity_with_pool(pool),
            "containers": [
                {"name": "prefill",
                 "resources": {"limits": {"nvidia.com/gpu": gpus_per_engine}}},
                {"name": "decode",
                 "resources": {"limits": {"nvidia.com/gpu": gpus_per_engine}}},
                {"name": "router",
                 "resources": {"limits": {}}},
            ],
        },
    }


def _ds_spec(*, slices=1, roles, per_pod_gpus=8, pool=POOL_H100):
    """Generic DS builder. `roles` is a list of (name, replicas, lwt_size)."""
    role_objs = []
    for name, replicas, lwt_size in roles:
        role_objs.append({
            "name": name,
            "spec": {
                "replicas": replicas,
                "leaderWorkerTemplate": {
                    "size": lwt_size,
                    "leaderTemplate": _pod_template(gpu_limit=per_pod_gpus, pool=pool),
                    "workerTemplate": _pod_template(gpu_limit=per_pod_gpus, pool=pool),
                },
            },
        })
    return {"spec": {"slices": slices, "roles": role_objs}}


def _ds_2p1d_spec(per_pod_gpus=8, pool=POOL_H100, slices=1):
    """2P1D DisaggregatedSet shape — mirrors `vllm-qwen36-2p1d` in test-dsset.
    Live DS has slices=1; the bundle knob turns this into N × 2P1D."""
    return _ds_spec(
        slices=slices,
        per_pod_gpus=per_pod_gpus, pool=pool,
        roles=[("prefill", 2, 1), ("decode", 1, 1)],
    )


def _ds_1p1d_spec(per_pod_gpus=8, pool=POOL_H100, slices=1):
    return _ds_spec(
        slices=slices,
        per_pod_gpus=per_pod_gpus, pool=pool,
        roles=[("prefill", 1, 1), ("decode", 1, 1)],
    )


def _ds_2p3d_spec(per_pod_gpus=8, pool=POOL_H100, slices=1):
    return _ds_spec(
        slices=slices,
        per_pod_gpus=per_pod_gpus, pool=pool,
        roles=[("prefill", 2, 1), ("decode", 3, 1)],
    )


def test_colocate_deployment_sums_container_gpus():
    """Shape C: ONE Deployment, 3 containers each owning a slice of the 8 GPUs.
    Sum-of-containers must equal scheduler-admission-total (8), not max (4).
    Regression: previous max() accounting made a colocate pod look half price,
    which would let the allocator oversubscribe the pool."""
    deploy = {"spec": {"template": _colocate_pod_template(gpus_per_engine=4)}}
    apps, core, custom = _fake_apis(deploy=deploy)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("test-dsset", "sglang-qwen36-colocate")
    assert p is not None
    assert p.kind == "deployment"
    assert p.gpus_per_replica == 8     # 4 (prefill) + 4 (decode) + 0 (router)


def test_ds_2p1d_bundle_gpus_and_replicas():
    """Shape I live today: DS slices=1, prefill.replicas=2, decode.replicas=1
    → ONE bundle of 2 prefill pods + 1 decode pod = 24 GPUs.
    Bundle knob is `spec.slices` (=1 here), NOT any role's replicas."""
    ds = _ds_2p1d_spec(per_pod_gpus=8)
    apps, core, custom = _fake_apis(ds=ds)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("test-dsset", "vllm-qwen36-2p1d")
    assert p is not None
    assert p.kind == "disaggregatedset"
    assert p.workload_name == "vllm-qwen36-2p1d"
    assert p.gpus_per_replica == 24    # per-bundle cost: 2×8 + 1×8
    assert p.spec_replicas == 1        # slices (not role replicas) is the knob
    assert p.members == (
        ("lws", "vllm-qwen36-2p1d-prefill"),
        ("lws", "vllm-qwen36-2p1d-decode"),
    )


def test_ds_2p1d_slices_5_is_five_bundles():
    """5 × 2p3d-style scaling: slices is the bundle count. Per-bundle cost
    is unchanged (gpus_per_replica is per-slice); the planner multiplies
    slices × per-bundle-cost to get the total."""
    ds = _ds_2p1d_spec(per_pod_gpus=8, slices=5)
    apps, core, custom = _fake_apis(ds=ds)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("test-dsset", "vllm-qwen36-2p1d")
    assert p is not None
    assert p.gpus_per_replica == 24    # unchanged: per-slice cost
    assert p.spec_replicas == 5        # 5 slices → planner sees "5 bundles"


def test_ds_1p1d_bundle():
    ds = _ds_1p1d_spec(per_pod_gpus=8)
    apps, core, custom = _fake_apis(ds=ds)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("test-dsset", "vllm-qwen36-1p1d")
    assert p is not None
    assert p.kind == "disaggregatedset"
    assert p.gpus_per_replica == 16    # 1×8 + 1×8
    assert p.spec_replicas == 1


def test_ds_2p3d_asymmetric_ratio():
    """2P3D: prefill.replicas=2, decode.replicas=3. Per-bundle cost is
    2×8 prefill + 3×8 decode = 40 GPUs. spec_replicas still = slices."""
    ds = _ds_2p3d_spec(per_pod_gpus=8)
    apps, core, custom = _fake_apis(ds=ds)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("test-dsset", "vllm-qwen36-2p3d")
    assert p is not None
    assert p.kind == "disaggregatedset"
    assert p.gpus_per_replica == 40    # 2×8 + 3×8
    assert p.spec_replicas == 1


def test_ds_role_with_lws_size_4_is_group_of_pods():
    """lwt.size=4 → each LWS group spans 4 pods (e.g. TP32 across pods).
    1 prefill group of size 4 × 8 GPU/pod = 32 GPUs just for prefill.
    Bundle = 1 prefill group (32) + 1 decode group (32) = 64 GPUs."""
    ds = _ds_spec(
        slices=1, per_pod_gpus=8, pool=POOL_H100,
        roles=[("prefill", 1, 4), ("decode", 1, 4)],
    )
    apps, core, custom = _fake_apis(ds=ds)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("test-dsset", "vllm-tp32")
    assert p is not None
    assert p.gpus_per_replica == 64    # 1×4×8 + 1×4×8
    assert p.spec_replicas == 1


def test_ds_5x_2p3d_with_tp16_across_2_pods():
    """Composition of slices × replicas × size, all three multiplied:
    slices=5 bundles, 2P3D ratio, each role's LWS group spans 2 pods.
    Per-bundle: 2×(2×8) prefill + 3×(2×8) decode = 32+48 = 80 GPUs.
    Total footprint on cluster = 5 × 80 = 400 GPUs (planner's job)."""
    ds = _ds_spec(
        slices=5, per_pod_gpus=8, pool=POOL_H100,
        roles=[("prefill", 2, 2), ("decode", 3, 2)],
    )
    apps, core, custom = _fake_apis(ds=ds)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("test-dsset", "vllm-5x2p3d-tp16")
    assert p is not None
    assert p.gpus_per_replica == 80    # per-bundle
    assert p.spec_replicas == 5        # slices


def test_ds_missing_falls_through_to_deployment_unchanged():
    """No DS, no LWS, no STS — Deployment branch is byte-for-byte unchanged."""
    deploy = {"spec": {"template": _pod_template(gpu_limit=2)}}
    apps, core, custom = _fake_apis(deploy=deploy)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("ns", "plain-deployment")
    assert p is not None
    assert p.kind == "deployment"
    assert p.gpus_per_replica == 2
    # DS was actually looked up first (and missed) before falling through.
    plurals_checked = [
        c.args[3] for c in custom.get_namespaced_custom_object.call_args_list
    ]
    assert k8s_state.DS_PLURAL in plurals_checked
    assert k8s_state.LWS_PLURAL in plurals_checked


def test_ds_403_falls_through_with_warning(caplog):
    """DS lookup hits a 403 (missing ClusterRole rule) → resolver falls through
    to LWS/STS/Deploy instead of raising. Only this service stays unmanaged;
    a raise would abort the whole tick for every service. Warning names the
    missing rule so the operator's fix is unambiguous — the follow-up
    'no workload found' log doesn't hint at RBAC on its own."""
    deploy = {"spec": {"template": _pod_template(gpu_limit=2)}}
    apps, core, custom = _fake_apis(deploy=deploy, ds_error=_Api403())
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    with caplog.at_level(logging.WARNING, logger="decision_gen.k8s_state"):
        p = state.resolve_placement("ns", "plain-deployment")
    assert p is not None
    assert p.kind == "deployment"          # fell through, service still resolves
    # Warning names the missing rule explicitly.
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "forbidden reading DS" in msgs
    assert "disaggregatedsets.disaggregatedset.x-k8s.io" in msgs


def test_ds_500_raises():
    """Non-4xx ApiExceptions (a real apiserver failure) still raise — the
    controller's tick-level catch handles them. Swallowing 5xx would hide
    genuine connectivity issues behind 'service unmanaged'."""
    deploy = {"spec": {"template": _pod_template(gpu_limit=2)}}

    class _Api500(_Api404):
        def __init__(self):
            super().__init__()
            self.status = 500

    apps, core, custom = _fake_apis(deploy=deploy, ds_error=_Api500())
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    with pytest.raises(_Api500):
        state.resolve_placement("ns", "plain-deployment")


def test_ds_malformed_does_not_fall_through():
    """DS exists but is broken (empty roles) → unmanageable, no silent fallback.
    A silent fall-through to a same-named LWS would mis-price the bundle."""
    ds = {"spec": {"slices": 1, "roles": []}}
    apps, core, custom = _fake_apis(ds=ds)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    assert state.resolve_placement("test-dsset", "vllm-qwen36-2p1d") is None


def test_ds_role_pool_mismatch_unmanageable():
    ds = {
        "spec": {
            "slices": 1,
            "roles": [
                {"name": "prefill", "spec": {
                    "replicas": 1,
                    "leaderWorkerTemplate": {
                        "size": 1,
                        "leaderTemplate": _pod_template(gpu_limit=8, pool=POOL_H100),
                        "workerTemplate": _pod_template(gpu_limit=8, pool=POOL_H100),
                    },
                }},
                {"name": "decode", "spec": {
                    "replicas": 1,
                    "leaderWorkerTemplate": {
                        "size": 1,
                        "leaderTemplate": _pod_template(gpu_limit=8, pool=POOL_A100),
                        "workerTemplate": _pod_template(gpu_limit=8, pool=POOL_A100),
                    },
                }},
            ],
        },
    }
    apps, core, custom = _fake_apis(ds=ds)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    assert state.resolve_placement("test-dsset", "vllm-qwen36-x") is None


def _cpu_only_pod_template():
    """A role pod that requests no GPUs (e.g. a CPU router folded into the DS)."""
    return {
        "spec": {
            "containers": [
                {"name": "router", "resources": {"limits": {}}},
            ],
        },
    }


def test_ds_with_cpu_router_role_contributes_zero():
    """Forward-compat: if a DS grows a CPU-only router role (CRD allows it;
    chart may someday fold the router Deployment into a 3rd role), the
    role contributes 0 GPUs and the bundle remains a valid placement.
    Today: bundle = 2P + 1D + 1 router = 24 GPUs (router free)."""
    ds = {
        "spec": {
            "slices": 1,
            "roles": [
                {"name": "prefill", "spec": {
                    "replicas": 2,
                    "leaderWorkerTemplate": {
                        "size": 1,
                        "leaderTemplate": _pod_template(gpu_limit=8, pool=POOL_H100),
                        "workerTemplate": _pod_template(gpu_limit=8, pool=POOL_H100),
                    },
                }},
                {"name": "decode", "spec": {
                    "replicas": 1,
                    "leaderWorkerTemplate": {
                        "size": 1,
                        "leaderTemplate": _pod_template(gpu_limit=8, pool=POOL_H100),
                        "workerTemplate": _pod_template(gpu_limit=8, pool=POOL_H100),
                    },
                }},
                {"name": "router", "spec": {
                    "replicas": 1,
                    "leaderWorkerTemplate": {
                        "size": 1,
                        "leaderTemplate": _cpu_only_pod_template(),
                        "workerTemplate": _cpu_only_pod_template(),
                    },
                }},
            ],
        },
    }
    apps, core, custom = _fake_apis(ds=ds)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("test-dsset", "vllm-qwen36-2p1d")
    assert p is not None
    assert p.gpus_per_replica == 24   # router contributes 0
    assert p.spec_replicas == 1
    # Router role is surfaced in members for tracing even though it's free.
    assert ("lws", "vllm-qwen36-2p1d-router") in p.members


def test_ds_all_cpu_roles_unmanageable():
    """A DS where every role is GPU-less has no pool consumption to ledger."""
    ds = {
        "spec": {
            "slices": 1,
            "roles": [
                {"name": "router", "spec": {
                    "replicas": 1,
                    "leaderWorkerTemplate": {
                        "size": 1,
                        "leaderTemplate": _cpu_only_pod_template(),
                        "workerTemplate": _cpu_only_pod_template(),
                    },
                }},
            ],
        },
    }
    apps, core, custom = _fake_apis(ds=ds)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    assert state.resolve_placement("test-dsset", "cpu-only-ds") is None


def test_placement_member_workloads_default_back_compat():
    """Non-DS placements expose ((kind, workload_name),) via member_workloads();
    planner / controller don't need to special-case DS bundles."""
    deploy = {"spec": {"template": _pod_template(gpu_limit=2)}}
    apps, core, custom = _fake_apis(deploy=deploy)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("ns", "plain")
    assert p.member_workloads() == (("deployment", "plain"),)


# ---------- pool_capacity ----------

def _node(name, pool, gpu_allocatable, ready=True, cordoned=False, present="true"):
    cond = SimpleNamespace(type="Ready", status="True" if ready else "False")
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name=name,
            labels={
                "nvidia.com/gpu.present": present,
                "nvidia.com/gpu.product": pool,
            },
        ),
        spec=SimpleNamespace(unschedulable=cordoned),
        status=SimpleNamespace(
            conditions=[cond],
            allocatable={"nvidia.com/gpu": gpu_allocatable},
        ),
    )


def _pod(name, node, gpu_requests):
    def _container(req):
        return SimpleNamespace(
            resources=SimpleNamespace(requests={"nvidia.com/gpu": req}),
        )
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name),
        spec=SimpleNamespace(
            node_name=node,
            containers=[_container(r) for r in gpu_requests],
        ),
    )


def test_pool_capacity_basic():
    apps, core, custom = _fake_apis(
        nodes=[
            _node("n1", POOL_H100, 8),
            _node("n2", POOL_H100, 8),
            _node("n3", POOL_A100, 4),
        ],
    )
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    cap = state.pool_capacity()
    assert cap[POOL_H100] == 16
    assert cap[POOL_A100] == 4


def test_pool_capacity_excludes_cordoned():
    apps, core, custom = _fake_apis(
        nodes=[
            _node("n1", POOL_H100, 8),
            _node("n2", POOL_H100, 8, cordoned=True),
        ],
    )
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    assert state.pool_capacity()[POOL_H100] == 8


def test_pool_capacity_excludes_not_ready():
    apps, core, custom = _fake_apis(
        nodes=[
            _node("n1", POOL_H100, 8),
            _node("n2", POOL_H100, 8, ready=False),
        ],
    )
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    assert state.pool_capacity()[POOL_H100] == 8


def test_pool_capacity_excludes_non_gpu_nodes():
    apps, core, custom = _fake_apis(
        nodes=[
            _node("n1", POOL_H100, 8),
            SimpleNamespace(
                metadata=SimpleNamespace(name="cpu-only", labels={}),
                spec=SimpleNamespace(unschedulable=False),
                status=SimpleNamespace(
                    conditions=[SimpleNamespace(type="Ready", status="True")],
                    allocatable={},
                ),
            ),
        ],
    )
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    assert state.pool_capacity() == {POOL_H100: 8}


def test_pool_capacity_uses_allocatable_not_labels():
    """A node misreporting gpus via labels: allocatable is authoritative."""
    node = _node("sketchy", POOL_H100, 7)   # allocatable says 7
    node.metadata.labels["nvidia.com/gpu.count"] = "8"
    apps, core, custom = _fake_apis(nodes=[node])
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    assert state.pool_capacity()[POOL_H100] == 7


def test_pool_capacity_no_pods_call():
    """We never list pods anymore — ledger accounting moved that to controller."""
    apps, core, custom = _fake_apis(nodes=[_node("n1", POOL_H100, 8)])
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    state.pool_capacity()
    core.list_pod_for_all_namespaces.assert_not_called()


# ---------- TEMP-REVERT: H800-as-H100 pool merge ----------
# Pins for the alias. On revert, delete this whole block and POOL_ALIASES.
# The intended-future failure is that polymorphic placement makes the
# merge redundant; until then these are load-bearing.

def _affinity_multi_pool(pools):
    return {
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [{
                    "matchExpressions": [{
                        "key": "nvidia.com/gpu.product",
                        "operator": "In",
                        "values": pools,
                    }],
                }],
            },
        },
    }


def test_h800_only_affinity_maps_to_h100():
    deploy = {"spec": {"template": {"spec": {
        "affinity": _affinity_multi_pool([POOL_H800]),
        "containers": [{"name": "m", "resources": {"limits": {"nvidia.com/gpu": 2}}}],
    }}}}
    apps, core, custom = _fake_apis(deploy=deploy)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("ns", "svc")
    assert p is not None and p.pool == POOL_H100


def test_h100_h800_mixed_affinity_maps_to_h100():
    deploy = {"spec": {"template": {"spec": {
        "affinity": _affinity_multi_pool([POOL_H100, POOL_H800]),
        "containers": [{"name": "m", "resources": {"limits": {"nvidia.com/gpu": 2}}}],
    }}}}
    apps, core, custom = _fake_apis(deploy=deploy)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("ns", "svc")
    assert p is not None and p.pool == POOL_H100


def test_h100_a100_mixed_affinity_still_unmanageable():
    """Alias must not paper over genuine ambiguity — {H100, A100} is a real
    cross-pool conflict and must stay unmanageable (fail closed)."""
    deploy = {"spec": {"template": {"spec": {
        "affinity": _affinity_multi_pool([POOL_H100, POOL_A100]),
        "containers": [{"name": "m", "resources": {"limits": {"nvidia.com/gpu": 2}}}],
    }}}}
    apps, core, custom = _fake_apis(deploy=deploy)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    assert state.resolve_placement("ns", "svc") is None


def test_a100_only_affinity_untouched_by_alias():
    deploy = {"spec": {"template": _pod_template(gpu_limit=2, pool=POOL_A100)}}
    apps, core, custom = _fake_apis(deploy=deploy)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("ns", "svc")
    assert p is not None and p.pool == POOL_A100


def test_pool_capacity_merges_h800_into_h100():
    apps, core, custom = _fake_apis(
        nodes=[
            _node("h100-1", POOL_H100, 8),
            _node("h100-2", POOL_H100, 8),
            _node("h800-1", POOL_H800, 8),
            _node("a100-1", POOL_A100, 4),
        ],
    )
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    cap = state.pool_capacity()
    assert cap[POOL_H100] == 24     # 16 h100 + 8 h800
    assert POOL_H800 not in cap
    assert cap[POOL_A100] == 4
