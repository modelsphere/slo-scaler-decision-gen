"""Tests for decision_gen.k8s_state — workload resolution + pool gap."""

from unittest.mock import MagicMock
from types import SimpleNamespace

import pytest

from decision_gen import k8s_state
from decision_gen.k8s_state import K8sState, Placement


POOL_H100 = "NVIDIA-H100-80GB-HBM3"
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


def _fake_apis(*, lws=None, sts=None, deploy=None, nodes=None, pods=None):
    """Build a (apps, core, custom) triple; missing workloads raise ApiException(404)."""
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
    if lws is None:
        custom.get_namespaced_custom_object.side_effect = _Api404()
    else:
        custom.get_namespaced_custom_object.return_value = lws
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


def test_alias_resolves_to_deployment():
    deploy = {"spec": {"template": _pod_template(gpu_limit=2)}}
    apps, core, custom = _fake_apis(deploy=deploy)
    state = K8sState(apps_v1=apps, core_v1=core, custom_v1=custom)
    p = state.resolve_placement("modelforge", "fallback-modelforge-01")
    assert p is not None
    assert p.kind == "deployment"
    assert p.gpus_per_replica == 2
    # confirm alias name was used, not the serviceId
    call_args = apps.read_namespaced_deployment.call_args
    assert call_args.args[0] == "modelforge-fallback-sglang"
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
