"""Kubernetes reads: workload resolution, placement, pool capacity.

No watches — each controller tick re-reads what it needs. All functions
return plain values; no kubernetes-client objects escape this module.

Workloads come in two access patterns:
  - Deployment / StatefulSet (typed clients, attribute-style access)
  - LeaderWorkerSet (CustomObjectsApi, dict-style access)

Helpers normalize both into "pick the right template dict; compute GPU
limits; find pool affinity" so the downstream code is uniform.
"""

import logging
import os
import time
from dataclasses import dataclass

from kubernetes import client
from kubernetes.client.rest import ApiException

log = logging.getLogger(__name__)


LWS_GROUP = "leaderworkerset.x-k8s.io"
LWS_VERSION = "v1"
LWS_PLURAL = "leaderworkersets"

# DisaggregatedSet (PD-disagg bundles). The `ds` shortname on the cluster
# currently resolves to disaggregatedsetrolescalers, so all client calls
# must use the full plural here.
DS_GROUP = "disaggregatedset.x-k8s.io"
DS_VERSION = "v1"
DS_PLURAL = "disaggregatedsets"

GPU_LIMIT_KEY = "nvidia.com/gpu"
GPU_PRODUCT_LABEL = "nvidia.com/gpu.product"
GPU_PRESENT_LABEL = "nvidia.com/gpu.present"

DEFAULT_GPU_POOL = os.environ.get("DEFAULT_GPU_POOL", "NVIDIA-H100-80GB-HBM3")

# TEMP-REVERT — H800-as-H100 pool merge. To revoke: delete POOL_ALIASES
# and the `_canonical_pool()` indirection below (revert to direct product
# lookups). Real fix is polymorphic pool placement, an R10 contract change;
# filed against scaling-requirements R10. Workloads pinned to H800-only
# get re-pooled into H100 by this alias — acceptable iff H800-compat
# workloads may also schedule on H100. See commit that introduced this.
POOL_ALIASES = {
    "NVIDIA-H800": "NVIDIA-H100-80GB-HBM3",
}


def _canonical_pool(product):
    """TEMP-REVERT — see POOL_ALIASES above."""
    return POOL_ALIASES.get(product, product)


@dataclass(frozen=True)
class Placement:
    """Per-service contract read from its workload template."""
    namespace: str
    name: str
    kind: str                # "deployment" | "statefulset" | "lws" | "disaggregatedset"
    pool: str
    gpus_per_replica: int
    workload_name: str = ""  # actual workload name (after alias)
    spec_replicas: int = 0   # declared replicas on the workload spec
    # Member workloads composing this placement — used for log tracing.
    # Empty means "single workload" — consumers see it as ((kind, workload_name),).
    # DisaggregatedSet bundles pass the per-role LWS list explicitly.
    members: tuple = ()

    def member_workloads(self):
        """Return the (kind, name) tuple list. Defaults to the single-workload
        identity for non-DS placements."""
        if self.members:
            return self.members
        return ((self.kind, self.workload_name or self.name),)


def _to_dict(obj):
    """kubernetes-python objects expose .attribute_map; some custom
    objects come back as dicts already. Normalize to dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return {}


class K8sState:
    # Hardcoded aliases: (namespace, service_id) -> workload name.
    # Consulted first during resolution. Remove as the cluster fixes
    # workload naming to match serviceId. NOTE: a stale alias that points
    # at a deleted/renamed workload is worse than no alias — resolution
    # will try the alias name (404s everywhere) and never try serviceId.
    WORKLOAD_ALIASES = {}

    def __init__(self, apps_v1=None, core_v1=None, custom_v1=None):
        """Dependency-injectable for tests."""
        self._apps = apps_v1
        self._core = core_v1
        self._custom = custom_v1

    # ---------- lazy api construction ----------

    def _apps_api(self):
        if self._apps is None:
            t = time.monotonic()
            self._apps = client.AppsV1Api()
            log.info("k8s api init: AppsV1Api %.3fs", time.monotonic() - t)
        return self._apps

    def _core_api(self):
        if self._core is None:
            t = time.monotonic()
            self._core = client.CoreV1Api()
            log.info("k8s api init: CoreV1Api %.3fs", time.monotonic() - t)
        return self._core

    def _custom_api(self):
        if self._custom is None:
            t = time.monotonic()
            self._custom = client.CustomObjectsApi()
            log.info("k8s api init: CustomObjectsApi %.3fs", time.monotonic() - t)
        return self._custom

    # ---------- resolve_placement ----------

    def resolve_placement(self, namespace, service_id):
        """Return Placement or None if the service is unmanageable this tick.

        Order: alias → DisaggregatedSet → LWS → StatefulSet → Deployment.
        """
        target_name = self.WORKLOAD_ALIASES.get((namespace, service_id), service_id)

        # 0. DisaggregatedSet (PD-disagg bundle; disaggregatedset.x-k8s.io/v1)
        ds = self._try_get_ds(namespace, target_name)
        if ds is not None:
            placement = self._placement_from_ds(namespace, target_name, ds)
            if placement is not None:
                log.info("%s/%s: ds pool=%s gpr=%d spec_replicas=%d members=%s",
                          namespace, service_id, placement.pool,
                          placement.gpus_per_replica, placement.spec_replicas,
                          placement.members)
                return placement
            # DS exists but malformed → unmanageable; do not fall through
            log.warning(
                "DisaggregatedSet %s/%s exists but malformed; service %s unmanageable this tick",
                namespace, target_name, service_id,
            )
            return None

        # 1. LWS (dict-based; leaderworkerset.x-k8s.io/v1)
        lws = self._try_get_lws(namespace, target_name)
        if lws is not None:
            placement = self._placement_from_lws(namespace, target_name, lws)
            if placement is not None:
                log.info("%s/%s: lws pool=%s gpr=%d spec_replicas=%d",
                          namespace, service_id, placement.pool,
                          placement.gpus_per_replica, placement.spec_replicas)
                return placement
            # LWS exists but is malformed → unmanageable here; don't fall through
            log.warning(
                "LWS %s/%s exists but malformed; service %s unmanageable this tick",
                namespace, target_name, service_id,
            )
            return None

        # 2. StatefulSet
        sts = self._try_get_sts(namespace, target_name)
        if sts is not None:
            sts_spec = _to_dict(sts).get("spec", {})
            placement = self._placement_from_pod_template(
                namespace, target_name, "statefulset",
                sts_spec.get("template", {}),
                # apiserver defaults spec.replicas to 1 server-side;
                # mirror that in case a test fake omits it.
                spec_replicas=int(sts_spec.get("replicas", 1)),
            )
            if placement is not None:
                log.info("%s/%s: sts pool=%s gpr=%d spec_replicas=%d",
                          namespace, service_id, placement.pool,
                          placement.gpus_per_replica, placement.spec_replicas)
                return placement
            log.warning(
                "StatefulSet %s/%s exists but malformed; service %s unmanageable this tick",
                namespace, target_name, service_id,
            )
            return None

        # 3. Deployment
        deploy = self._try_get_deploy(namespace, target_name)
        if deploy is not None:
            deploy_spec = _to_dict(deploy).get("spec", {})
            placement = self._placement_from_pod_template(
                namespace, target_name, "deployment",
                deploy_spec.get("template", {}),
                spec_replicas=int(deploy_spec.get("replicas", 0)),
            )
            if placement is not None:
                log.info("%s/%s: deploy pool=%s gpr=%d spec_replicas=%d",
                          namespace, service_id, placement.pool,
                          placement.gpus_per_replica, placement.spec_replicas)
                return placement
            log.warning(
                "Deployment %s/%s exists but malformed; service %s unmanageable this tick",
                namespace, target_name, service_id,
            )
            return None

        log.info("no workload found for %s/%s", namespace, service_id)
        return None

    def _try_get_lws(self, namespace, name):
        try:
            return self._custom_api().get_namespaced_custom_object(
                LWS_GROUP, LWS_VERSION, namespace, LWS_PLURAL, name,
            )
        except ApiException as e:
            if e.status == 404:
                return None
            raise

    def _try_get_ds(self, namespace, name):
        try:
            return self._custom_api().get_namespaced_custom_object(
                DS_GROUP, DS_VERSION, namespace, DS_PLURAL, name,
            )
        except ApiException as e:
            if e.status == 404:
                return None
            if e.status == 403:
                # Missing ClusterRole rule for disaggregatedsets. Fall through
                # to LWS/STS/Deploy so only this service is unmanaged; raising
                # would freeze every service's decision this tick. Call out
                # the fix explicitly — the follow-up log ("no workload found")
                # does not hint at RBAC.
                log.warning(
                    "forbidden reading DS %s/%s (rule %s %s): treating as absent; "
                    "falling back to LWS/STS/Deploy. Grant get/list/watch on %s to fix",
                    namespace, name, DS_GROUP, DS_PLURAL, DS_PLURAL + "." + DS_GROUP,
                )
                return None
            raise

    def _try_get_sts(self, namespace, name):
        try:
            return self._apps_api().read_namespaced_stateful_set(name, namespace)
        except ApiException as e:
            if e.status == 404:
                return None
            raise

    def _try_get_deploy(self, namespace, name):
        try:
            return self._apps_api().read_namespaced_deployment(name, namespace)
        except ApiException as e:
            if e.status == 404:
                return None
            raise

    # ---------- placement extraction ----------

    def _placement_from_ds(self, namespace, ds_name, ds_obj):
        """DisaggregatedSet (PD-disagg bundle).

        A DS "replica" = one slice = a bundle of
          (P prefill LWS groups × lwt.size pods each)
        + (D decode  LWS groups × lwt.size pods each)
        + optional CPU-only roles (router), which contribute 0 GPUs.

        Per-bundle GPU cost (gpus_per_replica):
          Σ over GPU-requesting roles of
            role.spec.replicas × leaderWorkerTemplate.size × per_pod_gpus.

        The bundle knob is `spec.slices` — scaling the service means
        bumping slices. Per-role `replicas` is the P:D ratio inside one
        slice. (Verified on live `vllm-qwen36-2p1d` in test-dsset: slices=1,
        prefill.replicas=2 spawn single LWS `…-0-…-prefill` with replicas=2.)
        spec_replicas therefore reads slices, NOT any role's replicas.
        """
        spec = ds_obj.get("spec", {}) or {}
        # None-safe int: spec field defaulted by CRD can still be null if a
        # test fixture or hand-edited object omits it; int(None) TypeErrors
        # and aborts the whole tick.
        def _int(d, key, default):
            v = d.get(key)
            return default if v is None else int(v)

        slices = _int(spec, "slices", 1)
        roles = spec.get("roles", []) or []
        if not roles:
            log.warning("DS %s/%s has empty spec.roles; unmanageable",
                        namespace, ds_name)
            return None

        pools = set()
        gpus_per_slice = 0
        members = []
        for role in roles:
            role_name = role.get("name", "") or ""
            role_spec = role.get("spec", {}) or {}
            role_replicas = _int(role_spec, "replicas", 1)
            lwt = role_spec.get("leaderWorkerTemplate", {}) or {}
            lwt_size = _int(lwt, "size", 1)
            leader_tmpl = lwt.get("leaderTemplate", {}) or {}
            worker_tmpl = lwt.get("workerTemplate", {}) or {}

            leader_place = self._placement_from_pod_template(
                namespace, ds_name, "lws",
                {"spec": leader_tmpl.get("spec", {})},
            )
            worker_place = self._placement_from_pod_template(
                namespace, ds_name, "lws",
                {"spec": worker_tmpl.get("spec", {})},
            )
            # Zero-GPU role (a CPU router folded into the DS) contributes 0
            # to the GPU pool and the bundle remains manageable. Note that
            # _placement_from_pod_template returns None silently on all its
            # failure paths (no containers, no GPU limits, ambiguous pool),
            # so "both None" can also mean "templates are malformed" — the
            # apiserver usually rejects a DS whose templates have no
            # containers before we'd see it, so we accept the ambiguity
            # rather than complicate the helper's return type.
            if leader_place is None and worker_place is None:
                members.append(("lws", f"{ds_name}-{role_name}"))
                continue
            if leader_place is None or worker_place is None:
                log.warning(
                    "DS %s/%s role=%s: only one of leader/worker templates is GPU-less; unmanageable",
                    namespace, ds_name, role_name,
                )
                return None
            if leader_place.pool != worker_place.pool:
                log.warning(
                    "DS %s/%s role=%s leader/worker pools disagree (%s vs %s)",
                    namespace, ds_name, role_name,
                    leader_place.pool, worker_place.pool,
                )
                return None
            if leader_place.gpus_per_replica != worker_place.gpus_per_replica:
                log.warning(
                    "DS %s/%s role=%s leader/worker gpus_per_pod differ (%d vs %d)",
                    namespace, ds_name, role_name,
                    leader_place.gpus_per_replica, worker_place.gpus_per_replica,
                )
                return None

            pools.add(leader_place.pool)
            per_group = lwt_size * leader_place.gpus_per_replica
            gpus_per_slice += role_replicas * per_group
            members.append(("lws", f"{ds_name}-{role_name}"))

        if len(pools) != 1:
            log.warning(
                "DS %s/%s roles span multiple pools %s; unmanageable",
                namespace, ds_name, sorted(pools),
            )
            return None
        if gpus_per_slice <= 0:
            # All roles were CPU-only (e.g. only-router DS) — no GPU pool
            # consumption to ledger. Mirror the "filters sidecars" rule in
            # _placement_from_pod_template and refuse to manage.
            log.warning("DS %s/%s has no GPU-requesting roles; unmanageable",
                        namespace, ds_name)
            return None
        pool = next(iter(pools))

        return Placement(
            namespace=namespace, name=ds_name, kind="disaggregatedset",
            pool=pool,
            gpus_per_replica=gpus_per_slice,
            workload_name=ds_name,
            spec_replicas=slices,
            members=tuple(members),
        )

    def _placement_from_lws(self, namespace, workload_name, lws_obj):
        """LWS: gpus_per_replica = leaderWorkerTemplate.size × per-pod GPUs."""
        spec = lws_obj.get("spec", {}) or {}
        lwt = spec.get("leaderWorkerTemplate", {}) or {}
        size = int(lwt.get("size", 1))
        replicas = int(spec.get("replicas", 1))
        leader_tmpl = lwt.get("leaderTemplate", {}) or {}
        worker_tmpl = lwt.get("workerTemplate", {}) or {}

        leader_place = self._placement_from_pod_template(
            namespace, workload_name, "lws",
            {"spec": leader_tmpl.get("spec", {})},
        )
        worker_place = self._placement_from_pod_template(
            namespace, workload_name, "lws",
            {"spec": worker_tmpl.get("spec", {})},
        )
        if leader_place is None or worker_place is None:
            return None
        if leader_place.pool != worker_place.pool:
            log.warning(
                "LWS %s/%s leader/worker pools disagree (%s vs %s); unmanageable",
                namespace, workload_name, leader_place.pool, worker_place.pool,
            )
            return None
        per_pod = leader_place.gpus_per_replica
        if per_pod != worker_place.gpus_per_replica:
            log.warning(
                "LWS %s/%s leader/worker gpus_per_pod differ (%d vs %d); unmanageable",
                namespace, workload_name, per_pod, worker_place.gpus_per_replica,
            )
            return None
        return Placement(
            namespace=namespace, name=workload_name, kind="lws",
            pool=leader_place.pool,
            gpus_per_replica=size * per_pod,
            workload_name=workload_name,
            spec_replicas=replicas,
        )

    def _placement_from_pod_template(self, namespace, workload_name, kind, template,
                                     spec_replicas=0):
        """Shared for Deployment / StatefulSet / one LWS member template."""
        spec = template.get("spec", {}) or {}
        containers = spec.get("containers", []) or []
        if not containers:
            return None
        # Sum the per-container GPU limits to get pod-level accounting.
        # Colocate (P+D engines in one pod) splits GPUs across engine
        # containers — max() would undercount; sum() matches the pod's
        # scheduler-admission total. Sidecars with no GPU limit contribute 0.
        total_gpus = 0
        for c in containers:
            limits = ((c.get("resources") or {}).get("limits")) or {}
            v = limits.get(GPU_LIMIT_KEY, 0)
            try:
                v = int(v)
            except (TypeError, ValueError):
                v = 0
            total_gpus += v
        if total_gpus <= 0:
            return None  # filters sidecars (cart-*) with no GPU

        # Pool decision:
        #   AMBIGUOUS → refuse to guess (workload unmanageable this tick).
        #   None      → no GPU-product pin; use v0 default. Note that a
        #               workload with *other* node affinity (disk, zone,
        #               hostname) but no GPU-product selector also lands
        #               here — that's correct: the default is about GPU
        #               booking, not about other constraints.
        #   str       → the pinned pool.
        pool = self._extract_pool_from_affinity(spec.get("affinity"))
        if pool is K8sState.AMBIGUOUS:
            return None
        if pool is None:
            log.info("%s/%s: no GPU affinity; defaulting pool=%s",
                     namespace, workload_name, DEFAULT_GPU_POOL)
            pool = DEFAULT_GPU_POOL
        return Placement(
            namespace=namespace, name=workload_name, kind=kind,
            pool=pool, gpus_per_replica=total_gpus,
            workload_name=workload_name,
            spec_replicas=spec_replicas,
        )

    # Sentinel returned by _extract_pool_from_affinity when the workload
    # pins to ≥2 distinct GPU pools after canonicalization — genuinely
    # ambiguous, refuse to guess. Distinct from "no GPU-product pin",
    # which falls through to DEFAULT_GPU_POOL.
    AMBIGUOUS = object()

    @staticmethod
    def _extract_pool_from_affinity(affinity):
        """Pool = the single GPU-product value the workload pins to.

        Returns:
          pool name (str)      — one canonical pool after alias merge.
          None                 — no GPU-product `In` selector found.
                                 Callers treat this as "unconstrained";
                                 default rules may apply.
          K8sState.AMBIGUOUS   — GPU-product selector listed ≥2 canonical
                                 pools. Callers must refuse to place.

        kubernetes-python's `.to_dict()` uses the *attribute* name
        (snake_case) for nested sub-objects, not the JSON key (camelCase).
        LWS custom objects come through as plain dicts — JSON keys. Accept
        both.
        """
        if not affinity:
            return None
        node_aff = (affinity.get("nodeAffinity")
                    or affinity.get("node_affinity") or {})
        req = (node_aff.get("requiredDuringSchedulingIgnoredDuringExecution")
               or node_aff.get("required_during_scheduling_ignored_during_execution")
               or {})
        terms = (req.get("nodeSelectorTerms")
                 or req.get("node_selector_terms") or [])
        values = []
        for t in terms:
            mes = (t.get("matchExpressions")
                   or t.get("match_expressions") or [])
            for me in mes:
                if me.get("key") == GPU_PRODUCT_LABEL and me.get("operator") == "In":
                    values.extend(me.get("values") or [])
        # TEMP-REVERT — map each candidate product through POOL_ALIASES
        # so {H100}, {H800}, and {H100, H800} all collapse to a single
        # canonical pool. To revoke: drop the _canonical_pool wrapper,
        # restore `if len(values) == 1: return values[0]`.
        values = sorted(set(values))
        if not values:
            return None
        canonical = {_canonical_pool(v) for v in values}
        if len(canonical) == 1:
            return canonical.pop()
        return K8sState.AMBIGUOUS

    # ---------- pool_capacity ----------

    def pool_capacity(self):
        """Return {pool: allocatable_gpus} across Ready & uncordoned GPU nodes.

        This is *gross* capacity — what we subtract against is the
        controller's per-service ledger (last-served decisions), not
        live pod consumption. See controller._tick.

        Reads `status.allocatable`, not labels: at least one live node
        advertises gpu.count=8 in labels but only 7 allocatable after a
        DevicePlugin partial failure; the scheduler trusts allocatable,
        so we do too. Cordoned and NotReady nodes contribute zero.
        """
        core = self._core_api()
        t = time.monotonic()
        nodes = core.list_node().items or []
        log.info("k8s list_node: %.3fs, %d nodes", time.monotonic() - t, len(nodes))

        pools = {}
        for n in nodes:
            labels = getattr(n.metadata, "labels", None) or {}
            if labels.get(GPU_PRESENT_LABEL) != "true":
                continue
            if getattr(n.spec, "unschedulable", False):
                continue
            conditions = getattr(n.status, "conditions", None) or []
            if not any(c.type == "Ready" and c.status == "True" for c in conditions):
                continue
            product = labels.get(GPU_PRODUCT_LABEL)
            if not product:
                continue
            alloc = getattr(n.status, "allocatable", None) or {}
            try:
                gpus = int(alloc.get(GPU_LIMIT_KEY, 0))
            except (TypeError, ValueError):
                gpus = 0
            # TEMP-REVERT — fold aliased products (H800) into their
            # canonical pool so `pools` has a single entry for the
            # merged hardware class. To revoke: use `product` directly.
            pools[_canonical_pool(product)] = pools.get(_canonical_pool(product), 0) + gpus
        if not pools:
            log.warning("pool_capacity: no GPU nodes found")
        return pools
