"""Entry point. Wires the four collaborators and starts both threads.

Process layout:
  - controller thread: 60s ticker, owns writes to the decision snapshot
  - HTTP thread:       read-only handlers over the snapshot
"""

import logging
import os

from kubernetes import config as kube_config
from kubernetes.config.config_exception import ConfigException

from decision_gen.controller import Controller
from decision_gen.k8s_state import K8sState
from decision_gen.server import serve
from decision_gen.signals import Signals
from decision_gen.slo_store import SLOStore


def env_int(name, default):
    return int(os.environ.get(name, default))


def _configure_logging():
    """Apply LOG_LEVEL to our loggers only. Third-party libraries spew
    raw response bodies at DEBUG (kubernetes.client.rest dumps the full
    HTTP body of every API call), which buries our own logs. Cap them
    at WARNING so their errors still surface."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    for name in ("kubernetes", "urllib3", "requests"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main():
    _configure_logging()
    log = logging.getLogger("decision_gen")

    port = env_int("PORT", 8080)
    tick_seconds = env_int("TICK_SECONDS", 60)

    # Kubernetes client config. In-cluster when running as a pod;
    # fall back to local kubeconfig for `make run` development.
    try:
        kube_config.load_incluster_config()
        log.info("kubernetes: using in-cluster config")
    except ConfigException:
        kube_config.load_kube_config()
        log.info("kubernetes: using local kubeconfig")

    slo = SLOStore()
    k8s = K8sState()
    signals = Signals(
        base_url=os.environ.get(
            "PROM_URL",
            "http://kube-prometheus-stack-prometheus.monitoring:9090",
        ),
        timeout_s=env_int("PROM_TIMEOUT_S", 5),
    )
    controller = Controller(
        slo=slo, k8s=k8s, signals=signals, tick_seconds=tick_seconds,
    )

    try:
        slo.start()             # bg CR watch thread
        controller.start()      # bg ticker thread
        serve(port=port, snapshot=controller.snapshot)  # blocks
    finally:
        controller.stop()
        slo.stop()
        log.info("shutting down")


if __name__ == "__main__":
    main()
