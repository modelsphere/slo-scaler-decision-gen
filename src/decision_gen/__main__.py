"""Entry point. Wires the four collaborators and starts both threads.

Process layout:
  - controller thread: 60s ticker, owns writes to the decision snapshot
  - HTTP thread:       read-only handlers over the snapshot
"""

import logging
import os

from decision_gen.controller import Controller
from decision_gen.k8s_state import K8sState
from decision_gen.metrics import Prometheus
from decision_gen.server import serve
from decision_gen.slo_store import SLOStore


def env_int(name, default):
    return int(os.environ.get(name, default))


def main():
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log = logging.getLogger("decision_gen")

    port = env_int("PORT", 8080)
    tick_seconds = env_int("TICK_SECONDS", 60)

    slo = SLOStore()
    k8s = K8sState()
    prom = Prometheus(
        base_url=os.environ.get("PROM_URL", "http://kube-prometheus-stack-prometheus.monitoring:9090"),
        timeout_s=env_int("PROM_TIMEOUT_S", 5),
    )
    controller = Controller(
        slo=slo, k8s=k8s, prom=prom, tick_seconds=tick_seconds,
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
