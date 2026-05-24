"""Computational experiment stub — the in-silico stand-in for the wet lab.

FROZEN — this stub is part of the frozen harness. Do not modify.
"""
from __future__ import annotations
import random, uuid
from datetime import datetime, timezone
from autoradionuclide.domain.models import (
    ExperimentRequest, ResultRecord, AssayResult, CandidateStatus
)
from autoradionuclide.interfaces.contract import WetLabInterface
import frozen.harness as harness


class StubWetLab(WetLabInterface):
    """Returns simulated results instantly by running the frozen scoring functions.

    In a real system, replace this with an adapter to your LIMS or robotics scheduler.
    The ResultRecord schema and WetLabInterface contract remain unchanged.
    """

    def __init__(self, noise_scale: float = 0.05, seed: int = 42) -> None:
        self._noise_scale = noise_scale
        self._rng = random.Random(seed)
        self._pending: dict[str, ExperimentRequest] = {}

    def submit(self, request: ExperimentRequest) -> str:
        job_id = str(uuid.uuid4())
        self._pending[job_id] = request
        return job_id

    def poll(self, job_id: str) -> ResultRecord | None:
        req = self._pending.pop(job_id, None)
        if req is None:
            return None
        return self._simulate(req)

    def submit_and_wait(self, request: ExperimentRequest) -> ResultRecord:
        job_id = self.submit(request)
        return self.poll(job_id)

    def _simulate(self, request: ExperimentRequest) -> ResultRecord:
        assay_results = []
        for construct in request.constructs:
            scores = harness.score_all(construct)
            for obj_name, obj_val in scores.items():
                noise = self._rng.gauss(0, self._noise_scale)
                measured = max(0.0, min(1.0, obj_val.estimate + noise))
                assay_results.append(AssayResult(
                    assay_name=obj_name,
                    construct_id=construct.id,
                    value=measured,
                    unit="normalized_score",
                    uncertainty=abs(noise) + obj_val.uncertainty,
                    passed=measured >= 0.3,
                ))
        return ResultRecord(
            id=str(uuid.uuid4()),
            request_id=request.id,
            campaign_id=request.campaign_id,
            cycle_id=request.cycle_id,
            construct_results=assay_results,
            radiochemical_yield=self._rng.uniform(0.6, 0.98),
            radiochemical_purity=self._rng.uniform(0.85, 0.99),
        )
