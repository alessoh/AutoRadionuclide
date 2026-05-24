"""Outward experiment-request contract — the only thing the reasoning layer pushes out.

Connecting a real wet lab means implementing WetLabInterface and passing it
to the planner. The stub is the default in-silico implementation.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from autoradionuclide.domain.models import ExperimentRequest, ResultRecord


class WetLabInterface(ABC):
    """The single interface a real facility must implement.

    Replace StubWetLab with a class that talks to your LIMS, robotics
    scheduler, or partner API. Nothing in the reasoning engine changes.
    """

    @abstractmethod
    def submit(self, request: ExperimentRequest) -> str:
        """Submit a batch request; return an opaque job id."""
        ...

    @abstractmethod
    def poll(self, job_id: str) -> ResultRecord | None:
        """Return the result when ready, else None."""
        ...

    @abstractmethod
    def submit_and_wait(self, request: ExperimentRequest) -> ResultRecord:
        """Blocking submit — the inner loop calls this."""
        ...
