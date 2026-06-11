"""QSWorkflow equivalent — automate the end-to-end process.

:class:`Pipeline` wires the other three layers into one repeatable run:

    ingest  -> QSConnect : pull data from a source into the store
    research-> QSResearch: generate signals and backtest them
    execute -> Omega     : turn the latest target position into a broker order

Run the whole thing with ``Pipeline(...).run()`` or call the stages
individually. By default it uses synthetic data, a local store, and a paper
broker, so a full run needs no credentials.
"""

from qstack.workflow.pipeline import Pipeline, PipelineResult

__all__ = ["Pipeline", "PipelineResult"]
