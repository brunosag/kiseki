from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse

from .analysis import AnalysisComparisonError, AnalysisParameterError
from .checkpoint import CheckpointNotFoundError
from .experiment import ExperimentManager
from .schemas import (
    AnalysisComparisonJobRequest,
    AnalysisComparisonJobStatus,
    AnalysisComparisonReport,
    CheckpointListMode,
    CheckpointSelection,
    CheckpointSummary,
    ExperimentControlsUpdate,
    ExperimentStatus,
    SchemaResponse,
    StartExperimentRequest,
    schema_response,
)


def create_app(manager: ExperimentManager | None = None) -> FastAPI:
    app = FastAPI(title="Kiseki Backend")
    experiment_manager = manager or ExperimentManager()

    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/schema", response_model=SchemaResponse)
    def get_schema() -> SchemaResponse:
        return schema_response()

    @app.get("/api/checkpoints", response_model=list[CheckpointSummary])
    def list_checkpoints(mode: CheckpointListMode = "training") -> list[CheckpointSummary]:
        return experiment_manager.checkpoints(mode)

    @app.delete("/api/checkpoints/{run_id}", status_code=204)
    def delete_checkpoint(run_id: str) -> None:
        try:
            experiment_manager.delete_checkpoint(run_id)
        except CheckpointNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/checkpoints/load", response_model=ExperimentStatus)
    def load_checkpoint(selection: CheckpointSelection) -> ExperimentStatus:
        try:
            return experiment_manager.load_checkpoint(selection)
        except CheckpointNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post(
        "/api/analysis/comparisons/jobs",
        response_model=AnalysisComparisonJobStatus,
    )
    def create_analysis_comparison_job(
        request: AnalysisComparisonJobRequest,
    ) -> AnalysisComparisonJobStatus:
        try:
            return experiment_manager.create_analysis_comparison_job(request)
        except CheckpointNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (AnalysisComparisonError, AnalysisParameterError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get(
        "/api/analysis/comparisons/jobs/{job_id}",
        response_model=AnalysisComparisonJobStatus,
    )
    def get_analysis_comparison_job(job_id: str) -> AnalysisComparisonJobStatus:
        try:
            return experiment_manager.get_analysis_comparison_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get(
        "/api/analysis/comparisons/jobs/{job_id}/report",
        response_model=AnalysisComparisonReport,
    )
    def get_analysis_comparison_report(job_id: str) -> AnalysisComparisonReport:
        try:
            return experiment_manager.get_analysis_comparison_report(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except AnalysisComparisonError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/analysis/comparisons/jobs/{job_id}/events")
    def stream_analysis_comparison_events(job_id: str) -> StreamingResponse:
        try:
            events = experiment_manager.analysis_comparison_events(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return StreamingResponse(
            events,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.post("/api/experiments/start", response_model=ExperimentStatus)
    def start_experiment(request: StartExperimentRequest) -> ExperimentStatus:
        try:
            return experiment_manager.start(request)
        except CheckpointNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/experiments/stop", response_model=ExperimentStatus)
    def stop_experiment() -> ExperimentStatus:
        return experiment_manager.stop()

    @app.post("/api/experiments/reset", response_model=ExperimentStatus)
    def reset_experiment() -> ExperimentStatus:
        try:
            return experiment_manager.reset()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/experiments/pause", response_model=ExperimentStatus)
    def pause_experiment() -> ExperimentStatus:
        try:
            return experiment_manager.pause()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/experiments/resume", response_model=ExperimentStatus)
    def resume_experiment(update: ExperimentControlsUpdate | None = None) -> ExperimentStatus:
        try:
            return experiment_manager.resume(update)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/experiments/status", response_model=ExperimentStatus)
    def get_status() -> ExperimentStatus:
        return experiment_manager.status()

    @app.get("/api/experiments/events")
    def stream_events() -> StreamingResponse:
        return StreamingResponse(
            experiment_manager.events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return app
