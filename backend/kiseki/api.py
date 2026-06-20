from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .experiment import ExperimentManager
from .schemas import ExperimentStatus, SchemaResponse, StartExperimentRequest, schema_response


def create_app(manager: ExperimentManager | None = None) -> FastAPI:
    app = FastAPI(title="Kiseki Backend")
    experiment_manager = manager or ExperimentManager()

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

    @app.post("/api/experiments/start", response_model=ExperimentStatus)
    def start_experiment(request: StartExperimentRequest) -> ExperimentStatus:
        try:
            return experiment_manager.start(request)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/experiments/stop", response_model=ExperimentStatus)
    def stop_experiment() -> ExperimentStatus:
        return experiment_manager.stop()

    @app.post("/api/experiments/pause", response_model=ExperimentStatus)
    def pause_experiment() -> ExperimentStatus:
        try:
            return experiment_manager.pause()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/experiments/resume", response_model=ExperimentStatus)
    def resume_experiment() -> ExperimentStatus:
        try:
            return experiment_manager.resume()
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
