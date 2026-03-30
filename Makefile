.PHONY: dev

dev:
	@trap 'kill %1; kill %2' SIGINT; \
	julia --project=. scripts/serve.jl & \
	cd frontend && npm run dev & \
	wait
