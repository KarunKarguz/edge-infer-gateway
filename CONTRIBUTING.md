# Contributing

Thanks for your interest in improving edge-infer-gateway! To keep the project predictable and safe for edge deployments, please follow the checklist below before opening a pull request.

## Workflow
- Fork the repository and create a feature branch (`feature/<short-description>`).
- Keep contributions focused; prefer smaller PRs that address a single issue.
- Run `clang-format`/`cmake --build` for C++ changes and `ruff`/`black` if you use formatters locally (optional, but appreciated).
- Ensure all automated tests pass (`pytest` plus any platform-specific checks you add).
- Include documentation updates when behaviour changes (README, docs/, sample configs).

## Coding Guidelines
- Source files must begin with the SPDX header `Apache-2.0` (already present in existing files).
- Favour clear, maintainable code—avoid hidden side effects.
- Keep configuration defaults safe for edge deployments (no credentials, sensible timeouts).
- When adding new models/pipelines, provide example configs or docs to help adopters reproduce results.

## Testing
- Run the orchestrator integration suite:
  ```bash
  python3 -m venv .venv && . .venv/bin/activate
  pip install -r requirements.dev.txt
  pytest
  ```
- For C++ runtime changes, run the existing sample clients against a running gateway.
- If you touch Docker packaging, rebuild the container locally before submitting.

## Reporting Issues
- Use GitHub Issues; include hardware (GPU/SoC), CUDA/TensorRT versions, and reproduction steps.
- Attach logs (`gateway` JSON lines, orchestrator metrics) when possible.

We appreciate your contributions—thank you for helping push edge AI forward!
