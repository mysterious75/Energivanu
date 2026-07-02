# Changelog

All notable changes to Energivanu are documented here.

## [Unreleased]

### Added
- 103 tests (up from 13) covering API, BESS, grid, optimizer, scheduler, and telemetry
- GPU Docker profile (`docker-compose --profile gpu up`)
- `Dockerfile.gpu` for CUDA 12.4-based GPU inference
- `CONTRIBUTING.md`, `CHANGELOG.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`
- GitHub issue templates (bug report, feature request) and PR template
- CI/CD badge and test count badge in README
- **Multi-GPU DDP training** (`src/energivanu/distributed.py`)
  - `torchrun`-compatible: `torchrun --nproc_per_node=N -m energivanu.train_commercial --distributed`
  - `DistributedSampler`, automated batch size scaling, rank-aware checkpoint saving
  - Works seamlessly in single-GPU mode (no flags needed)

### Changed
- README test count updated from 13 to 103 passing, 8 skipped
- WHITEPAPER.md corrected: MPC uses heuristic trajectory optimization, not CVXPY/QP
- README validation disclaimer expanded for transparency
- MPC_IMPLEMENTATION.md corrected to reflect grid-search approach

### Fixed
- MPC documentation falsely claimed CVXPY/OSQP solver usage
- docker-compose.yml lacked GPU passthrough for GPU-accelerated inference
- Critical test coverage gap (only 13 tests existed)

## [0.1.0] - 2026-06-01

### Added
- Initial release
- TCN+Attention power prediction model
- BESS MPC heuristic controller
- OpenADR 2.0b VEN client
- ERCOT SCED parser with PCLR compliance
- Phase staggering scheduler
- GPU telemetry collector (nvidia-smi)
- Docker deployment
- Dual licensing (AGPL-3.0 + Commercial)
- Professional magazine publication
- ROI calculator
