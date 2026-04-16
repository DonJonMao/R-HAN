# Stage2-UCC

This folder is the isolated home for the current UCC Stage2 implementation.

Layout:
- `stage2_phase3a_unified/`: UCC runtime, pipeline, verifier, correction, prompts, and orchestration glue.
- `vendor/stage2_gcr_plus/`: vendored Stage2-GCR+ package used by UCC so future UCC changes do not need to edit `Stage2-GCR+`.
- `train_mas_stage2_phase3a_unified_target_suite.py`: UCC training entrypoint.
- `train_mas_stage2_target_suite.py`: local copy of the shared target-suite training helpers used by the UCC trainer.
- `tests/`: UCC-specific tests.

Path resolution:
- UCC entrypoints prepend `Stage2-UCC/` and `Stage2-UCC/vendor/` before the repo root.
- `stage2_gcr_plus` imports inside this folder therefore resolve to the vendored copy in `Stage2-UCC/vendor/stage2_gcr_plus/`.

Scope:
- Keep UCC-specific code changes inside `Stage2-UCC/`.
- Shared libraries such as `mas_stage2/` and `mas_treesearch/` are still read from the repo root unless a future UCC-only fork is explicitly needed.
