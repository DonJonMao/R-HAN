from __future__ import annotations

from pathlib import Path

from stage2_phase3a_unified.orchestration.runner import _parser as _base_parser, run_sample_parallel


def _parser():
    parser = _base_parser()
    parser.description = "Phase2 unified local correction sample-parallel runner"
    parser.set_defaults(
        train_script=str(Path(__file__).resolve().parents[2] / "train_mas_stage2_phase2_unified_local_correction_target_suite.py")
    )
    return parser


def main(argv=None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    summary = run_sample_parallel(args)
    import json

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
