from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from mas_stage2 import PreparedStage1Artifact, load_prepared_stage1_artifact


@dataclass
class RollbackPreparedStage1Artifact:
    base_prepared: Optional[PreparedStage1Artifact]
    anchor_artifact: Dict[str, Any]
    rollback_trace: Dict[str, Any]
    anchor_replay_cache: Dict[str, Any] = field(default_factory=dict)


def load_optional_prepared_artifact(path: Optional[str]) -> Optional[PreparedStage1Artifact]:
    if not path:
        return None
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None
    return load_prepared_stage1_artifact(str(artifact_path))
