# stage2-rollback

这是新的独立 Stage2 home，目录层级与 `mas_stage2_v4_4/` 同级。

这条线的定位非常明确：

- 它不是 `Stage2-UCC/stage2_phase3a_unified/` 的原地修补版。
- 它也不是 `Stage2-GCR+` 或 `mas_stage2_v4_4/` 的直接 fork。
- 它是一条新的、独立落码的 `pure rollback / local rerun` 路线。

代码边界要求：

- 后续 `stage2-rollback/` 下的 runtime / pipeline / verifier / operator / trainer 必须独立实现，且老代码保持不动。
- 允许把 `Stage2-UCC/stage2_phase3a_unified@362aff3`、`runtime_v2.py`、`runtime_v41.py`、`runtime_v44.py` 当作参考对象阅读。
- 不允许在正式实现中直接复用或 import 这些旧版 stage2 专有实现文件。
- 共享的底层仓库能力仅限通用依赖，例如 `mas_treesearch` 的数据加载、evaluator、agent pool、基础类型等；如果后续这些依赖也妨碍独立性，再继续下沉 fork。

正式设计见：[ROLLBACK整体设计.md](/mnt/nvme/projects/R-HAN/stage2-rollback/ROLLBACK整体设计.md)
