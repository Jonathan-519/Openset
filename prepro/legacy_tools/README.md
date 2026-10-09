# 历史数据工具

这些工具从原 `prepro/` 主目录移入，算法和历史固定划分保留，入口改为：

```bash
python -m prepro.legacy_tools.build_taxosafe_v11_known_splits --help
python -m prepro.legacy_tools.rebuild_taxosafe_v9_dataset --help
python -m prepro.legacy_tools.refresh_filenames --help
```

当前 TaxoSieve 使用 `prepro/build_taxosieve_dataset.py`。不要在当前图像上运行历史 v9 移动工具，也不要用17叶工具覆盖23叶分类树。历史默认目录、固定数量及数据协议只适用于对应版本；使用独立项目/配置/输出目录恢复历史实验。

`refresh_filenames.py` 仅允许原3157图像内容守恒的纯文件名变更。其固定基准存放于 `reproducibility/history/filename_baseline.json`；当前样本数量已经改变，不适用该工具。原运行备份保存在 `reproducibility/history/filename_refresh/`，内部旧路径记录作为历史证据保留原文，不应当作当前路径执行。
