# 私密研究上传仓库 Agent 规范

本仓库是原 meeti 研究树的干净上传入口，远程必须是 `https://github.com/stevenson2/meeti-research-private.git`，可见性必须保持 private。

1. 上传、提交和 Git 配置在 Windows PowerShell 完成；训练与评测仍在原 WSL 研究树。不要在上传仓库自动运行训练。
2. 原研究树的提交历史不得合并、推送或用 bundle 导入本仓库。保持原研究树的 HEAD、分支、工作区和冻结产物不变。
3. 不上传 `runs/`、`data/`、原始 ECG、逐拍真值、权重、缓存、日志、论文全文、虚拟环境、凭据、会话状态；即使远程私密也遵守此规则。
4. `tools/Sync-ResearchScripts.ps1` 只复制已提交的脚本，不自动提交或推送。路径从原项目 `project_paths.json` 读取。
5. 只添加经过检查的具体文件，提交前查看暂存差异。禁止 `git add -f`、无差别暂存和 force-push。
6. `docs/source_snapshot.json` 记录脚本的来源提交与逐文件哈希。修改脚本或同步新快照时说明来源；复制快照不代表实验验收。
7. 用户要求上传其他研究实现时，可将核实过的纯代码/说明复制到本仓库，并保留来源与限制；不要因此打开旧 test 或封存 val 的波形、标签或分布。
8. 不更改实验数字或冻结裁决；文档结论必须带可核对的来源。研究工作仍遵守原项目 AGENTS.md 及其适用约束。
