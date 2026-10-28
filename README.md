# meeti-research-private

ESP32 单导联可穿戴 ECG 项目的私密研究代码仓库。

- 私密远程：<https://github.com/stevenson2/meeti-research-private>
- 产品仓：<https://github.com/stevenson2/ecg-programme-tju-flex.electron>
- 本仓库从 2026-10-08 建立独立提交历史，初始内容为本机研究树已提交的 `scripts/` 快照。
- 原研究树继续负责本机训练、评测与冻结产物。本仓库负责可上传的代码与经检查的研究说明。

## 内容与来源

`scripts/` 来自原研究树的明确 Git 提交，逐文件来源和 SHA-256 见 `docs/source_snapshot.json`。复制不执行这些脚本；初次上传不构成模型或实验验收。

这里没有复制原研究树的 Git 历史，也没有上传 `runs/`、原始 ECG、训练缓存、权重、日志、论文全文或会话状态。原研究树的历史中包含项目规定永不推送的文件，不能直接把它的历史推到本仓库。

初始快照仅包含已提交的研究脚本；原树未提交的新脚本和位于 `runs/` 的实验实现不在初始快照内。未来需要上传这些实现时，应挑选代码与说明文件，并检查其数据边界。

## 使用

在本仓库目录打开 Windows PowerShell，普通上传流程：

```powershell
git status --short
git add -- <准备上传的具体文件或目录>
git diff --cached --stat
git commit -m "Describe the research change"
git push origin main
```

尖括号中的内容需要替换为实际路径。研究树脚本快照的同步方式和完整上传说明见 [docs/UPLOAD_GUIDE.md](docs/UPLOAD_GUIDE.md)。

本仓库没有内置数据或运行环境；复制的历史脚本仍需原项目路径配置与依赖。目录存在不代表脚本均可独立运行。

## 本地文件边界

`.gitignore` 覆盖原始数据、实验目录、模型文件、缓存、环境、日志、论文全文和会话状态。研究报告可以放在 `docs/reports/`，确认不包含原始波形、逐拍真值或敏感配置后再明确添加。

不使用 `git add -f` 绕过忽略规则，不向本仓库推送原研究树分支或 bundle，不推 tag，不 force-push `main`。

## 最近研究阶段

- [2026-10-28：新增200患者的N/V四格三种子与一次独立评价](docs/reports/nv_expand200_20261028/README.md)。包含经检查的代码、聚合报告、图表及来源哈希；不包含数据、权重或会话状态。
