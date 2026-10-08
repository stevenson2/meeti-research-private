# 上传与同步说明

私密仓库：<https://github.com/stevenson2/meeti-research-private>

## 两个工作区

原 WSL 研究树保留完整本地研究历史，继续用于训练与实验。本仓库在 Windows 工作区下，拥有独立 `main` 和 `origin`，用于上传。初始仓库只包含已提交的研究脚本和本次新增说明。

本仓库不是原研究树的完整备份。数据、权重、运行产物及冻结账本仍保留在原研究树；本次没有读取旧 test/封存 val 的原始波形或标签。

## 以后上传本仓库的文件

在本仓库目录打开 PowerShell：

```powershell
git status --short
# 示例：上传已检查的说明或报告。
git add -- docs/reports/my_report.md
git diff --cached --stat
git diff --cached
git commit -m "Add research report"
git push origin main
```

示例路径需替换为实际文件。原始波形、逐拍标签、模型文件和秘密配置不进入 Git。不要用 `git add -f`。

## 本机网络设置

首次推送时，Git 直接连接 GitHub 遇到连接重置和超时。为本仓库单独设置 Windows 当前代理后，推送成功。该设置仅在本地 `.git/config`，没有修改全局 Git 配置，也不会上传。

在当前电脑上传时，请保持该代理可用。以后换网络、停用代理或更换代理端口，可以在本仓库调整：

```powershell
# 网络已能直接访问 GitHub 时，移除本仓库的代理配置。
git config --local --unset http.proxy
```

其他电脑 clone 时按该电脑的实际网络配置，服务器仓库不携带这项本地设置。

## 同步原研究树已提交的脚本

原树作者先在原树完成明确的代码提交。随后在本仓库 PowerShell 执行：

```powershell
.\tools\Sync-ResearchScripts.ps1
git status --short
git diff --stat
git add -- scripts docs/source_snapshot.json
# 保留源中标记为可执行的脚本模式，尤其是新增加的 .sh。
$snapshot = Get-Content docs/source_snapshot.json -Raw -Encoding UTF8 | ConvertFrom-Json
$snapshot.files | Where-Object source_git_mode -eq '100755' | ForEach-Object { git update-index --chmod=+x -- $_.path }
git diff --cached --stat
git commit -m "Sync committed research scripts"
git push origin main
```

同步脚本使用相邻产品仓 `scripts/project_paths.json` 解析原研究树的 Windows 路径，也可以通过 `-PathsJson` 指定配置位置。它锁定一个源提交、复制代码并校验哈希；不运行研究脚本、不复制源 Git 历史、不自动提交或推送。

为保护本仓库的未提交脚本修改，目标 `scripts/` 不干净时同步会停止。先检查并处理这些修改。同步不会自动删除源中已删除的旧脚本；需要删除时请检查来源清单后明确操作。

原树中位于 `runs/` 的纯代码或未提交文件不会被自动同步。需要上传时，单独复制经过检查的代码到适合的目录，例如 `experiments/`，并在说明中记录原路径、版本与限制；不要复制整个运行目录。

## 其他电脑使用

```powershell
git clone https://github.com/stevenson2/meeti-research-private.git
cd meeti-research-private
```

需要登录有权限的 GitHub 账号。本次没有邀请协作者；如以后给其他人权限，先明确账号和访问范围。

## 来源与验证

初始脚本来源和哈希见 `source_snapshot.json`。该清单证明复制来源，不证明模型质量、运行成功或独立实验验收。研究结论继续引用原始有效报告和账本。
