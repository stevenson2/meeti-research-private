#!/bin/bash
set -e
cd /home/devcontainers/meeti
git config --global credential.helper "/mnt/d/Git/mingw64/bin/git-credential-manager.exe"
rm -rf /tmp/meeti_pub
git clone -q /home/devcontainers/meeti /tmp/meeti_pub
cd /tmp/meeti_pub
git config user.name "devcontainers"
git config user.email "devcontainers@local"
git checkout -q --orphan meeti-research
git rm -rq --cached .
git add -A
git rm -rqf papers_fulltext
git add -A
git commit -qm "meeti research snapshot: unified benchmark v1.2 (24 models) + r9_form_ptbbal candidate lineage + scripts/reports (paper fulltexts excluded)"
echo "paper files in snapshot: $(git ls-files | grep -icE 'papers_fulltext|\.pdf' || true)"
git remote add gh https://github.com/stevenson2/ecg-programme-tju-flex.electron.git
git push gh meeti-research:meeti-research 2>&1 | tail -3
echo PUSH_DONE
