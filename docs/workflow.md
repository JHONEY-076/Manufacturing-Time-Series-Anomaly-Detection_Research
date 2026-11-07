# 저장과 업데이트

전체 실험 코드는 최상위 Git 저장소 하나로 관리합니다. 새 결과는 `Research/runs/`에 저장하므로 연구 이력과 자동으로 섞이지 않습니다. 공유할 결과를 선별해 docs/results에 복사하고 실행 설정과 해시를 함께 설명하세요.

```bash
git pull --ff-only
python Research/run.py check
git status
git diff
git add README.md .gitignore Research docs
git diff --cached --stat
git commit -m "Update research experiment and results"
git push origin main
```

원본·분할·생성 후보를 수정해 실험을 이어갈 수 있습니다. 기존 입력 해시와 달라지는 경우 새 입력의 출처·생성 방법과 해시를 기록하세요. 기존 연구 기록은 덮어쓰지 말고 구분합니다.

PPT·Word·PDF·논문 원고와 참고논문, codex_tmp 계열 임시파일, 가상환경과 로그는 업로드하지 않습니다. 이미 추적 중인 파일에는 .gitignore만으로 제외가 적용되지 않으므로 커밋 목록도 확인해야 합니다. 과거 공개 커밋에 있던 파일은 현재 트리에서 제외해도 기록에 남습니다.
