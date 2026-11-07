# 포트폴리오 업데이트

이 폴더의 최상위 Git 저장소 하나로 공개 파일을 관리합니다. 기존 Research 내부 Git 메타데이터와 양쪽 커밋 백업은 `.local-only/`에 보관하며 업로드하지 않습니다.

## 평소 작업

```powershell
git pull --ff-only
# 코드 수정 및 공개할 결과 정리
git status
git diff
git add README.md docs Research
git diff --cached --stat
git commit -m "Update public research code and results"
git push origin main
```

루트 .gitignore는 공개 폴더를 허용하는 방식입니다. 원고, 원본 데이터, 모델과 미선별 연구 출력은 로컬에 남습니다. 노트북은 실행 출력과 개인정보를 확인한 뒤 공개하세요. 강제 추가(`git add -f`)는 제외 규칙을 우회하므로 사용 전에 공개 범위를 확인하세요.

## 다른 컴퓨터에서 받기

```powershell
git clone https://github.com/JHONEY-076/Manufacturing-Time-Series-Anomaly-Detection_Research.git
```

clone으로 받는 것은 공개용 코드·문서·대표 결과입니다. 제외한 원고와 연구 자료는 별도 백업이 필요합니다. 이미 과거 커밋에 올라간 자료는 .gitignore나 현재 트리에서의 제외만으로 지워지지 않습니다.
