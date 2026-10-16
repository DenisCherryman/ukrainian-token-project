# Публікація UTP на GitHub

**Поточна ситуація:** у підключеному GitHub-акаунті не знайдено репозиторію `ukrainian-token-project`; можливість створити новий репозиторій через доступний конектор не надається. Цей ZIP **не опублікований** на GitHub. Нижче — точний спосіб опублікувати його власноруч.

## Найшвидший варіант з GitHub CLI

Після встановлення [GitHub CLI](https://cli.github.com/):

```bash
unzip Ukrainian-Token-Project-v0.3.zip
cd ukrainian-token-project-v3
python -m pytest -q
git init
git add .
git commit -m "Initial public UTP v0.3 research toolkit"
gh auth login
gh repo create ukrainian-token-project --public --source . --remote origin --push
```

Якщо у GitHub CLI ще немає логіну, `gh auth login` відкриє стандартний потік авторизації. Для вже створеного репозиторію скористайтеся його URL і `git remote add origin <URL>` та `git push -u origin main` (назва гілки залежить від локальної конфігурації).

## Перед публікацією

- Перевірте `git status` та `git diff --cached --stat`.
- **Не публікуйте** директорії `data/`, приватні корпуси, API-ключі, комерційно обмежені моделі, тестові вхідні тексти третіх осіб.
- Проєктний код MIT; сторонні дані (зокрема UD Ukrainian-IU, CC BY-NC-SA 4.0) — не MIT.
- Не називайте синтетичні відсотки доказом економії токенів у ChatGPT.
- Після публікації перевірте GitHub Actions (CI) і відкрийте issue на наступні кроки: *Real-corpus document-disjoint benchmark (non-commercial)*.

## Для колабораторів

У README є чіткі інструкції повторення результатів. Приймайте PR з новими джерелами тільки з описом ліцензій, hash, документного розбиття, перевірок round-trip та негативних результатів.
