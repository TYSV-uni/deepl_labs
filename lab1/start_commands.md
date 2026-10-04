### Без навмисної помилки:

```bash
uv venv
uv pip install -r requirements.txt
uv run python lab1.py
```

### З навмисною помилкою:

```bash
uv venv
uv pip install -r requirements.txt
uv run python lab1.py --no-normed
```