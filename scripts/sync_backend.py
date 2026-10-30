"""Copy shared search/evaluation data into backend code and the Sites checkout."""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
for target in (root / 'backend' / 'worker', root / 'llm-backend' / 'worker'):
    if not target.parent.exists():
        continue
    target.mkdir(exist_ok=True)
    for source, name in ((root/'search.mjs','search.mjs'), (root/'qa_data.json','qa_data.json'), (root/'tests'/'retrieval-cases.json','retrieval-cases.json')):
        shutil.copy2(source, target/name)
print('Shared backend knowledge and search synchronized')
