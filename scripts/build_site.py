"""Stage only the public website files for GitHub Pages (no Python server)."""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
target = root / '_site'
target.mkdir(exist_ok=True)
for name in ('index.html', 'styles.css', 'app.mjs', 'search.mjs', 'chat-config.mjs', 'qa_data.json'):
    shutil.copy2(root / name, target / name)
(target / '.nojekyll').touch()
print(f'Static website ready: {target}')
