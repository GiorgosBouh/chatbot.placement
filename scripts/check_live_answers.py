"""Small live evaluation using synthetic questions, never credentials."""
import json
import sys
import subprocess
from pathlib import Path

cases = [
    {"name": "weekly_journal", "messages": [{"role": "user", "content": "Τι γράφω κάθε εβδομάδα στο ημερολόγιο της πρακτικής;"}]},
    {"name": "insurance", "messages": [{"role": "user", "content": "Έχω κωδικούς taxisnet αλλά δεν έχω εργαστεί ποτέ, πώς βγάζω ασφάλεια για την πρακτική;"}]},
    {"name": "new_topic", "messages": [{"role": "user", "content": "Ποιος είναι ο υπεύθυνος;"}, {"role": "assistant", "content": "Γεώργιος Σοφιανίδης, gsofianidis@mitropolitiko.edu.gr"}, {"role": "user", "content": "Αν κάνω 6 ώρες την ημέρα, σε πόσες ημέρες συμπληρώνω τις 240;"}]},
]
results = []
for case in cases:
    response = subprocess.run([
        'curl', '--silent', '--show-error', '--fail-with-body', '--max-time', '30',
        '-H', 'Origin: https://giorgosbouh.github.io', '-H', 'Content-Type: application/json',
        '--data-binary', '@-', 'https://placement-se5117-ai.leafyduck18.chatgpt.site/chat',
    ], input=json.dumps({"messages": case["messages"]}), text=True, capture_output=True, check=False)
    try:
        result = json.loads(response.stdout)
    except json.JSONDecodeError:
        result = {"error": "transport_error", "exit_code": response.returncode}
    results.append({"name": case["name"], **result})
Path(sys.argv[1]).write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(json.dumps(results, ensure_ascii=False))
