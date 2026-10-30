import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {selectKnowledge} from '../worker/context.mjs';
import {rankQuestions} from '../worker/search.mjs';
const knowledge=JSON.parse(readFileSync(new URL('../worker/qa_data.json',import.meta.url)));
const cases=JSON.parse(readFileSync(new URL('../worker/retrieval-cases.json',import.meta.url)));
const select=messages=>selectKnowledge(messages,knowledge,rankQuestions).map(entry=>entry.id);
test('Relevant knowledge is found for all evaluation paraphrases',()=>{
  for(const example of cases){const ids=select([{role:'user',content:example.question}]);for(const expected of example.expected)assert.ok(ids.includes(expected),`${example.question}: missing ${expected}, got ${ids}`);}
});
test('Answer text provides details missing from question titles and keywords',()=>{
  const ids=select([{role:'user',content:'Βάζω τελεία στο πεδίο αριθμός έγκριτης απόφασης;'}]);assert.ok(ids.includes(24));
});
test('A new topic does not inherit unrelated contact questions',()=>{
  const ids=select([{role:'user',content:'Ποιος είναι ο υπεύθυνος της πρακτικής;'},{role:'assistant',content:'Γεώργιος Σοφιανίδης, gsofianidis@mitropolitiko.edu.gr'},{role:'user',content:'Πόσες ώρες πρακτικής χρειάζομαι;'}]);assert.equal(ids[0],4);assert.equal(ids.includes(40),false);assert.equal(ids.includes(41),false);
});
test('A pronoun follow-up retains the previous contact context',()=>{
  const ids=select([{role:'user',content:'Ποιος είναι ο υπεύθυνος της πρακτικής;'},{role:'assistant',content:'Γεώργιος Σοφιανίδης'},{role:'user',content:'Και ποιο είναι το email του;'}]);assert.ok(ids.includes(11));
});
test('Multiple requested topics are included without unbounded context',()=>{
  const ids=select([{role:'user',content:'Τι δικαιολογητικά χρειάζομαι και πού τα ανεβάζω;'}]);assert.ok(ids.includes(2));assert.ok(ids.includes(31));assert.ok(ids.length<=5);
});
test('Unrelated questions do not reuse knowledge from an earlier topic',()=>{
  assert.deepEqual(select([{role:'user',content:'Πόσες ώρες χρειάζομαι;'},{role:'assistant',content:'240 ώρες'},{role:'user',content:'Ποιος είναι ο πρωθυπουργός;'}]),[]);
});
