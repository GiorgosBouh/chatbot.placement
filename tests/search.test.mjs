import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {findAnswer,normalize} from '../search.mjs';
const data=JSON.parse(readFileSync(new URL('../qa_data.json',import.meta.url)));
test('All curated questions return their own answer',()=>{for(const entry of data){const result=findAnswer(entry.question,data);assert.equal(result.kind,'answer');assert.equal(result.entry.id,entry.id);}});
test('Greek accents, capitalization and final sigma are normalized',()=>{assert.equal(normalize('ΩΡΕΣ;'),normalize('ώρες'));for(const entry of data){const result=findAnswer(normalize(entry.question),data);assert.equal(result.kind,'answer');assert.equal(result.entry.id,entry.id);}});
test('Short payment question retrieves the existing payment answer',()=>{const result=findAnswer('Παίρνω αμοιβή;',data);assert.equal(result.kind,'answer');assert.equal(result.entry.id,30);});
test('Hours question retrieves hours requirements',()=>{const result=findAnswer('Πόσες ώρες πρακτικής χρειάζομαι;',data);assert.equal(result.kind,'answer');assert.equal(result.entry.id,4);});
test('Ambiguous document question offers curated choices',()=>{const result=findAnswer('Τι έγγραφα χρειάζομαι;',data);assert.equal(result.kind,'choices');assert.ok(result.matches.some(entry=>entry.id===2));});
test('Unknown or empty questions do not produce made-up answers',()=>{for(const query of ['', 'καλημέρα', 'Ποιος κέρδισε το πρωτάθλημα;', 'Πώς φτιάχνω μακαρόνια;','<script>alert(1)</script>']) assert.equal(findAnswer(query,data).kind,'unknown');});
test('Responsible-person question accepts genitive and the reported typo',()=>{
  for(const query of ['ποιος είναι ο υπέυθυνος της πρκτικής?', 'Ποιος είναι ο υπεύθυνος της πρακτικής;', 'ποιος ειναι ο υπευθυνος της πρακτικης ασκησης;', 'Ποιος είναι ο υπέυθυνος της πρκτικης ασκησης;']) {
    const result=findAnswer(query,data);assert.equal(result.kind,'answer',query);assert.equal(result.entry.id,11,query);
  }
});
test('Single-letter typos in payment questions remain searchable',()=>{
  for(const query of ['Παίρνω αμιβή;', 'Παίρνω αμοβιή;', 'Παίρνω αμοιιβή;']) {
    const result=findAnswer(query,data);assert.equal(result.kind,'answer',query);assert.equal(result.entry.id,30,query);
  }
});
test('Unrelated and ambiguous misspellings do not invent a match',()=>{
  for(const query of ['Ποιος είναι ο πρωθυπουργός;', 'Θέλω συνταγή μαγειρικής', 'Πότε φεύγει το αεροπλάνο;']) assert.equal(findAnswer(query,data).kind,'unknown',query);
});
