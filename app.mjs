import {findAnswer} from './search.mjs';
const $ = id => document.getElementById(id);
let data = [];
function message(role, content, source) {
  const article = document.createElement('article'); article.className = `message ${role}`;
  const speaker = document.createElement('span'); speaker.className = 'speaker'; speaker.textContent = role === 'user' ? 'ΕΣΥ' : 'ΟΔΗΓΟΣ ΠΡΑΚΤΙΚΗΣ';
  const bubble = document.createElement('div'); bubble.className = 'bubble'; bubble.textContent = content;
  article.append(speaker, bubble);
  if (source) { const note = document.createElement('span'); note.className='source'; note.textContent=`Πηγή: Βάση γνώσης μαθήματος · ${source.category} · Ερώτηση ${source.id}`; article.append(note); }
  $('messages').append(article); $('messages').scrollTop = $('messages').scrollHeight;
  return article;
}
function welcome() { message('assistant','Γεια σου! Είμαι ο οδηγός πρακτικής άσκησης. Ρώτησέ με για ώρες, δικαιολογητικά, φορείς ή διαδικασίες — ή διάλεξε μια από τις συχνές ερωτήσεις.'); }
function ask(question, selected) {
  question=question.trim(); if (!question || !data.length) return;
  message('user',question);
  const result=selected ? {kind:'answer',entry:selected} : findAnswer(question,data);
  if (result.kind==='answer') message('assistant',result.entry.answer,result.entry);
  else if (result.kind==='choices') {
    const article=message('assistant','Βρήκα περισσότερες σχετικές ερωτήσεις. Ποια σε ενδιαφέρει;');
    const choices=document.createElement('div'); choices.className='choices';
    for(const entry of result.matches){ const button=document.createElement('button'); button.type='button';button.textContent=entry.question;button.addEventListener('click',()=>ask(entry.question,entry));choices.append(button); }
    article.append(choices); $('messages').scrollTop=$('messages').scrollHeight;
  } else message('assistant','Δεν βρήκα σαφή απάντηση στη βάση γνώσης. Δοκίμασε μια πιο συγκεκριμένη ερώτηση ή επίλεξε από τις συχνές ερωτήσεις. Για διευκρινίσεις: gsofianidis@mitropolitiko.edu.gr · 2314 409000.');
  $('question').value='';
}
function renderFaq(){
  $('faq-list').replaceChildren();
  for(const entry of data.filter(entry=>!$('category').value || entry.category===$('category').value)){
    const button=document.createElement('button');button.type='button';button.textContent=entry.question;button.addEventListener('click',()=>ask(entry.question,entry));$('faq-list').append(button);
  }
}
$('question-form').addEventListener('submit',event=>{event.preventDefault();ask($('question').value);});
$('clear').addEventListener('click',()=>{$('messages').replaceChildren();welcome();$('question').value='';$('question').focus();});
$('category').addEventListener('change',renderFaq);
for(const button of document.querySelectorAll('[data-question]')) button.addEventListener('click',()=>ask(button.dataset.question));
welcome();
async function load(){
  try {
    const response=await fetch('./qa_data.json');if(!response.ok) throw new Error('Knowledge base unavailable');
    const entries=await response.json();
    if(!Array.isArray(entries)||!entries.length||!entries.every(entry=>Number.isInteger(entry.id)&&typeof entry.question==='string'&&typeof entry.answer==='string'&&typeof entry.category==='string'&&Array.isArray(entry.keywords))) throw new Error('Invalid knowledge base');
    data=entries;
    for(const category of [...new Set(data.map(entry=>entry.category))].sort((a,b)=>a.localeCompare(b,'el'))){const option=document.createElement('option');option.value=category;option.textContent=category;$('category').append(option);}
    renderFaq();document.querySelectorAll('button[disabled],input[disabled],select[disabled]').forEach(element=>element.disabled=false);
    $('status').textContent=`${data.length} ερωτήσεις διαθέσιμες · Απαντήσεις από τη βάση γνώσης`;
  }catch(error){
    $('status').textContent='Η βάση γνώσης δεν φορτώθηκε. Ανανέωσε τη σελίδα για νέα προσπάθεια.';$('status').classList.add('error');
    message('assistant','Προσωρινά δεν είναι διαθέσιμη η αναζήτηση. Μπορείς να επικοινωνήσεις με τον υπεύθυνο από τον σύνδεσμο επικοινωνίας.');
  }
}
load();
