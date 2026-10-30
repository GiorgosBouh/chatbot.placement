import {findAnswer} from './search.mjs?v=4';
import {CHAT_API_BASE} from './chat-config.mjs?v=3';
const $ = id => document.getElementById(id);
let data = [];
let aiReady=false, busy=false, conversation=[], activeRequest=null, generation=0;
function setBusy(value){busy=value;$('question').disabled=value;$('send').disabled=value;document.querySelectorAll('[data-question],#faq-list button,.choices button').forEach(button=>button.disabled=value);}
function status(){ $('status').textContent=aiReady?'AI Groq ενεργό · Απαντήσεις με βάση το υλικό του μαθήματος':`${data.length} ερωτήσεις διαθέσιμες · Απαντήσεις από τη βάση γνώσης`; }
function message(role, content, source) {
  const article = document.createElement('article'); article.className = `message ${role}`;
  const speaker = document.createElement('span'); speaker.className = 'speaker'; speaker.textContent = role === 'user' ? 'ΕΣΥ' : 'ΟΔΗΓΟΣ ΠΡΑΚΤΙΚΗΣ';
  const bubble = document.createElement('div'); bubble.className = 'bubble'; bubble.textContent = content;
  article.append(speaker, bubble);
  if (source) { const note = document.createElement('span'); note.className='source'; note.textContent=source.label||`Πηγή: Βάση γνώσης μαθήματος · ${source.category} · Ερώτηση ${source.id}`; article.append(note); }
  $('messages').append(article); $('messages').scrollTop = $('messages').scrollHeight;
  return article;
}
function welcome() { message('assistant','Γεια σου! Είμαι ο οδηγός πρακτικής άσκησης. Ρώτησέ με για ώρες, δικαιολογητικά, φορείς ή διαδικασίες — ή διάλεξε μια από τις συχνές ερωτήσεις.'); }
function remember(role,content){conversation.push({role,content:content.slice(0,1500)});while(conversation.length>8 || conversation.reduce((n,m)=>n+m.content.length,0)>5800)conversation.shift();while(conversation[0]?.role==='assistant')conversation.shift();}
async function ask(question, selected) {
  question=question.trim(); if (!question || !data.length || busy) return;
  message('user',question);
  $('question').value='';
  if(aiReady && !selected){
    const turn=++generation;remember('user',question);setBusy(true);
    const pending=message('assistant','Ετοιμάζω απάντηση…');$('status').textContent='Το AI επεξεργάζεται την ερώτησή σου…';
    const controller=new AbortController();activeRequest=controller;const timer=setTimeout(()=>controller.abort(),25000);
    try{
      const response=await fetch(`${CHAT_API_BASE}/chat`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:conversation}),signal:controller.signal,credentials:'omit'});
      const result=await response.json().catch(()=>({}));
      if(!response.ok)throw new Error(result.error||'AI unavailable');
      if(typeof result.answer!=='string'||!result.answer.trim())throw new Error('Empty AI answer');
      if(turn!==generation)return;pending.remove();message('assistant',result.answer,{label:'Απάντηση AI · Groq'+(result.sources?.length?` · Σχετικές ενότητες: ${[...new Set(result.sources.map(source=>source.category))].join(' · ')}`:'')});remember('assistant',result.answer);return;
    }catch(error){
      if(turn!==generation)return;pending.remove();conversation.pop();
      if(error.message==='rate_limited'){
        message('assistant','Έφτασες το προσωρινό όριο αιτημάτων του chat. Περίμενε περίπου ένα λεπτό και δοκίμασε ξανά.');
        return;
      }
      if(error.message==='provider_rate_limited'){
        message('assistant','Το AI έφτασε προσωρινά το όριο χρήσης του Groq. Δοκίμασε ξανά αργότερα, όταν ανανεωθεί το όριο του λογαριασμού.');
        return;
      }
      if(error.message==='provider_quota_exceeded'){
        message('assistant','Δεν μπορώ να απαντήσω αυτή τη στιγμή. Ξαναδοκιμάστε σε λίγο.');
        return;
      }
      message('assistant','Το AI δεν είναι διαθέσιμο αυτή τη στιγμή. Θα προσπαθήσω να σε βοηθήσω από τις έτοιμες απαντήσεις.');
    }finally{clearTimeout(timer);if(turn===generation){activeRequest=null;setBusy(false);status();}}
  }
  const result=selected ? {kind:'answer',entry:selected} : findAnswer(question,data);
  if (result.kind==='answer') {message('assistant',result.entry.answer,result.entry);remember('user',question);remember('assistant',result.entry.answer);}
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
$('clear').addEventListener('click',()=>{generation++;activeRequest?.abort();activeRequest=null;conversation=[];setBusy(false);$('messages').replaceChildren();welcome();$('question').value='';if(data.length)status();else{$('question').disabled=true;$('send').disabled=true;}$('question').focus();});
$('category').addEventListener('change',renderFaq);
for(const button of document.querySelectorAll('[data-question]')) button.addEventListener('click',()=>ask(button.dataset.question));
async function load(state={configured:false}){
  try {
    const response=await fetch('./qa_data.json');if(!response.ok) throw new Error('Knowledge base unavailable');
    const entries=await response.json();
    if(!Array.isArray(entries)||!entries.length||!entries.every(entry=>Number.isInteger(entry.id)&&typeof entry.question==='string'&&typeof entry.answer==='string'&&typeof entry.category==='string'&&Array.isArray(entry.keywords))) throw new Error('Invalid knowledge base');
    data=entries;
    for(const category of [...new Set(data.map(entry=>entry.category))].sort((a,b)=>a.localeCompare(b,'el'))){const option=document.createElement('option');option.value=category;option.textContent=category;$('category').append(option);}
    renderFaq();document.querySelectorAll('button[disabled],input[disabled],select[disabled]').forEach(element=>element.disabled=false);
    aiReady=state.configured===true;
    status();
  }catch(error){
    $('status').textContent='Η βάση γνώσης δεν φορτώθηκε. Ανανέωσε τη σελίδα για νέα προσπάθεια.';$('status').classList.add('error');
    message('assistant','Προσωρινά δεν είναι διαθέσιμη η αναζήτηση. Μπορείς να επικοινωνήσεις με τον υπεύθυνο από τον σύνδεσμο επικοινωνίας.');
  }
}
async function initialize(){
  welcome();
  let state={configured:false};
  try{
    const response=await fetch(`${CHAT_API_BASE}/status`,{signal:AbortSignal.timeout(8000),credentials:'omit'});
    if(response.ok) state=await response.json();
  }catch{}
  await load(state);
}
initialize();
