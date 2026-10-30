import {normalize,tokens} from './search.mjs';

// Keep a new topic independent of older questions. Resolve short follow-ups
// against the most recent exchange instead of mixing the last three topics.
export function selectKnowledge(messages,knowledge,rank) {
  const current=messages.at(-1)?.content||'';
  const normalized=normalize(current);
  const previousUser=messages.slice(0,-1).findLast(message=>message.role==='user');
  const previousAssistant=messages.slice(0,-1).findLast(message=>message.role==='assistant');
  const refersBack=/(^| )(του|τησ|αυτο|αυτον|αυτη|αυτοσ|αυτα|αυτων|εκει|τοτε|εννοω)( |$)/.test(normalized) || /^(και |δηλαδη |αρα )/.test(normalized);
  const followUp=!!previousUser && (tokens(current).length===0 || (refersBack && tokens(current).length<=4));
  const queries=[{text:current,weight:1}];
  const clauses=current.split(/\s+και\s+(?=(?:πώς|πως|πού|που|τι)(?:\s|$)|ποιο|ποια|πόσ|ποσ)/iu);
  const required=[];
  if(clauses.length>1){
    for(const clause of clauses.slice(0,3)){
      const best=rank(clause,knowledge).find(row=>row.score>.2);
      if(best&&!required.some(entry=>entry.id===best.entry.id))required.push(best.entry);
      queries.push({text:clause,weight:.6});
    }
  }
  if(followUp){queries.push({text:previousUser.content,weight:.6});if(previousAssistant)queries.push({text:previousAssistant.content,weight:.25});}
  const scores=new Map();
  for(const query of queries){
    for(const row of rank(query.text,knowledge).filter(row=>row.score>.15).slice(0,8)){
      const existing=scores.get(row.entry.id)||{entry:row.entry,score:0};existing.score+=row.score*query.weight;scores.set(row.entry.id,existing);
    }
  }
  const ranked=[...scores.values()].sort((a,b)=>b.score-a.score||a.entry.id-b.entry.id).map(row=>row.entry);
  return [...required,...ranked.filter(entry=>!required.some(other=>other.id===entry.id))].slice(0,5);
}
