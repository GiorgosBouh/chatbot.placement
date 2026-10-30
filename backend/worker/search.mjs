// Search only the curated knowledge base; never invent an answer.
export const normalize = value => String(value).normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().replace(/ς/g, 'σ').replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
export const needsFrameworkClarification = value => normalize(value).split(' ').some(word => word.startsWith('πλαισι'));
const stop = new Set(normalize('τι πως που ποιο ποιος ποια ποιες ποιοι ποιον ποιας πότε μπορω μπορει μπορουμε να σε για την τη το τον τα του της τους τις των μου με μας και η ο οι ενα μια εναν απο ως αν δεν εχω ειναι θα πρεπει κανω κανει κανετε χρειαζομαι χρειάζεται παιρνω γινεται σχετικα πρακτικη πρακτικης ασκηση ασκησης θελω θελει εγω εσεις αυτο ειναι καθε αλλά ποτέ πριν μετα αυριο').split(' '));
const groups = [
 ['ωρα','ωρες','ωρων'], ['εγγραφα','εγγραφο','εγγραφων','δικαιολογητικα','χαρτια'],
 ['αμοιβη','αμοιβης','πληρωμη','πληρωνομαι','πληρωθω','πληρωθώ','μισθο','μισθος','λεφτα','χρηματα'],
 ['ξεκιναω','ξεκινησω','ξεκινω','αρχιζω','αρχισω','εναρξη','ξεκινημα'],
 ['δομη','δομες','δομης','δομων','φορεα','φορεας','φορεις','φορεων'],
 ['επικοινωνια','επικοινωνω','επικοινωνησω'], ['προθεσμια','προθεσμιες','deadline'],
 ['απουσια','απουσιες','απουσιασω'], ['αρρωστησω','αρρωστος','ασθενεια'],
 ['αλλαξω','αλλαγη'], ['συμβαση','συμβασεις','συμβασης'],
 ['αξιολογηση','αξιολογουμαι','βαθμολογει','βαθμολογια','βαθμος'], ['βιβλιο','βιβλιου','ημερολογιο'], ['τηλεφωνο','τηλεφωνα'],
 ['υπογραφη','υπογραφες','υπογραφης','υπογραφων','υπογραφω','υπογραψω','υπογραψει','υπεγραψα'],
 ['σφραγιδα','σφραγιδες','σφραγιδων'], ['ασφαλιση','ασφαλεια','ασφαλιστικη','ασφαλιστικης','ασφαλιστικα'],
 ['ικανοτητα','ικανοτητας'], ['ενημεροτητα','ενημεροτητας'],
 ['εβδομαδα','εβδομαδες','εβδομαδιαιων','εβδομαδιαιες','εβδομαδιαια'],
 ['υποβολη','υποβαλλω','υποβαλω','ανεβαζω','ανεβασω','στελνω','στειλω','ανεβασμα'],
 ['συμπληρωση','συμπληρωνω','συμπληρωσω','συμπληρωσει','συμπληρωνεται','συμπληρωμενο'],
 ['αθλημα','αθληματα','αθλητισμοσ','προπονητικη','ενοργανη','γυμναστικη','ποδοσφαιρο','μπασκετ','κολυμβηση'],
 ['παραλληλα','ταυτοχρονα'], ['εργασια','εργαζομαι','εργαστει','εργαζομενοι','δουλευω','δουλεψω']
];
const aliases = new Map(groups.flatMap(group => group.map(word => [normalize(word), normalize(group[0])])));
export function tokens(value) { return [...new Set(normalize(value).split(' ').filter(word => word && !stop.has(word)).map(word => aliases.get(word) || word))]; }
// Accept one missing, extra, mistyped or transposed letter in longer words.
// Correct only when all nearest spellings have the same meaning; leave unknown
// topics untouched rather than turning unrelated queries into confident answers.
function oneEditApart(a, b) {
  if (Math.abs(a.length-b.length)>1) return false;
  if (a.length===b.length) {
    const differences=[];
    for(let i=0;i<a.length;i++) if(a[i]!==b[i]) differences.push(i);
    return differences.length===1 || (differences.length===2 && differences[1]===differences[0]+1 && a[differences[0]]===b[differences[1]] && a[differences[1]]===b[differences[0]]);
  }
  const shorter=a.length<b.length?a:b, longer=a.length<b.length?b:a;
  let i=0;while(i<shorter.length && shorter[i]===longer[i]) i++;
  return shorter.slice(i)===longer.slice(i+1);
}
function queryTokens(question, data) {
  const vocabulary=new Set([...stop,...aliases.keys(),...data.flatMap(entry=>normalize([entry.question,entry.category,entry.answer,...entry.keywords].join(' ')).split(' '))]);
  const words=normalize(question).split(' ').map(word=>{
    if(word.length<5 || vocabulary.has(word)) return word;
    const alternatives=[...vocabulary].filter(candidate=>candidate.length>=5 && oneEditApart(word,candidate));
    const meanings=new Set(alternatives.map(candidate=>stop.has(candidate)?'':aliases.get(candidate)||candidate));
    return meanings.size===1 ? [...meanings][0] : word;
  });
  return tokens(words.join(' '));
}
function match(word, list) { return list.some(other => word === other || (Math.min(word.length, other.length) >= 6 && word.slice(0, -2) === other.slice(0, -2))); }
export function rankQuestions(question, data) {
  const query = queryTokens(question, data);
  const indexed = data.map(entry => ({entry, title:tokens(entry.question), keys:tokens((entry.keywords || []).join(' ')), category:tokens(entry.category), body:tokens(entry.answer||'')}));
  return indexed.map(item => {
    if (normalize(question) === normalize(item.entry.question)) return {entry:item.entry, score:1};
    let total = 0, hit = 0;
    for (const word of query) {
      const count = indexed.filter(row => match(word, [...row.title, ...row.keys, ...row.body])).length;
      const weight = 1 + Math.log((data.length + 1) / (count + 1));
      total += weight;
      hit += weight * (match(word, item.title) ? 1 : match(word, item.keys) ? .8 : match(word, item.category) ? .65 : match(word, item.body) ? .5 : 0);
    }
    return {entry:item.entry, score:total ? hit / total : 0};
  }).filter(item => item.score > 0).sort((a,b) => b.score-a.score || a.entry.id-b.entry.id);
}
export function findAnswer(question, data) {
  const exact = data.find(entry => normalize(entry.question) === normalize(question));
  if (exact) return {kind:'answer', entry:exact};
  const ranked = rankQuestions(question, data);
  if (!ranked.length || ranked[0].score < .55) return {kind:'unknown', matches:[]};
  const [first, second] = ranked;
  if (normalize(question) === normalize(first.entry.question)) return {kind:'answer', entry:first.entry};
  if (second && first.score - second.score < .1) return {kind:'choices', matches:ranked.filter(row => row.score >= first.score-.1).slice(0,3).map(row => row.entry)};
  return {kind:'answer', entry:first.entry};
}
