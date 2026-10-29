// Search only the curated knowledge base; never invent an answer.
export const normalize = value => String(value).normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().replace(/ς/g, 'σ').replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
const stop = new Set(normalize('τι πως που ποιο ποιος ποια ποιες ποιοι ποιον ποιας πότε μπορω μπορει μπορουμε να σε για την τη το τον τα του της τους τις των μου με μας και η ο οι ενα μια εναν απο ως αν δεν εχω ειναι θα πρεπει κανω κανει κανετε χρειαζομαι χρειάζεται παιρνω γινεται σχετικα πρακτικη ασκηση').split(' '));
const groups = [
 ['ωρα','ωρες','ωρων'], ['εγγραφα','εγγραφο','εγγραφων','δικαιολογητικα','χαρτια'],
 ['αμοιβη','αμοιβης','πληρωμη','πληρωνομαι','μισθο','μισθος','λεφτα','χρηματα'],
 ['ξεκιναω','ξεκινησω','ξεκινω','αρχιζω','αρχισω','εναρξη','ξεκινημα'],
 ['δομη','δομες','δομης','δομων','φορεα','φορεας','φορεις','φορεων'],
 ['επικοινωνια','επικοινωνω','επικοινωνησω'], ['προθεσμια','προθεσμιες','deadline'],
 ['απουσια','απουσιες','απουσιασω'], ['αρρωστησω','αρρωστος','ασθενεια'],
 ['αλλαξω','αλλαγη'], ['συμβαση','συμβασεις','συμβασης'],
 ['αξιολογηση','αξιολογουμαι'], ['βιβλιο','βιβλιου'], ['τηλεφωνο','τηλεφωνα']
];
const aliases = new Map(groups.flatMap(group => group.map(word => [normalize(word), normalize(group[0])])));
export function tokens(value) { return [...new Set(normalize(value).split(' ').filter(word => word && !stop.has(word)).map(word => aliases.get(word) || word))]; }
function match(word, list) { return list.some(other => word === other || (Math.min(word.length, other.length) >= 6 && word.slice(0, -2) === other.slice(0, -2))); }
export function rankQuestions(question, data) {
  const query = tokens(question);
  const indexed = data.map(entry => ({entry, title:tokens(entry.question), keys:tokens((entry.keywords || []).join(' ')), category:tokens(entry.category)}));
  return indexed.map(item => {
    if (normalize(question) === normalize(item.entry.question)) return {entry:item.entry, score:1};
    let total = 0, hit = 0;
    for (const word of query) {
      const count = indexed.filter(row => match(word, [...row.title, ...row.keys])).length;
      const weight = 1 + Math.log((data.length + 1) / (count + 1));
      total += weight;
      hit += weight * (match(word, item.title) ? 1 : match(word, item.keys) ? .8 : match(word, item.category) ? .65 : 0);
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
