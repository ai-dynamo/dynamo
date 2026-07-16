const ical = require('node-ical');
const fs = require('fs');

const ICS_URL = 'https://calendar.google.com/calendar/ical/c_c2448d2efb09eac2ddee1f34524124135bd3f4554868769059105e18e1b97e8f%40group.calendar.google.com/public/full.ics';

async function main() {
  const events = await ical.async.fromURL(ICS_URL);
  const now = new Date();

  const all = Object.values(events)
    .filter(e => e.type === 'VEVENT' && e.start)
    .sort((a, b) => a.start - b.start);

  const past = all
    .filter(e => e.start < now)
    .slice(-2);

  const future = all
    .filter(e => e.start >= now)
    .slice(0, 3);

  const selected = [...future.reverse(), ...past.reverse()];

  const lines = ['| Date | Event |', '|------|-------|'];

  if (selected.length === 0) {
    lines.push('| – | No events to show |');
  } else {
    for (const e of selected) {
      const date = e.start.toDateString();
      const label = e.url ? `[${e.summary}](${e.url})` : e.summary;
      lines.push(`| ${date} | ${label} |`);
    }
  }

  const md = fs.readFileSync('README.md', 'utf8');
  const updated = md.replace(
    /<!-- EVENTS:START -->[\s\S]*?<!-- EVENTS:END -->/,
    `<!-- EVENTS:START -->\n${lines.join('\n')}\n<!-- EVENTS:END -->`
  );
  fs.writeFileSync('README.md', updated);
  console.log(`Updated README with ${selected.length} events (${past.length} past, ${future.length} upcoming).`);
}

main().catch(err => { console.error(err); process.exit(1); });
