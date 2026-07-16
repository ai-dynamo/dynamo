const ical = require('node-ical');
const fs = require('fs');

const ICS_URL = 'https://calendar.google.com/calendar/ical/c_c2448d2efb09eac2ddee1f34524124135bd3f4554868769059105e18e1b97e8f%40group.calendar.google.com/public/full.ics';

function formatLocation(location) {
  if (!location) return '–';
  if (/^https?:\/\//.test(location)) return `[Online](${location})`;
  const parts = location.split(',').map(s => s.trim());
  const city = parts.length >= 2 ? parts[parts.length - 2] : parts[0];
  return city || '–';
}

function buildAddToCalendarURL(e) {
  const fmt = d => d.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}/, '');
  const params = new URLSearchParams({
    action: 'TEMPLATE',
    text: e.summary,
    dates: `${fmt(e.start)}/${fmt(e.end || e.start)}`,
    ...(e.location && { location: e.location }),
    ...(e.description && { details: e.description }),
  });
  return `https://calendar.google.com/calendar/render?${params.toString()}`;
}

async function main() {
  const events = await ical.async.fromURL(ICS_URL);
  const now = new Date();

  const all = Object.values(events)
    .filter(e => e.type === 'VEVENT' && e.start)
    .sort((a, b) => a.start - b.start);

  const past = all.filter(e => e.start < now).slice(-2);
  const future = all.filter(e => e.start >= now).slice(0, 3);
  const selected = [...future.reverse(), ...past.reverse()];

  const lines = ['| Date | Event | Location |', '|------|-------|----------|'];

  if (selected.length === 0) {
    lines.push('| – | No events to show | |');
  } else {
    for (const e of selected) {
      const date = e.start.toDateString();
      const addUrl = buildAddToCalendarURL(e);
      const label = `[${e.summary}](${addUrl})`;
      const location = formatLocation(e.location);
      lines.push(`| ${date} | ${label} | ${location} |`);
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
