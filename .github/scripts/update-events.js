const ical = require('node-ical');
const fs = require('fs');
const path = require('path');

const ICS_URL = 'https://calendar.google.com/calendar/ical/c_c2448d2efb09eac2ddee1f34524124135bd3f4554868769059105e18e1b97e8f%40group.calendar.google.com/public/full.ics';
const REPO = 'ai-dynamo/dynamo';
const BRANCH = 'main';
const EVENTS_DIR = '.github/events';
const RAW_BASE = `https://raw.githubusercontent.com/${REPO}/${BRANCH}/${EVENTS_DIR}`;

function formatLocation(location) {
  if (!location) return '–';
  if (/^https?:\/\//.test(location)) return `[Online](${location})`;
  const parts = location.split(',').map(s => s.trim());
  const city = parts.length >= 2 ? parts[parts.length - 2] : parts[0];
  return city || '–';
}

function slugify(str) {
  return str.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
}

function buildICS(e) {
  const fmt = d => d.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}/, '');
  return [
    'BEGIN:VCALENDAR',
    'VERSION:2.0',
    'PRODID:-//ai-dynamo//dynamo//EN',
    'BEGIN:VEVENT',
    `UID:${e.uid}`,
    `SUMMARY:${e.summary}`,
    `DTSTART:${fmt(e.start)}`,
    `DTEND:${fmt(e.end || e.start)}`,
    e.location ? `LOCATION:${e.location}` : '',
    e.description ? `DESCRIPTION:${e.description.replace(/\n/g, '\\n')}` : '',
    'END:VEVENT',
    'END:VCALENDAR',
  ].filter(Boolean).join('\r\n');
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

  fs.mkdirSync(EVENTS_DIR, { recursive: true });

  const existingFiles = new Set(fs.readdirSync(EVENTS_DIR).filter(f => f.endsWith('.ics')));
  const newFiles = new Set();

  const lines = ['| Date | Event | Location |', '|------|-------|----------|'];

  if (selected.length === 0) {
    lines.push('| – | No events to show | |');
  } else {
    for (const e of selected) {
      const date = e.start.toDateString();
      const filename = `${slugify(e.summary)}.ics`;
      const filepath = path.join(EVENTS_DIR, filename);
      fs.writeFileSync(filepath, buildICS(e));
      newFiles.add(filename);

      const icsLink = `${RAW_BASE}/${filename}`;
      const label = `[${e.summary}](${icsLink})`;
      const location = formatLocation(e.location);
      lines.push(`| ${date} | ${label} | ${location} |`);
    }
  }

  for (const f of existingFiles) {
    if (!newFiles.has(f)) fs.unlinkSync(path.join(EVENTS_DIR, f));
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
