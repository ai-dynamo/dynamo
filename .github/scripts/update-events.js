const ical = require('node-ical');
const fs = require('fs');

const ICS_URL = 'https://calendar.google.com/calendar/ical/c_c2448d2efb09eac2ddee1f34524124135bd3f4554868769059105e18e1b97e8f%40group.calendar.google.com/public/full.ics';

const ALLOWED_ONLINE_HOSTS = ['meet.google.com', 'zoom.us', 'teams.microsoft.com', 'webex.com'];

function escapeMd(str) {
  return str.replace(/\r?\n/g, ' ').replace(/[[\]|`*_~]/g, '\\$&');
}

function safeUrl(url) {
  return url.replace(/\)/g, '%29');
}

function formatLocation(location) {
  if (!location) return '–';
  try {
    const parsed = new URL(location);
    const host = parsed.hostname.replace(/^www\./, '');
    if (/^(lu\.ma|luma\.com)$/i.test(host)) return `[Luma](<${safeUrl(location)}>)`;
    if (ALLOWED_ONLINE_HOSTS.some(h => host === h || host.endsWith(`.${h}`))) return `[Online](<${safeUrl(location)}>)`;
  } catch (_) {}
  const parts = location.split(',').map(s => s.trim());
  const city = parts.length >= 3 ? parts[parts.length - 3] : parts[0];
  return escapeMd(city || '–');
}

function buildAddToCalendarURL(e) {
  const fmt = d => d.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}/, '');
  const params = new URLSearchParams({
    action: 'TEMPLATE',
    text: e.summary || 'Event',
    dates: `${fmt(e.start)}/${fmt(e.end || e.start)}`,
    ...(e.location && { location: e.location }),
    ...(e.description && { details: e.description }),
  });
  return `https://calendar.google.com/calendar/render?${params.toString()}`;
}

function formatDate(d, isAllDay) {
  const opts = { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' };
  if (isAllDay) return d.toLocaleDateString('en-US', { ...opts, timeZone: 'UTC' });
  return d.toLocaleDateString('en-US', { ...opts, timeZone: 'America/Los_Angeles' });
}

async function main() {
  const events = await ical.async.fromURL(ICS_URL);
  const now = new Date();

  const all = Object.values(events)
    .filter(e => e.type === 'VEVENT' && e.start && e.summary && String(e.status || '').toUpperCase() !== 'CANCELLED')
    .sort((a, b) => a.start - b.start);

  const past = all.filter(e => (e.end || e.start) < now).slice(-2);
  const future = all.filter(e => (e.end || e.start) >= now).slice(0, 3);
  const selected = [...future, ...past.reverse()];

  const lines = ['| 📅 Date | 🎤 Event | 📍 Location |', '|:--------|:---------|:------------|'];

  if (selected.length === 0) {
    lines.push('| – | No upcoming events | |');
  } else {
    for (const e of selected) {
      const isPast = (e.end || e.start) < now;
      const isAllDay = e.datetype === 'date';
      const date = formatDate(e.start, isAllDay);
      const addUrl = buildAddToCalendarURL(e);
      const safeTitle = escapeMd(e.summary);
      const label = isPast ? `~~[${safeTitle}](${addUrl})~~` : `**[${safeTitle}](${addUrl})**`;
      const location = formatLocation(e.location);
      lines.push(`| ${date} | ${label} | ${location} |`);
    }
  }

  const md = fs.readFileSync('README.md', 'utf8');
  const marker = /<!-- EVENTS:START -->[\s\S]*?<!-- EVENTS:END -->/;
  if (!marker.test(md)) throw new Error('EVENTS markers not found in README.md');
  const updated = md.replace(marker, `<!-- EVENTS:START -->\n${lines.join('\n')}\n<!-- EVENTS:END -->`);
  fs.writeFileSync('README.md', updated);
  console.log(`Updated README with ${selected.length} events (${past.length} past, ${future.length} upcoming).`);
}

main().catch(err => { console.error(err); process.exit(1); });
