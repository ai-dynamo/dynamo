"""Print compact error evidence without loading a full request export."""
import json
import sys
from collections import Counter

counts = Counter()
examples = []
with open(sys.argv[1]) as f:
    for line in f:
        row = json.loads(line)
        counts['records'] += 1
        m = row.get('metadata', {})
        error = m.get('error') or m.get('error_code') or row.get('error')
        if error:
            counts[str(error)] += 1
            if len(examples) < 5:
                examples.append({'error': error, 'request_id': m.get('x_request_id'),
                                 'metrics': row.get('metrics')})
print(json.dumps({'counts': dict(counts), 'examples': examples}, indent=2))
