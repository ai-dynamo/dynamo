#!/bin/bash
# Creates the input data file for aiperf benchmarking

IMG_URL="https://huggingface.co/datasets/Sayali9141/traffic_signal_images/resolve/main/61.jpg"

python3 -c "
import json

prompt = (
    'Provide an extremely detailed and comprehensive analysis of this image. '
    'Describe every single object, person, color, texture, shadow, and spatial relationship you can observe. '
    'Then discuss the possible context, setting, time of day, weather conditions, and cultural significance. '
    'Follow that with a creative story inspired by this image that is at least 3000 words long. '
    'Then provide a technical photography analysis covering composition, lighting, depth of field, and color grading. '
    'Finally, list 50 specific observations about minor details in the image. '
    'Do not stop until you have written at least 4000 words.\n'
    '\n'
    'Additional analysis guidelines:\n'
    'For the visual description, consider foreground elements, midground composition, and background details separately. '
    'Note any text, signage, or symbols visible. Describe the color palette using specific color names. '
    'Identify materials and surfaces. Estimate distances between objects.\n'
    '\n'
    'For the creative story, develop characters with names and backstories. Include dialogue. '
    'Create a narrative arc with a beginning, rising action, climax, and resolution. '
    'Set the story in the location depicted. '
    'Include sensory details beyond just visual - sounds, smells, temperature, tactile sensations.\n'
    '\n'
    'For the photography analysis, discuss the rule of thirds, leading lines, framing, symmetry or asymmetry, '
    'negative space, and visual weight. Analyze whether the image uses natural or artificial lighting, '
    'the direction and quality of light, and any color temperature shifts.\n'
    '\n'
    'For the observations list, look for subtle details like reflections, partial objects at frame edges, '
    'weathering patterns, brand logos, license plates, architectural styles, vegetation types, cloud formations, '
    'and any motion blur or artifacts.'
)

record = {'texts': [prompt], 'images': ['$IMG_URL']}
print(json.dumps(record))
" > /tmp/dynamo/osl_high.jsonl
echo "Created /tmp/dynamo/osl_high.jsonl"
