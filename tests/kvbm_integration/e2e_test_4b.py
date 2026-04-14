#!/usr/bin/env python3
"""Quick E2E test for KVBM with Nemotron 4B."""
import requests
import json

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
BASE = "http://localhost:8000/v1/completions"
METRICS = "http://localhost:6880/metrics"

PROMPT = (
    "In the heart of Eldoria, an ancient land of boundless magic and mysterious "
    "creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge "
    "and power, Aeloria was buried beneath the shifting sands of time, lost to the "
    "world for centuries. You are an intrepid explorer, known for your unparalleled "
    "curiosity and courage, who has stumbled upon an ancient map hinting at secrets "
    "that Aeloria holds a secret so profound that it has the potential to reshape the "
    "very fabric of reality. Your journey will take you through treacherous deserts, "
    "enchanted forests, and across perilous mountain ranges. The ancient prophecy "
    "speaks of a chosen one who will unlock the gates of Aeloria and harness the "
    "power of the Eternal Flame, a source of energy so vast that it once powered "
    "an entire civilization. But the path is fraught with danger. The Guardians of "
    "the Threshold, spectral beings bound to protect the city secrets, will test "
    "your resolve at every turn. You must solve the riddles of the Stone Pillars, "
    "navigate the Maze of Whispers where illusions dance at the edge of perception, "
    "and face the Shadow Drake in its volcanic lair. Along the way, you will "
    "encounter allies and enemies alike: the enigmatic Sage of the Silver Tower, "
    "the treacherous merchant lord of Blackport, and the warrior queen of the "
    "nomadic Windborn tribes. Each holds a piece of the puzzle that will lead you "
    "to the heart of Aeloria. The question is: do you have what it takes to "
    "unravel the mysteries of a lost civilization and claim the Eternal Flame "
    "before the forces of darkness consume everything in their path? "
    "The chronicles speak of five trials that guard the entrance to the inner "
    "sanctum of Aeloria. The first trial is the Bridge of Echoes, a seemingly "
    "infinite span across a bottomless chasm where every footstep echoes with "
    "the voices of those who came before. The second is the Garden of Living "
    "Shadows, where the plants themselves are sentient and will entangle any who "
    "dare pass without offering tribute. The third trial takes place in the Hall "
    "of Mirrors, where reflections show not your face but your deepest fears. "
    "The fourth is the Crucible of Elements, a chamber where fire, water, earth, "
    "and air converge in a maelstrom of raw power. And the fifth, most dreaded "
    "of all, is the Audience with the Oracle, an ancient being of immense wisdom "
    "who poses three questions that test the very essence of your character."
)


def get_metrics():
    r = requests.get(METRICS)
    m = {}
    for line in r.text.split("\n"):
        if line.startswith("kvbm_") and not line.startswith("#"):
            parts = line.split()
            if len(parts) == 2:
                m[parts[0]] = int(float(parts[1]))
    return m


def make_request(prompt, max_tokens=600):
    r = requests.post(BASE, json={
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
    })
    return r.json()


print("=" * 60)
print("PHASE 1: Initial request (expect offload)")
print("=" * 60)
r1 = make_request(PROMPT)
text1 = r1["choices"][0]["text"]
usage1 = r1["usage"]
pt = usage1["prompt_tokens"]
tt = usage1["total_tokens"]
print(f"Tokens: prompt={pt}, total={tt}")
print(f"Text[:150]: {text1[:150]}")
m1 = get_metrics()
d2h_1 = m1.get("kvbm_offload_blocks_d2h", 0)
print(f"offload_d2h: {d2h_1}")
print(f"matched_tokens: {m1.get('kvbm_matched_tokens', 0)}")

print()
print("=" * 60)
print("PHASE 2: Eviction prompts")
print("=" * 60)
for i in range(10):
    make_request(f"Eviction {i}: " + "x " * 200, max_tokens=400)
    print(f"  eviction {i+1}/10 done")

m2 = get_metrics()
d2h_2 = m2.get("kvbm_offload_blocks_d2h", 0)
print(f"offload_d2h after eviction: {d2h_2}")

print()
print("=" * 60)
print("PHASE 3: Repeat original prompt (expect cache hit + onboard)")
print("=" * 60)
r2 = make_request(PROMPT)
text2 = r2["choices"][0]["text"]
usage2 = r2["usage"]
pt2 = usage2["prompt_tokens"]
tt2 = usage2["total_tokens"]
print(f"Tokens: prompt={pt2}, total={tt2}")
print(f"Text[:150]: {text2[:150]}")
m3 = get_metrics()
d2h_3 = m3.get("kvbm_offload_blocks_d2h", 0)
h2d_3 = m3.get("kvbm_onboard_blocks_h2d", 0)
match_3 = m3.get("kvbm_matched_tokens", 0)
print(f"offload_d2h: {d2h_3}")
print(f"onboard_h2d: {h2d_3}")
print(f"matched_tokens: {match_3}")

print()
print("=" * 60)
print("PHASE 4: Determinism check")
print("=" * 60)
if text1 == text2:
    print("PASS: Responses are identical")
else:
    min_len = min(len(text1), len(text2))
    diverge = next((i for i in range(min_len) if text1[i] != text2[i]), min_len)
    print(f"DIVERGE at char {diverge}/{min_len}")
    print(f"  R1: {text1[diverge:diverge+80]!r}")
    print(f"  R2: {text2[diverge:diverge+80]!r}")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
offload_ok = d2h_1 > 0
onboard_ok = h2d_3 > 0
match_ok = match_3 > 0
determ_ok = text1 == text2
print(f"Offload:      {'PASS' if offload_ok else 'FAIL'} ({d2h_1} blocks)")
print(f"Cache hit:    {'PASS' if match_ok else 'FAIL'} ({match_3} tokens)")
print(f"Onboard:      {'PASS' if onboard_ok else 'FAIL'} ({h2d_3} blocks)")
print(f"Determinism:  {'PASS' if determ_ok else 'FAIL'}")
