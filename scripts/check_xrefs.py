"""Check cross-references between chapters.

Verifies references of the forms
    "Module 05 section 3.4"   -> chapter 05 must contain a '### 3.4 ...' sub-heading or a '## 5.3/3.4' style section
    "section 14.5" / "see 8.7" / "§ 13.2"  where the first number is a chapter that has an explicit '## N.M' section
and reports references whose target does not exist. Ignores code fences. Prints a summary; exit 1 if any broken.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
chapters = {}
for md in sorted((ROOT / "chapters").glob("*.md")):
    n = int(md.name[:2])
    text = md.read_text(encoding="utf-8")
    prose, fence = [], False
    for ln in text.split("\n"):
        if ln.lstrip().startswith("```"):
            fence = not fence
            continue
        if not fence:
            prose.append(ln)
    body = "\n".join(prose)
    chapters[n] = {
        "name": md.name,
        "prose": body,
        "h2": set(re.findall(r"^## (\d+\.\d+)\s", body, flags=re.M)),                   # explicit N.M sections
        "legacy_h2": set(re.findall(r"^## (\d+)\.\s", body, flags=re.M)),               # '## 3.' legacy sections
        "h3": set(re.findall(r"^#{3,4} (\d+\.\d+(?:\.\d+)?)\s", body, flags=re.M)),     # '### 3.4' sub-headings
    }

broken, checked = [], 0
for n, c in chapters.items():
    for i, ln in enumerate(c["prose"].split("\n"), 1):
        if ln.startswith("#"):
            continue
        # Module NN section X.Y
        for m in re.finditer(r"Module\s+(\d{1,2})\s+(?:sections?|§)\s+(\d+(?:\.\d+)+)", ln):
            tgt, sec = int(m.group(1)), m.group(2)
            checked += 1
            t = chapters.get(tgt)
            if t is None:
                broken.append((c["name"], ln.strip()[:90], f"module {tgt} missing"))
            elif sec not in t["h3"] and sec not in t["h2"] and sec.split(".")[0] not in t["legacy_h2"]:
                broken.append((c["name"], ln.strip()[:90], f"Module {tgt} has no {sec}"))
        # plain "section N.M" / "see N.M" / "§N.M" where N is a chapter with explicit sections
        for m in re.finditer(r"(?:section|sections|see|§)\s*(\d{1,2})\.(\d{1,2})\b(?!\.\d)", ln):
            tgt, sub = int(m.group(1)), m.group(2)
            if "Module" in ln[max(0, m.start() - 12):m.start()]:
                continue
            t = chapters.get(tgt)
            sec = f"{tgt}.{sub}"
            if t is None or not t["h2"]:
                continue                      # chapter without explicit sections: legacy numbering, skip
            checked += 1
            if sec not in t["h2"] and sec not in t["h3"]:
                broken.append((c["name"], ln.strip()[:90], f"no section {sec}"))

print(f"cross-references checked: {checked}; broken: {len(broken)}")
for b in broken[:60]:
    print(" BROKEN", b)
sys.exit(1 if broken else 0)
