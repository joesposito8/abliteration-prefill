"""Render session-2-outline.md into a Google Slides deck via the `gws` CLI.

Usage:
    python build_deck.py                 # create a new deck
    python build_deck.py <presentationId>  # replace the slides in an existing deck

The first `##` section becomes the title slide; the rest become title-and-body slides.
"""

import json
import subprocess
import sys
from pathlib import Path

OUTLINE = Path(__file__).parent / "session-2-outline.md"
TITLE = "Prefilling x Abliteration - BlueDot Session 2"


def gws(*args, params=None, body=None):
    cmd = ["gws", *args]
    if params:
        cmd += ["--params", json.dumps(params)]
    if body:
        cmd += ["--json", json.dumps(body)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"gws failed: {' '.join(cmd)}\n{out.stderr}")
    # gws prints a keyring banner on stderr; stdout is the JSON payload.
    return json.loads(out.stdout)


def parse_slides(text):
    slides, title, body = [], None, []
    for line in text.splitlines():
        if line.startswith("## "):
            if title:
                slides.append((title, body))
            title, body = line[3:].strip(), []
        elif title and line.strip():
            body.append(line.strip())
    if title:
        slides.append((title, body))
    return slides


def build_requests(slides):
    """createSlide + insertText for each slide, with stable placeholder object IDs."""
    requests = []
    for i, (title, body) in enumerate(slides):
        slide_id, title_id, body_id = f"slide{i:02d}", f"title{i:02d}", f"body{i:02d}"
        # Slide 0 is the deck title. The TITLE layout names its placeholders
        # CENTERED_TITLE/SUBTITLE, not TITLE/BODY.
        layout = "TITLE" if i == 0 else "TITLE_AND_BODY"
        title_type = "CENTERED_TITLE" if i == 0 else "TITLE"
        body_type = "SUBTITLE" if i == 0 else "BODY"
        requests.append({
            "createSlide": {
                "objectId": slide_id,
                "insertionIndex": i,
                "slideLayoutReference": {"predefinedLayout": layout},
                "placeholderIdMappings": [
                    {"layoutPlaceholder": {"type": title_type, "index": 0}, "objectId": title_id},
                    {"layoutPlaceholder": {"type": body_type, "index": 0}, "objectId": body_id},
                ],
            }
        })
        requests.append({"insertText": {"objectId": title_id, "text": title}})
        if body:
            requests.append({"insertText": {"objectId": body_id, "text": "\n".join(body)}})
            if i:  # the title slide's subtitle stays unbulleted
                requests.append({
                    "createParagraphBullets": {
                        "objectId": body_id,
                        "textRange": {"type": "ALL"},
                        "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE",
                    }
                })
    return requests


def main():
    slides = parse_slides(OUTLINE.read_text())
    print(f"{len(slides)} slides parsed from {OUTLINE.name}")

    if len(sys.argv) > 1:
        pid = sys.argv[1]
    else:
        pid = gws("slides", "presentations", "create",
                  body={"title": TITLE})["presentationId"]
        print(f"created presentation {pid}")

    # A new presentation ships with a default slide; a rebuild has the previous run's.
    # Either way, clear the deck before inserting so slide indexes stay predictable.
    existing = gws("slides", "presentations", "get",
                   params={"presentationId": pid, "fields": "slides(objectId)"})
    deletes = [{"deleteObject": {"objectId": s["objectId"]}}
               for s in existing.get("slides", [])]
    print(f"clearing {len(deletes)} existing slide(s)")

    gws("slides", "presentations", "batchUpdate", params={"presentationId": pid},
        body={"requests": deletes + build_requests(slides)})
    print(f"https://docs.google.com/presentation/d/{pid}/edit")


if __name__ == "__main__":
    main()
