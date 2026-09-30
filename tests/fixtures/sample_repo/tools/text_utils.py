import re


def slugify(title):
    """Turn a title into a lowercase URL slug."""
    title = re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-")
    return title.lower()


def levenshtein(a, b):
    """Edit distance between two strings using dynamic programming."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class WordCounter:
    """Count word frequencies in text."""

    def __init__(self):
        self.counts = {}

    def add(self, text):
        for w in text.lower().split():
            self.counts[w] = self.counts.get(w, 0) + 1

    def most_common(self, n):
        return sorted(self.counts.items(), key=lambda kv: -kv[1])[:n]
