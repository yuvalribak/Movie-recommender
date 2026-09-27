"""
Movie recommender

1. Builds a list of the most popular films in MovieLens (IMDb-style weighted rating).
2. Serves a web page where you pick your favourite films from a dropdown you can type into.
3. Uses item-based k-nearest-neighbours collaborative filtering on the MovieLens
   rating matrix to find films rated similarly by the same people.

 http://127.0.0.1:5000

"""

import difflib
import io
import os
import re
import ssl
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request
from scipy.sparse import csr_matrix

#settings
TOP_N_POPULAR = 400        # size of the popular-films list shown in the dropdown
NUM_FAVOURITES = 3         # how many films the user is asked for
MAX_FAVOURITES = 10        # upper limit if they add more boxes
MIN_RATINGS_POPULAR = 30   # a film needs this many ratings to be on the popular list
MIN_RATINGS_KNN = 10       # films with fewer ratings are left out of the kNN model
NEIGHBOURS = 10            # "also liked" films shown per chosen film
SHRINKAGE = 10             # damps similarities built on only a handful of shared raters
DROPDOWN_ALL_FILMS = False # True = dropdown offers every film in the kNN model, not just the top 400

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ml-latest-small")


# data (only on first run)
def download_movielens():

    print("Downloading MovieLens latest-small ...")
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(DATA_URL, context=context) as resp:
            data = resp.read()
    except urllib.error.URLError as err:
        raise SystemExit(
            f"\nCouldn't download the dataset: {err.reason}\n\n"
            "If this is a certificate error on a Mac, do one of these, then run again:\n"
            "  * pip install certifi\n"
            "  * or double-click 'Install Certificates.command' in your Applications/Python 3.x folder\n\n"
            f"Or download it yourself from {DATA_URL}\n"
            f"and unzip it so this file exists:\n  {os.path.join(DATA_DIR, 'ratings.csv')}\n"
        )
    zipfile.ZipFile(io.BytesIO(data)).extractall(os.path.dirname(DATA_DIR))


def load_movielens():
    """Download MovieLens latest-small if needed, then load ratings and movies."""
    if not os.path.exists(os.path.join(DATA_DIR, "ratings.csv")):
        download_movielens()
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"), usecols=["userId", "movieId", "rating"])
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    movies["title"] = movies.title.map(readable_title)
    return ratings, movies


def readable_title(title):
    """MovieLens writes 'Matrix, The (1999)'; turn that into 'The Matrix (1999)'."""
    m = re.match(r"^(.*?), (The|A|An|Les|Le|La|L'|Il|Der|Die|Das|El|Los|Las)( \(.*)?$", title.strip())
    if not m:
        return title.strip()
    article = m.group(2)
    joiner = "" if article.endswith("'") else " "
    return article + joiner + m.group(1) + (m.group(3) or "")


# popular films weighted rating
def popular_movies(ratings, movies, top_n=TOP_N_POPULAR, min_ratings=MIN_RATINGS_POPULAR):
    """
    IMDb-style weighted rating:
        WR = (v / (v + m)) * R + (m / (v + m)) * C
    v = number of ratings, R = the film's mean rating,
    C = mean rating across all eligible films, m = 75th percentile of v.
    Films with few ratings are pulled toward the overall mean.
    """
    stats = ratings.groupby("movieId").rating.agg(avg="mean", count="count").reset_index()
    stats = stats[stats["count"] >= min_ratings]
    m = stats["count"].quantile(0.75)
    C = stats["avg"].mean()
    stats["weighted"] = (stats["count"] * stats["avg"] + m * C) / (stats["count"] + m)
    top = stats.sort_values("weighted", ascending=False).head(top_n)
    return top.merge(movies, on="movieId")[["movieId", "title", "genres", "avg", "count", "weighted"]]


# item-based kNN
class ItemKNN:
    """
    Item-based collaborative filtering.

    Each film is a vector of the ratings every user gave it (a film x user matrix). Two films are similar when the same
    people rate them the same way, measured by cosine similarity.

    Ratings are centred on each user's own average
    Similarities built on few shared raters are shrunk toward zero.
    """

    def __init__(self, ratings, movies, min_ratings=MIN_RATINGS_KNN, shrinkage=SHRINKAGE):
        counts = ratings.movieId.value_counts()
        keep = counts[counts >= min_ratings].index
        r = ratings[ratings.movieId.isin(keep)].copy()
        r["centred"] = r.rating - r.groupby("userId").rating.transform("mean") #adjusting for different raters (adjusted cosine similarity)

        self.movie_ids = np.sort(r.movieId.unique())
        self.index_of = {mid: i for i, mid in enumerate(self.movie_ids)}
        user_ids = r.userId.unique()
        user_index = {u: i for i, u in enumerate(user_ids)}
        rows = r.movieId.map(self.index_of).values
        cols = r.userId.map(user_index).values
        shape = (len(self.movie_ids), len(user_ids))

        X = csr_matrix((r.centred.values, (rows, cols)), shape=shape)
        B = csr_matrix((np.ones(len(r)), (rows, cols)), shape=shape)  # who rated what- sparse matrix with 1 row per film, 1 column per user
        #comparing films (dot product)
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        cosine = (X @ X.T).toarray() / np.outer(norms, norms) 
        shared = (B @ B.T).toarray()    
        #scale by no. users who rated both films                
        self.sim = cosine * shared / (shared + shrinkage)
        np.fill_diagonal(self.sim, -np.inf)             # a film is not its own neighbour

        info = movies.set_index("movieId").loc[self.movie_ids]
        self.titles = info.title.values
        self.genres = info.genres.str.replace("|", ", ", regex=False).values
        self.counts = counts.loc[self.movie_ids].values

    def neighbours(self, movie_id, k=NEIGHBOURS, exclude=()):
        """The k films most similar to one film."""
        i = self.index_of[movie_id]
        order = np.argsort(-self.sim[i])
        out = []
        for j in order:
            if self.movie_ids[j] in exclude or self.sim[i, j] <= 0:
                continue
            out.append(self._row(j, self.sim[i, j]))
            if len(out) == k:
                break
        return out

    def recommend(self, movie_ids, k=NEIGHBOURS, only=None, skip=None):
        """
        Combined picks for several favourites: add up each candidate's similarity
        to every chosen film (negative similarities count as zero).
        `only` / `skip` restrict candidates to, or away from, a set of movieIds.
        """
        idx = [self.index_of[m] for m in movie_ids]
        score = np.clip(self.sim[idx], 0, None).sum(axis=0)
        best_match = np.array(idx)[np.argmax(self.sim[idx], axis=0)]
        out = []
        for j in np.argsort(-score):
            mid = self.movie_ids[j]
            if mid in movie_ids or score[j] <= 0:
                continue
            if only is not None and mid not in only:
                continue
            if skip is not None and mid in skip:
                continue
            row = self._row(j, score[j] / len(idx))
            row["because"] = self.titles[best_match[j]]
            out.append(row)
            if len(out) == k:
                break
        return out

    def _row(self, j, similarity):
        return {"title": self.titles[j], "genres": self.genres[j],
                "ratings": int(self.counts[j]), "similarity": float(similarity)}


# build everything once at start-up
ratings, movies = load_movielens()
popular = popular_movies(ratings, movies)
knn = ItemKNN(ratings, movies)

popular_ids = set(popular.movieId)
dropdown = (popular[popular.movieId.isin(knn.index_of)] if not DROPDOWN_ALL_FILMS
            else movies[movies.movieId.isin(knn.index_of)])
title_to_id = dict(zip(dropdown.title, dropdown.movieId))
dropdown_titles = sorted(title_to_id)
print(f"Popular list: {len(popular)} films. kNN model: {len(knn.movie_ids)} films. "
      f"Dropdown: {len(dropdown_titles)} films.")


def find_title(text):
    """Exact (case-insensitive) match against the dropdown list, plus close suggestions."""
    text = (text or "").strip()
    if not text:
        return None, []
    lookup = {t.lower(): t for t in dropdown_titles}
    if text.lower() in lookup:
        return lookup[text.lower()], []
    starts = [t for t in dropdown_titles if t.lower().startswith(text.lower())]
    if len(starts) == 1:
        return starts[0], []
    contains = [t for t in dropdown_titles if text.lower() in t.lower()]
    suggestions = contains[:5] or difflib.get_close_matches(text, dropdown_titles, n=5, cutoff=0.5)
    return None, suggestions


# web page
app = Flask(__name__)

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Also Liked</title>
<style>
  :root { --ink:#1b1d22; --muted:#5f6570; --line:#d6d9de; --bg:#f3f4f6; --card:#fff; --accent:#1f5f8b; --ok:#2a7a46; --bad:#a3392f; }
  * { box-sizing:border-box; }
  body { margin:0; padding:24px 16px 48px; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  .wrap { max-width:900px; margin:0 auto; display:flex; flex-direction:column; gap:24px; }
  h1 { margin:0; font-size:30px; }
  h2 { margin:0 0 8px; font-size:19px; }
  p.lead { margin:4px 0 0; color:var(--muted); }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px; }
  .slot { display:grid; grid-template-columns:28px 1fr; gap:4px 10px; align-items:center; margin-bottom:12px; }
  .slot span.n { color:var(--muted); font-variant-numeric:tabular-nums; }
  .slot input { width:100%; padding:9px 10px; border:1px solid var(--line); border-radius:8px; font:inherit; }
  .slot input:focus { outline:2px solid var(--accent); outline-offset:1px; }
  .slot small { grid-column:2; min-height:1.2em; font-size:13px; }
  .ok { color:var(--ok); } .bad { color:var(--bad); }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  button { font:inherit; padding:8px 14px; border-radius:8px; border:1px solid var(--line); background:var(--card); cursor:pointer; }
  button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
  .errors { background:#fbeceb; color:var(--bad); border-radius:8px; padding:10px 12px; }
  .errors ul { margin:4px 0 0; padding-left:20px; }
  table { width:100%; border-collapse:collapse; font-size:14px; }
  th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
  th { color:var(--muted); font-weight:500; font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
  td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
  .genres { color:var(--muted); font-size:12.5px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(380px, 1fr)); gap:16px; }
  .scroll { overflow-x:auto; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Also Liked</h1>
    <p class="lead">Pick {{ num }} films you love from the {{ dropdown_size }} most popular films in MovieLens.
      Start typing in a box to search. You'll get films that the same people rated highly.</p>
  </header>

  <form class="card" method="post" action="/" id="form">
    {% if errors %}
      <div class="errors">Some films weren't found:
        <ul>{% for e in errors %}<li>{{ e }}</li>{% endfor %}</ul>
      </div><br>
    {% endif %}
    <div id="slots">
      {% for value in entries %}
      <label class="slot">
        <span class="n">{{ loop.index }}.</span>
        <input name="film" list="films" value="{{ value }}" placeholder="Start typing a film title…" autocomplete="off">
        <small></small>
      </label>
      {% endfor %}
    </div>
    <div class="row">
      <button class="primary" type="submit">Recommend</button>
      <button type="button" id="add">Add another film</button>
    </div>
  </form>

  <datalist id="films">
    {% for t in titles %}<option value="{{ t }}">{% endfor %}
  </datalist>

  {% if combined %}
  <section class="card">
    <h2>Your picks</h2>
    <p class="lead">Films most similar to all of your choices together, split into popular films and hidden gems.</p>
    <div class="grid" style="margin-top:12px">
      {% for heading, rows in combined %}
      <div class="scroll">
        <h3 style="margin:0 0 6px;font-size:15px">{{ heading }}</h3>
        <table>
          <tr><th>Film</th><th>Because you like</th><th class="num">Match</th></tr>
          {% for r in rows %}
          <tr><td>{{ r.title }}<div class="genres">{{ r.genres }}</div></td>
              <td>{{ r.because }}</td><td class="num">{{ "%.0f"|format(r.similarity*100) }}%</td></tr>
          {% else %}<tr><td colspan="3">No matches.</td></tr>{% endfor %}
        </table>
      </div>
      {% endfor %}
    </div>
  </section>

  {% for film, rows in per_film %}
  <section class="card scroll">
    <h2>People who liked {{ film }} also liked</h2>
    <table>
      <tr><th>#</th><th>Film</th><th class="num">Ratings</th><th class="num">Similarity</th></tr>
      {% for r in rows %}
      <tr><td class="num">{{ loop.index }}</td><td>{{ r.title }}<div class="genres">{{ r.genres }}</div></td>
          <td class="num">{{ r.ratings }}</td><td class="num">{{ "%.2f"|format(r.similarity) }}</td></tr>
      {% endfor %}
    </table>
  </section>
  {% endfor %}
  {% endif %}
</div>

<script>
  const titles = new Set({{ titles|tojson }}.map(t => t.toLowerCase()));
  function check(input) {
    const note = input.parentElement.querySelector("small");
    const v = input.value.trim();
    if (!v) { note.textContent = ""; return; }
    if (titles.has(v.toLowerCase())) { note.textContent = "✓ In the list"; note.className = "ok"; }
    else { note.textContent = "Not in the list yet. Keep typing or pick from the suggestions."; note.className = "bad"; }
  }
  const slots = document.getElementById("slots");
  slots.addEventListener("input", e => { if (e.target.matches("input")) check(e.target); });
  slots.querySelectorAll("input").forEach(check);
  document.getElementById("add").addEventListener("click", () => {
    const n = slots.children.length;
    if (n >= {{ max_fav }}) return;
    const label = slots.children[0].cloneNode(true);
    label.querySelector(".n").textContent = (n + 1) + ".";
    label.querySelector("input").value = "";
    label.querySelector("small").textContent = "";
    slots.appendChild(label);
    label.querySelector("input").focus();
  });
</script>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    entries = [""] * NUM_FAVOURITES
    errors, per_film, combined = [], [], []

    if request.method == "POST":
        entries = request.form.getlist("film")[:MAX_FAVOURITES]
        chosen = []
        for text in entries:
            if not text.strip():
                continue
            title, suggestions = find_title(text)
            if title is None:
                hint = f" Did you mean: {'; '.join(suggestions)}?" if suggestions else ""
                errors.append(f"“{text}” isn't in the list.{hint}")
            elif title_to_id[title] not in chosen:
                chosen.append(title_to_id[title])
        if not chosen and not errors:
            errors.append("Enter at least one film.")
        entries = [find_title(t)[0] or t for t in entries] + [""] * max(0, NUM_FAVOURITES - len(entries))

        if chosen and not errors:
            for mid in chosen:
                per_film.append((knn.titles[knn.index_of[mid]], knn.neighbours(mid, exclude=set(chosen))))
            combined = [
                ("Popular picks", knn.recommend(chosen, only=popular_ids)),
                ("Hidden gems (outside the top %d)" % TOP_N_POPULAR, knn.recommend(chosen, skip=popular_ids)),
            ]

    return render_template_string(
        PAGE, entries=entries, errors=errors, per_film=per_film, combined=combined,
        titles=dropdown_titles, num=NUM_FAVOURITES, max_fav=MAX_FAVOURITES,
        dropdown_size=len(dropdown_titles),
    )


if __name__ == "__main__":
    app.run(debug=False)