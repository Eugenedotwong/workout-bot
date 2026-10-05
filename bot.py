import os, re, io, json, math, asyncio, datetime as dt
from collections import defaultdict
from zoneinfo import ZoneInfo

import discord
from discord.ext import commands, tasks
import gspread
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

load_dotenv()

CHANNEL_ID = int(os.environ["WORKOUT_CHANNEL_ID"])
SHEET_ID = os.environ["SHEET_ID"]
PREFIX = os.getenv("PREFIX", "w!")  # change if another bot in your server uses the same prefix
CREDS = os.getenv("GOOGLE_CREDS", "credentials.json")
TZ = ZoneInfo(os.getenv("TIMEZONE", "America/New_York"))
SUMMARY_HOUR = int(os.getenv("SUMMARY_HOUR", "20"))  # Sunday, local time, 24h

HEADERS = ["Date/Time", "Exercise", "Muscle", "Weight (lb)", "Sets",
           "Reps/set", "Total reps", "Minutes", "Distance"]
USERS_TAB, SUMMARY_TAB = "Users", "Weekly Summary"
TS_FMT = "%Y-%m-%d %H:%M:%S"

# Order matters: first match wins. Edit freely to add your own exercises.
MUSCLE_MAP = [
    ("forearms", ["wrist", "forearm", "farmer"]),
    ("triceps", ["tricep", "skull", "pushdown", "dip", "close grip", "close-grip"]),
    ("legs", ["squat", "leg press", "lunge", "leg extension", "leg curl", "hamstring", "romanian",
              "rdl", "stiff", "good morning", "hip thrust", "glute", "bridge", "kickback", "calf",
              "calves", "step up", "step-up", "bulgarian", "adductor", "abductor", "quad", "nordic"]),
    ("shoulders", ["shoulder", "overhead", "ohp", "military", "lateral raise", "front raise",
                   "arnold", "face pull", "rear delt", "reverse fly", "upright row"]),
    ("chest", ["bench", "chest", "fly", "flye", "pec", "push up", "pushup", "push-up",
               "incline press"]),
    ("back", ["row", "pull up", "pullup", "pull-up", "chin up", "chin-up", "pulldown",
              "pull down", "deadlift", "shrug"]),
    ("biceps", ["bicep", "curl", "hammer"]),
    ("core", ["crunch", "plank", "sit up", "situp", "sit-up", "leg raise", "ab wheel",
              "abs", "core", "twist"]),
]

# "bench press 100 10x3" or "zercher squat 100 10x3 legs" -> exercise, weight, reps x sets, optional muscle
LOG_RE = re.compile(r"^(?P<ex>.+?)\s+(?P<w>\d+(?:\.\d+)?)\s+(?P<r>\d+)\s*[xX×]\s*(?P<s>\d+)(?P<extra>(?:\s+\S+)*)$")
# "running 30" or "running 30 3.2" -> exercise, minutes, optional distance
CARDIO_RE = re.compile(r"^(?P<ex>.+?)\s+(?P<m>\d+(?:\.\d+)?)(?:\s+(?P<d>\d+(?:\.\d+)?))?(?P<extra>(?:\s+\S+)*)$")


ALL_MUSCLES = [m for m, _ in MUSCLE_MAP] + ["other"]
ALIASES = {"quad": "legs", "quads": "legs", "hamstring": "legs", "hamstrings": "legs",
           "glute": "legs", "glutes": "legs", "calf": "legs", "calves": "legs", "leg": "legs",
           "bicep": "biceps", "tricep": "triceps", "shoulder": "shoulders", "delt": "shoulders",
           "delts": "shoulders", "forearm": "forearms", "ab": "core", "abs": "core",
           "pec": "chest", "pecs": "chest", "lat": "back", "lats": "back", "trap": "back",
           "traps": "back"}
LEGACY = {"quads": "legs", "hamstrings": "legs", "glutes": "legs", "calves": "legs"}


def canon(muscle):
    """Old rows used quads/hamstrings/glutes/calves; treat them all as legs."""
    return LEGACY.get(muscle, muscle)


def normalize_muscle(word):
    w = word.lower()
    w = ALIASES.get(w, w)
    return w if w in ALL_MUSCLES else None


def squash(s):
    return re.sub(r"[^a-z]", "", s.lower())


def osa(a, b):
    """Edit distance where swapping two adjacent letters counts as one edit (bnech -> bench = 1)."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def classify(exercise):
    """Returns (muscle, matched_keyword, was_fuzzy). muscle is None if the exercise is unknown."""
    e = exercise.lower()
    sq = squash(e)
    for muscle, kws in MUSCLE_MAP:  # exact: keyword appears in the name (spaces ignored for 5+ letters)
        for k in kws:
            ks = squash(k)
            if k in e or (len(ks) >= 5 and ks in sq):
                return muscle, k, False
    best, order = None, 0  # fuzzy: closest keyword within 1-2 typos
    for muscle, kws in MUSCLE_MAP:
        for k in kws:
            order += 1
            ks = squash(k)
            if len(ks) < 4:
                continue
            tol = 1 if len(ks) < 7 else 2
            for size in (len(ks) - 1, len(ks), len(ks) + 1):
                for i in range(len(sq) - size + 1):
                    w = sq[i:i + size]
                    if w[0] != ks[0]:
                        continue
                    dist = osa(w, ks)
                    if 0 < dist <= tol and (best is None or (dist, order) < best[:2]):
                        best = (dist, order, muscle, k)
    return (best[2], best[3], True) if best else (None, None, False)


def muscle_for(exercise: str) -> str:
    return classify(exercise)[0] or "other"


def closest_known(ex, known):
    """If ex is a 1-letter typo of an exercise this member already logged, reuse that name."""
    if ex in known:
        return ex
    sq = squash(ex)
    tol = 0 if len(sq) < 4 else 1
    best = None
    for k in known:
        d = osa(sq, squash(k))
        if d <= tol and (best is None or d < best[0]):
            best = (d, k)
    return best[1] if best else ex


def num(s):
    f = float(s)
    return int(f) if f.is_integer() else f


def now_str() -> str:
    return dt.datetime.now(TZ).strftime(TS_FMT)


def parse_date(tok):
    """today, yesterday, 2d (days ago), 10/3, 10/3/2026, 2026-10-03 -> date, or None if not a date."""
    t = tok.lower()
    today = dt.datetime.now(TZ).date()
    if t == "today":
        return today
    if t == "yesterday":
        return today - dt.timedelta(days=1)
    m = re.fullmatch(r"(\d{1,3})d", t)
    if m:
        return today - dt.timedelta(days=int(m[1]))
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})", t)
    if m:  # no year: this year, or last year if that would be in the future
        try:
            d = dt.date(today.year, int(m[1]), int(m[2]))
        except ValueError:
            return None
        return d if d <= today else d.replace(year=today.year - 1)
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(t, fmt).date()
        except ValueError:
            pass
    return None


def split_extras(extra):
    """Trailing words after the numbers: any mix of one date and one muscle. -> (date, muscle_word, error)"""
    date, word = None, None
    for tok in extra.split():
        d = parse_date(tok)
        if d and date is None:
            date = d
        elif word is None and not d:
            word = tok
        else:
            return None, None, f"Couldn't read `{tok}`."
    return date, word, None


def stamp_for(date):
    """Timestamp string for a (possibly past) date. Past days are stamped at noon. Returns (ts, error)."""
    today = dt.datetime.now(TZ).date()
    if date is None or date == today:
        return now_str(), None
    if date > today:
        return None, "That date is in the future."
    if (today - date).days > 365:
        return None, "That date is more than a year ago."
    return dt.datetime.combine(date, dt.time(12, 0)).strftime(TS_FMT), None


def this_monday() -> dt.datetime:
    now = dt.datetime.now(TZ).replace(tzinfo=None)
    return (now - dt.timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)


def aggregate(rows, week_start):
    """rows = sheet values incl. header. Returns (sets_by_muscle, minutes_by_muscle) for the week."""
    end = week_start + dt.timedelta(days=7)
    sets, mins = defaultdict(float), defaultdict(float)
    for r in rows[1:]:
        r = list(r) + [""] * 9
        try:
            ts = dt.datetime.strptime(r[0], TS_FMT)
        except ValueError:
            continue
        if not (week_start <= ts < end):
            continue
        if r[4]:
            sets[canon(r[2])] += float(r[4])
        if r[7]:
            mins[canon(r[2])] += float(r[7])
    return sets, mins


def week_of(ts):
    return (ts - dt.timedelta(days=ts.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)


def week_starts(n=12):
    mon = this_monday()
    return [mon - dt.timedelta(weeks=i) for i in range(n - 1, -1, -1)]


def parsed(rows):
    """Yield (timestamp, exercise, muscle, weight, sets, minutes) from sheet values."""
    f = lambda x: float(x) if x not in ("", None) else 0.0
    for r in rows[1:]:
        r = list(r) + [""] * 9
        try:
            ts = dt.datetime.strptime(r[0], TS_FMT)
        except ValueError:
            continue
        yield ts, r[1].lower(), canon(r[2]), f(r[3]), f(r[4]), f(r[7])


def weekly_series(rows, weeks, kind, key):
    """kind 'muscle': sets per week (minutes for cardio). kind 'exercise': top weight per week (nan = none)."""
    idx = {w: i for i, w in enumerate(weeks)}
    out = [0.0] * len(weeks) if kind == "muscle" else [float("nan")] * len(weeks)
    for ts, ex, muscle, wt, sets, mins in parsed(rows):
        i = idx.get(week_of(ts))
        if i is None:
            continue
        if kind == "muscle" and muscle == key:
            out[i] += mins if key == "cardio" else sets
        elif kind == "exercise" and key in ex:
            out[i] = wt if math.isnan(out[i]) else max(out[i], wt)
    return out


def render(title, ylabel, weeks, series, kind, labels=None, xlabel="Week starting"):
    """series = {label: [values per week]}; kind = 'line' | 'bar' | 'stack'. Returns PNG bytes buffer."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = labels or [w.strftime("%b %d") for w in weeks]
    x = list(range(len(labels)))
    if kind == "stack":
        bottom = [0.0] * len(labels)
        for name, vals in series.items():
            ax.bar(x, vals, bottom=bottom, label=name)
            bottom = [b + v for b, v in zip(bottom, vals)]
    elif kind == "bar":
        width = 0.8 / len(series)
        for j, (name, vals) in enumerate(series.items()):
            ax.bar([i - 0.4 + width * (j + 0.5) for i in x], vals, width, label=name)
    else:
        for name, vals in series.items():
            ax.plot(x, vals, marker="o", label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", alpha=0.3)
    if len(series) > 1:
        ax.legend()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------- Google Sheets ----------
# Hosting: set GOOGLE_CREDS_JSON (the full contents of credentials.json) instead of uploading the file
CREDS_JSON = os.getenv("GOOGLE_CREDS_JSON")
gc = (gspread.service_account_from_dict(json.loads(CREDS_JSON)) if CREDS_JSON
      else gspread.service_account(filename=CREDS))
book = gc.open_by_key(SHEET_ID)


def get_or_create(title, headers):
    try:
        return book.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = book.add_worksheet(title=title, rows=1000, cols=len(headers))
        ws.append_row(headers)
        ws.freeze(rows=1)
        return ws


users_ws = get_or_create(USERS_TAB, ["Discord user ID", "Tab name"])
summary_ws = get_or_create(SUMMARY_TAB, ["Week starting", "Member", "Muscle", "Total sets", "Minutes"])
users = {r[0]: r[1] for r in users_ws.get_all_values()[1:] if len(r) > 1}  # user id -> tab name


def user_ws(user):
    uid = str(user.id)
    if uid not in users:
        base = re.sub(r"[\[\]*?:/\\]", "", user.name).strip()[:80] or uid
        existing = {w.title for w in book.worksheets()}
        title, i = base, 2
        while title in existing:
            title = f"{base} {i}"
            i += 1
        users_ws.append_row([uid, title])
        users[uid] = title
    return get_or_create(users[uid], HEADERS)


def rows_for(user):
    title = users.get(str(user.id))
    return book.worksheet(title).get_all_values() if title else None


def known_exercises(user):
    """{exercise name: muscle} from this member's past entries (latest wins)."""
    out = {}
    for r in (rows_for(user) or [])[1:]:
        if len(r) > 2 and r[1]:
            out[r[1]] = canon(r[2])
    return out


def append_row(user, row):
    user_ws(user).append_row(row, value_input_option="RAW")


def undo_last(user):
    ws = user_ws(user)
    n = len(ws.col_values(1))
    if n <= 1:
        return None
    last = ws.row_values(n)
    ws.delete_rows(n)
    return last


def weekly_summary(week_start) -> str:
    lines, new_rows = [], []
    for ws in book.worksheets():
        if ws.title in (USERS_TAB, SUMMARY_TAB):
            continue
        sets, mins = aggregate(ws.get_all_values(), week_start)
        parts = []
        for m in sorted(set(sets) | set(mins)):
            new_rows.append([str(week_start.date()), ws.title, m,
                             sets.get(m) or "", mins.get(m) or ""])
            parts.append(f"{m} {sets[m]:g} sets" if m in sets else f"{m} {mins[m]:g} min")
        if parts:
            lines.append(f"**{ws.title}**: " + ", ".join(parts))
    if new_rows:
        summary_ws.append_rows(new_rows, value_input_option="RAW")
    head = f"📊 **Weekly summary: week of {week_start:%b %d}**\n"
    return (head + ("\n".join(lines) if lines else "No workouts logged this week."))[:1990]


# ---------- Discord ----------
PENDING = {}  # user id -> (expires, timestamp, exercise, weight, sets, reps) awaiting "yes" to log as other


class WorkoutBot(commands.Bot):
    async def setup_hook(self):
        weekly_job.start()


intents = discord.Intents.default()
intents.message_content = True
bot = WorkoutBot(command_prefix=PREFIX, intents=intents)


@bot.check
def only_workout_channel(ctx):
    return ctx.channel.id == CHANNEL_ID


@tasks.loop(time=dt.time(hour=SUMMARY_HOUR, tzinfo=TZ))
async def weekly_job():
    if dt.datetime.now(TZ).weekday() != 6:  # Sunday only
        return
    ch = bot.get_channel(CHANNEL_ID)
    if ch:
        await ch.send(await asyncio.to_thread(weekly_summary, this_monday()))


@bot.command(name="log")
async def log_cmd(ctx, *, text: str = ""):
    """!log bench press 100 10x3  (weight, then reps x sets)"""
    m = LOG_RE.match(text.strip())
    if not m:
        return await ctx.reply(f"Format: `{PREFIX}log bench press 100 10x3` (weight, then reps x sets). "
                               "Use 0 for bodyweight. Add a date to backfill: `yesterday`, `2d`, or `10/3`.",
                               mention_author=False)
    ex, w, reps, sets = m["ex"].strip().lower(), num(m["w"]), int(m["r"]), int(m["s"])
    if not (0 < reps <= 1000 and 0 < sets <= 100):
        return await ctx.reply("Those reps/sets look off. Try again.", mention_author=False)
    date, mu_word, err = split_extras(m["extra"])
    if err:
        return await ctx.reply(err + " Put at most one date and one muscle after the sets.",
                               mention_author=False)
    ts, err = stamp_for(date)
    if err:
        return await ctx.reply(err, mention_author=False)
    explicit = None
    if mu_word:
        explicit = normalize_muscle(mu_word)
        if not explicit:
            return await ctx.reply(f"Couldn't read `{mu_word}` as a date or muscle. "
                                   f"Muscles: {', '.join(ALL_MUSCLES)}. Dates: yesterday, 2d, 10/3.",
                                   mention_author=False)
    known = {k: v for k, v in (await asyncio.to_thread(known_exercises, ctx.author)).items()
             if v != "cardio"}
    fixed = closest_known(ex, known)  # fix 1-letter typos of exercises you've logged before
    muscle = explicit
    if not muscle:
        km = known.get(fixed)  # muscle you used for this exercise before
        muscle = km if km and km != "other" else (classify(fixed)[0] or km)
    if not muscle:
        PENDING[ctx.author.id] = (dt.datetime.now(TZ) + dt.timedelta(minutes=5), ts, fixed, w, sets, reps)
        await ctx.message.add_reaction("❓")
        return await ctx.reply(
            f"I don't recognize **{fixed}**, so it isn't logged yet. Reply **yes** to log it as `other`, "
            f"or resend with the muscle, e.g. `{PREFIX}log {fixed} {w} {reps}x{sets} legs`. "
            f"Muscles: {', '.join(ALL_MUSCLES)}. (Expires in 5 min.)", mention_author=False)
    note = f" | dated {ts[:10]}" if date and date != dt.datetime.now(TZ).date() else ""
    if fixed != ex:
        note += f" | corrected from '{ex}'"
    if not explicit and not known.get(fixed) and classify(fixed)[2]:
        note += f" | guessed from '{classify(fixed)[1]}', check spelling"
    await asyncio.to_thread(append_row, ctx.author,
                            [ts, fixed, muscle, w, sets, reps, reps * sets, "", ""])
    await ctx.message.add_reaction("✅")
    await ctx.reply(f"**{fixed}** {w} lb, {reps}x{sets} = {reps * sets} reps ({muscle}){note}",
                    mention_author=False)


@bot.command(name="cardio")
async def cardio_cmd(ctx, *, text: str = ""):
    """!cardio running 30 3.2  (minutes, optional distance)"""
    m = CARDIO_RE.match(text.strip())
    if not m:
        return await ctx.reply(f"Format: `{PREFIX}cardio running 30` (minutes) or "
                               f"`{PREFIX}cardio running 30 3.2` (minutes, distance). Add `yesterday`, `2d` or `10/3` to backfill.",
                               mention_author=False)
    ex, mins, dist = m["ex"].strip().lower(), num(m["m"]), num(m["d"]) if m["d"] else ""
    date, extra_word, err = split_extras(m["extra"])
    if err or extra_word:
        return await ctx.reply((err or f"Couldn't read `{extra_word}` as a date.") +
                               " Dates: yesterday, 2d, 10/3.", mention_author=False)
    ts, err = stamp_for(date)
    if err:
        return await ctx.reply(err, mention_author=False)
    known = {k for k, v in (await asyncio.to_thread(known_exercises, ctx.author)).items() if v == "cardio"}
    ex = closest_known(ex, known)
    await asyncio.to_thread(append_row, ctx.author,
                            [ts, ex, "cardio", "", "", "", "", mins, dist])
    await ctx.message.add_reaction("✅")
    when = f" | dated {ts[:10]}" if date and date != dt.datetime.now(TZ).date() else ""
    await ctx.reply(f"**{ex}** {mins} min" + (f", {dist} distance" if dist != "" else "") + when,
                    mention_author=False)


@bot.command(name="undo")
async def undo_cmd(ctx):
    """Delete your most recent entry."""
    last = await asyncio.to_thread(undo_last, ctx.author)
    await ctx.reply(f"Removed: {last[1]} ({last[0]})" if last else "Nothing to undo.",
                    mention_author=False)


@bot.command(name="summary")
async def summary_cmd(ctx):
    """Post this week's summary now."""
    await ctx.send(await asyncio.to_thread(weekly_summary, this_monday()))


@bot.command(name="chart")
async def chart_cmd(ctx, *, text: str = ""):
    """!chart  |  !chart chest  |  !chart bench press  |  !chart chest @Alan  |  !chart @Alan"""
    members = [ctx.author] + [u for u in ctx.message.mentions
                              if u.id != ctx.author.id and not u.bot]
    query = re.sub(r"<@!?\d+>", "", text).strip().lower()
    query = ALIASES.get(query, query)
    weeks = week_starts()
    async with ctx.typing():
        data = {u.name: await asyncio.to_thread(rows_for, u) for u in members}
    data = {name: rows for name, rows in data.items() if rows}
    if not data:
        return await ctx.reply("No workouts logged yet.", mention_author=False)

    labels, xlabel = None, "Week starting"
    if not query and len(members) > 1:  # compare members across all muscles (total sets, 12 weeks)
        if len(data) < 2:
            return await ctx.reply("Someone you mentioned has no workouts logged yet.",
                                   mention_author=False)
        total = lambda r, m: sum(weekly_series(r, weeks, "muscle", m))
        labels = [m for m in ALL_MUSCLES if any(total(r, m) > 0 for r in data.values())]
        series = {n: [total(r, m) for m in labels] for n, r in data.items()}
        title, ylabel, kind, xlabel = "Total sets by muscle, last 12 weeks", "Sets", "bar", "Muscle"
    elif not query:  # all muscles, stacked, for the author
        name = ctx.author.name
        if name not in data:
            return await ctx.reply("No workouts logged yet.", mention_author=False)
        series = {m: weekly_series(data[name], weeks, "muscle", m) for m in ALL_MUSCLES}
        series = {m: v for m, v in series.items() if any(v)}
        title, ylabel, kind = f"{name}: sets per week by muscle", "Sets", "stack"
    elif query in ALL_MUSCLES or query == "cardio":
        series = {n: weekly_series(r, weeks, "muscle", query) for n, r in data.items()}
        unit = "Minutes" if query == "cardio" else "Sets"
        title, ylabel, kind = f"{query.title()}: {unit.lower()} per week", unit, "bar"
    else:  # exercise name, top weight per week
        series = {n: weekly_series(r, weeks, "exercise", query) for n, r in data.items()}
        series = {n: v for n, v in series.items() if not all(math.isnan(x) for x in v)}
        title, ylabel, kind = f"{query.title()}: top weight per week", "Weight (lb)", "line"

    vals = [x for v in series.values() for x in v if not math.isnan(x)]
    if not vals or (kind != "line" and not any(x > 0 for x in vals)):
        return await ctx.reply(f"No data for `{query or 'anything'}` in the last 12 weeks.",
                               mention_author=False)
    buf = await asyncio.to_thread(render, title, ylabel, weeks, series, kind, labels, xlabel)
    await ctx.send(file=discord.File(buf, "chart.png"))


@bot.event
async def on_message(msg):
    if msg.content.startswith(PREFIX) and not msg.author.bot:  # debug: shows what the bot receives
        ok = "OK" if msg.channel.id == CHANNEL_ID else f"WRONG CHANNEL (expected {CHANNEL_ID})"
        print(f"[command seen] channel {msg.channel.id} {ok}: {msg.content!r}")
    if (not msg.author.bot and msg.channel.id == CHANNEL_ID and msg.author.id in PENDING
            and not msg.content.startswith(PREFIX)):
        expires, ts, ex, w, sets, reps = PENDING[msg.author.id]
        answer = msg.content.strip().lower()
        if dt.datetime.now(TZ) > expires:
            PENDING.pop(msg.author.id, None)
        elif answer in ("yes", "y", "confirm"):
            PENDING.pop(msg.author.id, None)
            try:
                await asyncio.to_thread(append_row, msg.author,
                                        [ts, ex, "other", w, sets, reps, reps * sets, "", ""])
            except Exception as e:
                print("error:", repr(e))
                return await msg.add_reaction("❌")
            await msg.add_reaction("✅")
            return await msg.reply(f"Logged **{ex}** as `other` and I'll remember that. To categorize it "
                                   f"later, resend with a muscle.", mention_author=False)
        elif answer in ("no", "n", "cancel"):
            PENDING.pop(msg.author.id, None)
            return await msg.reply("Cancelled, nothing logged.", mention_author=False)
    await bot.process_commands(msg)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}. Watching channel {CHANNEL_ID}.")


@bot.event
async def on_command_error(ctx, err):
    if isinstance(err, (commands.CheckFailure, commands.CommandNotFound)):
        print("[ignored]", type(err).__name__)
        return
    print("error:", repr(err))
    await ctx.message.add_reaction("❌")


bot.run(os.environ["DISCORD_TOKEN"])
