import streamlit as st
import pandas as pd
import random
import time
import os
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import numpy as np

# =====================
# OPTIONAL: OPENAI API KEY
# Set here or via environment variable OPENAI_API_KEY
# =====================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")  # or paste key directly: "sk-..."

# =====================
# QUESTION BANK
# difficulty: "easy" | "medium" | "hard"
# tags: topic tags for grouping/reporting
# =====================
question_bank = {
    "Python": [
        {"q": "Which keyword is used to define a function in Python?", "options": ["def", "func", "define", "fun"], "answer": "def", "difficulty": "easy", "tags": ["basics"]},
        {"q": "What data type is the result of type([])?", "options": ["list", "tuple", "dict", "set"], "answer": "list", "difficulty": "easy", "tags": ["basics"]},
        {"q": "Which of these is used to handle exceptions in Python?", "options": ["try/except", "catch/throw", "error/handle", "check/fail"], "answer": "try/except", "difficulty": "easy", "tags": ["error-handling"]},
        {"q": "What is the output of len('hello')?", "options": ["4", "5", "6", "error"], "answer": "5", "difficulty": "easy", "tags": ["basics"]},
        {"q": "Which method adds an element to the end of a list?", "options": ["append()", "add()", "insert()", "push()"], "answer": "append()", "difficulty": "easy", "tags": ["lists"]},
        {"q": "What does the 'self' keyword refer to in a Python class?", "options": ["The current instance", "The class itself", "The parent class", "A global variable"], "answer": "The current instance", "difficulty": "medium", "tags": ["oop"]},
        {"q": "Which of the following is immutable in Python?", "options": ["tuple", "list", "dict", "set"], "answer": "tuple", "difficulty": "medium", "tags": ["data-types"]},
        {"q": "What is a lambda function in Python?", "options": ["An anonymous function", "A recursive function", "A built-in function", "A class method"], "answer": "An anonymous function", "difficulty": "medium", "tags": ["functions"]},
        {"q": "What does the 'yield' keyword do in Python?", "options": ["Returns a generator", "Exits the program", "Raises an exception", "Imports a module"], "answer": "Returns a generator", "difficulty": "hard", "tags": ["advanced"]},
        {"q": "Which module is used for regular expressions in Python?", "options": ["re", "regex", "regexp", "string"], "answer": "re", "difficulty": "medium", "tags": ["modules"]},
    ],
    "DSA": [
        {"q": "Which data structure follows FIFO order?", "options": ["Queue", "Stack", "Tree", "Graph"], "answer": "Queue", "difficulty": "easy", "tags": ["basics"]},
        {"q": "Which data structure uses LIFO order?", "options": ["Stack", "Queue", "Heap", "LinkedList"], "answer": "Stack", "difficulty": "easy", "tags": ["basics"]},
        {"q": "What is the time complexity of Binary Search?", "options": ["O(log n)", "O(n)", "O(n^2)", "O(1)"], "answer": "O(log n)", "difficulty": "medium", "tags": ["searching"]},
        {"q": "Which traversal visits the root node first?", "options": ["Pre-order", "In-order", "Post-order", "Level-order"], "answer": "Pre-order", "difficulty": "medium", "tags": ["trees"]},
        {"q": "In a min-heap, the root contains the?", "options": ["Minimum element", "Maximum element", "Median element", "Random element"], "answer": "Minimum element", "difficulty": "medium", "tags": ["heaps"]},
        {"q": "What is the time complexity of accessing an element in a hash table on average?", "options": ["O(1)", "O(n)", "O(log n)", "O(n^2)"], "answer": "O(1)", "difficulty": "medium", "tags": ["hashing"]},
        {"q": "Which sorting algorithm is stable and has O(n log n) average complexity?", "options": ["Merge Sort", "Quick Sort", "Heap Sort", "Selection Sort"], "answer": "Merge Sort", "difficulty": "medium", "tags": ["sorting"]},
        {"q": "What is the worst-case time complexity of QuickSort?", "options": ["O(n^2)", "O(n log n)", "O(n)", "O(log n)"], "answer": "O(n^2)", "difficulty": "hard", "tags": ["sorting"]},
        {"q": "What is the space complexity of Merge Sort?", "options": ["O(n)", "O(1)", "O(log n)", "O(n^2)"], "answer": "O(n)", "difficulty": "hard", "tags": ["sorting"]},
        {"q": "Which algorithm is used to find the shortest path in a weighted graph?", "options": ["Dijkstra's", "BFS", "DFS", "Prim's"], "answer": "Dijkstra's", "difficulty": "hard", "tags": ["graphs"]},
    ],
    "DBMS": [
        {"q": "What does a primary key ensure in a table?", "options": ["Uniqueness of each row", "Foreign key reference", "Null values allowed", "Duplicate rows"], "answer": "Uniqueness of each row", "difficulty": "easy", "tags": ["basics"]},
        {"q": "Which SQL command is used to retrieve data from a table?", "options": ["SELECT", "INSERT", "UPDATE", "DELETE"], "answer": "SELECT", "difficulty": "easy", "tags": ["sql"]},
        {"q": "Which command permanently saves a transaction in SQL?", "options": ["COMMIT", "ROLLBACK", "SAVEPOINT", "BEGIN"], "answer": "COMMIT", "difficulty": "easy", "tags": ["transactions"]},
        {"q": "What is a foreign key?", "options": ["A key that references a primary key in another table", "A key that is always unique", "A key that cannot be null", "A key used for indexing"], "answer": "A key that references a primary key in another table", "difficulty": "medium", "tags": ["keys"]},
        {"q": "What does ACID stand for in database transactions?", "options": ["Atomicity, Consistency, Isolation, Durability", "Access, Control, Index, Data", "Aggregate, Commit, Insert, Delete", "None of the above"], "answer": "Atomicity, Consistency, Isolation, Durability", "difficulty": "medium", "tags": ["transactions"]},
        {"q": "Which JOIN returns all rows from both tables, with NULLs where no match?", "options": ["FULL OUTER JOIN", "INNER JOIN", "LEFT JOIN", "CROSS JOIN"], "answer": "FULL OUTER JOIN", "difficulty": "medium", "tags": ["joins"]},
        {"q": "What is a view in SQL?", "options": ["A virtual table based on a query", "A physical copy of a table", "An index on a table", "A stored procedure"], "answer": "A virtual table based on a query", "difficulty": "medium", "tags": ["sql"]},
        {"q": "Which SQL clause is used to filter groups after GROUP BY?", "options": ["HAVING", "WHERE", "FILTER", "ORDER BY"], "answer": "HAVING", "difficulty": "medium", "tags": ["sql"]},
        {"q": "Which normal form eliminates partial dependencies?", "options": ["2NF", "1NF", "3NF", "BCNF"], "answer": "2NF", "difficulty": "hard", "tags": ["normalization"]},
        {"q": "What is denormalization?", "options": ["Adding redundancy to improve read performance", "Removing duplicate data", "Splitting tables into smaller ones", "Creating indexes"], "answer": "Adding redundancy to improve read performance", "difficulty": "hard", "tags": ["normalization"]},
    ],
    "General": [
        {"q": "What does CPU stand for?", "options": ["Central Processing Unit", "Core Processing Unit", "Central Program Utility", "Computer Processing Unit"], "answer": "Central Processing Unit", "difficulty": "easy", "tags": ["hardware"]},
        {"q": "What does RAM stand for?", "options": ["Random Access Memory", "Read Access Memory", "Rapid Access Module", "Runtime Access Memory"], "answer": "Random Access Memory", "difficulty": "easy", "tags": ["hardware"]},
        {"q": "What does HTTP stand for?", "options": ["HyperText Transfer Protocol", "High Transfer Text Protocol", "HyperText Transmission Process", "Host Transfer Text Protocol"], "answer": "HyperText Transfer Protocol", "difficulty": "easy", "tags": ["networking"]},
        {"q": "What is the full form of URL?", "options": ["Uniform Resource Locator", "Universal Resource Link", "Uniform Retrieval Locator", "Universal Retrieval Link"], "answer": "Uniform Resource Locator", "difficulty": "easy", "tags": ["networking"]},
        {"q": "Which of the following is an example of an operating system?", "options": ["Linux", "Python", "MySQL", "Apache"], "answer": "Linux", "difficulty": "easy", "tags": ["os"]},
        {"q": "Which protocol is used to send emails?", "options": ["SMTP", "HTTP", "FTP", "SSH"], "answer": "SMTP", "difficulty": "medium", "tags": ["networking"]},
        {"q": "What is the binary representation of the decimal number 10?", "options": ["1010", "1001", "1100", "0110"], "answer": "1010", "difficulty": "medium", "tags": ["binary"]},
        {"q": "Which layer of the OSI model handles routing?", "options": ["Network Layer", "Transport Layer", "Data Link Layer", "Session Layer"], "answer": "Network Layer", "difficulty": "medium", "tags": ["networking"]},
        {"q": "What is the purpose of a firewall?", "options": ["To monitor and control network traffic", "To speed up internet connection", "To store data securely", "To compress files"], "answer": "To monitor and control network traffic", "difficulty": "medium", "tags": ["security"]},
        {"q": "What does SSL stand for?", "options": ["Secure Sockets Layer", "Safe Server Link", "Secure System Login", "Standard Socket Layer"], "answer": "Secure Sockets Layer", "difficulty": "hard", "tags": ["security"]},
    ]
}

DIFFICULTY_ORDER = ["easy", "medium", "hard"]
DIFFICULTY_EMOJI = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
DIFF_SCORE = {"easy": 1, "medium": 2, "hard": 3}


# =====================
# ADAPTIVE ENGINE
# =====================
def get_next_question(domain, asked_indices, current_difficulty, correct_streak=0):
    pool = question_bank.get(domain, [])
    if correct_streak >= 5:
        return None, current_difficulty  # early stop

    candidates = [(i, q) for i, q in enumerate(pool)
                  if i not in asked_indices and q["difficulty"] == current_difficulty]
    if not candidates:
        for diff in DIFFICULTY_ORDER:
            candidates = [(i, q) for i, q in enumerate(pool)
                          if i not in asked_indices and q["difficulty"] == diff]
            if candidates:
                current_difficulty = diff
                break
    if not candidates:
        return None, current_difficulty

    idx, question = random.choice(candidates)
    return (idx, question), current_difficulty


def update_difficulty(current_difficulty, correct_streak, wrong_count):
    if correct_streak >= 2:
        if current_difficulty == "easy":   return "medium"
        elif current_difficulty == "medium": return "hard"
    if wrong_count >= 2:
        if current_difficulty == "hard":   return "medium"
        elif current_difficulty == "medium": return "easy"
    return current_difficulty


# =====================
# AI FEATURE 1: SKILL GAP DETECTION (Random Forest)
# Input: per-answer features → Output: weak/strong label per tag
# =====================
def detect_skill_gaps(df):
    if len(df) < 3:
        return {}, {}

    diff_map = {"easy": 1, "medium": 2, "hard": 3}
    df = df.copy()
    df["diff_num"] = df["difficulty"].map(diff_map)
    df["time_norm"] = (df["time_taken"] - df["time_taken"].min()) / (df["time_taken"].max() - df["time_taken"].min() + 1e-9)

    # Build tag-level feature matrix
    tag_rows = []
    for _, row in df.iterrows():
        for tag in row["tags"].split(", "):
            tag_rows.append({
                "tag": tag,
                "correct": row["correct"],
                "diff_num": row["diff_num"],
                "time_norm": row["time_norm"],
                "is_guess": row["is_guess"],
            })
    tag_df = pd.DataFrame(tag_rows)

    tag_acc = tag_df.groupby("tag")["correct"].mean()
    tag_avg_time = tag_df.groupby("tag")["time_norm"].mean()

    weak_tags = tag_acc[tag_acc < 0.6].index.tolist()
    strong_tags = tag_acc[tag_acc >= 0.6].index.tolist()

    # RF model: predict correctness from difficulty + time + guess flag
    X = df[["diff_num", "time_norm", "is_guess"]].values
    y = df["correct"].values
    if len(set(y)) > 1:
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)
        importances = dict(zip(["difficulty", "time_taken", "guessing"], clf.feature_importances_))
    else:
        importances = {}

    return {"weak": weak_tags, "strong": strong_tags, "tag_acc": tag_acc.to_dict()}, importances


# =====================
# AI FEATURE 2: PERSONALIZED RECOMMENDATIONS (OpenAI or rule-based fallback)
# =====================
RESOURCES = {
    "oop":           ("Practice OOP problems on LeetCode",        "https://leetcode.com/tag/design/"),
    "basics":        ("Review Python basics on W3Schools",         "https://www.w3schools.com/python/"),
    "error-handling":("Python exception handling guide",           "https://docs.python.org/3/tutorial/errors.html"),
    "functions":     ("Deep dive into Python functions",           "https://realpython.com/defining-your-own-python-function/"),
    "advanced":      ("Advanced Python concepts on Real Python",   "https://realpython.com/"),
    "data-types":    ("Python data types reference",               "https://docs.python.org/3/library/stdtypes.html"),
    "lists":         ("Python lists tutorial",                     "https://realpython.com/python-lists-tuples/"),
    "modules":       ("Python modules and packages",               "https://docs.python.org/3/tutorial/modules.html"),
    "sorting":       ("Sorting algorithms visualized",             "https://visualgo.net/en/sorting"),
    "searching":     ("Binary search explained",                   "https://www.geeksforgeeks.org/binary-search/"),
    "trees":         ("Tree traversal guide",                      "https://www.geeksforgeeks.org/tree-traversals-inorder-preorder-and-postorder/"),
    "graphs":        ("Graph algorithms on GeeksForGeeks",         "https://www.geeksforgeeks.org/graph-data-structure-and-algorithms/"),
    "heaps":         ("Heap data structure explained",             "https://www.geeksforgeeks.org/heap-data-structure/"),
    "hashing":       ("Hash tables explained",                     "https://www.geeksforgeeks.org/hashing-data-structure/"),
    "basics":        ("DSA basics on GeeksForGeeks",               "https://www.geeksforgeeks.org/data-structures/"),
    "sql":           ("SQL practice on SQLZoo",                    "https://sqlzoo.net/"),
    "joins":         ("SQL JOINs explained",                       "https://www.w3schools.com/sql/sql_join.asp"),
    "transactions":  ("Database transactions guide",               "https://www.geeksforgeeks.org/transaction-management-in-dbms/"),
    "normalization": ("Database normalization explained",          "https://www.geeksforgeeks.org/normal-forms-in-dbms/"),
    "keys":          ("Primary and foreign keys guide",            "https://www.geeksforgeeks.org/types-of-keys-in-relational-model/"),
    "networking":    ("Computer networking basics",                "https://www.geeksforgeeks.org/computer-network-tutorials/"),
    "hardware":      ("Computer hardware overview",                "https://www.geeksforgeeks.org/computer-organization-and-architecture-tutorials/"),
    "security":      ("Cybersecurity fundamentals",                "https://www.geeksforgeeks.org/cyber-security-tutorial/"),
    "binary":        ("Number systems explained",                  "https://www.geeksforgeeks.org/number-system-and-base-conversions/"),
    "os":            ("Operating systems basics",                  "https://www.geeksforgeeks.org/operating-systems/"),
}


def get_llm_recommendation(name, domain, weak_tags, strong_tags, accuracy, api_key, behavior=None):
    """
    Calls OpenAI GPT with a rich prompt that returns 3 structured sections:
    1. Personalized Feedback
    2. Study Plan (week-by-week)
    3. Curated Resources
    Falls back to rule-based if no key provided.
    """
    if not api_key:
        return None, "no_key"

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        behavior_note = ""
        if behavior:
            behavior_note = f"""
- Avg time per question: {behavior.get('avg_time', 0):.1f}s
- Likely guesses: {behavior.get('guess_count', 0)}
- Struggled questions: {behavior.get('struggle_count', 0)}
- Confidence score: {behavior.get('confidence', 0):.0f}/100"""

        prompt = f"""You are an expert technical learning coach reviewing a quiz result.

Student: {name or 'the student'}
Domain: {domain}
Score: {accuracy:.1f}%
Strong topics: {', '.join(strong_tags) if strong_tags else 'None'}
Weak topics: {', '.join(weak_tags) if weak_tags else 'None'}{behavior_note}

Respond in EXACTLY this format with these three labeled sections:

## 1. Personalized Feedback
Write 3-4 sentences. Be specific about what they did well and where they struggled. Reference their actual topics. Be encouraging but honest.

## 2. Study Plan
Give a realistic 2-week study plan with daily focus areas. Format as:
Week 1: [focus]
- Day 1-2: ...
- Day 3-4: ...
- Day 5-7: ...
Week 2: [focus]
- Day 1-3: ...
- Day 4-7: ...

## 3. Recommended Resources
List 4-5 specific resources (name + URL) tailored to their weak topics. Format as:
- [Resource name](URL) — one line reason why

Be specific, practical, and avoid generic advice."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip(), "gpt"
    except Exception as e:
        return None, str(e)


def rule_based_recommendation(name, domain, weak_tags, strong_tags, accuracy):
    lines = []
    n = name or "You"
    if strong_tags:
        lines.append(f"✅ {n} showed strong understanding in: **{', '.join(strong_tags)}**.")
    if weak_tags:
        lines.append(f"⚠️ Areas needing work: **{', '.join(weak_tags)}**.")
        lines.append("\n**Recommended resources:**")
        for tag in weak_tags:
            if tag in RESOURCES:
                label, url = RESOURCES[tag]
                lines.append(f"- [{label}]({url})")
    if accuracy >= 80:
        lines.append(f"\n🚀 Excellent overall — {n} is ready for advanced {domain} topics!")
    elif accuracy >= 60:
        lines.append(f"\n📈 Good progress — focus on the weak areas above to level up.")
    else:
        lines.append(f"\n📚 Keep practicing — revisit the fundamentals and try again.")
    return "\n".join(lines)


# =====================
# AI FEATURE 3: BEHAVIORAL ANALYSIS
# Tracks time per question, guessing patterns, confidence
# =====================
def analyze_behavior(df):
    if df.empty:
        return {}

    avg_time = df["time_taken"].mean()
    fast_threshold = avg_time * 0.5   # answered in less than half avg time = likely guessing
    slow_threshold = avg_time * 1.8   # took much longer = struggled

    df = df.copy()
    df["behavior"] = "normal"
    df.loc[df["time_taken"] < fast_threshold, "behavior"] = "fast/guess"
    df.loc[df["time_taken"] > slow_threshold, "behavior"] = "slow/struggled"

    guess_count = (df["behavior"] == "fast/guess").sum()
    struggle_count = (df["behavior"] == "slow/struggled").sum()

    # Confidence: correct on hard = high confidence, wrong on easy = low confidence
    diff_map = {"easy": 1, "medium": 2, "hard": 3}
    df["diff_num"] = df["difficulty"].map(diff_map)
    df["confidence_score"] = df["correct"] * df["diff_num"]
    confidence = df["confidence_score"].mean() / 3 * 100  # normalize to 0-100

    return {
        "avg_time": avg_time,
        "guess_count": int(guess_count),
        "struggle_count": int(struggle_count),
        "confidence": confidence,
        "behavior_df": df[["difficulty", "tags", "correct", "time_taken", "behavior"]],
    }


# =====================
# PDF REPORT GENERATOR
# =====================
def generate_pdf(name, domain, accuracy, questions_count, final_difficulty,
                 confidence, strong_tags, weak_tags, tag_acc,
                 behavior, llm_text):
    import io
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    # Custom styles
    title_style   = ParagraphStyle("title",   fontSize=22, textColor=colors.HexColor("#1e293b"),
                                   spaceAfter=6, alignment=TA_CENTER, fontName="Helvetica-Bold")
    sub_style     = ParagraphStyle("sub",     fontSize=11, textColor=colors.HexColor("#64748b"),
                                   spaceAfter=16, alignment=TA_CENTER)
    heading_style = ParagraphStyle("heading", fontSize=13, textColor=colors.HexColor("#1e40af"),
                                   spaceBefore=14, spaceAfter=6, fontName="Helvetica-Bold")
    body_style    = ParagraphStyle("body",    fontSize=10, textColor=colors.HexColor("#1e293b"),
                                   spaceAfter=4, leading=15)
    tag_good      = ParagraphStyle("tg",      fontSize=10, textColor=colors.HexColor("#065f46"),
                                   spaceAfter=3)
    tag_bad       = ParagraphStyle("tb",      fontSize=10, textColor=colors.HexColor("#991b1b"),
                                   spaceAfter=3)

    story = []

    # ── Header ──
    story.append(Paragraph("Skill Gap Intelligence Report", title_style))
    story.append(Paragraph(f"Adaptive AI Quiz · {domain} Domain", sub_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#3b82f6")))
    story.append(Spacer(1, 0.4*cm))

    # ── Candidate info ──
    story.append(Paragraph("Candidate Information", heading_style))
    info_data = [
        ["Name",             name or "—"],
        ["Domain",           domain],
        ["Date",             pd.Timestamp.now().strftime("%B %d, %Y")],
    ]
    info_table = Table(info_data, colWidths=[5*cm, 12*cm])
    info_table.setStyle(TableStyle([
        ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 10),
        ("TEXTCOLOR",   (0,0), (0,-1), colors.HexColor("#1e40af")),
        ("TEXTCOLOR",   (1,0), (1,-1), colors.HexColor("#1e293b")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#f8fafc"), colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING",     (0,0), (-1,-1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Score summary ──
    story.append(Paragraph("Performance Summary", heading_style))
    score_data = [
        ["Metric", "Value"],
        ["Overall Accuracy",      f"{accuracy:.1f}%"],
        ["Questions Answered",    str(questions_count)],
        ["Final Difficulty Level", final_difficulty.capitalize()],
        ["Confidence Score",      f"{confidence:.0f} / 100"],
        ["Avg Time per Question", f"{behavior.get('avg_time', 0):.1f}s" if behavior else "—"],
        ["Likely Guesses",        str(behavior.get('guess_count', 0)) if behavior else "—"],
    ]
    score_table = Table(score_data, colWidths=[9*cm, 8*cm])
    score_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#1e40af")),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f0f9ff"), colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING",     (0,0), (-1,-1), 7),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Topic accuracy table ──
    story.append(Paragraph("Topic-Level Accuracy", heading_style))
    topic_data = [["Topic", "Accuracy", "Status"]]
    for tag, acc in sorted(tag_acc.items(), key=lambda x: x[1]):
        pct = f"{acc*100:.0f}%"
        status = "✗ Needs Work" if acc < 0.6 else "✓ Strong"
        topic_data.append([tag, pct, status])
    topic_table = Table(topic_data, colWidths=[8*cm, 4*cm, 5*cm])
    topic_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#1e40af")),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f8fafc"), colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING",     (0,0), (-1,-1), 7),
    ]))
    story.append(topic_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Strengths & Weaknesses ──
    story.append(Paragraph("Strengths & Weaknesses", heading_style))
    sw_data = [["✅ Strong Topics", "❌ Weak Topics"]]
    max_len = max(len(strong_tags), len(weak_tags), 1)
    for i in range(max_len):
        s = strong_tags[i] if i < len(strong_tags) else ""
        w = weak_tags[i]   if i < len(weak_tags)   else ""
        sw_data.append([s, w])
    sw_table = Table(sw_data, colWidths=[8.5*cm, 8.5*cm])
    sw_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#1e40af")),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 10),
        ("TEXTCOLOR",   (0,1), (0,-1), colors.HexColor("#065f46")),
        ("TEXTCOLOR",   (1,1), (1,-1), colors.HexColor("#991b1b")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f0fdf4"), colors.HexColor("#fef2f2")]),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING",     (0,0), (-1,-1), 7),
    ]))
    story.append(sw_table)
    story.append(Spacer(1, 0.4*cm))

    # ── AI Recommendations ──
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(Paragraph("AI-Generated Recommendations", heading_style))
    if llm_text:
        import re
        sections = re.split(r"##\s*\d+\.\s*", llm_text)
        sec_labels = ["Personalized Feedback", "Study Plan", "Recommended Resources"]
        sec_colors = ["#1e40af", "#065f46", "#6d28d9"]
        for i, sec in enumerate(s.strip() for s in sections if s.strip()):
            if i >= 3:
                break
            label = sec_labels[i] if i < len(sec_labels) else f"Section {i+1}"
            color = sec_colors[i] if i < len(sec_colors) else "#1e293b"
            story.append(Paragraph(label, ParagraphStyle(
                f"sh{i}", fontSize=11, textColor=colors.HexColor(color),
                fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4
            )))
            # Strip markdown bold/links for PDF
            clean = re.sub(r"\*\*(.*?)\*\*", r"\1", sec)
            clean = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", clean)
            for line in clean.split("\n"):
                line = line.strip()
                if line:
                    story.append(Paragraph(line, body_style))
    else:
        story.append(Paragraph("No GPT recommendations available. Add an OpenAI API key for AI-powered feedback.", body_style))
        if weak_tags:
            story.append(Paragraph(f"Focus on: {', '.join(weak_tags)}", body_style))

    # ── Footer ──
    story.append(Spacer(1, 0.6*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(Paragraph("Generated by Adaptive Skill Gap Intelligence Engine · AI Powered",
                            ParagraphStyle("footer", fontSize=8, textColor=colors.HexColor("#94a3b8"),
                                           alignment=TA_CENTER, spaceBefore=6)))

    doc.build(story)
    buf.seek(0)
    return buf


# =====================
# UI CONFIG
# =====================
st.set_page_config(page_title="Skill Gap AI", layout="wide", page_icon="🎯")

st.markdown("""
<style>
.main { background-color: #0f172a; }
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: white;
}
.domain-card {
    background: linear-gradient(135deg, #1e3a5f, #0f172a);
    border: 1px solid #2563eb;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background:linear-gradient(135deg,#1e293b,#0f172a);
padding:30px;border-radius:16px;text-align:center;color:white;margin-bottom:20px;
border: 1px solid #334155;'>
<div style='font-size:36px;font-weight:bold;'>🎯 Adaptive Skill Gap Intelligence Engine</div>
<div style='font-size:14px;color:#94a3b8;margin-top:8px;'>AI-Powered • Adaptive Difficulty • Real-time Analysis</div>
</div>
""", unsafe_allow_html=True)

# =====================
# SESSION STATE
# =====================
defaults = {
    "page": 1,
    "domain": "Python",
    "current_difficulty": "easy",
    "asked_indices": [],
    "answers": [],
    "current_q": None,
    "current_q_idx": None,
    "early_stop": False,
    "name": "",
    "correct_streak": 0,
    "wrong_count": 0,
    "q_start_time": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =====================
# PAGE 1 — DASHBOARD + START
# =====================
if st.session_state.page == 1:

    # ── Domain Overview Dashboard ──
    st.markdown("### 📊 Domain Overview")
    domains = ["Python", "DSA", "DBMS", "General"]
    domain_info = {
        "Python":  {"icon": "🐍", "topics": 10, "levels": "Easy → Hard", "color": "#3b82f6"},
        "DSA":     {"icon": "🌲", "topics": 10, "levels": "Easy → Hard", "color": "#10b981"},
        "DBMS":    {"icon": "🗄️",  "topics": 10, "levels": "Easy → Hard", "color": "#f59e0b"},
        "General": {"icon": "🌐", "topics": 10, "levels": "Easy → Hard", "color": "#8b5cf6"},
    }

    dcols = st.columns(4)
    for i, (dom, info) in enumerate(domain_info.items()):
        with dcols[i]:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1e293b,#0f172a);
            border:1px solid {info["color"]};border-radius:12px;padding:16px;text-align:center;color:white;'>
            <div style='font-size:28px;'>{info["icon"]}</div>
            <div style='font-size:16px;font-weight:bold;margin:6px 0;'>{dom}</div>
            <div style='font-size:12px;color:#94a3b8;'>{info["topics"]} Questions</div>
            <div style='font-size:12px;color:{info["color"]};'>{info["levels"]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── How It Works ──
    st.markdown("### ⚙️ How the Adaptive Engine Works")
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.markdown("""<div style='background:#1e293b;border-radius:10px;padding:14px;text-align:center;color:white;border:1px solid #334155;'>
        <div style='font-size:22px;'>🟢</div><div style='font-weight:bold;margin:4px 0;'>Start Easy</div>
        <div style='font-size:12px;color:#94a3b8;'>Every quiz begins at easy level</div></div>""", unsafe_allow_html=True)
    with h2:
        st.markdown("""<div style='background:#1e293b;border-radius:10px;padding:14px;text-align:center;color:white;border:1px solid #334155;'>
        <div style='font-size:22px;'>📈</div><div style='font-weight:bold;margin:4px 0;'>2 Correct → Harder</div>
        <div style='font-size:12px;color:#94a3b8;'>Streak promotes difficulty</div></div>""", unsafe_allow_html=True)
    with h3:
        st.markdown("""<div style='background:#1e293b;border-radius:10px;padding:14px;text-align:center;color:white;border:1px solid #334155;'>
        <div style='font-size:22px;'>📉</div><div style='font-weight:bold;margin:4px 0;'>2 Wrong → Easier</div>
        <div style='font-size:12px;color:#94a3b8;'>Struggles demote difficulty</div></div>""", unsafe_allow_html=True)
    with h4:
        st.markdown("""<div style='background:#1e293b;border-radius:10px;padding:14px;text-align:center;color:white;border:1px solid #334155;'>
        <div style='font-size:22px;'>🏆</div><div style='font-weight:bold;margin:4px 0;'>5 Streak → Stop</div>
        <div style='font-size:12px;color:#94a3b8;'>Early exit if you're an expert</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── Quiz Setup ──
    st.markdown("### 🚀 Start Your Quiz")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.name = st.text_input("Your Name", placeholder="Enter your name")
        domain = st.selectbox("Select Domain", domains,
                              format_func=lambda d: f"{domain_info[d]['icon']} {d}")
        st.session_state.domain = domain
    with c2:
        api_key_input = st.text_input("OpenAI API Key (optional)",
                                      type="password",
                                      placeholder="sk-... for GPT-powered recommendations")
        if api_key_input:
            OPENAI_API_KEY = api_key_input
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎯 Start Adaptive Quiz", use_container_width=True, type="primary"):
            st.session_state.current_difficulty = "easy"
            st.session_state.asked_indices = []
            st.session_state.answers = []
            st.session_state.early_stop = False
            st.session_state.correct_streak = 0
            st.session_state.wrong_count = 0

            result, diff = get_next_question(domain, [], "easy", 0)
            if result:
                idx, q = result
                st.session_state.current_q_idx = idx
                st.session_state.current_q = q
                st.session_state.current_difficulty = diff
                st.session_state.q_start_time = time.time()
            st.session_state.page = 2
            st.rerun()
    st.stop()

# =====================
# PAGE 2 — QUIZ
# =====================
if st.session_state.page == 2:

    if st.session_state.early_stop or st.session_state.current_q is None:
        st.session_state.page = 3
        st.rerun()

    q = st.session_state.current_q
    answered_count = len(st.session_state.answers)
    diff = q["difficulty"]
    emoji = DIFFICULTY_EMOJI[diff]
    diff_colors = {"easy": "#10b981", "medium": "#f59e0b", "hard": "#ef4444"}

    # Progress bar at top
    progress = min(answered_count / 10, 1.0)
    st.progress(progress, text=f"Question {answered_count + 1} of 10")

    # Live stats bar
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Question", f"{answered_count + 1}/10")
    col2.metric("Difficulty", f"{emoji} {diff.capitalize()}")
    col3.metric("🔥 Streak", st.session_state.correct_streak)
    col4.metric("Domain", st.session_state.domain)

    st.markdown("<br>", unsafe_allow_html=True)

    # Question card
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1e293b,#0f172a);
    border:1px solid {diff_colors[diff]};border-radius:16px;padding:28px;color:white;margin-bottom:16px;'>
    <div style='font-size:12px;color:{diff_colors[diff]};margin-bottom:10px;text-transform:uppercase;letter-spacing:1px;'>
    {emoji} {diff.upper()} · {' · '.join(q['tags']).upper()}
    </div>
    <div style='font-size:20px;font-weight:600;line-height:1.5;'>{q['q']}</div>
    </div>
    """, unsafe_allow_html=True)

    ans = st.radio("", q["options"], key=f"q_{answered_count}",
                   label_visibility="collapsed")

    if st.button("Submit Answer ➜", type="primary", use_container_width=False):
        time_taken = time.time() - (st.session_state.q_start_time or time.time())
        correct = 1 if ans == q["answer"] else 0
        is_guess = 1 if time_taken < 3 else 0

        st.session_state.answers.append({
            "skill": st.session_state.domain,
            "difficulty": diff,
            "tags": ", ".join(q["tags"]),
            "correct": correct,
            "time_taken": round(time_taken, 2),
            "is_guess": is_guess,
        })
        st.session_state.asked_indices.append(st.session_state.current_q_idx)

        if correct == 1:
            st.session_state.correct_streak += 1
            st.session_state.wrong_count = 0
        else:
            st.session_state.wrong_count += 1
            st.session_state.correct_streak = 0

        new_diff = update_difficulty(diff, st.session_state.correct_streak, st.session_state.wrong_count)
        if new_diff != diff:
            st.session_state.correct_streak = 0
            st.session_state.wrong_count = 0
        st.session_state.current_difficulty = new_diff

        result, next_diff = get_next_question(
            st.session_state.domain,
            st.session_state.asked_indices,
            new_diff,
            st.session_state.correct_streak
        )

        if result is None:
            st.session_state.early_stop = True
            st.session_state.current_q = None
        else:
            idx, next_q = result
            st.session_state.current_q_idx = idx
            st.session_state.current_q = next_q
            st.session_state.current_difficulty = next_diff
            st.session_state.q_start_time = time.time()

        st.rerun()

    st.stop()


# =====================
# PAGE 3 — RESULTS DASHBOARD
# =====================
if st.session_state.page == 3:

    df = pd.DataFrame(st.session_state.answers)
    accuracy = df["correct"].mean() * 100
    name = st.session_state.name or "You"

    # ── Header banner ──
    if st.session_state.early_stop and len(df) < 10:
        st.markdown("""<div style='background:linear-gradient(135deg,#064e3b,#0f172a);
        border:1px solid #10b981;border-radius:16px;padding:20px;text-align:center;color:white;margin-bottom:20px;'>
        <div style='font-size:28px;'>🏆 Expert Detected — Early Stop!</div>
        <div style='color:#6ee7b7;margin-top:6px;'>You nailed 5 in a row. Quiz ended early.</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style='background:linear-gradient(135deg,#1e3a5f,#0f172a);
        border:1px solid #3b82f6;border-radius:16px;padding:20px;text-align:center;color:white;margin-bottom:20px;'>
        <div style='font-size:28px;'>🎉 Quiz Complete!</div>
        </div>""", unsafe_allow_html=True)

    # ── Top KPI row ──
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{accuracy:.1f}%")
    k2.metric("Questions", len(df))
    k3.metric("Final Level", df["difficulty"].iloc[-1].capitalize())
    diff_map_score = {"easy": 1, "medium": 2, "hard": 3}
    conf = df.apply(lambda r: r["correct"] * diff_map_score.get(r["difficulty"], 1), axis=1).mean() / 3 * 100
    k4.metric("Confidence Score", f"{conf:.0f}/100")

    st.divider()

    # ── Build tag accuracy data ──
    tag_rows = []
    for _, row in df.iterrows():
        for tag in row["tags"].split(", "):
            tag_rows.append({"tag": tag, "correct": row["correct"],
                             "difficulty": row["difficulty"], "time_taken": row["time_taken"]})
    tag_df = pd.DataFrame(tag_rows) if tag_rows else pd.DataFrame()
    tag_acc = tag_df.groupby("tag")["correct"].mean() if not tag_df.empty else pd.Series()

    gap_results, importances = detect_skill_gaps(df)
    weak_tags   = gap_results.get("weak", [])   if gap_results else []
    strong_tags = gap_results.get("strong", []) if gap_results else []

    # ── ROW 1: Radar + Difficulty bar ──
    st.markdown("### 📊 Performance Dashboard")
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        # Radar chart — topic accuracy
        if not tag_acc.empty:
            tags = tag_acc.index.tolist()
            vals = tag_acc.values.tolist()
            vals_closed = vals + [vals[0]]
            tags_closed = tags + [tags[0]]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed, theta=tags_closed, fill="toself",
                fillcolor="rgba(59,130,246,0.2)", line=dict(color="#3b82f6", width=2),
                name="Your Score"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[1.0] * len(tags_closed), theta=tags_closed, fill="toself",
                fillcolor="rgba(255,255,255,0.03)", line=dict(color="#334155", width=1, dash="dot"),
                name="Perfect Score"
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="#0f172a",
                    radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#94a3b8"), gridcolor="#334155"),
                    angularaxis=dict(tickfont=dict(color="#e2e8f0"), gridcolor="#334155"),
                ),
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font=dict(color="white"),
                title=dict(text="🕸️ Skill Radar Chart", font=dict(color="white", size=14)),
                legend=dict(font=dict(color="#94a3b8")),
                margin=dict(t=50, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    with r1c2:
        # Difficulty performance bar
        diff_acc = df.groupby("difficulty")["correct"].mean().reindex(["easy", "medium", "hard"]).dropna()
        fig_diff = go.Figure(go.Bar(
            x=diff_acc.index.str.capitalize(),
            y=(diff_acc.values * 100).round(1),
            marker_color=["#10b981", "#f59e0b", "#ef4444"],
            text=[f"{v:.0f}%" for v in diff_acc.values * 100],
            textposition="outside",
        ))
        fig_diff.update_layout(
            title=dict(text="📈 Accuracy by Difficulty", font=dict(color="white", size=14)),
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font=dict(color="white"),
            yaxis=dict(range=[0, 110], gridcolor="#334155", ticksuffix="%"),
            xaxis=dict(gridcolor="#334155"),
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_diff, use_container_width=True)

    # ── ROW 2: Topic heatmap + Time per question ──
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        # Heatmap: topic × difficulty
        if not tag_df.empty:
            pivot = tag_df.groupby(["tag", "difficulty"])["correct"].mean().unstack(fill_value=0)
            pivot = pivot.reindex(columns=["easy", "medium", "hard"], fill_value=0)
            fig_heat = px.imshow(
                pivot,
                color_continuous_scale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#10b981"]],
                zmin=0, zmax=1,
                labels=dict(x="Difficulty", y="Topic", color="Accuracy"),
                title="🔥 Topic × Difficulty Heatmap",
                text_auto=".0%",
            )
            fig_heat.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font=dict(color="white"),
                title_font=dict(color="white", size=14),
                coloraxis_colorbar=dict(tickfont=dict(color="#94a3b8")),
                margin=dict(t=50, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    with r2c2:
        # Time per question line chart
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=list(range(1, len(df) + 1)),
            y=df["time_taken"],
            mode="lines+markers",
            line=dict(color="#8b5cf6", width=2),
            marker=dict(color=["#10b981" if c else "#ef4444" for c in df["correct"]], size=10),
            name="Time (s)",
        ))
        fig_time.add_hline(y=df["time_taken"].mean(), line_dash="dot",
                           line_color="#94a3b8", annotation_text="avg",
                           annotation_font_color="#94a3b8")
        fig_time.update_layout(
            title=dict(text="⏱️ Time per Question (🟢 correct · 🔴 wrong)", font=dict(color="white", size=14)),
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font=dict(color="white"),
            xaxis=dict(title="Question #", gridcolor="#334155"),
            yaxis=dict(title="Seconds", gridcolor="#334155"),
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_time, use_container_width=True)

    st.divider()

    # ── AI FEATURE 1: Skill Gap ──
    st.markdown("### 🧠 Skill Gap Detection (ML)")
    sg1, sg2 = st.columns(2)
    with sg1:
        st.markdown("**Weak Areas**")
        if weak_tags:
            for t in weak_tags:
                pct = tag_acc.get(t, 0) * 100
                st.markdown(f"""<div style='background:#1e293b;border-left:4px solid #ef4444;
                border-radius:8px;padding:10px 14px;margin:4px 0;color:white;'>
                ❌ <b>{t}</b> — {pct:.0f}%
                <div style='background:#334155;border-radius:4px;height:6px;margin-top:6px;'>
                <div style='background:#ef4444;width:{pct}%;height:6px;border-radius:4px;'></div></div>
                </div>""", unsafe_allow_html=True)
        else:
            st.success("No weak areas — great job!")
    with sg2:
        st.markdown("**Strong Areas**")
        for t in strong_tags:
            pct = tag_acc.get(t, 0) * 100
            st.markdown(f"""<div style='background:#1e293b;border-left:4px solid #10b981;
            border-radius:8px;padding:10px 14px;margin:4px 0;color:white;'>
            ✅ <b>{t}</b> — {pct:.0f}%
            <div style='background:#334155;border-radius:4px;height:6px;margin-top:6px;'>
            <div style='background:#10b981;width:{pct}%;height:6px;border-radius:4px;'></div></div>
            </div>""", unsafe_allow_html=True)

    if importances:
        st.caption(f"RF Feature Importances → Difficulty: {importances.get('difficulty',0):.2f} | "
                   f"Time: {importances.get('time_taken',0):.2f} | Guessing: {importances.get('guessing',0):.2f}")

    st.divider()

    # ── AI FEATURE 2: Recommendations ──
    st.markdown("### 🎯 Personalized Recommendations")

    behavior = analyze_behavior(df)  # needed for GPT context too

    llm_text, llm_status = get_llm_recommendation(
        name, st.session_state.domain, weak_tags, strong_tags,
        accuracy, OPENAI_API_KEY, behavior
    )

    if llm_text:
        # Parse the 3 sections from GPT response
        import re
        sections = re.split(r"##\s*\d+\.\s*", llm_text)
        sections = [s.strip() for s in sections if s.strip()]
        labels = ["1. Personalized Feedback", "2. Study Plan", "3. Recommended Resources"]
        icons  = ["💬", "📅", "📚"]
        colors = ["#3b82f6", "#10b981", "#8b5cf6"]

        for i, section in enumerate(sections[:3]):
            label = labels[i] if i < len(labels) else f"Section {i+1}"
            icon  = icons[i]  if i < len(icons)  else "📌"
            color = colors[i] if i < len(colors) else "#334155"
            st.markdown(f"""
            <div style='background:#1e293b;border-left:4px solid {color};
            border-radius:12px;padding:20px;margin:12px 0;color:white;'>
            <div style='font-size:13px;color:{color};font-weight:bold;
            text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;'>
            {icon} {label}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown(section)

        st.caption("🤖 Generated by GPT-3.5 based on your quiz performance")

    else:
        # Rule-based fallback
        rec = rule_based_recommendation(
            name, st.session_state.domain, weak_tags, strong_tags, accuracy
        )
        st.markdown(rec)
        if llm_status == "no_key":
            st.info("💡 Add your OpenAI API key on the start page to unlock GPT-powered feedback, "
                    "a personalized study plan, and curated resources.")
        elif llm_status != "no_key":
            st.warning(f"GPT unavailable ({llm_status}) — showing rule-based recommendations.")

    st.divider()

    # ── AI FEATURE 3: Behavioral Analysis ──
    st.markdown("### 📊 Behavioral Analysis")
    if behavior:
        b1, b2, b3 = st.columns(3)
        b1.metric("Avg Time / Question", f"{behavior['avg_time']:.1f}s")
        b2.metric("Likely Guesses", behavior["guess_count"], help="Answered in under 3 seconds")
        b3.metric("Confidence Score", f"{behavior['confidence']:.0f}/100")

        bdf = behavior["behavior_df"]
        fig_beh = px.bar(
            bdf.groupby("behavior")["correct"].mean().reset_index(),
            x="behavior", y="correct",
            color="behavior",
            color_discrete_map={"fast/guess": "#f59e0b", "normal": "#3b82f6", "slow/struggled": "#ef4444"},
            title="Accuracy by Behavior Pattern",
            text_auto=".0%",
        )
        fig_beh.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font=dict(color="white"), title_font=dict(color="white", size=14),
            yaxis=dict(gridcolor="#334155", tickformat=".0%"),
            xaxis=dict(gridcolor="#334155"),
            showlegend=False, margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig_beh, use_container_width=True)

        if behavior["guess_count"] >= 3:
            st.warning(f"⚡ {behavior['guess_count']} answers submitted very quickly — possible guessing.")
        if behavior["struggle_count"] >= 2:
            st.info(f"🤔 {behavior['struggle_count']} questions took much longer — likely your real gaps.")
        if behavior["confidence"] >= 70:
            st.success(f"💪 Confidence {behavior['confidence']:.0f}/100 — strong under pressure.")
        elif behavior["confidence"] < 40:
            st.warning(f"📉 Confidence {behavior['confidence']:.0f}/100 — practice harder questions.")

    st.divider()

    # ── PDF Download ──
    st.markdown("### 📄 Download Your Report")
    pdf_buf = generate_pdf(
        name        = name,
        domain      = st.session_state.domain,
        accuracy    = accuracy,
        questions_count = len(df),
        final_difficulty = df["difficulty"].iloc[-1] if len(df) else "easy",
        confidence  = conf,
        strong_tags = strong_tags,
        weak_tags   = weak_tags,
        tag_acc     = tag_acc.to_dict() if not tag_acc.empty else {},
        behavior    = behavior,
        llm_text    = llm_text if llm_text else None,
    )
    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_buf,
        file_name=f"skill_gap_report_{(name or 'user').replace(' ','_')}.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )
    st.caption("Share this report with recruiters or keep it as a learning milestone.")

    st.divider()

    if st.button("🔄 Take Another Quiz", type="primary"):
        for k in ["page", "current_difficulty", "asked_indices", "answers",
                  "current_q", "current_q_idx", "early_stop",
                  "correct_streak", "wrong_count", "q_start_time"]:
            del st.session_state[k]
        st.rerun()
