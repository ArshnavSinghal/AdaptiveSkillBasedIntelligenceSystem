import streamlit as st
import pandas as pd
import random
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

# =====================
# QUESTION BANK
# 10 unique questions per domain, no cross-domain repeats
# =====================
question_bank = {
    "Python": [
        {"q": "Which keyword is used to define a function in Python?", "options": ["def", "func", "define", "fun"], "answer": "def", "difficulty": 1},
        {"q": "What data type is the result of: type([])?", "options": ["list", "tuple", "dict", "set"], "answer": "list", "difficulty": 1},
        {"q": "Which of these is used to handle exceptions in Python?", "options": ["try/except", "catch/throw", "error/handle", "check/fail"], "answer": "try/except", "difficulty": 1},
        {"q": "What is the output of len('hello')?", "options": ["4", "5", "6", "error"], "answer": "5", "difficulty": 1},
        {"q": "Which method adds an element to the end of a list?", "options": ["append()", "add()", "insert()", "push()"], "answer": "append()", "difficulty": 1},
        {"q": "What does the 'self' keyword refer to in a Python class?", "options": ["The current instance", "The class itself", "The parent class", "A global variable"], "answer": "The current instance", "difficulty": 2},
        {"q": "Which of the following is immutable in Python?", "options": ["tuple", "list", "dict", "set"], "answer": "tuple", "difficulty": 2},
        {"q": "What is a lambda function in Python?", "options": ["An anonymous function", "A recursive function", "A built-in function", "A class method"], "answer": "An anonymous function", "difficulty": 2},
        {"q": "What does the 'yield' keyword do in Python?", "options": ["Returns a generator", "Exits the program", "Raises an exception", "Imports a module"], "answer": "Returns a generator", "difficulty": 3},
        {"q": "Which module is used for regular expressions in Python?", "options": ["re", "regex", "regexp", "string"], "answer": "re", "difficulty": 2},
    ],
    "DSA": [
        {"q": "Which data structure follows FIFO order?", "options": ["Queue", "Stack", "Tree", "Graph"], "answer": "Queue", "difficulty": 1},
        {"q": "What is the time complexity of Binary Search?", "options": ["O(log n)", "O(n)", "O(n^2)", "O(1)"], "answer": "O(log n)", "difficulty": 2},
        {"q": "Which traversal visits the root node first?", "options": ["Pre-order", "In-order", "Post-order", "Level-order"], "answer": "Pre-order", "difficulty": 2},
        {"q": "What is the worst-case time complexity of QuickSort?", "options": ["O(n^2)", "O(n log n)", "O(n)", "O(log n)"], "answer": "O(n^2)", "difficulty": 3},
        {"q": "Which data structure uses LIFO order?", "options": ["Stack", "Queue", "Heap", "LinkedList"], "answer": "Stack", "difficulty": 1},
        {"q": "What is the space complexity of Merge Sort?", "options": ["O(n)", "O(1)", "O(log n)", "O(n^2)"], "answer": "O(n)", "difficulty": 3},
        {"q": "In a min-heap, the root contains the?", "options": ["Minimum element", "Maximum element", "Median element", "Random element"], "answer": "Minimum element", "difficulty": 2},
        {"q": "Which algorithm is used to find the shortest path in a weighted graph?", "options": ["Dijkstra's", "BFS", "DFS", "Prim's"], "answer": "Dijkstra's", "difficulty": 3},
        {"q": "What is the time complexity of accessing an element in a hash table on average?", "options": ["O(1)", "O(n)", "O(log n)", "O(n^2)"], "answer": "O(1)", "difficulty": 2},
        {"q": "Which sorting algorithm is stable and has O(n log n) average complexity?", "options": ["Merge Sort", "Quick Sort", "Heap Sort", "Selection Sort"], "answer": "Merge Sort", "difficulty": 2},
    ],
    "DBMS": [
        {"q": "What does a primary key ensure in a table?", "options": ["Uniqueness of each row", "Foreign key reference", "Null values allowed", "Duplicate rows"], "answer": "Uniqueness of each row", "difficulty": 1},
        {"q": "Which SQL command is used to retrieve data from a table?", "options": ["SELECT", "INSERT", "UPDATE", "DELETE"], "answer": "SELECT", "difficulty": 1},
        {"q": "What is a foreign key?", "options": ["A key that references a primary key in another table", "A key that is always unique", "A key that cannot be null", "A key used for indexing"], "answer": "A key that references a primary key in another table", "difficulty": 2},
        {"q": "Which normal form eliminates partial dependencies?", "options": ["2NF", "1NF", "3NF", "BCNF"], "answer": "2NF", "difficulty": 3},
        {"q": "What does ACID stand for in database transactions?", "options": ["Atomicity, Consistency, Isolation, Durability", "Access, Control, Index, Data", "Aggregate, Commit, Insert, Delete", "None of the above"], "answer": "Atomicity, Consistency, Isolation, Durability", "difficulty": 2},
        {"q": "Which JOIN returns all rows from both tables, with NULLs where no match?", "options": ["FULL OUTER JOIN", "INNER JOIN", "LEFT JOIN", "CROSS JOIN"], "answer": "FULL OUTER JOIN", "difficulty": 2},
        {"q": "What is a view in SQL?", "options": ["A virtual table based on a query", "A physical copy of a table", "An index on a table", "A stored procedure"], "answer": "A virtual table based on a query", "difficulty": 2},
        {"q": "Which command permanently saves a transaction in SQL?", "options": ["COMMIT", "ROLLBACK", "SAVEPOINT", "BEGIN"], "answer": "COMMIT", "difficulty": 1},
        {"q": "What is denormalization?", "options": ["Adding redundancy to improve read performance", "Removing duplicate data", "Splitting tables into smaller ones", "Creating indexes"], "answer": "Adding redundancy to improve read performance", "difficulty": 3},
        {"q": "Which SQL clause is used to filter groups after GROUP BY?", "options": ["HAVING", "WHERE", "FILTER", "ORDER BY"], "answer": "HAVING", "difficulty": 2},
    ],
    "General": [
        {"q": "What does CPU stand for?", "options": ["Central Processing Unit", "Core Processing Unit", "Central Program Utility", "Computer Processing Unit"], "answer": "Central Processing Unit", "difficulty": 1},
        {"q": "Which protocol is used to send emails?", "options": ["SMTP", "HTTP", "FTP", "SSH"], "answer": "SMTP", "difficulty": 2},
        {"q": "What does RAM stand for?", "options": ["Random Access Memory", "Read Access Memory", "Rapid Access Module", "Runtime Access Memory"], "answer": "Random Access Memory", "difficulty": 1},
        {"q": "What is the binary representation of the decimal number 10?", "options": ["1010", "1001", "1100", "0110"], "answer": "1010", "difficulty": 2},
        {"q": "Which layer of the OSI model handles routing?", "options": ["Network Layer", "Transport Layer", "Data Link Layer", "Session Layer"], "answer": "Network Layer", "difficulty": 2},
        {"q": "What does HTTP stand for?", "options": ["HyperText Transfer Protocol", "High Transfer Text Protocol", "HyperText Transmission Process", "Host Transfer Text Protocol"], "answer": "HyperText Transfer Protocol", "difficulty": 1},
        {"q": "What is the full form of URL?", "options": ["Uniform Resource Locator", "Universal Resource Link", "Uniform Retrieval Locator", "Universal Retrieval Link"], "answer": "Uniform Resource Locator", "difficulty": 1},
        {"q": "Which of the following is an example of an operating system?", "options": ["Linux", "Python", "MySQL", "Apache"], "answer": "Linux", "difficulty": 1},
        {"q": "What is the purpose of a firewall?", "options": ["To monitor and control network traffic", "To speed up internet connection", "To store data securely", "To compress files"], "answer": "To monitor and control network traffic", "difficulty": 2},
        {"q": "What does SSL stand for?", "options": ["Secure Sockets Layer", "Safe Server Link", "Secure System Login", "Standard Socket Layer"], "answer": "Secure Sockets Layer", "difficulty": 2},
    ]
}


# =====================
# OPTIONAL: OPENAI API KEY
# Drop your key here to enrich questions via GPT
# Leave as empty string to use local bank only
# =====================
OPENAI_API_KEY = ""  # e.g. "sk-..."

# =====================
# GENERATE QUESTIONS
# Always uses local bank (guaranteed 10 unique per domain)
# =====================
def generate_questions(domain, n=10):
    pool = question_bank.get(domain, [])
    questions = []
    for q in pool:
        q_copy = q.copy()
        q_copy["skill"] = domain
        questions.append(q_copy)
    random.shuffle(questions)
    return questions[:n]

# =====================
# UI CONFIG
# =====================

st.set_page_config(page_title="Skill Gap AI", layout="wide")

st.markdown("""
<div style='background:linear-gradient(135deg,#1e293b,#0f172a);
padding:25px;border-radius:12px;text-align:center;color:white;
font-size:28px;font-weight:bold;'>
🎯 Skill Gap Intelligence Engine (AI Powered)
</div>
""", unsafe_allow_html=True)

# =====================
# SESSION STATE
# =====================
if "page" not in st.session_state:
    st.session_state.page = 1
if "questions" not in st.session_state:
    st.session_state.questions = []
if "answers" not in st.session_state:
    st.session_state.answers = []
if "current_q" not in st.session_state:
    st.session_state.current_q = 0
if "domain" not in st.session_state:
    st.session_state.domain = "Python"

# =====================
# PAGE 1 (INPUT)
# =====================
if st.session_state.page == 1:

    st.header("Start Quiz")

    name = st.text_input("Enter Name")

    domain = st.selectbox("Select Domain", ["Python", "DSA", "DBMS", "General"])
    st.session_state.domain = domain

    if st.button("Start Quiz"):
        st.session_state.questions = generate_questions(domain)
        st.session_state.answers = []
        st.session_state.current_q = 0
        st.session_state.page = 2

    st.stop()

# =====================
# PAGE 2 (QUIZ)
# =====================
if st.session_state.page == 2:

    q = st.session_state.questions[st.session_state.current_q]

    st.subheader(f"Question {st.session_state.current_q + 1} of {len(st.session_state.questions)}")
    st.markdown(f"**{q['q']}**")
    st.caption(f"Domain: {q['skill']} | Difficulty: {q['difficulty']}")

    ans = st.radio("Choose answer:", q["options"], key=str(st.session_state.current_q))

    if st.button("Next"):
        st.session_state.answers.append({
            "skill": q["skill"],
            "correct": 1 if ans == q["answer"] else 0
        })

        if st.session_state.current_q < len(st.session_state.questions) - 1:
            st.session_state.current_q += 1
        else:
            st.session_state.page = 3

    st.progress((st.session_state.current_q + 1) / len(st.session_state.questions))
    st.stop()

# =====================
# PAGE 3 (RESULTS)
# =====================
if st.session_state.page == 3:

    df = pd.DataFrame(st.session_state.answers)

    st.success("Quiz Completed 🎉")

    skill_acc = df.groupby("skill")["correct"].mean()

    st.subheader("Skill Performance")
    st.bar_chart(skill_acc)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{df['correct'].mean()*100:.1f}%")
    col2.metric("Total Questions", len(df))
    col3.metric("Weak Skills", len(skill_acc[skill_acc < 0.6]))

    if len(df) > 3:
        kmeans = KMeans(n_clusters=2, random_state=42)
        df["cluster"] = kmeans.fit_predict(df[["correct"]])
        st.subheader("Cluster Analysis")
        st.dataframe(df)

    if len(df) > 3:
        model = RandomForestRegressor()
        model.fit(df[["correct"]], df["correct"])
        st.subheader("Feature Importance")
        st.write(model.feature_importances_)

    weak = skill_acc[skill_acc < 0.6]
    st.subheader("Recommendation")
    if len(weak) > 0:
        st.error(f"Focus on: {', '.join(weak.index)}")
    else:
        st.success("Excellent Performance 🚀")

    if st.button("Restart"):
        st.session_state.page = 1
        st.session_state.current_q = 0
        st.session_state.answers = []
