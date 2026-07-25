"""
Synthetic Resume Section Data Generator

Generates realistic resume section text across 8 categories for training
a text classifier. Uses template-based generation with randomized entities,
synonym replacement, and structural variation to produce diverse examples.

Author: Lorenzo Scaturchio (gr8monk3ys)
"""

import csv
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Entity pools – used to fill templates with realistic variation
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Daniel",
    "Lisa", "Matthew", "Nancy", "Anthony", "Betty", "Mark", "Sandra",
    "Aisha", "Wei", "Carlos", "Priya", "Olga", "Hiroshi", "Fatima", "Liam",
    "Sofia", "Andrei", "Mei", "Alejandro", "Yuki", "Omar", "Elena", "Raj",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Patel", "Chen", "Kim", "Nakamura", "Ivanov", "Silva", "Okafor",
]

COMPANIES = [
    "Google", "Microsoft", "Amazon", "Apple", "Meta", "Netflix", "Stripe",
    "Airbnb", "Uber", "Salesforce", "Adobe", "IBM", "Oracle", "Intel",
    "Tesla", "SpaceX", "Palantir", "Snowflake", "Databricks", "Confluent",
    "JPMorgan Chase", "Goldman Sachs", "Morgan Stanley", "Deloitte",
    "McKinsey & Company", "Boston Consulting Group", "Accenture",
    "Lockheed Martin", "Boeing", "Raytheon", "General Electric",
    "Procter & Gamble", "Johnson & Johnson", "Pfizer", "Moderna",
    "Shopify", "Square", "Twilio", "Cloudflare", "HashiCorp",
    "DataRobot", "Hugging Face", "OpenAI", "Anthropic", "Cohere",
    "Startup XYZ", "TechCorp Inc.", "InnovateTech", "DataDriven LLC",
]

UNIVERSITIES = [
    "Massachusetts Institute of Technology", "Stanford University",
    "Harvard University", "University of California, Berkeley",
    "Carnegie Mellon University", "Georgia Institute of Technology",
    "University of Michigan", "University of Illinois Urbana-Champaign",
    "California Institute of Technology", "Princeton University",
    "Columbia University", "University of Washington",
    "University of Texas at Austin", "Cornell University",
    "University of Pennsylvania", "University of Southern California",
    "New York University", "University of Wisconsin-Madison",
    "Duke University", "Northwestern University",
    "University of California, Los Angeles", "Rice University",
    "University of Maryland", "Purdue University",
    "Ohio State University", "Arizona State University",
    "University of Virginia", "University of Florida",
    "Boston University", "Northeastern University",
]

DEGREES = [
    ("Bachelor of Science", "B.S."),
    ("Bachelor of Arts", "B.A."),
    ("Master of Science", "M.S."),
    ("Master of Arts", "M.A."),
    ("Master of Business Administration", "MBA"),
    ("Doctor of Philosophy", "Ph.D."),
    ("Associate of Science", "A.S."),
    ("Bachelor of Engineering", "B.Eng."),
    ("Master of Engineering", "M.Eng."),
]

MAJORS = [
    "Computer Science", "Software Engineering", "Data Science",
    "Electrical Engineering", "Mechanical Engineering",
    "Information Technology", "Mathematics", "Statistics",
    "Business Administration", "Economics", "Finance",
    "Biomedical Engineering", "Chemical Engineering",
    "Civil Engineering", "Physics", "Biology",
    "Artificial Intelligence", "Machine Learning",
    "Human-Computer Interaction", "Cybersecurity",
    "Information Systems", "Operations Research",
]

MINORS = [
    "Mathematics", "Statistics", "Psychology", "Business",
    "Economics", "Philosophy", "Linguistics", "Physics",
    "Data Science", "Communication", "Sociology", "History",
]

GPA_VALUES = [
    "3.5", "3.6", "3.7", "3.8", "3.9", "4.0",
    "3.52", "3.65", "3.78", "3.85", "3.92", "3.45",
]

GRAD_YEARS = list(range(2015, 2027))

JOB_TITLES = [
    "Software Engineer", "Senior Software Engineer", "Staff Engineer",
    "Principal Engineer", "Engineering Manager", "Tech Lead",
    "Data Scientist", "Senior Data Scientist", "Machine Learning Engineer",
    "ML Research Scientist", "Data Engineer", "Data Analyst",
    "Product Manager", "Senior Product Manager", "Program Manager",
    "DevOps Engineer", "Site Reliability Engineer", "Cloud Architect",
    "Full Stack Developer", "Frontend Engineer", "Backend Engineer",
    "Mobile Developer", "iOS Engineer", "Android Developer",
    "QA Engineer", "Security Engineer", "Solutions Architect",
    "Research Scientist", "AI Engineer", "NLP Engineer",
    "Quantitative Analyst", "Financial Analyst", "Business Analyst",
    "UX Designer", "UI Engineer", "Technical Writer",
    "Intern", "Software Engineering Intern", "Data Science Intern",
]

PROGRAMMING_LANGUAGES = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C", "C#",
    "Go", "Rust", "Kotlin", "Swift", "Ruby", "PHP", "Scala",
    "R", "MATLAB", "Julia", "Haskell", "Elixir", "Dart",
]

FRAMEWORKS = [
    "React", "Angular", "Vue.js", "Next.js", "Django", "Flask",
    "FastAPI", "Spring Boot", "Express.js", "Node.js", "Rails",
    "TensorFlow", "PyTorch", "Keras", "scikit-learn", "Pandas",
    "NumPy", "Spark", "Hadoop", "Kubernetes", "Docker",
    "AWS", "GCP", "Azure", "Terraform", "Ansible",
    ".NET", "Laravel", "Svelte", "Remix", "Astro",
]

TOOLS = [
    "Git", "GitHub", "GitLab", "Jira", "Confluence", "Slack",
    "VS Code", "IntelliJ", "PyCharm", "Vim", "Emacs",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
    "Kafka", "RabbitMQ", "Airflow", "dbt", "Snowflake",
    "Tableau", "Power BI", "Grafana", "Prometheus", "Datadog",
    "Jenkins", "CircleCI", "GitHub Actions", "ArgoCD",
    "Figma", "Sketch", "Adobe XD", "Postman", "Swagger",
]

SOFT_SKILLS = [
    "Leadership", "Communication", "Team Collaboration",
    "Problem Solving", "Critical Thinking", "Time Management",
    "Project Management", "Agile Methodologies", "Scrum",
    "Cross-functional Collaboration", "Mentoring",
    "Strategic Planning", "Stakeholder Management",
    "Technical Writing", "Public Speaking", "Negotiation",
]

CERTIFICATIONS_LIST = [
    "AWS Certified Solutions Architect - Associate",
    "AWS Certified Developer - Associate",
    "AWS Certified Machine Learning - Specialty",
    "Google Cloud Professional Data Engineer",
    "Google Cloud Professional ML Engineer",
    "Microsoft Azure Fundamentals (AZ-900)",
    "Microsoft Azure Data Scientist Associate (DP-100)",
    "Certified Kubernetes Administrator (CKA)",
    "Certified Kubernetes Application Developer (CKAD)",
    "Certified Information Systems Security Professional (CISSP)",
    "CompTIA Security+",
    "Project Management Professional (PMP)",
    "Certified ScrumMaster (CSM)",
    "TensorFlow Developer Certificate",
    "Databricks Certified Data Engineer Associate",
    "Snowflake SnowPro Core Certification",
    "HashiCorp Terraform Associate",
    "Cisco Certified Network Associate (CCNA)",
    "Oracle Certified Professional, Java SE",
    "Red Hat Certified System Administrator (RHCSA)",
    "Deep Learning Specialization (Coursera)",
    "Machine Learning by Stanford (Coursera)",
    "Professional Scrum Master I (PSM I)",
]

AWARDS_LIST = [
    "Dean's List", "Summa Cum Laude", "Magna Cum Laude", "Cum Laude",
    "Phi Beta Kappa", "Tau Beta Pi", "National Merit Scholar",
    "Employee of the Quarter", "Spot Bonus Award", "President's Club",
    "Best Paper Award", "Innovation Award", "Hackathon Winner",
    "Outstanding Graduate Student Award", "Research Fellowship",
    "Teaching Assistant Excellence Award", "Community Service Award",
    "IEEE Best Student Paper", "ACM ICPC Regional Finalist",
    "Google Code Jam Qualifier", "Facebook Hacker Cup Participant",
    "Patent Holder", "Top Performer Award", "Rising Star Award",
]

CITIES = [
    "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
    "Boston, MA", "Chicago, IL", "Los Angeles, CA", "Denver, CO",
    "Portland, OR", "Atlanta, GA", "Washington, DC", "San Jose, CA",
    "Raleigh, NC", "Pittsburgh, PA", "Minneapolis, MN", "Dallas, TX",
    "Miami, FL", "Phoenix, AZ", "San Diego, CA", "Philadelphia, PA",
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

MONTHS_SHORT = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

PROJECT_ADJECTIVES = [
    "Real-time", "Scalable", "Distributed", "Cloud-native",
    "AI-powered", "Automated", "Interactive", "Cross-platform",
    "Open-source", "End-to-end", "High-performance", "Serverless",
    "Event-driven", "Microservice-based", "Full-stack",
]

PROJECT_NOUNS = [
    "Dashboard", "Platform", "Pipeline", "Application", "System",
    "API", "Framework", "Tool", "Service", "Engine",
    "Chatbot", "Recommendation System", "Search Engine",
    "Analytics Platform", "Monitoring System", "Marketplace",
]

IMPACT_METRICS = [
    "reduced latency by {pct}%",
    "improved throughput by {pct}%",
    "increased user engagement by {pct}%",
    "decreased error rate by {pct}%",
    "saved ${amount}K annually",
    "reduced costs by {pct}%",
    "improved accuracy by {pct}%",
    "increased conversion rate by {pct}%",
    "served {users} daily active users",
    "processed {events} events per second",
    "reduced deployment time from hours to minutes",
    "cut onboarding time by {pct}%",
    "automated {pct}% of manual processes",
    "improved model F1 score from 0.{f1_old} to 0.{f1_new}",
]

PHONE_AREA_CODES = [
    "415", "650", "408", "510", "212", "646", "718", "206",
    "512", "617", "312", "213", "303", "503", "404", "202",
]

LINKEDIN_PREFIXES = [
    "linkedin.com/in/", "www.linkedin.com/in/",
]

GITHUB_PREFIXES = [
    "github.com/", "www.github.com/",
]

DOMAINS = [
    "gmail.com", "outlook.com", "yahoo.com", "protonmail.com",
    "icloud.com", "hotmail.com", "mail.com",
]

# ---------------------------------------------------------------------------
# Synonym replacement pools for augmentation
# ---------------------------------------------------------------------------

SYNONYMS = {
    "developed": ["built", "created", "engineered", "designed", "implemented", "constructed", "authored"],
    "managed": ["led", "oversaw", "directed", "supervised", "coordinated", "administered"],
    "improved": ["enhanced", "optimized", "upgraded", "refined", "boosted", "strengthened"],
    "implemented": ["deployed", "executed", "delivered", "rolled out", "launched", "shipped"],
    "analyzed": ["examined", "evaluated", "assessed", "investigated", "studied", "reviewed"],
    "collaborated": ["partnered", "worked closely with", "teamed up with", "cooperated with"],
    "responsible for": ["in charge of", "accountable for", "tasked with", "owned"],
    "utilized": ["leveraged", "employed", "used", "applied", "harnessed"],
    "achieved": ["accomplished", "attained", "reached", "secured", "delivered"],
    "experience": ["expertise", "background", "proficiency", "track record"],
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _pick(pool, k=1):
    """Return k unique random items from a pool."""
    k = min(k, len(pool))
    return random.sample(pool, k)


def _pick_one(pool):
    return random.choice(pool)


def _date_range(allow_present: bool = True):
    """Return a random date range string."""
    start_year = random.randint(2014, 2024)
    start_month = _pick_one(MONTHS_SHORT)
    fmt = random.choice(["short", "long", "year_only"])

    if allow_present and random.random() < 0.3:
        end_str = random.choice(["Present", "Current", "Now"])
    else:
        end_year = random.randint(start_year, min(start_year + 6, 2026))
        end_month = _pick_one(MONTHS_SHORT)
        if fmt == "short":
            end_str = f"{end_month} {end_year}"
        elif fmt == "long":
            end_str = f"{_pick_one(MONTHS)} {end_year}"
        else:
            end_str = str(end_year)

    if fmt == "short":
        start_str = f"{start_month} {start_year}"
    elif fmt == "long":
        start_str = f"{_pick_one(MONTHS)} {start_year}"
    else:
        start_str = str(start_year)

    sep = random.choice([" - ", " – ", " to ", "–", "-"])
    return f"{start_str}{sep}{end_str}"


def _impact():
    """Generate a random impact metric string."""
    template = _pick_one(IMPACT_METRICS)
    return template.format(
        pct=random.randint(10, 85),
        amount=random.randint(50, 500),
        users=random.choice(["10K", "50K", "100K", "500K", "1M", "5M"]),
        events=random.choice(["1K", "10K", "50K", "100K", "1M"]),
        f1_old=random.randint(65, 80),
        f1_new=random.randint(82, 97),
    )


def _synonym_replace(text: str) -> str:
    """Randomly replace words with synonyms for augmentation."""
    words = text.split()
    result = []
    for w in words:
        lower = w.lower().rstrip(".,;:")
        if lower in SYNONYMS and random.random() < 0.3:
            replacement = _pick_one(SYNONYMS[lower])
            # Preserve original capitalization of first char
            if w[0].isupper():
                replacement = replacement.capitalize()
            # Preserve trailing punctuation
            trailing = w[len(lower):]
            result.append(replacement + trailing)
        else:
            result.append(w)
    return " ".join(result)


def _bullet():
    """Return a random bullet character."""
    return random.choice(["•", "-", "●", "*", "▪", ""])


def _reorder_bullets(bullets: list) -> list:
    """Shuffle bullet points for variation."""
    shuffled = bullets.copy()
    random.shuffle(shuffled)
    return shuffled


# ---------------------------------------------------------------------------
# Section generators – each returns a string of realistic text
# ---------------------------------------------------------------------------

def generate_education() -> str:
    """Generate a realistic education section."""
    templates = []

    # Template 1: Full formal entry
    def _t1():
        uni = _pick_one(UNIVERSITIES)
        deg_full, deg_short = _pick_one(DEGREES)
        major = _pick_one(MAJORS)
        year = _pick_one(GRAD_YEARS)
        lines = []

        header_style = random.choice(["full", "short", "inline"])
        if header_style == "full":
            lines.append(f"{deg_full} in {major}")
            lines.append(f"{uni}")
            lines.append(f"Graduated: {_pick_one(MONTHS)} {year}")
        elif header_style == "short":
            lines.append(f"{deg_short} {major}, {uni} ({year})")
        else:
            lines.append(f"{uni} — {deg_full} in {major}, {year}")

        # Optional GPA
        if random.random() < 0.6:
            gpa = _pick_one(GPA_VALUES)
            lines.append(f"GPA: {gpa}/4.0")

        # Optional minor
        if random.random() < 0.3:
            minor = _pick_one(MINORS)
            lines.append(f"Minor in {minor}")

        # Optional coursework
        if random.random() < 0.5:
            courses = _pick(MAJORS + ["Algorithms", "Data Structures",
                                       "Operating Systems", "Database Systems",
                                       "Computer Networks", "Linear Algebra",
                                       "Probability and Statistics",
                                       "Deep Learning", "Natural Language Processing",
                                       "Computer Vision", "Distributed Systems"], k=random.randint(3, 6))
            prefix = random.choice(["Relevant Coursework:", "Key Courses:", "Coursework:"])
            lines.append(f"{prefix} {', '.join(courses)}")

        # Optional honors
        if random.random() < 0.3:
            honor = random.choice(["Summa Cum Laude", "Magna Cum Laude",
                                    "Cum Laude", "Dean's List (all semesters)",
                                    "Honors Program", "University Scholar"])
            lines.append(honor)

        # Optional thesis
        if "Ph.D." in deg_short or ("M.S." in deg_short and random.random() < 0.4):
            topic = random.choice([
                "Transformer-based approaches to document classification",
                "Scalable distributed systems for real-time data processing",
                "Graph neural networks for molecular property prediction",
                "Federated learning in healthcare applications",
                "Efficient attention mechanisms for long-sequence modeling",
                "Reinforcement learning for autonomous navigation",
            ])
            label = "Dissertation" if "Ph.D." in deg_short else "Thesis"
            lines.append(f"{label}: \"{topic}\"")

        return "\n".join(lines)

    # Template 2: Multiple degrees
    def _t2():
        entries = []
        for _ in range(random.randint(2, 3)):
            uni = _pick_one(UNIVERSITIES)
            deg_full, deg_short = _pick_one(DEGREES)
            major = _pick_one(MAJORS)
            year = _pick_one(GRAD_YEARS)
            gpa_line = f" | GPA: {_pick_one(GPA_VALUES)}" if random.random() < 0.5 else ""
            entries.append(f"{deg_short} in {major}, {uni}, {year}{gpa_line}")
        return "\n".join(entries)

    # Template 3: Education with activities
    def _t3():
        uni = _pick_one(UNIVERSITIES)
        deg_full, deg_short = _pick_one(DEGREES)
        major = _pick_one(MAJORS)
        year = _pick_one(GRAD_YEARS)
        lines = [f"{uni}", f"{deg_full} in {major} | {_pick_one(MONTHS)} {year}"]

        activities = random.sample([
            "Teaching Assistant for Introduction to Computer Science",
            "President, Computer Science Student Association",
            "Member, ACM Student Chapter",
            "Undergraduate Research Assistant, ML Lab",
            "Peer Tutor, Mathematics Department",
            "Captain, University Programming Competition Team",
            "Volunteer, Engineering Outreach Program",
            "Member, Honors College",
            "Study Abroad Program, Technical University of Munich",
            "Resident Advisor, Engineering Living-Learning Community",
        ], k=random.randint(1, 3))

        b = _bullet()
        for a in activities:
            lines.append(f"{b} {a}" if b else a)

        return "\n".join(lines)

    templates = [_t1, _t2, _t3]
    return random.choice(templates)()


def generate_experience() -> str:
    """Generate a realistic work experience section."""

    def _single_role():
        title = _pick_one(JOB_TITLES)
        company = _pick_one(COMPANIES)
        city = _pick_one(CITIES)
        date_range = _date_range()

        header_styles = [
            f"{title} | {company} | {city} | {date_range}",
            f"{title}, {company}\n{city} | {date_range}",
            f"{company} — {title}\n{date_range} | {city}",
            f"{title}\n{company}, {city}\n{date_range}",
        ]
        lines = [random.choice(header_styles)]

        # Generate bullet points
        bullet_templates = [
            f"Developed and maintained {random.choice(['microservices', 'APIs', 'web applications', 'data pipelines', 'ML models', 'backend systems', 'frontend components'])} using {', '.join(_pick(PROGRAMMING_LANGUAGES, k=random.randint(1,3)))} and {', '.join(_pick(FRAMEWORKS, k=random.randint(1,2)))}",
            f"Collaborated with cross-functional teams of {random.randint(3,15)} engineers to deliver {random.choice(['product features', 'platform improvements', 'system migrations', 'infrastructure upgrades'])} on schedule",
            f"Designed and implemented {random.choice(['CI/CD pipelines', 'testing frameworks', 'monitoring solutions', 'data models', 'caching strategies', 'authentication systems'])} that {_impact()}",
            f"Led migration of {random.choice(['legacy monolith', 'on-premise infrastructure', 'batch processing system', 'manual workflows'])} to {random.choice(['cloud-native architecture', 'microservices', 'real-time streaming', 'automated pipelines'])}",
            f"Mentored {random.randint(2,8)} junior engineers through code reviews, pair programming, and technical design sessions",
            f"Optimized {random.choice(['database queries', 'API response times', 'model inference', 'data processing pipelines', 'search indexing'])} resulting in {_impact()}",
            f"Wrote comprehensive technical documentation and {random.choice(['RFCs', 'design docs', 'runbooks', 'architecture decision records'])} for {random.choice(['system design', 'API contracts', 'deployment procedures', 'incident response'])}",
            f"Built {random.choice(['real-time', 'batch', 'streaming', 'event-driven'])} {random.choice(['data pipeline', 'ETL process', 'analytics system', 'feature store'])} processing {random.choice(['1M+', '10M+', '100M+', '1B+'])} records {random.choice(['daily', 'per hour', 'in real-time'])}",
            f"Spearheaded adoption of {_pick_one(FRAMEWORKS)} and {_pick_one(TOOLS)}, {_impact()}",
            f"Conducted A/B testing and experimentation for {random.choice(['recommendation engine', 'search ranking', 'pricing model', 'onboarding flow', 'notification system'])}, {_impact()}",
            f"Architected {random.choice(['distributed', 'fault-tolerant', 'highly available', 'horizontally scalable'])} system handling {random.choice(['10K', '50K', '100K', '1M'])} requests per second with {random.choice(['99.9%', '99.95%', '99.99%'])} uptime",
        ]

        n_bullets = random.randint(2, 5)
        selected = random.sample(bullet_templates, min(n_bullets, len(bullet_templates)))
        selected = _reorder_bullets(selected)
        b = _bullet()
        for bullet in selected:
            lines.append(f"{b} {bullet}" if b else bullet)

        return "\n".join(lines)

    # Sometimes include multiple roles
    n_roles = random.choices([1, 2], weights=[0.7, 0.3])[0]
    roles = [_single_role() for _ in range(n_roles)]
    return "\n\n".join(roles)


def generate_skills() -> str:
    """Generate a realistic skills section."""
    templates = []

    def _t_categorized():
        lines = []
        categories = []

        if random.random() < 0.9:
            langs = _pick(PROGRAMMING_LANGUAGES, k=random.randint(3, 7))
            label = random.choice(["Languages", "Programming Languages", "Programming"])
            categories.append((label, langs))

        if random.random() < 0.9:
            fws = _pick(FRAMEWORKS, k=random.randint(3, 7))
            label = random.choice(["Frameworks", "Frameworks & Libraries", "Technologies"])
            categories.append((label, fws))

        if random.random() < 0.8:
            tls = _pick(TOOLS, k=random.randint(3, 7))
            label = random.choice(["Tools", "Developer Tools", "Tools & Platforms"])
            categories.append((label, tls))

        if random.random() < 0.4:
            ss = _pick(SOFT_SKILLS, k=random.randint(2, 5))
            label = random.choice(["Soft Skills", "Other Skills", "Additional Skills"])
            categories.append((label, ss))

        sep = random.choice([": ", " - ", " — "])
        for label, items in categories:
            joiner = random.choice([", ", " | ", " · ", " / "])
            lines.append(f"{label}{sep}{joiner.join(items)}")

        return "\n".join(lines)

    def _t_flat():
        all_skills = (_pick(PROGRAMMING_LANGUAGES, k=random.randint(3, 6)) +
                      _pick(FRAMEWORKS, k=random.randint(3, 6)) +
                      _pick(TOOLS, k=random.randint(2, 4)))
        random.shuffle(all_skills)
        joiner = random.choice([", ", " | ", " · ", " • "])
        return joiner.join(all_skills)

    def _t_proficiency():
        lines = []
        levels = ["Expert", "Advanced", "Proficient", "Intermediate", "Familiar"]
        used = set()
        for level in random.sample(levels, k=random.randint(2, 4)):
            pool = [s for s in PROGRAMMING_LANGUAGES + FRAMEWORKS + TOOLS if s not in used]
            items = _pick(pool, k=random.randint(2, 5))
            used.update(items)
            lines.append(f"{level}: {', '.join(items)}")
        return "\n".join(lines)

    templates = [_t_categorized, _t_flat, _t_proficiency]
    return random.choice(templates)()


def generate_projects() -> str:
    """Generate a realistic projects section."""

    def _single_project():
        adj = _pick_one(PROJECT_ADJECTIVES)
        noun = _pick_one(PROJECT_NOUNS)
        name = f"{adj} {noun}"
        techs = _pick(PROGRAMMING_LANGUAGES + FRAMEWORKS, k=random.randint(2, 5))

        header_styles = [
            f"{name} | {', '.join(techs)}",
            f"{name}\nTechnologies: {', '.join(techs)}",
            f"{name} ({', '.join(techs)})",
        ]
        lines = [random.choice(header_styles)]

        # Optional link
        if random.random() < 0.3:
            username = _pick_one(FIRST_NAMES).lower() + _pick_one(LAST_NAMES).lower()
            lines.append(f"github.com/{username}/{name.lower().replace(' ', '-')}")

        descriptions = [
            f"Built a {noun.lower()} that {random.choice(['processes', 'analyzes', 'visualizes', 'aggregates', 'transforms'])} {random.choice(['user data', 'financial data', 'text documents', 'sensor data', 'social media feeds', 'medical records'])} in real-time",
            f"Implemented {random.choice(['REST API', 'GraphQL API', 'gRPC service', 'WebSocket server', 'event-driven architecture'])} with {random.choice(['authentication', 'rate limiting', 'caching', 'pagination', 'logging'])} support",
            f"Trained {random.choice(['classification', 'regression', 'NLP', 'computer vision', 'recommendation'])} model achieving {random.choice(['92%', '95%', '97%', '89%', '94%'])} {random.choice(['accuracy', 'F1 score', 'AUC-ROC'])} on test set",
            f"Deployed to {random.choice(['AWS', 'GCP', 'Azure', 'Heroku', 'Vercel', 'Railway'])} with {random.choice(['Docker', 'Kubernetes', 'serverless', 'auto-scaling'])} configuration",
            f"Attracted {random.choice(['100+', '500+', '1K+', '5K+'])} GitHub stars and {random.choice(['20+', '50+', '100+'])} contributors from the open-source community",
            f"Features {random.choice(['real-time notifications', 'responsive UI', 'role-based access control', 'data export', 'interactive visualizations', 'natural language search'])}",
        ]

        b = _bullet()
        for desc in random.sample(descriptions, k=random.randint(2, 4)):
            lines.append(f"{b} {desc}" if b else desc)

        return "\n".join(lines)

    n_projects = random.randint(1, 3)
    return "\n\n".join([_single_project() for _ in range(n_projects)])


def generate_summary() -> str:
    """Generate a realistic professional summary / objective section."""
    years = random.randint(2, 15)
    specialties = _pick(MAJORS + [
        "full-stack development", "distributed systems", "machine learning",
        "data engineering", "cloud architecture", "mobile development",
        "DevOps", "backend development", "frontend development",
        "natural language processing", "computer vision",
    ], k=random.randint(1, 3))

    templates = [
        # Template 1: Traditional summary
        lambda: f"Results-driven {_pick_one(JOB_TITLES).lower()} with {years}+ years of experience in {' and '.join(specialties)}. Proven track record of {random.choice(['delivering high-impact solutions', 'building scalable systems', 'driving technical excellence', 'leading cross-functional teams'])} at companies like {_pick_one(COMPANIES)} and {_pick_one(COMPANIES)}. Passionate about {random.choice(['clean code', 'system design', 'open source', 'mentorship', 'continuous learning', 'innovation'])} and {random.choice(['building products that scale', 'solving complex problems', 'leveraging data-driven insights', 'improving developer experience'])}.",

        # Template 2: Technical focus
        lambda: f"Experienced {_pick_one(JOB_TITLES).lower()} specializing in {', '.join(specialties)}. Skilled in {', '.join(_pick(PROGRAMMING_LANGUAGES, k=3))} with deep expertise in {', '.join(_pick(FRAMEWORKS, k=2))}. {random.choice(['Strong background in', 'Demonstrated ability in', 'Track record of'])} {random.choice(['building distributed systems at scale', 'developing ML models for production', 'architecting cloud-native applications', 'leading agile engineering teams'])}. Seeking to {random.choice(['contribute to cutting-edge products', 'drive technical innovation', 'solve challenging problems', 'build impactful technology'])} at a {random.choice(['fast-growing startup', 'leading technology company', 'mission-driven organization'])}.",

        # Template 3: Achievement-oriented
        lambda: f"{_pick_one(JOB_TITLES)} with {years} years of experience building {random.choice(['enterprise-scale', 'consumer-facing', 'B2B', 'data-intensive'])} applications. Key achievements include {_impact()}, {_impact()}, and {_impact()}. Proficient in {', '.join(_pick(PROGRAMMING_LANGUAGES, k=3))} and {', '.join(_pick(FRAMEWORKS, k=2))}.",

        # Template 4: Brief objective
        lambda: f"Motivated {random.choice(['professional', 'engineer', 'developer', 'technologist'])} seeking a {_pick_one(JOB_TITLES).lower()} role where I can apply my expertise in {' and '.join(specialties)} to {random.choice(['build innovative products', 'solve real-world problems', 'drive business impact', 'push the boundaries of technology'])}.",

        # Template 5: Narrative style
        lambda: f"I am a {_pick_one(JOB_TITLES).lower()} who thrives at the intersection of {_pick_one(specialties)} and {_pick_one(specialties)}. Over the past {years} years, I have {random.choice(['shipped products used by millions', 'built ML systems processing petabytes of data', 'led engineering teams through rapid growth', 'contributed to open-source projects with thousands of stars'])}. I bring a {random.choice(['data-driven', 'user-centric', 'systems-thinking', 'first-principles'])} approach to every problem I tackle.",
    ]

    return random.choice(templates)()


def generate_certifications() -> str:
    """Generate a realistic certifications section."""
    n = random.randint(2, 6)
    certs = _pick(CERTIFICATIONS_LIST, k=n)

    lines = []
    for cert in certs:
        year = random.randint(2019, 2025)
        styles = [
            f"{cert} ({year})",
            f"{cert} — Issued {_pick_one(MONTHS)} {year}",
            f"{cert}, {year}",
            f"{cert}\n  Issued: {_pick_one(MONTHS_SHORT)} {year}" + (
                f" | Expires: {_pick_one(MONTHS_SHORT)} {year + random.randint(2, 3)}"
                if random.random() < 0.3 else ""
            ),
        ]
        lines.append(random.choice(styles))

    b = _bullet()
    if b and random.random() < 0.5:
        return "\n".join(f"{b} {line}" for line in lines)
    return "\n".join(lines)


def generate_contact() -> str:
    """Generate a realistic contact information section."""
    first = _pick_one(FIRST_NAMES)
    last = _pick_one(LAST_NAMES)
    city = _pick_one(CITIES)
    area_code = _pick_one(PHONE_AREA_CODES)
    email_user = random.choice([
        f"{first.lower()}.{last.lower()}",
        f"{first.lower()}{last.lower()}",
        f"{first[0].lower()}{last.lower()}",
        f"{first.lower()}_{last.lower()}",
        f"{first.lower()}{random.randint(1, 99)}",
    ])
    email = f"{email_user}@{_pick_one(DOMAINS)}"
    phone = f"({area_code}) {random.randint(100,999)}-{random.randint(1000,9999)}"
    linkedin_user = f"{first.lower()}-{last.lower()}-{random.randint(100, 999)}"
    github_user = f"{first.lower()}{last.lower()}"

    parts = [f"{first} {last}"]

    if random.random() < 0.8:
        parts.append(email)
    if random.random() < 0.7:
        parts.append(phone)
    if random.random() < 0.6:
        parts.append(city)
    if random.random() < 0.5:
        parts.append(f"{_pick_one(LINKEDIN_PREFIXES)}{linkedin_user}")
    if random.random() < 0.4:
        parts.append(f"{_pick_one(GITHUB_PREFIXES)}{github_user}")
    if random.random() < 0.2:
        parts.append(f"{github_user}.dev" if random.random() < 0.5 else f"{first.lower()}{last.lower()}.com")

    sep = random.choice(["\n", " | ", " · ", "\n"])
    return sep.join(parts)


def generate_awards() -> str:
    """Generate a realistic awards & honors section."""
    n = random.randint(2, 6)
    awards = _pick(AWARDS_LIST, k=n)
    lines = []

    for award in awards:
        year = random.randint(2015, 2025)
        org = random.choice([
            _pick_one(UNIVERSITIES),
            _pick_one(COMPANIES),
            random.choice(["ACM", "IEEE", "Google", "Facebook", "Microsoft",
                          "National Science Foundation", "Department of Education"]),
        ])
        styles = [
            f"{award}, {org} ({year})",
            f"{award} — {org}, {year}",
            f"{award} ({year})\n  Awarded by {org}",
            f"{award}, {year}",
        ]
        lines.append(random.choice(styles))

    b = _bullet()
    if b and random.random() < 0.6:
        return "\n".join(f"{b} {line}" for line in lines)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optional section headers – sometimes sections include a heading
# ---------------------------------------------------------------------------

SECTION_HEADERS = {
    "education": ["EDUCATION", "Education", "Academic Background", "ACADEMIC BACKGROUND", "Education & Training"],
    "experience": ["EXPERIENCE", "Experience", "WORK EXPERIENCE", "Work Experience", "PROFESSIONAL EXPERIENCE", "Professional Experience", "Employment History"],
    "skills": ["SKILLS", "Skills", "TECHNICAL SKILLS", "Technical Skills", "Core Competencies", "CORE COMPETENCIES", "Technologies"],
    "projects": ["PROJECTS", "Projects", "PERSONAL PROJECTS", "Personal Projects", "SIDE PROJECTS", "Selected Projects", "Portfolio"],
    "summary": ["SUMMARY", "Summary", "PROFESSIONAL SUMMARY", "Professional Summary", "OBJECTIVE", "Objective", "PROFILE", "Profile", "About Me", "ABOUT"],
    "certifications": ["CERTIFICATIONS", "Certifications", "CERTIFICATES", "Certificates", "Licenses & Certifications", "PROFESSIONAL CERTIFICATIONS"],
    "contact": ["CONTACT", "Contact", "CONTACT INFORMATION", "Contact Information", "Personal Information"],
    "awards": ["AWARDS", "Awards", "HONORS & AWARDS", "Honors & Awards", "ACHIEVEMENTS", "Achievements", "Awards & Honors", "RECOGNITION"],
}

GENERATORS = {
    "education": generate_education,
    "experience": generate_experience,
    "skills": generate_skills,
    "projects": generate_projects,
    "summary": generate_summary,
    "certifications": generate_certifications,
    "contact": generate_contact,
    "awards": generate_awards,
}


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_example(label: str, include_header: bool = False, augment: bool = False) -> str:
    """
    Generate a single synthetic example for the given label.

    Args:
        label: One of the 8 section categories.
        include_header: Whether to prepend a section header.
        augment: Whether to apply text augmentation.

    Returns:
        Generated text string.
    """
    text = GENERATORS[label]()

    # Optionally prepend a section header
    if include_header and random.random() < 0.5:
        header = _pick_one(SECTION_HEADERS[label])
        sep = random.choice(["\n", "\n\n", "\n---\n"])
        text = f"{header}{sep}{text}"

    # Augmentation
    if augment:
        if random.random() < 0.4:
            text = _synonym_replace(text)
        # Randomly add/remove trailing whitespace or newlines
        if random.random() < 0.2:
            text = text.strip() + "\n"
        if random.random() < 0.1:
            text = "  " + text

    return text


def generate_dataset(
    examples_per_category: int = 80,
    augmented_copies: int = 2,
    include_header_prob: float = 0.4,
    seed: int = 42,
) -> list[dict]:
    """
    Generate a complete synthetic dataset.

    Args:
        examples_per_category: Base examples per category.
        augmented_copies: Number of augmented copies per base example.
        include_header_prob: Probability of including section header.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with 'text' and 'label' keys.
    """
    random.seed(seed)
    labels = list(GENERATORS.keys())
    dataset = []

    for label in labels:
        for i in range(examples_per_category):
            include_header = random.random() < include_header_prob
            text = generate_example(label, include_header=include_header, augment=False)
            dataset.append({"text": text, "label": label})

            # Generate augmented versions
            for _ in range(augmented_copies):
                aug_text = generate_example(label, include_header=include_header, augment=True)
                dataset.append({"text": aug_text, "label": label})

    random.shuffle(dataset)
    return dataset


def save_to_csv(dataset: list[dict], path: str) -> None:
    """Save dataset to CSV."""
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(dataset)
    print(f"Saved {len(dataset)} examples to {filepath}")


def load_as_hf_dataset(dataset: list[dict]):
    """Convert to HuggingFace Dataset with stratified train/val/test splits.

    ``train_test_split(stratify_by_column=...)`` only accepts a ``ClassLabel``
    column, so the label is cast for the split and decoded back to its string
    name afterwards -- ``train.py`` looks labels up by name via ``label2id``.
    """
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

    ds = Dataset.from_list(dataset)
    class_label = ClassLabel(names=sorted({row["label"] for row in dataset}))
    ds = ds.cast_column("label", class_label)

    # 80/10/10 split, stratified at both steps
    train_test = ds.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    val_test = train_test["test"].train_test_split(test_size=0.5, seed=42, stratify_by_column="label")

    # Decode ids back to names. Casting ClassLabel -> string does NOT do this
    # (it stringifies the ids, giving "0"/"1"), and mapping into a column that
    # is still typed ClassLabel silently re-encodes the names straight back to
    # ids -- hence int2str plus an explicit output schema.
    string_features = Features({"text": Value("string"), "label": Value("string")})

    def to_label_name(batch):
        return {"label": class_label.int2str(batch["label"])}

    splits = {
        "train": train_test["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    }
    return DatasetDict({
        name: split.map(to_label_name, batched=True, features=string_features)
        for name, split in splits.items()
    })


def get_label_mapping(dataset: list[dict]) -> tuple[dict, dict]:
    """Create label <-> id mappings."""
    labels = sorted(set(d["label"] for d in dataset))
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic resume section data")
    parser.add_argument("--examples-per-category", type=int, default=80,
                        help="Number of base examples per category (default: 80)")
    parser.add_argument("--augmented-copies", type=int, default=2,
                        help="Number of augmented copies per example (default: 2)")
    parser.add_argument("--output", type=str, default="data/resume_sections.csv",
                        help="Output CSV path (default: data/resume_sections.csv)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--print-stats", action="store_true",
                        help="Print dataset statistics")
    parser.add_argument("--print-samples", type=int, default=0,
                        help="Print N sample examples")

    args = parser.parse_args()

    print(f"Generating dataset with {args.examples_per_category} base examples per category...")
    print(f"Augmented copies per example: {args.augmented_copies}")
    print(f"Total expected examples: {args.examples_per_category * (1 + args.augmented_copies) * 8}")

    dataset = generate_dataset(
        examples_per_category=args.examples_per_category,
        augmented_copies=args.augmented_copies,
        seed=args.seed,
    )

    save_to_csv(dataset, args.output)

    if args.print_stats:
        from collections import Counter
        counts = Counter(d["label"] for d in dataset)
        print("\nDataset Statistics:")
        print(f"  Total examples: {len(dataset)}")
        print(f"  Categories: {len(counts)}")
        for label, count in sorted(counts.items()):
            print(f"    {label}: {count}")
        avg_len = sum(len(d["text"]) for d in dataset) / len(dataset)
        print(f"  Average text length: {avg_len:.0f} chars")

    if args.print_samples > 0:
        print(f"\n{'='*60}")
        print(f"Sample Examples (first {args.print_samples}):")
        print(f"{'='*60}")
        for i, example in enumerate(dataset[:args.print_samples]):
            print(f"\n--- Example {i+1} [{example['label']}] ---")
            print(example["text"][:300])
            if len(example["text"]) > 300:
                print("...")
