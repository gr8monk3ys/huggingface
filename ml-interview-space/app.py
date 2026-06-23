"""
ML Interview Prep - Interactive practice for ML/DS interview questions.
"""

import gradio as gr
import pandas as pd

# ---------------------------------------------------------------------------
# Sample Question Database
# ---------------------------------------------------------------------------

# Embedded sample questions (in production, load from dataset)
QUESTIONS = [
    # Statistics
    {"id": "1", "question": "Explain the difference between Type I and Type II errors.", "answer": "Type I error (false positive) occurs when we reject a true null hypothesis. Type II error (false negative) occurs when we fail to reject a false null hypothesis. In ML terms, Type I is like flagging a legitimate transaction as fraud, while Type II is missing actual fraud. The tradeoff between these errors is controlled by the significance level (alpha) and power (1-beta) of the test.", "category": "Statistics", "difficulty": "easy", "company_tags": "Google|Meta|Amazon", "topic_tags": "hypothesis testing|statistical inference"},
    {"id": "2", "question": "What is the Central Limit Theorem and why is it important?", "answer": "The Central Limit Theorem states that the sampling distribution of the sample mean approaches a normal distribution as sample size increases, regardless of the population's distribution. This is crucial because it allows us to make inferences about population parameters using normal distribution properties, enables hypothesis testing and confidence intervals, and justifies many statistical methods even when the underlying data isn't normally distributed.", "category": "Statistics", "difficulty": "easy", "company_tags": "Google|Meta|Netflix", "topic_tags": "probability|distributions"},
    {"id": "3", "question": "How would you handle multicollinearity in a regression model?", "answer": "Multicollinearity can be addressed by: 1) Removing highly correlated features based on VIF (Variance Inflation Factor) > 5-10, 2) Using regularization (Ridge/Lasso) which shrinks correlated coefficients, 3) PCA to create uncorrelated components, 4) Domain knowledge to select the most meaningful feature among correlated ones. The choice depends on whether you need interpretable coefficients (remove features) or just prediction (regularization).", "category": "Statistics", "difficulty": "medium", "company_tags": "Meta|Airbnb|Uber", "topic_tags": "regression|feature selection"},

    # ML Theory
    {"id": "4", "question": "Explain the bias-variance tradeoff.", "answer": "The bias-variance tradeoff describes the tension between two sources of prediction error. High bias means the model is too simple and underfits (e.g., linear regression on nonlinear data). High variance means the model is too complex and overfits (e.g., unpruned decision tree). Total error = Bias² + Variance + Irreducible noise. We balance this through model selection, regularization, and ensemble methods. Cross-validation helps find the sweet spot.", "category": "ML Theory", "difficulty": "easy", "company_tags": "Google|Amazon|Microsoft", "topic_tags": "fundamentals|model selection"},
    {"id": "5", "question": "What's the difference between bagging and boosting?", "answer": "Bagging (Bootstrap Aggregating) trains models in parallel on random subsets of data, then averages predictions. It reduces variance (Random Forest). Boosting trains models sequentially, with each model focusing on errors of previous ones. It reduces bias (XGBoost, AdaBoost). Bagging works well when base models overfit; boosting works well when they underfit. Boosting is more prone to overfitting but often achieves higher accuracy with proper tuning.", "category": "ML Theory", "difficulty": "medium", "company_tags": "Google|Meta|Apple", "topic_tags": "ensemble|algorithms"},
    {"id": "6", "question": "How does gradient descent work and what are its variants?", "answer": "Gradient descent iteratively updates parameters in the direction of steepest descent of the loss function: θ = θ - α∇L(θ). Variants include: Batch GD (uses all data, stable but slow), Stochastic GD (one sample, noisy but fast), Mini-batch GD (compromise). Advanced optimizers like Adam combine momentum (accumulates past gradients) and RMSprop (adaptive learning rates). Choice depends on dataset size, convergence needs, and computational resources.", "category": "ML Theory", "difficulty": "medium", "company_tags": "Google|DeepMind|OpenAI", "topic_tags": "optimization|training"},

    # Deep Learning
    {"id": "7", "question": "Explain the vanishing gradient problem and how to address it.", "answer": "Vanishing gradients occur when gradients become very small during backpropagation in deep networks, preventing early layers from learning. Causes include sigmoid/tanh activations (derivatives < 1). Solutions: 1) ReLU activation (gradient = 1 for positive inputs), 2) Batch/Layer normalization (stabilizes activations), 3) Residual connections (skip connections allow gradient flow), 4) Proper weight initialization (Xavier/He). Modern architectures like ResNet and Transformers incorporate these solutions.", "category": "Deep Learning", "difficulty": "medium", "company_tags": "Google|Meta|OpenAI", "topic_tags": "neural networks|training"},
    {"id": "8", "question": "What is the attention mechanism and why is it important?", "answer": "Attention allows models to focus on relevant parts of input when producing output. It computes weighted combinations of values based on query-key similarity: Attention(Q,K,V) = softmax(QK^T/√d)V. Importance: 1) Captures long-range dependencies without sequential processing, 2) Provides interpretability through attention weights, 3) Enables parallelization (vs RNNs). Self-attention (Transformers) revolutionized NLP and is now used in vision (ViT) and other domains.", "category": "Deep Learning", "difficulty": "medium", "company_tags": "Google|OpenAI|Anthropic", "topic_tags": "transformers|architectures"},
    {"id": "9", "question": "How would you handle class imbalance in deep learning?", "answer": "Strategies include: 1) Data-level: oversampling minority (SMOTE), undersampling majority, data augmentation, 2) Algorithm-level: class weights in loss function, focal loss (down-weights easy examples), threshold adjustment, 3) Ensemble: combine models trained on balanced subsets. For neural networks specifically: stratified batching, two-phase training (pretrain on balanced, fine-tune on original). Evaluation should use precision-recall curves and F1 rather than accuracy.", "category": "Deep Learning", "difficulty": "hard", "company_tags": "Amazon|PayPal|Stripe", "topic_tags": "imbalanced data|training"},

    # NLP
    {"id": "10", "question": "Explain word embeddings and their evolution.", "answer": "Word embeddings map words to dense vectors capturing semantic meaning. Evolution: 1) One-hot encoding (sparse, no semantics), 2) Word2Vec/GloVe (static embeddings from co-occurrence), 3) ELMo (contextualized via bidirectional LSTM), 4) BERT/GPT (contextualized via Transformers). Key insight: words with similar contexts have similar vectors. Modern embeddings are contextual (same word gets different vectors based on context) and can be fine-tuned for downstream tasks.", "category": "NLP", "difficulty": "medium", "company_tags": "Google|OpenAI|Meta", "topic_tags": "embeddings|representations"},
    {"id": "11", "question": "What are the key differences between BERT and GPT?", "answer": "BERT (Bidirectional Encoder Representations from Transformers) uses masked language modeling and sees full context bidirectionally. Best for understanding tasks (classification, NER, QA). GPT (Generative Pre-trained Transformer) uses autoregressive language modeling, predicting next token left-to-right. Best for generation tasks. BERT is encoder-only, GPT is decoder-only. For tasks needing both understanding and generation, encoder-decoder (T5) or large autoregressive models (GPT-4) with in-context learning work well.", "category": "NLP", "difficulty": "medium", "company_tags": "Google|OpenAI|Microsoft", "topic_tags": "transformers|language models"},

    # System Design
    {"id": "12", "question": "Design a recommendation system for an e-commerce platform.", "answer": "Components: 1) Data collection: user behavior (clicks, purchases, time), item features, context (time, device), 2) Candidate generation: collaborative filtering (user-item matrix factorization), content-based (item similarity), 3) Ranking: ML model combining features (user, item, context) to predict engagement, 4) Serving: precompute for cold start, real-time for logged-in users, 5) Feedback loop: A/B testing, handling cold start (popular items, explore/exploit). Key tradeoffs: latency vs personalization, diversity vs relevance, short vs long-term engagement.", "category": "System Design", "difficulty": "hard", "company_tags": "Amazon|Netflix|Spotify", "topic_tags": "recommendations|architecture"},
    {"id": "13", "question": "How would you design an ML pipeline for real-time fraud detection?", "answer": "Architecture: 1) Data ingestion: Kafka for streaming transactions, 2) Feature engineering: real-time features (velocity, device fingerprint) + batch features (historical patterns), 3) Model: ensemble of rules + ML (isolation forest, XGBoost) for sub-100ms latency, 4) Serving: feature store for consistency, model versioning, 5) Feedback: human review loop, delayed labels, continuous retraining. Key considerations: class imbalance, adversarial adaptation, explainability for disputes, cost of false positives/negatives.", "category": "System Design", "difficulty": "hard", "company_tags": "PayPal|Stripe|Square", "topic_tags": "fraud|real-time systems"},

    # Feature Engineering
    {"id": "14", "question": "What are the most important feature engineering techniques?", "answer": "Key techniques: 1) Numerical: scaling (StandardScaler, MinMax), log transform for skewed data, binning, polynomial features, 2) Categorical: one-hot encoding, target encoding (with smoothing to avoid leakage), frequency encoding, 3) Temporal: lag features, rolling statistics, cyclical encoding (sin/cos for hours), 4) Text: TF-IDF, embeddings, 5) Interactions: domain-driven feature combinations. The best technique depends on the algorithm (trees don't need scaling) and domain knowledge. Always validate with cross-validation.", "category": "Feature Engineering", "difficulty": "medium", "company_tags": "Airbnb|Uber|Meta", "topic_tags": "preprocessing|features"},

    # A/B Testing
    {"id": "15", "question": "How do you determine sample size for an A/B test?", "answer": "Sample size depends on: 1) Baseline conversion rate (p), 2) Minimum detectable effect (MDE), 3) Significance level (α, typically 0.05), 4) Power (1-β, typically 0.8). Formula: n = 2 * ((z_α/2 + z_β)² * p(1-p)) / MDE². Practical considerations: higher baseline = more power, smaller MDE needs more samples, account for multiple testing if many variants. Use power calculators. For ratio metrics, variance is harder to estimate—consider pre-experiment data analysis.", "category": "A/B Testing", "difficulty": "medium", "company_tags": "Google|Meta|Netflix", "topic_tags": "experimentation|statistics"},
]

# Convert to DataFrame
questions_df = pd.DataFrame(QUESTIONS)

# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------


def new_session_state() -> dict:
    """Return a fresh per-session quiz state.

    Held in a ``gr.State`` so each visitor has independent progress and the
    revealed answer can never leak across concurrent users.
    """
    return {"current": None, "answered": 0, "seen": []}

# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def get_random_question(categories: list, difficulties: list, state: dict) -> dict:
    """Get a random question matching filters, updating *state* in place."""
    filtered = questions_df.copy()

    if categories and "All" not in categories:
        filtered = filtered[filtered["category"].isin(categories)]

    if difficulties and "All" not in difficulties:
        filtered = filtered[filtered["difficulty"].isin([d.lower() for d in difficulties])]

    if filtered.empty:
        return None

    # Avoid repeating recent questions
    available = filtered[~filtered["id"].isin(state["seen"][-10:])]
    if available.empty:
        available = filtered

    question = available.sample(1).iloc[0].to_dict()
    state["seen"].append(question["id"])
    state["current"] = question

    return question


def format_question(question: dict) -> str:
    """Format question for display."""
    if not question:
        return "No questions match your filters. Try selecting different categories or difficulties."

    companies = question.get("company_tags", "").replace("|", ", ")

    output = f"""## {question['question']}

**Category:** {question['category']} | **Difficulty:** {question['difficulty'].title()}

**Common at:** {companies}
"""
    return output


def format_answer(question: dict) -> str:
    """Format answer for display."""
    if not question:
        return ""

    topics = question.get("topic_tags", "").replace("|", ", ")

    output = f"""## Answer

{question['answer']}

---

**Topics:** {topics}
"""
    return output


def start_quiz(categories: list, difficulties: list, state: dict) -> tuple[str, str, str, dict]:
    """Start a new quiz question for this session."""
    if state is None:
        state = new_session_state()

    question = get_random_question(categories, difficulties, state)
    if question is None:
        return format_question(None), "", "No questions match your filters.", state

    # Only count questions that were actually served.
    state["answered"] += 1
    return format_question(question), "", f"Question #{state['answered']}", state


def reveal_answer(state: dict) -> str:
    """Reveal the answer to this session's current question."""
    if state and state.get("current"):
        return format_answer(state["current"])
    return "No question loaded. Click 'Next Question' first."


def browse_questions(category: str, difficulty: str, search: str) -> str:
    """Browse and filter all questions."""
    filtered = questions_df.copy()

    if category and category != "All":
        filtered = filtered[filtered["category"] == category]

    if difficulty and difficulty != "All":
        filtered = filtered[filtered["difficulty"] == difficulty.lower()]

    if search:
        mask = (
            filtered["question"].str.contains(search, case=False) |
            filtered["answer"].str.contains(search, case=False)
        )
        filtered = filtered[mask]

    if filtered.empty:
        return "No questions match your search."

    output = f"## Found {len(filtered)} Questions\n\n"

    for _, row in filtered.iterrows():
        output += f"### {row['question']}\n\n"
        output += f"**{row['category']}** | **{row['difficulty'].title()}**\n\n"
        output += f"{row['answer']}\n\n"
        output += "---\n\n"

    return output


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

CATEGORIES = ["All"] + sorted(questions_df["category"].unique().tolist())
DIFFICULTIES = ["All", "Easy", "Medium", "Hard"]

with gr.Blocks(title="ML Interview Prep", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # ML Interview Prep

    Practice machine learning and data science interview questions.
    Choose your categories and difficulty, then test your knowledge!
    """)

    # Per-session state (deep-copied by Gradio for each visitor)
    quiz_session = gr.State(new_session_state())

    with gr.Tabs():
        # Quiz Mode Tab
        with gr.TabItem("Quiz Mode"):
            gr.Markdown("### Practice with random questions")

            with gr.Row():
                category_select = gr.Dropdown(
                    choices=CATEGORIES,
                    value=["All"],
                    multiselect=True,
                    label="Categories",
                )
                difficulty_select = gr.Dropdown(
                    choices=DIFFICULTIES,
                    value=["All"],
                    multiselect=True,
                    label="Difficulties",
                )

            with gr.Row():
                next_btn = gr.Button("Next Question", variant="primary")
                reveal_btn = gr.Button("Show Answer", variant="secondary")

            status_text = gr.Textbox(label="Status", value="Click 'Next Question' to start")
            question_output = gr.Markdown(label="Question")
            answer_output = gr.Markdown(label="Answer")

            next_btn.click(
                fn=start_quiz,
                inputs=[category_select, difficulty_select, quiz_session],
                outputs=[question_output, answer_output, status_text, quiz_session],
            )

            reveal_btn.click(
                fn=reveal_answer,
                inputs=[quiz_session],
                outputs=answer_output,
            )

        # Browse Mode Tab
        with gr.TabItem("Browse All"):
            gr.Markdown("### Search and filter all questions")

            with gr.Row():
                browse_category = gr.Dropdown(
                    choices=CATEGORIES,
                    value="All",
                    label="Category",
                )
                browse_difficulty = gr.Dropdown(
                    choices=DIFFICULTIES,
                    value="All",
                    label="Difficulty",
                )
                search_box = gr.Textbox(
                    label="Search",
                    placeholder="Search questions and answers...",
                )

            search_btn = gr.Button("Search", variant="primary")
            browse_output = gr.Markdown()

            search_btn.click(
                fn=browse_questions,
                inputs=[browse_category, browse_difficulty, search_box],
                outputs=browse_output,
            )

        # Stats Tab
        with gr.TabItem("About"):
            stats = f"""
### Question Database Statistics

- **Total Questions:** {len(questions_df)}
- **Categories:** {', '.join(questions_df['category'].unique())}
- **Difficulty Distribution:**
  - Easy: {len(questions_df[questions_df['difficulty'] == 'easy'])}
  - Medium: {len(questions_df[questions_df['difficulty'] == 'medium'])}
  - Hard: {len(questions_df[questions_df['difficulty'] == 'hard'])}

### Tips for Interview Prep

1. **Start with fundamentals** - Ensure you understand basic concepts before advanced topics
2. **Practice explaining** - Say your answers out loud as if in an interview
3. **Understand trade-offs** - Most questions have nuanced answers depending on context
4. **Know your projects** - Be ready to connect concepts to your own experience
5. **Ask clarifying questions** - In real interviews, it's good to ask about constraints

### Company-Specific Prep

Questions are tagged with companies where similar questions are commonly asked.
Filter by company in the Browse tab to focus your preparation.

---

Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys)
"""
            gr.Markdown(stats)

if __name__ == "__main__":
    demo.launch()
