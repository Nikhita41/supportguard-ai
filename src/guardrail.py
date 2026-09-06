"""
guardrail.py
------------
Multi-stage guardrail layer for the Customer Support Chatbot.
Executes BEFORE the LoRA intent classifier to ensure queries are in-scope,
safe, and sufficiently specific.

Architecture:
  Stage 1: Safety, Injection & Task-Boundary Filter (regex heuristics for jailbreaks, code, creative writing, math)
  Stage 2: Ambiguity & Vagueness Filter (intercepts low-entropy queries and prompts for clarification)
  Stage 3: Domain Relevance Filter (Domain Concept Anchors + Calibrated TF-IDF Cosine Similarity, threshold=0.12)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class GuardrailResult:
    is_allowed: bool
    reason: str             # "in_scope" | "out_of_scope" | "ambiguous_query" | "prompt_injection" | "empty_query"
    confidence: float       # 0.0 to 1.0
    fallback_message: Optional[str] = None
    suggested_action: Optional[str] = None


class CustomerSupportGuardrail:
    """
    Modular, CPU-friendly guardrail for customer-support chatbots.
    Runs in < 2ms without requiring a secondary neural model.
    """

    # -----------------------------------------------------------------------
    # Stage 1: Prompt Injection & Task-Boundary Patterns (Isolated)
    # -----------------------------------------------------------------------
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior|above|system)\s+instructions?",
        r"forget\s+(that\s+)?you\s+are\s+(a\s+)?(customer\s+support|bot|assistant|ai)",
        r"(give|show|reveal|display|print)\s+(me\s+)?(your\s+)?(system\s+prompt|instructions?|hidden\s+rules?)",
        r"act\s+as\s+(a\s+)?(developer|coder|hacker|python\s+interpreter|terminal|linux|dan|jailbreak)",
        r"(write|generate|create|debug|execute|run)\s+.*(python|java|c\+\+|javascript|sql|code|script|function|program|html|css)",
        r"(write|generate|compose|draft|tell)\s+.*(essay|poem|song|story|haiku|joke|article|speech|riddle)",
        r"(solve|calculate|compute)\s+.*(math|equation|integral|derivative|algebra|\d+\s*[\+\-\*\/])",
        r"bypass\s+(safety|content|policy|filter)",
        r"you\s+are\s+now\s+in\s+developer\s+mode",
        r"disregard\s+(all\s+)?guidelines",
    ]

    # -----------------------------------------------------------------------
    # Stage 2: Ambiguous / Vague Query Patterns (Isolated)
    # -----------------------------------------------------------------------
    AMBIGUOUS_PATTERNS = [
        r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))[\.!\?]*$",
        r"^(help|help\s+me|i\s+need\s+help|please\s+help|assist\s+me)[\.!\?]*$",
        r"^(i\s+have\s+a\s+problem|there\s+is\s+a\s+problem|problem|issue)[\.!\?]*$",
        r"^(can\s+you\s+help(\s+me)?|what\s+can\s+you\s+do|who\s+are\s+you)[\.!\?]*$",
        r"^(i\s+have\s+a\s+question|question|support|customer\s+support)[\.!\?]*$",
    ]

    # -----------------------------------------------------------------------
    # Stage 3: Domain Concept Anchors & Prototypes
    # -----------------------------------------------------------------------
    DOMAIN_ANCHORS: Set[str] = {
        "order", "orders", "cancel", "cancellation", "shipping", "shipment",
        "delivery", "deliver", "delivered", "package", "parcel", "item", "items",
        "product", "products", "invoice", "invoices", "bill", "billing", "receipt",
        "refund", "refunds", "return", "returns", "reimbursement", "payment",
        "payments", "pay", "card", "credit", "debit", "paypal", "checkout",
        "account", "accounts", "password", "passwords", "login", "signin",
        "signup", "register", "registration", "profile", "user", "email",
        "address", "location", "destination", "postal", "zip", "courier",
        "agent", "representative", "human", "person", "operator", "someone",
        "speak", "talk", "contact", "service", "support", "helpdesk",
        "complaint", "complaints", "review", "reviews", "feedback", "rating",
        "newsletter", "subscription", "subscriptions", "subscribe", "unsubscribe",
        "fee", "charge", "charges", "cost", "policy", "track", "tracking",
        "transit", "arrival", "arrive", "purchase", "purchased", "bought",
    }

    DOMAIN_SEEDS = [
        "order cancel tracking package shipment delivery status address destination",
        "refund money reimbursement return charge fee policy cancellation invoice bill receipt",
        "account password login profile credentials register create delete switch edit email",
        "payment card credit debit transaction checkout failed declined issue",
        "customer service support agent representative human help assistance complaint review feedback",
        "newsletter subscription subscribe unsubscribe email marketing promotional",
        "change delivery shipping address update street city postal code apartment location",
        "estimated arrival date transit period delivery options courier express standard",
        "create open register profile sign up account membership",
        "delete remove terminate cancel close account user profile",
        "request claim refund get my money back reimbursement status",
        "leave submit product review rating feedback customer experience",
    ]

    def __init__(self, data_csv_path: Optional[Path] = None, similarity_threshold: float = 0.12):
        """
        Initializes guardrail rules and fits the TF-IDF domain vectorizer.
        """
        self.similarity_threshold = similarity_threshold
        self.injection_regex = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
        self.ambiguous_regex = [re.compile(p, re.IGNORECASE) for p in self.AMBIGUOUS_PATTERNS]

        domain_corpus = list(self.DOMAIN_SEEDS)

        # Supplement with individual sample instructions per intent
        if data_csv_path and Path(data_csv_path).exists():
            try:
                df = pd.read_csv(data_csv_path, usecols=["instruction", "intent"])
                sample_instructions = (
                    df.groupby("intent")
                    .apply(lambda s: s["instruction"].sample(min(30, len(s)), random_state=42))
                    .reset_index(drop=True)
                    .tolist()
                )
                domain_corpus.extend(sample_instructions)
            except Exception:
                pass

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True,
            sublinear_tf=True,
            max_features=4000,
        )
        self.domain_matrix = self.vectorizer.fit_transform(domain_corpus)

    def _has_domain_anchor(self, text: str) -> bool:
        """Checks if the query contains at least one domain-specific root term."""
        tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))
        return bool(tokens.intersection(self.DOMAIN_ANCHORS))

    def validate(self, query: str) -> GuardrailResult:
        """
        Runs the 3-stage validation pipeline on user query.
        Returns a structured GuardrailResult.
        """
        text = query.strip()

        # Sanity Check: Empty or near-empty
        if len(text) < 2:
            return GuardrailResult(
                is_allowed=False,
                reason="empty_query",
                confidence=0.0,
                fallback_message="Please enter a customer support inquiry so I can assist you.",
                suggested_action="prompt_user",
            )

        # Stage 1: Safety, Injection & Task-Boundary Check (Isolated)
        for pattern in self.injection_regex:
            if pattern.search(text):
                return GuardrailResult(
                    is_allowed=False,
                    reason="prompt_injection",
                    confidence=1.0,
                    fallback_message=(
                        "I am a specialized customer support assistant and cannot execute code, "
                        "follow instruction overrides, or generate creative writing/math. "
                        "How can I help you with your order, account, or billing?"
                    ),
                    suggested_action="refuse_override",
                )

        # Stage 2: Ambiguity & Vague Query Check (Isolated)
        for pattern in self.ambiguous_regex:
            if pattern.match(text):
                return GuardrailResult(
                    is_allowed=False,
                    reason="ambiguous_query",
                    confidence=0.5,
                    fallback_message=(
                        "Hello! I'd be glad to help you. Could you please specify what you need "
                        "assistance with (e.g., tracking an order, resetting a password, requesting a refund, "
                        "or updating a delivery address)?"
                    ),
                    suggested_action="clarify_query",
                )

        # Stage 3: Domain Relevance Check (Domain Anchors + TF-IDF)
        has_anchor = self._has_domain_anchor(text)

        query_vec = self.vectorizer.transform([text])
        similarities = cosine_similarity(query_vec, self.domain_matrix)[0]
        max_similarity = float(similarities.max())

        # If query has domain anchors and exceeds threshold -> In-scope
        if has_anchor and max_similarity >= self.similarity_threshold:
            return GuardrailResult(
                is_allowed=True,
                reason="in_scope",
                confidence=max_similarity,
                fallback_message=None,
                suggested_action="route_to_classifier",
            )
        else:
            return GuardrailResult(
                is_allowed=False,
                reason="out_of_scope",
                confidence=max_similarity,
                fallback_message=(
                    "I am a customer support assistant. I can only help with inquiries related to "
                    "orders, deliveries, shipping, payments, invoices, accounts, and refunds. "
                    "Please ask a question related to these services."
                ),
                suggested_action="refuse_out_of_scope",
            )
