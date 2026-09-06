"""
response_generator.py
---------------------
Grounded LLM Response Generation Layer for Customer Support Chatbot.
Generates conversational, helpful responses conditioned on:
  1. Original user query
  2. Verified predicted intent (from LoRA classifier)
  3. Intent guidance policy (procedural instructions without invented facts)

Reuses the base Qwen2.5-0.5B model with LoRA adapters temporarily disabled.
Includes strict anti-hallucination safeguards and output validation.
"""

from typing import Dict, Optional
import re
import torch
from transformers import AutoTokenizer, PreTrainedModel


class SupportResponseGenerator:
    """
    Generates grounded customer support responses using the base language model.
    Enforces strict safety and grounding constraints:
      - Never requests sensitive credentials (card numbers, CVV, PIN, passwords).
      - Never claims to execute backend actions (e.g. "I have updated", "I will process").
      - Never invents UI layout details, specific URLs, or unsupported policies.
      - Keeps responses concise (2-3 sentences).
    """

    INTENT_GUIDANCE: Dict[str, str] = {
        "cancel_order": (
            "Explain that order cancellations can be requested prior to dispatch, "
            "and ask the customer for their order details if not provided."
        ),
        "change_order": (
            "Explain that modifications can be requested before order fulfillment, "
            "and ask for their order details along with the items they wish to update."
        ),
        "change_shipping_address": (
            "Explain that delivery addresses can be updated before shipment, "
            "and ask for their order details and the new address."
        ),
        "check_cancellation_fee": (
            "Explain that cancellation terms depend on the order status and product type, "
            "and ask for their order details to assist further."
        ),
        "check_invoice": (
            "Acknowledge the invoice inquiry, explain that invoice records are linked to "
            "completed orders, and ask for their order details."
        ),
        "check_payment_methods": (
            "Explain that available payment options are presented during checkout, "
            "and offer assistance if they have questions about payment options."
        ),
        "check_refund_policy": (
            "Explain that refund eligibility depends on product condition and return "
            "request guidelines, and ask for details about their item."
        ),
        "complaint": (
            "Apologize for the inconvenience, acknowledge their feedback, and ask for "
            "the details of the issue so support can address it."
        ),
        "contact_customer_service": (
            "Acknowledge their request for customer service assistance and ask how "
            "support can help with their inquiry."
        ),
        "contact_human_agent": (
            "Acknowledge the request to connect with a representative and ask them to "
            "provide their issue details while the request is routed."
        ),
        "create_account": (
            "Guide the customer to complete registration by providing their email and "
            "setting up their account credentials on the sign-up page."
        ),
        "delete_account": (
            "Explain the general account closure procedure and mention that account data "
            "is permanently removed upon deletion."
        ),
        "delivery_options": (
            "Explain that available shipping methods and delivery speeds are displayed "
            "during checkout based on the destination address."
        ),
        "delivery_period": (
            "Explain that delivery timelines vary depending on the chosen delivery method "
            "and destination once an order has dispatched."
        ),
        "edit_account": (
            "Explain that personal information and contact details can be updated "
            "within their account profile settings."
        ),
        "get_invoice": (
            "Acknowledge the request for an invoice copy and ask for the order details "
            "so the invoice document can be provided."
        ),
        "get_refund": (
            "Explain the refund request process and ask for their order details and "
            "the reason for the refund request."
        ),
        "newsletter_subscription": (
            "Explain that newsletter preferences can be managed through subscription links "
            "in received emails or in communication settings."
        ),
        "payment_issue": (
            "Acknowledge the payment difficulty, suggest verifying entered payment details "
            "or trying an alternative payment method, and offer help."
        ),
        "place_order": (
            "Guide the customer to add their selected items to the shopping cart and complete "
            "the steps through the checkout process."
        ),
        "recover_password": (
            "Guide the customer to use the self-service password recovery option on the sign-in page to "
            "receive a secure reset link at their registered email. Never ask for passwords."
        ),
        "registration_problems": (
            "Acknowledge the sign-up issue, suggest verifying that all required fields are "
            "correctly completed, and offer further troubleshooting."
        ),
        "review": (
            "Guide the customer to submit product ratings and written feedback directly "
            "on the relevant product page."
        ),
        "set_up_shipping_address": (
            "Explain that shipping addresses can be added and saved during checkout or "
            "in the address section of their account."
        ),
        "switch_account": (
            "Guide the customer to sign out of the current session and sign in with the "
            "credentials of their other account."
        ),
        "track_order": (
            "Explain that order tracking updates are available with the order reference, "
            "and ask for their order details if not provided."
        ),
        "track_refund": (
            "Explain that refund progress can be checked once the return has been processed, "
            "and ask for their order or refund reference."
        ),
    }

    SYSTEM_PROMPT_TEMPLATE = (
        "You are a professional customer support assistant for an online store.\n"
        "Your task: Write a concise, helpful 2-sentence response based ONLY on the guidance below.\n\n"
        "Customer Intent: {intent}\n"
        "Guidance: {intent_guidance}\n\n"
        "STRICT SECURITY & CONSTRAINTS:\n"
        "1. NEVER ask the customer for their password, credit card numbers, CVV, or PIN under any circumstances.\n"
        "2. NEVER claim you directly updated an address, checked a database, or processed a refund.\n"
        "3. NEVER invent specific button locations, UI tabs, URLs, or unstated policies.\n"
        "4. Do NOT output numbered lists. Keep it in natural, polite paragraph form (2 sentences max)."
    )

    PROHIBITED_CREDENTIAL_PATTERNS = [
        r"\b(card\s+number|cvv|cvc|expiration\s+date|pin\s+number)\b",
        r"\b(credit\s+card\s+details|full\s+card\s+number)\b",
        r"\b(provide\s+(me\s+with\s+)?(your\s+)?(current\s+)?password|tell\s+me\s+your\s+password|enter\s+your\s+password)\b",
    ]

    UNSUPPORTED_ACTION_CLAIMS = [
        (r"i\s+(will|have)\s+update(d)?\s+your\s+(shipping\s+)?address", "you can request an address update by providing your order details and new address"),
        (r"i\s+(will|have)\s+process(ed)?\s+your\s+refund", "our support team can review your refund request once you provide the order details"),
        (r"i\s+(will|have)\s+cancel(led)?\s+your\s+order", "our support team can assist with cancelling your order once order details are provided"),
        (r"i\s+checked\s+your\s+order", "to check your order details"),
    ]

    def __init__(self, model: PreTrainedModel, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.prohibited_regex = [re.compile(p, re.IGNORECASE) for p in self.PROHIBITED_CREDENTIAL_PATTERNS]

    def _validate_and_sanitize(self, raw_text: str, intent: str) -> str:
        """
        Lightweight output validator to guarantee safety and concise formatting.
        """
        text = raw_text.strip()

        # Check 1: Sensitive credential request detection (passwords, cards, pins)
        for pattern in self.prohibited_regex:
            if pattern.search(text):
                if "password" in intent:
                    return (
                        "For your security, please never share your password. You can reset or update your "
                        "password using the self-service recovery option on the sign-in page to receive a reset link at your registered email."
                    )
                return (
                    f"For your security, please never share sensitive payment credentials or passwords. "
                    f"If you need assistance with {intent.replace('_', ' ')}, please provide your order reference."
                )

        # Check 2: Replace false claims of backend execution
        for pattern, replacement in self.UNSUPPORTED_ACTION_CLAIMS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Check 3: Flatten multi-line lists to clean paragraph form
        text = re.sub(r"\n+", " ", text).strip()
        text = re.sub(r"^\s*\d+\.\s*", "", text)
        text = re.sub(r"\s+\d+\.\s*", " ", text)

        # Check 4: Ensure clean sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        clean_sentences = [s for s in sentences if s.endswith((".", "!", "?"))]
        if clean_sentences:
            text = " ".join(clean_sentences[:2])

        return text

    def generate_response(
        self,
        user_query: str,
        predicted_intent: str,
        max_new_tokens: int = 80,
        temperature: float = 0.2,
    ) -> str:
        """
        Generates a natural, grounded response conditioned on the user query and intent.
        Reuses the base model by temporarily disabling the LoRA adapter.
        """
        guidance = self.INTENT_GUIDANCE.get(
            predicted_intent,
            "Acknowledge the customer's request and ask how you can assist them further."
        )

        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            intent=predicted_intent,
            intent_guidance=guidance,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query.strip()},
        ]

        # Use Qwen chat template
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt")

        # Disable LoRA adapter to utilize the base model's conversational capability
        is_peft = hasattr(self.model, "disable_adapter")

        with torch.no_grad():
            if is_peft:
                with self.model.disable_adapter():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.85,
                        repetition_penalty=1.15,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
            else:
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.85,
                    repetition_penalty=1.15,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

        generated_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        raw_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Run output validation
        clean_response = self._validate_and_sanitize(raw_response, predicted_intent)
        return clean_response
