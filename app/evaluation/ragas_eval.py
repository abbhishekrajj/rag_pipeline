from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy,context_precision,context_recall
from typing import List, Dict
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq


class RAGASEvaluator:

    def __init__(self):
        pass

    def evaluate_response(
        self,
        question: str,
        answer: str,
        contexts: List[str]
    ) -> Dict:

        try:
            # 🔥 Validate inputs
            if not question or not answer or not contexts:
                return {"error": "Missing input for evaluation"}

            # 🔥 Dataset format (VERY IMPORTANT)
            dataset = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],  # list of list
            }

            # 🔥 Run evaluation
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy]
            )

            # 🔥 Extract scores
            return {
                "faithfulness": float(result["faithfulness"][0]),
                "answer_relevancy": float(result["answer_relevancy"][0])
            }

        except Exception as e:
            return {
                "error": str(e)
            }