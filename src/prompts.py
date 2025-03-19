from langchain.prompts import PromptTemplate

basic_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=(
        "You are an AI assistant answering questions based strictly on the provided context. "
        "If the answer is not found in the context, say 'I don’t know based on the given information.'\n\n"
        "**Context:**\n{context}\n\n"
        "**User Question:** {question}\n\n"
        "**Answer:**"
    )
)

summary_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "Summarize the following conversation history concisely:\n\n"
        "{text}\n\n"
        "**Summary:**"
    )
)

retrieval_prompt = PromptTemplate(
    input_variables=["current_question", "history"],
    template=(
        "You are an AI deciding if new information needs to be retrieved from a database to answer a question. "
        "Given the current question and conversation history, respond with 'YES' if new retrieval is needed, "
        "or 'NO' if the existing context is sufficient.\n\n"
        "**Conversation History:**\n{history}\n\n"
        "**Current Question:** {current_question}\n\n"
        "**Response:** YES or NO"
    )
)