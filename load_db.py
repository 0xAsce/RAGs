import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace
from ollama import embeddings
from langchain_openai import ChatOpenAI


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("query_text", type=str, required=True, help="the query text to ")
    args = parse.parse_args()
    query_text = args.query_text


    #prepare the database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
    db = Chroma(persist_directory="chroma", embedding_function=embeddings)
    
    #search the database
    results = db.similarity_search_with_score(query_text, k=5)
    if len(results) == 0 or results[0][1] < 0.7:
        print("unable to find relevant information")
        return
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()