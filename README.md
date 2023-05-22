# Langchain_Semnatic_Search_Pinecone
This Repository is Text Based Q&amp;A model like chat-Gpt using langchain, pinecone, semantic search
# Creating a Document-Based System for Answering Questions with LangChain, Pinecone, and LLMs like GPT-4 and ChatGPT

This article explores how to build a system that can answer questions based on documents using LangChain and Pinecone, leveraging the state-of-the-art in large language models (LLMs), such as OpenAI GPT-4 and ChatGPT.
LangChain is a robust platform for creating applications powered by language models, while Pinecone acts as an effective vector database for developing high-performance vector search applications. Our use case concentrates on answering questions over specific documents, using only the information within those documents to produce precise and context-aware answers.
By harnessing the power of semantic search with the remarkable abilities of LLMs like GPT, we will show how to create a top-notch Document QnA system that utilizes advanced AI technologies.

Link for langchain semantic search Pinecone: https://github.com/amohini099/Langchain_Semnatic_Search_Pinecone/tree/main 

### 1. Semantic Search + GPT QnA VS fine-tuning GPT which is better and why?
Semantic Search and GPT QNA is better than fine-tuning GPT , before dive into implementing part let's understand the advantages of using semantic search with GPT QnA over fine-tuning GPT.
Using Semantic Search + GPT QnA, first relevant passages from a huge collection of documents are found and then answers based on those passages are generated. This method can offer more precise and current information, using the newest information from various sources. Fine-tuning GPT, however, depends on the knowledge embedded in the model during training, which may get outdated or incomplete over time.
### Context-specific answers:
By basing answers on specific passages from relevant documents, Semantic Search + GPT QnA can produce more context-relevant and accurate answers. However, fine-tuned GPT models may produce answers based on the general knowledge embedded in the model, which may be less accurate or unrelated to the question's context.
### Adaptability:
The Semantic Search part can be quickly updated with new information sources or adjusted to different domains, making it more flexible to specific use cases or industries. On the other hand, fine-tuning GPT needs re-training the model, which can be time-intensive and computationally costly.
## 2. Better handling of ambiguous queries:
By finding the most pertinent passages that match the query, Semantic Search can clarify queries that have multiple meanings. This can result in more precise and pertinent responses than a fine-tuned GPT model, which may have difficulty with ambiguity without adequate context.
## 3. LangChain Modules
LangChain  provides support for several main modules.
•	Models: The different kinds of models and model combinations that LangChain supports.
•	Indexes: Combining your own text data with language models can make them more powerful - this module covers the best ways to do that.
•	Chains: Chains are more than just one LLM call, and are sequences of calls (to an LLM or another utility). LangChain offers a standard interface for chains, many integrations with other tools, and complete chains for common uses.
## 4. Setting up the environment
First, we have to install the needed packages and import the essential libraries.
!pip install --upgrade langchain openai -q
!pip install unstructured -q
!pip install unstructured[local-inference] -q
!pip install detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2 -q
!apt-get install poppler-utils
Importing Necessary Libraries:
import os
import openai 
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
## 5. Loading documents
First, we have to load the documents from a directory using the DirectoryLoader from LangChain. In this example, we assume the documents are stored in a directory named 'data'.
directory = '/content/data'
def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents
documents = load_docs(directory)
len(documents)
## 6. Splitting documents
Now, we have to split the documents into smaller chunks for processing. We will use the RecursiveCharacterTextSplitter from LangChain, which by default attempts to split on the characters ["\n\n", "\n", " ", ""].
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs
docs = split_docs(documents)
print(len(docs))
## 7. Embedding documents with OpenAI
After the documents are split, we need to embed them using OpenAI's language model. First, we have to install the tiktoken library.
embeddings = OpenAIEmbeddings(model_name="ada")
query_result = embeddings.embed_query("Hello world")
len(query_result)
## 8. Vector search with Pinecone
Next, we will use Pinecone to create an index for our documents. First, we have to install the pinecone-client.
!pip install pinecone-client –q
Then, we can initialize Pinecone and create a Pinecone index
pinecone.init(
    api_key="pinecone api key",
    environment="env"
)
index_name = "langchain-demo"
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

Using the Pinecone.from_documents() method, we create a new vector index in Pinecone. This method requires three arguments:
#### 1.	docs: 
A list of documents that have been divided into smaller pieces using the RecursiveCharacterTextSplitter. These smaller pieces will be indexed in Pinecone to make it easier to search and retrieve relevant documents later on.
#### 2.	embeddings: 
An instance of the OpenAIEmbeddings class, which is in charge of converting text data into embeddings (i.e., numerical representations) using OpenAI's language model. These embeddings will be stored in the Pinecone index and used for similarity search.
#### 4.	index_name: 
A string that represents the name of the Pinecone index. This name is used to identify the index in Pinecone's database, and it should be unique to avoid conflicts with other indexes.
Using the OpenAIEmbeddings instance given, the Pinecone.from_documents() method creates embeddings from the input documents and makes a new Pinecone index with the name specified. The index object that results can do similarity searches and get related documents based on user queries.
### 9. Finding similar documents
Next, we can write a function to search for documents that are similar to a given query.
def get_similiar_docs(query, k=2, score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query, k=k)
  else:
    similar_docs = index.similarity_search(query, k=k)
  return similar_docs
### 10. Question answering using LangChain and OpenAI LLM
Following are examples of model name in Lang chain 
model_name = "text-davinci-003"
model_name = "gpt-3.5-turbo"
model_name = "gpt-4"
llm = OpenAI(model_name=model_name)
chain = load_qa_chain(llm, chain_type="stuff")
def get_answer(query):
      similar_docs = get_similiar_docs(query)
      answer = chain.run(input_documents=similar_docs, question=query)
  return answer
### 11. Example queries and answers
Finally, let's test our question answering system with some example queries.
query = "How is India's economy?"
answer = get_answer(query)
print(answer)

query = "How have relations between India and the US improved?"
answer = get_answer(query)
print(answer)

