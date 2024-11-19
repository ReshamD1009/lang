from langchain_community.embeddings import HuggingFaceEmbeddings

class EmbeddingGenerator:
    def __init__(self, model_name='BAAI/bge-small-en'):
        self.model = HuggingFaceEmbeddings(model_name=model_name)

    def generate_embedding(self, text):
        return self.model.embed_documents([text])[0] #passing the text as a list (even though it's only one document, it's passed as a list because the model expects a list of documents).
