from ragatouille import RAGPretrainedModel
import os

def get_docs(folderpath):
    """
    Get content of all markdown files in a specified folder, excluding 'template.md'.
    """
    # List all files in the specified folder
    files = os.listdir(folderpath)
    # Filter for files that end with .md
    md_files = [f for f in files if f.endswith('.md')]
    # Sort the list of markdown files
    md_files.sort()
    
    # remove 'template.md' from the list if it exists
    if 'template.md' in md_files:
        md_files.remove('template.md')
        
    # Read the content of each markdown file and store to list
    docs = []
    for md_file in md_files:
        with open(os.path.join(folderpath, md_file), 'r') as file:
            docs.append(file.read())

    return md_files, docs

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
document_ids, docs = get_docs("../autometrics/metrics/documentation/")
index_path = RAG.index(
    index_name="all_metrics",
    collection=docs,
    document_ids=document_ids,
)