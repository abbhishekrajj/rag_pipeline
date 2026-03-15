
import os
from langchain_community.document_loaders import PyMuPDFLoader
#from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import numpy
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_core.documents import Document
from pathlib import Path



class DocumentProcess:
    def __init__(self, data_dir="Data"):
        self.data_dir = Path(data_dir)
        self.all_documents = []

    ### Read all the Pdf inside the directory
    def process_all_pdfs(self):
        print("Current working directory:", os.getcwd())
        print("Data dir exists:", self.data_dir.exists())
        print("All files in Data:", list(self.data_dir.glob("*")))
        # Find all the pdf recursevely
        print("DATA DIR:", self.data_dir)
        pdf_files = list(self.data_dir.glob("**/*.pdf"))
        print("PDF FILES FOUND:", pdf_files)
            
        print(f"found {len(pdf_files)} Pdf file to process")

        for pdf_file in pdf_files:
            print(f"\nProcessing: {pdf_file.name}")
            try:
                # 'hi_res' strategy tables aur headings ko behtar pehchanti hai
                # 'mode="elements"' har paragraph/table ko alag document banata hai
                loader = PyMuPDFLoader(str(pdf_file))
                documents = loader.load()
                documnets =loader.load()
                #add source info to meta data
                for doc in documnets:
                    doc.metadata['source_file'] = pdf_file.name
                    doc.metadata['file_path'] = str(pdf_file)
                    doc.metadata['file_type'] = 'pdf'
                    doc.metadata['page_number'] = doc.metadata.get("page_number", None)

                self.all_documents.extend(documnets)
                print(f"  ✓ Loaded {len(documnets)} pages")

            except Exception as e:
                print(f" Error: {e}")
        print(f"\nTotal documents loaded: {len(self.all_documents)}")
        return self.all_documents

    ### Read all the CSV inside the directory



    def process_all_csv(self):
        """process all the csv files in a directory"""


        # Find all the pdf recursevely
        csv_files = list(self.data_dir.glob("**/*.csv"))
            
        print(f"found {len(csv_files)} csv file to process")

        for csv_file in csv_files:
            print(f"\nProcessing: {csv_file.name}")
            try:
                loader = CSVLoader(
                    str(csv_file),
                    encoding="utf-8",         # file encoding
                    csv_args={                # optional csv settings
                        "delimiter": ",",
                        "quotechar": '"'},
                    source_column="package_name"
                )
                documnets =loader.load()
                #add source info to meta data
                for i,doc in enumerate(documnets):
                    doc.metadata['source_file'] = csv_file.name
                    doc.metadata['file_path'] = str(csv_file)
                    doc.metadata['file_type'] = 'csv'
                    doc.metadata['page_number'] = i + 1

                self.all_documents.extend(documnets)
                
                print(f"  ✓ Loaded {len(documnets)} pages")
            except Exception as e:
                print(f" Error: {e}")
        print(f"\nTotal documents loaded: {len(self.all_documents)}")
        return self.all_documents
    
    base_url = "https://redcliffelabs.com/"

    def get_internal_links(self,Base_url):
        response = requests.get(Base_url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        links = set()
        for a in soup.find_all("a", href=True):
            full_url = urljoin(Base_url, a["href"])
            parsed = urlparse(full_url)

            # sirf same domain ke links
            if "redcliffelabs.com" in parsed.netloc:
                # fragment hata do
                clean_url = full_url.split("#")[0]
                links.add(clean_url)

        return list(links)

    def extract_page_data(self,url):
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.title.string.strip() if soup.title else ""
        
        meta_desc_tag = soup.find("meta", attrs={"name": "description"})
        meta_description = meta_desc_tag["content"].strip() if meta_desc_tag and meta_desc_tag.get("content") else ""

        h1_tag = soup.find("h1")
        h1 = h1_tag.get_text(strip=True) if h1_tag else ""

        # page text
        text = soup.get_text(separator=" ", strip=True)

        metadata = {
            "source": url,
            "title": title,
            "meta_description": meta_description,
            "h1": h1
        }

        return Document(page_content=text, metadata=metadata)
    
    def process_website(self, base_url):

        links = self.get_internal_links(base_url)

        for link in links:

            try:
                doc = self.extract_page_data(link)
                self.all_documents.append(doc)

            except Exception as e:
                print("Error:", link)

        print("Total after WEB:", len(self.all_documents))

    def load_all_data(self):

        self.process_all_pdfs()
        self.process_all_csv()
        self.process_website("https://redcliffelabs.com")

        return self.all_documents

if __name__ == "__main__":

    # 1 load docs
    loader = DocumentProcess()
    documents = loader.load_all_data()

    print("Total documents:", len(documents))

    for doc in documents[:3]:
        print("------")
        print(doc.page_content[:300])

