from langchain_community.document_loaders import DirectoryLoader, TextLoader

class LawDataLoader:
    def load_dir(self, dir_path="data"):
        print(f"Đang quét thư mục: {dir_path}")
        loader = DirectoryLoader(
            dir_path, 
            glob="**/*.md", 
            loader_cls=TextLoader, 
            show_progress=True,
            use_multithreading=True
        )
        documents = loader.load()
        print(f"Đã tìm thấy {len(documents)} file!")
        return documents