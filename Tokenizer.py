import unicodedata

class Tokenizer:
    def __init__(self, raw: str):
        cleaned_text = self.Clean(raw)
        vocab = sorted(set(cleaned_text)) #this gets a list of unique words in the text

        #Get integer->string and string->integer dictionaries for encoding and decoding
        self.stoi = {}
        self.itos = {}

    def Clean(self, raw: str):
        """
        This method just takes the raw text and attempts to clean it up to reduce the 
        vocab size.
        """
        text = unicodedata.normalize("NFKC", raw) #collapse weird unicode characters
        text = text.lower() #just make it all lowercase
        text = text.strip() #remove excess spaces if there are still any
        return text




########
# References
########
# https://chatgpt.com/share/692e5a2b-c190-8005-a2d5-d766e939262f 
# https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize
#       > ChatGPT suggested unicode normalization, very handy