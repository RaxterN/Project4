import unicodedata

class Tokenizer:
    def __init__(self, raw: str):
        cleaned_text = self.clean(raw)
        vocab = sorted(set(cleaned_text)) #this gets a list of unique words in the text

        #Get integer->string and string->integer dictionaries for encoding and decoding
        self.stoi = {word: i for i, word in enumerate(vocab)}
        self.itos = {i: word for word, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def clean(self, raw: str):
        """
        This method just takes the raw text and attempts to clean it up to reduce the 
        vocab size.
        """
        text = unicodedata.normalize("NFKC", raw) #collapse weird unicode characters
        text = text.lower() #just make it all lowercase
        text = text.split()
        return text
    
    def encode(self, text: str):
        """
        Returns the integer id's for any tokens passed in, if they are
        in the string->integer dict. (self.stoi) made from the original vocabulary
        """
        tokens = self.clean(text)
        return [self.stoi[token] for token in tokens if token in self.stoi]

    def decode(self, ids):
        """
        Returns the strings if they exist in from the interger->string
        vocab disctionary (self.itos)
        """
        return " ".join(self.itos[id] for id in ids)




########
# References
########
# https://chatgpt.com/share/692e5a2b-c190-8005-a2d5-d766e939262f 
# https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize
#       > ChatGPT suggested unicode normalization, very handy