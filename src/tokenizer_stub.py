from typing import List, Dict

class StubTokenizer:
    def __init__(self):
        # mapping word -> ID
        self.word2id: Dict[str, int] = {}
        # reverse mapping ID -> word
        self.id2word: Dict[int, str] = {}
        self.next_id = 1
        
        
    def encode(self, text:str) -> List[int]:
        tokens = text.strip().split()
        ids = []
        for tok in tokens:
            if tok not in self.word2id:
                self.word2id[tok] = self.next_id
                self.id2word[self.next_id] = tok
                self.next_id += 1
            ids.append(self.word2id[tok])
        return ids
    
    
    def decode(self, ids: List[int]) -> str:
        words = [self.id2word.get(i, "<UNK>") for i in ids]
        return " ".join(words)
    