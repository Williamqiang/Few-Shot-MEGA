import torch
from torch import nn, optim
from .base_model import SentenceRE


class SoftmaxNN(SentenceRE):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id,N,K,Q):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.fc = nn.Linear(self.sentence_encoder.hidden_size, num_class)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        self.N=N
        self.Q=Q
        self.K=K
        self.dot=False
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, item):
        self.eval()
        item = self.sentence_encoder.tokenize(item)
        logits = self.forward(*item)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score
    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)  # S: (B, 1, N, D), Q: (B, Q, 1, D)
    def forward(self,*args):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """

        batch_support,batch_query=zip(*zip(*args))
        support_eme = self.sentence_encoder(*batch_support)
        query_eme = self.sentence_encoder(*batch_query)
        hidden_size=support_eme.size(-1)
        
        # print("1support_shape:",support_eme.shape)
        # print("1query_shape:",query_eme.shape)
        # print("hidden_size:",hidden_size)
        support = self.drop(support_eme)
        query = self.drop(query_eme)
        support = support.view(-1, self.N, self.K, hidden_size) # (B, N, K, D)
        query = query.view(-1, self.Q, hidden_size) # (B, total_Q, D)
        # print("2support:",support.shape)
        # print("2query:",query.shape)
        B=support.size(0)

        # Prototypical Networks
        # Ignore NA policy
        support = torch.mean(support, 2) #(B, N, D)
        logits = self.__batch_dist__(support, query) # (B, total_Q, N)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, self.N+1), 1)

        # logits=torch.cat([logits], 1)
        logits=logits.view(-1,logits.size(-1))
        # print("logits:",logits.shape)
        # print("pred:",pred)
        #logits=logits.squeeze(1)
        return logits, pred

        # rep = self.drop(rep)
        # logits = self.fc(rep)  # (B, N)
        # return logits
