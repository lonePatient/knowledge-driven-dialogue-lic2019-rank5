from collections import Counter
from eval import calc_bleu, calc_distinct, calc_f1
import re
import os
import scipy
from utils.pkl_util import to_pkl, load_pkl
import operator

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import string
from collections import defaultdict
import gensim
import numpy as np
import pandas as pd
import config
from utils import dist_utils, ngram_utils, nlp_utils, np_utils
from feature_base import BaseEstimator
from multiprocessing import Pool


def get_goal_knowledge(src):
    goal_knowledge = src.split("[Q=0]")[0]
    goal_knowledge = goal_knowledge.replace("[KG] ", "").replace("[", "")
    goal_knowledge = re.sub("Goal=\d] ","", goal_knowledge)
    return goal_knowledge

def get_conver(src):
    conver = src.split("[Q=0]")[1].replace("[", "")
    conver = re.sub("Q=\d] ", "", conver)
    conver = re.sub("A=\d] ", "", conver).replace("]", "")
    return conver

def get_last_question(src):
    last = src.split("]")[-1]
    return last.strip()


def compute_bleu1(p, s):
    bleu1, bleu2 = calc_bleu([[p.split(), s.split()]])
    return bleu1

def compute_bleu2(p, s):
    bleu1, bleu2 = calc_bleu([[p.split(), s.split()]])
    return bleu2

def compute_f1(p, s):
    f1 = calc_f1([[p.split(), s.split()]])
    return f1

def compute_distinct1(p, s):
    d1, d2 = calc_distinct([[p.split(), s.split()]])
    return d1

def compute_distinct2(p, s):
    d1, d2 = calc_distinct([[p.split(), s.split()]])
    return d2

def is_question_sent(p):
    if "吗" in p or "？" in p:
        return 1
    else:
        return 0

def repeat_word_count(p):
    count = Counter()
    count.update(p.split())
    n = 0
    for k, v in count.items():
        n += v
    return n

def entity_overlap_num(p, gk):
    count = Counter()
    gk = gk.split()
    count.update(gk)
    p = p.split()
    n = 0
    for word in p:
        if count.get(word) is not None:
            n+=1
    return n


def get_goal_knowledge_list(src):
    goal_knowledge = src.split("[Q=0]")[0]
    goal_knowledge = re.sub("=\d] ", "", goal_knowledge)
    goal = goal_knowledge.split("[Goal")[1:]
    entitys = []
    for item in goal:
        if "[KG]" not in item:
            entitys.append(item.strip())
        else:
            entitys.extend([i.strip() for i in item.split("[KG]")])
    return entitys


def get_cooccur_pair(s):
    pairs = []
    s = s.split()
    for i in range(len(s)):
        j = i
        while j < len(s) - 1:
            pairs.append([s[i], s[j]])
            j += 1
    return pairs


def entity_cooccur(pred, src):
    entitys = get_goal_knowledge_list(src)
    n = 0
    for e in entitys:
        pairs = get_cooccur_pair(e)
        for pair in pairs:
            token1, token2 = pair
            if token1 in pred and token2 in pred:
                n += 1
    return n


class VectorSpace:
    ## word based
    def _init_word_bow(self, ngram, vocabulary=None):
        bow = CountVectorizer(min_df=3,
                                max_df=0.75,
                                max_features=None,
                                # norm="l2",
                                strip_accents="unicode",
                                analyzer="word",
                                token_pattern=r"\w{1,}",
                                ngram_range=(1, ngram),
                                vocabulary=vocabulary)
        return bow

    ## word based
    def _init_word_ngram_tfidf(self, ngram, vocabulary=None):
        tfidf = TfidfVectorizer(min_df=3,
                                max_df=0.75,
                                max_features=None,
                                norm="l2",
                                strip_accents="unicode",
                                analyzer="word",
                                token_pattern=r"\w{1,}",
                                ngram_range=(1, ngram),
                                use_idf=1,
                                smooth_idf=1,
                                sublinear_tf=1,
                                # stop_words="english",
                                vocabulary=vocabulary)
        return tfidf

    ## char based
    def _init_char_tfidf(self, include_digit=False):
        chars = list(string.ascii_lowercase)
        if include_digit:
            chars += list(string.digits)
        vocabulary = dict(zip(chars, range(len(chars))))
        tfidf = TfidfVectorizer(strip_accents="unicode",
                                analyzer="char",
                                norm=None,
                                token_pattern=r"\w{1,}",
                                ngram_range=(1, 1),
                                use_idf=0,
                                vocabulary=vocabulary)
        return tfidf

    ## char based ngram
    def _init_char_ngram_tfidf(self, ngram, vocabulary=None):
        tfidf = TfidfVectorizer(min_df=3,
                                max_df=0.75,
                                max_features=None,
                                norm="l2",
                                strip_accents="unicode",
                                analyzer="char",
                                token_pattern=r"\w{1,}",
                                ngram_range=(1, ngram),
                                use_idf=1,
                                smooth_idf=1,
                                sublinear_tf=1,
                                # stop_words="english",
                                vocabulary=vocabulary)
        return tfidf


# LSA
class LSA_Word_Ngram(VectorSpace):
    def __init__(self, obs_corpus, place_holder, ngram=3, svd_dim=100, svd_n_iter=5):
        self.obs_corpus = obs_corpus
        self.ngram = ngram
        self.svd_dim = svd_dim
        self.svd_n_iter = svd_n_iter
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "LSA%d_Word_%s" % (self.svd_dim, self.ngram_str)

    def transform(self):
        tfidf = self._init_word_ngram_tfidf(self.ngram)
        X = tfidf.fit_transform(self.obs_corpus)
        svd = TruncatedSVD(n_components=self.svd_dim,
                           n_iter=self.svd_n_iter, random_state=config.RANDOM_SEED)
        return svd.fit_transform(X)


class LSA_Char_Ngram(VectorSpace):
    def __init__(self, obs_corpus, place_holder, ngram=5, svd_dim=100, svd_n_iter=5):
        self.obs_corpus = obs_corpus
        self.ngram = ngram
        self.svd_dim = svd_dim
        self.svd_n_iter = svd_n_iter
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "LSA%d_Char_%s" % (self.svd_dim, self.ngram_str)

    def transform(self):
        tfidf = self._init_char_ngram_tfidf(self.ngram)
        X = tfidf.fit_transform(self.obs_corpus)
        svd = TruncatedSVD(n_components=self.svd_dim,
                           n_iter=self.svd_n_iter, random_state=config.RANDOM_SEED)
        return svd.fit_transform(X)

# ------------------------ Cooccurrence LSA -------------------------------
# 1st in CrowdFlower
class LSA_Word_Ngram_Cooc(VectorSpace):
    def __init__(self, obs_corpus, target_corpus,
            obs_ngram=1, target_ngram=1, svd_dim=100, svd_n_iter=5):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus
        self.obs_ngram = obs_ngram
        self.target_ngram = target_ngram
        self.svd_dim = svd_dim
        self.svd_n_iter = svd_n_iter
        self.obs_ngram_str = ngram_utils._ngram_str_map[self.obs_ngram]
        self.target_ngram_str = ngram_utils._ngram_str_map[self.target_ngram]

    def __name__(self):
        return "LSA%d_Word_Obs_%s_Target_%s_Cooc"%(
            self.svd_dim, self.obs_ngram_str, self.target_ngram_str)

    def _get_cooc_terms(self, lst1, lst2, join_str):
        out = [""] * len(lst1) * len(lst2)
        cnt =  0
        for item1 in lst1:
            for item2 in lst2:
                out[cnt] = item1 + join_str + item2
                cnt += 1
        res = " ".join(out)
        return res

    def transform(self):
        # ngrams
        obs_ngrams = list(map(lambda x: ngram_utils._ngrams(x.split(" "), self.obs_ngram, "_"), self.obs_corpus))
        target_ngrams = list(map(lambda x: ngram_utils._ngrams(x.split(" "), self.target_ngram, "_"), self.target_corpus))
        # cooccurrence ngrams
        cooc_terms = list(map(lambda lst1,lst2: self._get_cooc_terms(lst1, lst2, "X"), obs_ngrams, target_ngrams))
        ## tfidf
        tfidf = self._init_word_ngram_tfidf(ngram=1)
        X = tfidf.fit_transform(cooc_terms)
        ## svd
        svd = TruncatedSVD(n_components=self.svd_dim,
                n_iter=self.svd_n_iter, random_state=config.RANDOM_SEED)
        return svd.fit_transform(X)


# 2nd in CrowdFlower (preprocessing_mikhail.py)
class LSA_Word_Ngram_Pair(VectorSpace):
    def __init__(self, obs_corpus, target_corpus, ngram=2, svd_dim=100, svd_n_iter=5):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus
        self.ngram = ngram
        self.svd_dim = svd_dim
        self.svd_n_iter = svd_n_iter
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "LSA%d_Word_%s_Pair"%(self.svd_dim, self.ngram_str)

    def transform(self):
        ## tfidf
        tfidf = self._init_word_ngram_tfidf(ngram=self.ngram)
        X_obs = tfidf.fit_transform(self.obs_corpus)
        X_target = tfidf.fit_transform(self.target_corpus)
        X_tfidf = scipy.sparse.hstack([X_obs, X_target]).tocsr()
        ## svd
        svd = TruncatedSVD(n_components=self.svd_dim,
                n_iter=self.svd_n_iter, random_state=config.RANDOM_SEED)
        X_svd = svd.fit_transform(X_tfidf)
        return X_svd


# -------------------------------- TSNE ------------------------------------------
# 2nd in CrowdFlower (preprocessing_mikhail.py)
class TSNE_LSA_Word_Ngram(LSA_Word_Ngram):
    def __init__(self, obs_corpus, place_holder, ngram=3, svd_dim=100, svd_n_iter=5):
        super().__init__(obs_corpus, None, ngram, svd_dim, svd_n_iter)

    def __name__(self):
        return "TSNE_LSA%d_Word_%s" % (self.svd_dim, self.ngram_str)

    def transform(self):
        X_svd = super().transform()
        X_scaled = StandardScaler().fit_transform(X_svd)
        X_tsne = TSNE().fit_transform(X_scaled)
        return X_tsne


class TSNE_LSA_Char_Ngram(LSA_Char_Ngram):
    def __init__(self, obs_corpus, place_holder, ngram=5, svd_dim=100, svd_n_iter=5):
        super().__init__(obs_corpus, None, ngram, svd_dim, svd_n_iter)

    def __name__(self):
        return "TSNE_LSA%d_Char_%s" % (self.svd_dim, self.ngram_str)

    def transform(self):
        X_svd = super().transform()
        X_scaled = StandardScaler().fit_transform(X_svd)
        X_tsne = TSNE().fit_transform(X_scaled)
        return X_tsne


class TSNE_LSA_Word_Ngram_Pair(LSA_Word_Ngram_Pair):
    def __init__(self, obs_corpus, target_corpus, ngram=2, svd_dim=100, svd_n_iter=5):
        super().__init__(obs_corpus, target_corpus, ngram, svd_dim, svd_n_iter)

    def __name__(self):
        return "TSNE_LSA%d_Word_%s_Pair" % (self.svd_dim, self.ngram_str)

    def transform(self):
        X_svd = super().transform()
        X_scaled = StandardScaler().fit_transform(X_svd)
        X_tsne = TSNE().fit_transform(X_scaled)
        return X_tsne


# ------------------------ TFIDF Cosine Similarity -------------------------------
class TFIDF_Word_Ngram_CosineSim(VectorSpace):
    def __init__(self, obs_corpus, target_corpus, ngram=3):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "TFIDF_Word_%s_CosineSim" % self.ngram_str

    def transform(self):
        ## get common vocabulary
        tfidf = self._init_word_ngram_tfidf(self.ngram)
        tfidf.fit(list(self.obs_corpus) + list(self.target_corpus))
        vocabulary = tfidf.vocabulary_
        ## obs tfidf
        tfidf = self._init_word_ngram_tfidf(self.ngram, vocabulary)
        X_obs = tfidf.fit_transform(self.obs_corpus)
        ## targetument tfidf
        tfidf = self._init_word_ngram_tfidf(self.ngram, vocabulary)
        X_target = tfidf.fit_transform(self.target_corpus)
        ## cosine similarity
        sim = list(map(dist_utils._cosine_sim, X_obs, X_target))
        sim = np.asarray(sim).squeeze()
        return sim


class TFIDF_Char_Ngram_CosineSim(VectorSpace):
    def __init__(self, obs_corpus, target_corpus, ngram=5):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "TFIDF_Char_%s_CosineSim" % self.ngram_str

    def transform(self):
        ## get common vocabulary
        tfidf = self._init_char_ngram_tfidf(self.ngram)
        tfidf.fit(list(self.obs_corpus) + list(self.target_corpus))
        vocabulary = tfidf.vocabulary_
        ## obs tfidf
        tfidf = self._init_char_ngram_tfidf(self.ngram, vocabulary)
        X_obs = tfidf.fit_transform(self.obs_corpus)
        ## targetument tfidf
        tfidf = self._init_char_ngram_tfidf(self.ngram, vocabulary)
        X_target = tfidf.fit_transform(self.target_corpus)
        ## cosine similarity
        sim = list(map(dist_utils._cosine_sim, X_obs, X_target))
        sim = np.asarray(sim).squeeze()
        return sim


# ------------------------ LSA Cosine Similarity -------------------------------
class LSA_Word_Ngram_CosineSim(VectorSpace):
    def __init__(self, obs_corpus, target_corpus, ngram=3, svd_dim=100, svd_n_iter=5):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus
        self.ngram = ngram
        self.svd_dim = svd_dim
        self.svd_n_iter = svd_n_iter
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "LSA%d_Word_%s_CosineSim" % (self.svd_dim, self.ngram_str)

    def transform(self):
        ## get common vocabulary
        tfidf = self._init_word_ngram_tfidf(self.ngram)
        tfidf.fit(list(self.obs_corpus) + list(self.target_corpus))
        vocabulary = tfidf.vocabulary_
        ## obs tfidf
        tfidf = self._init_word_ngram_tfidf(self.ngram, vocabulary)
        X_obs = tfidf.fit_transform(self.obs_corpus)
        ## targetument tfidf
        tfidf = self._init_word_ngram_tfidf(self.ngram, vocabulary)
        X_target = tfidf.fit_transform(self.target_corpus)
        ## svd
        svd = TruncatedSVD(n_components=self.svd_dim,
                           n_iter=self.svd_n_iter, random_state=config.RANDOM_SEED)
        svd.fit(scipy.sparse.vstack((X_obs, X_target)))
        X_obs = svd.transform(X_obs)
        X_target = svd.transform(X_target)
        ## cosine similarity
        sim = list(map(dist_utils._cosine_sim, X_obs, X_target))
        sim = np.asarray(sim).squeeze()
        return sim


class LSA_Char_Ngram_CosineSim(VectorSpace):
    def __init__(self, obs_corpus, target_corpus, ngram=5, svd_dim=100, svd_n_iter=5):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus
        self.ngram = ngram
        self.svd_dim = svd_dim
        self.svd_n_iter = svd_n_iter
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "LSA%d_Char_%s_CosineSim" % (self.svd_dim, self.ngram_str)

    def transform(self):
        ## get common vocabulary
        tfidf = self._init_char_ngram_tfidf(self.ngram)
        tfidf.fit(list(self.obs_corpus) + list(self.target_corpus))
        vocabulary = tfidf.vocabulary_
        ## obs tfidf
        tfidf = self._init_char_ngram_tfidf(self.ngram, vocabulary)
        X_obs = tfidf.fit_transform(self.obs_corpus)
        ## targetument tfidf
        tfidf = self._init_char_ngram_tfidf(self.ngram, vocabulary)
        X_target = tfidf.fit_transform(self.target_corpus)
        ## svd
        svd = TruncatedSVD(n_components=self.svd_dim,
                           n_iter=self.svd_n_iter, random_state=config.RANDOM_SEED)
        svd.fit(scipy.sparse.vstack((X_obs, X_target)))
        X_obs = svd.transform(X_obs)
        X_target = svd.transform(X_target)
        ## cosine similarity
        sim = list(map(dist_utils._cosine_sim, X_obs, X_target))
        sim = np.asarray(sim).squeeze()
        return sim


# ------------------- Char Distribution Based features ------------------
# 2nd in CrowdFlower (preprocessing_stanislav.py)
class CharDistribution(VectorSpace):
    def __init__(self, obs_corpus, target_corpus):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus

    def normalize(self, text):
        # pat = re.compile("[a-z0-9]")
        pat = re.compile("[a-z]")
        group = pat.findall(text.lower())
        if group is None:
            res = " "
        else:
            res = "".join(group)
            res += " "
        return res

    def preprocess(self, corpus):
        return [self.normalize(text) for text in corpus]

    def get_distribution(self):
        ## obs tfidf
        tfidf = self._init_char_tfidf()
        X_obs = tfidf.fit_transform(self.preprocess(self.obs_corpus)).todense()
        X_obs = np.asarray(X_obs)
        # apply laplacian smoothing
        s = 1.
        X_obs = (X_obs + s) / (np.sum(X_obs, axis=1)[:, None] + X_obs.shape[1] * s)
        ## targetument tfidf
        tfidf = self._init_char_tfidf()
        X_target = tfidf.fit_transform(self.preprocess(self.target_corpus)).todense()
        X_target = np.asarray(X_target)
        X_target = (X_target + s) / (np.sum(X_target, axis=1)[:, None] + X_target.shape[1] * s)
        return X_obs, X_target


class CharDistribution_Ratio(CharDistribution):
    def __init__(self, obs_corpus, target_corpus, const_A=1., const_B=1.):
        super().__init__(obs_corpus, target_corpus)
        self.const_A = const_A
        self.const_B = const_B

    def __name__(self):
        return "CharDistribution_Ratio"

    def transform(self):
        X_obs, X_target = self.get_distribution()
        ratio = (X_obs + self.const_A) / (X_target + self.const_B)
        return ratio


class CharDistribution_CosineSim(CharDistribution):
    def __init__(self, obs_corpus, target_corpus):
        super().__init__(obs_corpus, target_corpus)

    def __name__(self):
        return "CharDistribution_CosineSim"

    def transform(self):
        X_obs, X_target = self.get_distribution()
        ## cosine similarity
        sim = list(map(dist_utils._cosine_sim, X_obs, X_target))
        sim = np.asarray(sim).squeeze()
        return sim


class CharDistribution_KL(CharDistribution):
    def __init__(self, obs_corpus, target_corpus):
        super().__init__(obs_corpus, target_corpus)

    def __name__(self):
        return "CharDistribution_KL"

    def transform(self):
        X_obs, X_target = self.get_distribution()
        ## KL
        kl = dist_utils._KL(X_obs, X_target)
        return kl


token_pattern = " "  # just split the text into tokens


# ----------------------------- TF ------------------------------------
# StatCooc stands for StatisticalCooccurrence
class StatCoocTF_Ngram(BaseEstimator):
    """Single aggregation features"""

    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="",
                 str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold

    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "StatCoocTF_%s_%s" % (
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["StatCoocTF_%s_%s" % (
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        val_list = []
        for w1 in obs_ngrams:
            s = 0.
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
            val_list.append(s)
        if len(val_list) == 0:
            val_list = [config.MISSING_VALUE_NUMERIC]
        return val_list


# ----------------------------- Normalized TF ------------------------------------
# StatCooc stands for StatisticalCooccurrence
class StatCoocNormTF_Ngram(BaseEstimator):
    """Single aggregation features"""

    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="",
                 str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold

    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "StatCoocNormTF_%s_%s" % (
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["StatCoocNormTF_%s_%s" % (
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        val_list = []
        for w1 in obs_ngrams:
            s = 0.
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
            val_list.append(np_utils._try_divide(s, len(target_ngrams)))
        if len(val_list) == 0:
            val_list = [config.MISSING_VALUE_NUMERIC]
        return val_list


# ------------------------------ TFIDF -----------------------------------
# StatCooc stands for StatisticalCooccurrence
class StatCoocTFIDF_Ngram(BaseEstimator):
    """Single aggregation features"""

    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="",
                 str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold
        self.df_dict = self._get_df_dict()

    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "StatCoocTFIDF_%s_%s" % (
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["StatCoocTFIDF_%s_%s" % (
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name

    def _get_df_dict(self):
        # smoothing
        d = defaultdict(lambda: 1)
        for target in self.target_corpus:
            target_tokens = nlp_utils._tokenize(target, token_pattern)
            target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
            for w in set(target_ngrams):
                d[w] += 1
        return d

    def _get_idf(self, word):
        return np.log((self.N - self.df_dict[word] + 0.5) / (self.df_dict[word] + 0.5))

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        val_list = []
        for w1 in obs_ngrams:
            s = 0.
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
            val_list.append(s * self._get_idf(w1))
        if len(val_list) == 0:
            val_list = [config.MISSING_VALUE_NUMERIC]
        return val_list


# ------------------------------ Normalized TFIDF -----------------------------------
# StatCooc stands for StatisticalCooccurrence
class StatCoocNormTFIDF_Ngram(BaseEstimator):
    """Single aggregation features"""

    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="",
                 str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold
        self.df_dict = self._get_df_dict()

    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "StatCoocNormTFIDF_%s_%s" % (
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["StatCoocNormTFIDF_%s_%s" % (
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name

    def _get_df_dict(self):
        # smoothing
        d = defaultdict(lambda: 1)
        for target in self.target_corpus:
            target_tokens = nlp_utils._tokenize(target, token_pattern)
            target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
            for w in set(target_ngrams):
                d[w] += 1
        return d

    def _get_idf(self, word):
        return np.log((self.N - self.df_dict[word] + 0.5) / (self.df_dict[word] + 0.5))

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        val_list = []
        for w1 in obs_ngrams:
            s = 0.
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
            val_list.append(np_utils._try_divide(s, len(target_ngrams)) * self._get_idf(w1))
        if len(val_list) == 0:
            val_list = [config.MISSING_VALUE_NUMERIC]
        return val_list


# ------------------------ BM25 ---------------------------------------------
# StatCooc stands for StatisticalCooccurrence
class StatCoocBM25_Ngram(BaseEstimator):
    """Single aggregation features"""

    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="",
                 str_match_threshold=config.STR_MATCH_THRESHOLD, k1=config.BM25_K1, b=config.BM25_B):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.k1 = k1
        self.b = b
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold
        self.df_dict = self._get_df_dict()
        self.avg_ngram_doc_len = self._get_avg_ngram_doc_len()

    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "StatCoocBM25_%s_%s" % (
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["StatCoocBM25_%s_%s" % (
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name

    def _get_df_dict(self):
        # smoothing
        d = defaultdict(lambda: 1)
        for target in self.target_corpus:
            target_tokens = nlp_utils._tokenize(target, token_pattern)
            target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
            for w in set(target_ngrams):
                d[w] += 1
        return d

    def _get_idf(self, word):
        return np.log((self.N - self.df_dict[word] + 0.5) / (self.df_dict[word] + 0.5))

    def _get_avg_ngram_doc_len(self):
        lst = []
        for target in self.target_corpus:
            target_tokens = nlp_utils._tokenize(target, token_pattern)
            target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
            lst.append(len(target_ngrams))
        return np.mean(lst)

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        K = self.k1 * (1 - self.b + self.b * np_utils._try_divide(len(target_ngrams), self.avg_ngram_doc_len))
        val_list = []
        for w1 in obs_ngrams:
            s = 0.
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
            bm25 = s * self._get_idf(w1) * np_utils._try_divide(1 + self.k1, s + K)
            val_list.append(bm25)
        if len(val_list) == 0:
            val_list = [config.MISSING_VALUE_NUMERIC]
        return val_list

def max_word_freq(s):
    s = s.split()
    count = Counter()
    count.update(s)
    sorted_count = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_count[0][1]

def mean_word_freq(s):
    s = s.split()
    count = Counter()
    count.update(s)
    n = 0
    for k, v in count.items():
        n +=v
    return n/len(count)


# How many ngrams of obs are in target?
# Obs: [AB, AB, AB, AC, DE, CD]
# Target: [AB, AC, AB, AD, ED]
# ->
# IntersectCount: 4 (i.e., AB, AB, AB, AC)
# IntersectRatio: 4/6
class IntersectCount_Ngram(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="",
                 str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold

    def __name__(self):
        return "IntersectCount_%s" % self.ngram_str

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        s = 0.
        for w1 in obs_ngrams:
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
                    break
        return s


class IntersectRatio_Ngram(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="",
                 str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold

    def __name__(self):
        return "IntersectRatio_%s" % self.ngram_str

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        s = 0.
        for w1 in obs_ngrams:
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
                    break
        return np_utils._try_divide(s, len(obs_ngrams))


# ----------------------------------------------------------------------------
# How many cooccurrence ngrams between obs and target?
# Obs: [AB, AB, AB, AC, DE, CD]
# Target: [AB, AC, AB, AD, ED]
# ->
# CooccurrenceCount: 7 (i.e., AB x 2 + AB x 2 + AB x 2 + AC x 1)
# CooccurrenceRatio: 7/(6 x 5)
class CooccurrenceCount_Ngram(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="",
                 str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold

    def __name__(self):
        return "CooccurrenceCount_%s" % self.ngram_str

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        s = 0.
        for w1 in obs_ngrams:
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
        return s


class CooccurrenceRatio_Ngram(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="",
                 str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold

    def __name__(self):
        return "CooccurrenceRatio_%s" % self.ngram_str

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        s = 0.
        for w1 in obs_ngrams:
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
        return np_utils._try_divide(s, len(obs_ngrams) * len(target_ngrams))


token_pattern = " "  # just split the text into tokens


def _inter_pos_list(obs, target):
    """
        Get the list of positions of obs in target
    """
    pos_list = [0]
    if len(obs) != 0:
        pos_list = [i for i, o in enumerate(obs, start=1) if o in target]
        if len(pos_list) == 0:
            pos_list = [0]
    return pos_list


def _inter_norm_pos_list(obs, target):
    pos_list = _inter_pos_list(obs, target)
    N = len(obs)
    return [np_utils._try_divide(i, N) for i in pos_list]


class IntersectPosition_Ngram(BaseEstimator):
    """Single aggregation features"""

    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "IntersectPosition_%s_%s" % (
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["IntersectPosition_%s_%s" % (
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        pos_list = _inter_pos_list(obs_ngrams, target_ngrams)
        return pos_list


class IntersectNormPosition_Ngram(BaseEstimator):
    """Single aggregation features"""

    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "IntersectNormPosition_%s_%s" % (
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["IntersectNormPosition_%s_%s" % (
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        pos_list = _inter_norm_pos_list(obs_ngrams, target_ngrams)
        return pos_list


class Count_Ngram_BaseEstimator(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, idx, aggregation_mode="",
                 str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.idx = idx
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold

    def _get_match_count(self, obs, target, idx):
        cnt = 0
        if (len(obs) != 0) and (len(target) != 0):
            for word in target:
                if dist_utils._is_str_match(word, obs[idx], self.str_match_threshold):
                    cnt += 1
        return cnt

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        return self._get_match_count(obs_ngrams, target_ngrams, self.idx)


class FirstIntersectCount_Ngram(Count_Ngram_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="",
                 str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, ngram, 0, aggregation_mode, str_match_threshold)

    def __name__(self):
        return "FirstIntersectCount_%s" % self.ngram_str


class LastIntersectCount_Ngram(Count_Ngram_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="",
                 str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, ngram, -1, aggregation_mode, str_match_threshold)

    def __name__(self):
        return "LastIntersectCount_%s" % self.ngram_str


# ------------------------- Ratio -------------------------------------------
class Ratio_Ngram_BaseEstimator(Count_Ngram_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, idx, aggregation_mode="",
                 str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, ngram, idx, aggregation_mode, str_match_threshold)

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        return np_utils._try_divide(self._get_match_count(obs_ngrams, target_ngrams, self.idx), len(target_ngrams))


class FirstIntersectRatio_Ngram(Ratio_Ngram_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="",
                 str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, ngram, 0, aggregation_mode, str_match_threshold)

    def __name__(self):
        return "FirstIntersectRatio_%s" % self.ngram_str


class LastIntersectRatio_Ngram(Ratio_Ngram_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="",
                 str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, ngram, -1, aggregation_mode, str_match_threshold)

    def __name__(self):
        return "LastIntersectRatio_%s" % self.ngram_str


# -------------------- Position ---------------------
class Position_Ngram_BaseEstimator(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, idx, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.idx = idx
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        return _inter_pos_list(target_ngrams, [obs_ngrams[self.idx]])


class FirstIntersectPosition_Ngram(Position_Ngram_BaseEstimator):
    """Single aggregation features"""

    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, ngram, 0, aggregation_mode)

    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "FirstIntersectPosition_%s_%s" % (
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["FirstIntersectPosition_%s_%s" % (
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name


class LastIntersectPosition_Ngram(Position_Ngram_BaseEstimator):
    """Single aggregation features"""

    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, ngram, -1, aggregation_mode)

    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "LastIntersectPosition_%s_%s" % (
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["LastIntersectPosition_%s_%s" % (
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name


# -------------------------- Norm Position ----------------------------------
class NormPosition_Ngram_BaseEstimator(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, idx, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.idx = idx
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        return _inter_norm_pos_list(target_ngrams, [obs_ngrams[self.idx]])


class FirstIntersectNormPosition_Ngram(NormPosition_Ngram_BaseEstimator):
    """Single aggregation features"""

    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, ngram, 0, aggregation_mode)

    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "FirstIntersectNormPosition_%s_%s" % (
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["FirstIntersectNormPosition_%s_%s" % (
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name


class LastIntersectNormPosition_Ngram(NormPosition_Ngram_BaseEstimator):
    """Single aggregation features"""

    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, ngram, -1, aggregation_mode)

    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "LastIntersectNormPosition_%s_%s" % (
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["LastIntersectNormPosition_%s_%s" % (
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name


# ------------------------ Word2Vec Features -------------------------
class Word2Vec_BaseEstimator(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix,
                 aggregation_mode="", aggregation_mode_prev=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode, None, aggregation_mode_prev)
        self.model = word2vec_model
        self.model_prefix = model_prefix
        self.vector_size = word2vec_model.vector_size

    def _get_valid_word_list(self, text):
        return [w for w in text.lower().split(" ") if w in self.model]

    def _get_importance(self, text1, text2):
        len_prev_1 = len(text1.split(" "))
        len_prev_2 = len(text2.split(" "))
        len1 = len(self._get_valid_word_list(text1))
        len2 = len(self._get_valid_word_list(text2))
        imp = np_utils._try_divide(len1 + len2, len_prev_1 + len_prev_2)
        return imp

    def _get_n_similarity(self, text1, text2):
        lst1 = self._get_valid_word_list(text1)
        lst2 = self._get_valid_word_list(text2)
        if len(lst1) > 0 and len(lst2) > 0:
            return self.model.n_similarity(lst1, lst2)
        else:
            return config.MISSING_VALUE_NUMERIC

    def _get_n_similarity_imp(self, text1, text2):
        sim = self._get_n_similarity(text1, text2)
        imp = self._get_importance(text1, text2)
        return sim * imp

    def _get_centroid_vector(self, text):
        lst = self._get_valid_word_list(text)
        centroid = np.zeros(self.vector_size)
        for w in lst:
            centroid += self.model[w]
        if len(lst) > 0:
            centroid /= float(len(lst))
        return centroid

    def _get_centroid_vdiff(self, text1, text2):
        centroid1 = self._get_centroid_vector(text1)
        centroid2 = self._get_centroid_vector(text2)
        return dist_utils._vdiff(centroid1, centroid2)

    def _get_centroid_rmse(self, text1, text2):
        centroid1 = self._get_centroid_vector(text1)
        centroid2 = self._get_centroid_vector(text2)
        return dist_utils._rmse(centroid1, centroid2)

    def _get_centroid_rmse_imp(self, text1, text2):
        rmse = self._get_centroid_rmse(text1, text2)
        imp = self._get_importance(text1, text2)
        return rmse * imp


class Word2Vec_Centroid_Vector(Word2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode)

    def __name__(self):
        return "Word2Vec_%s_D%d_Centroid_Vector" % (self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_centroid_vector(obs)


class Word2Vec_Importance(Word2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode)

    def __name__(self):
        return "Word2Vec_%s_D%d_Importance" % (self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_importance(obs, target)


class Word2Vec_N_Similarity(Word2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode)

    def __name__(self):
        return "Word2Vec_%s_D%d_N_Similarity" % (self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_n_similarity(obs, target)


class Word2Vec_N_Similarity_Imp(Word2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode)

    def __name__(self):
        return "Word2Vec_%s_D%d_N_Similarity_Imp" % (self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_n_similarity_imp(obs, target)


class Word2Vec_Centroid_RMSE(Word2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode)

    def __name__(self):
        return "Word2Vec_%s_D%d_Centroid_RMSE" % (self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_centroid_rmse(obs, target)


class Word2Vec_Centroid_RMSE_IMP(Word2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode)

    def __name__(self):
        return "Word2Vec_%s_D%d_Centroid_RMSE_IMP" % (self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_centroid_rmse_imp(obs, target)


class Word2Vec_Centroid_Vdiff(Word2Vec_BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix, aggregation_mode)

    def __name__(self):
        return "Word2Vec_%s_D%d_Centroid_Vdiff" % (self.model_prefix, self.vector_size)

    def transform_one(self, obs, target, id):
        return self._get_centroid_vdiff(obs, target)


class Word2Vec_CosineSim(Word2Vec_BaseEstimator):
    """Double aggregation features"""

    def __init__(self, obs_corpus, target_corpus, word2vec_model, model_prefix,
                 aggregation_mode="", aggregation_mode_prev=""):
        super().__init__(obs_corpus, target_corpus, word2vec_model, model_prefix,
                         aggregation_mode, aggregation_mode_prev)

    def __name__(self):
        feat_name = []
        for m1 in self.aggregation_mode_prev:
            for m in self.aggregation_mode:
                n = "Word2Vec_%s_D%d_CosineSim_%s_%s" % (
                    self.model_prefix, self.vector_size, string.capwords(m1), string.capwords(m))
                feat_name.append(n)
        return feat_name

    def transform_one(self, obs, target, id):
        val_list = []
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        for obs_token in obs_tokens:
            _val_list = []
            if obs_token in self.model:
                for target_token in target_tokens:
                    if target_token in self.model:
                        sim = dist_utils._cosine_sim(self.model[obs_token], self.model[target_token])
                        _val_list.append(sim)
            if len(_val_list) == 0:
                _val_list = [config.MISSING_VALUE_NUMERIC]
            val_list.append(_val_list)
        if len(val_list) == 0:
            val_list = [[config.MISSING_VALUE_NUMERIC]]
        return val_list

# df features
def run_df_feature(train_raw_data, test_raw_data):
    df_train = pd.read_csv(train_raw_data, sep="\t")
    df_test = pd.read_csv(test_raw_data, sep="\t")
    df = pd.concat((df_train, df_test), ignore_index=True)
    del df_train, df_test
    print(df.head())
    df["pred_word_length"] = df["preds"].apply(lambda s: len(s.split()))
    df["pred_char_length"] = df["preds"].apply(lambda s: len(s.replace(" ", "")))
    df["goal_and_knowledge"] = df["src"].apply(lambda s: get_goal_knowledge(s))
    df["conver"] = df["src"].apply(lambda s: get_conver(s))
    df["last_conver"] = df["src"].apply(lambda s: get_last_question(s))
    df["bleu1_pred_src"] = df.apply(lambda row: compute_bleu1(row["preds"], row["src"]), axis=1)
    df["bleu2_pred_src"] = df.apply(lambda row: compute_bleu2(row["preds"], row["src"]), axis=1)
    df["bleu1_pred_conver"] = df.apply(lambda row: compute_bleu1(row["preds"], row["conver"]), axis=1)
    df["bleu2_pred_conver"] = df.apply(lambda row: compute_bleu2(row["preds"], row["conver"]), axis=1)
    df["bleu1_pred_last_conver"] = df.apply(lambda row: compute_bleu1(row["preds"], row["last_conver"]), axis=1)
    df["bleu2_pred_last_conver"] = df.apply(lambda row: compute_bleu2(row["preds"], row["last_conver"]), axis=1)
    df["pred_is_question_sent"] = df["preds"].apply(lambda s: is_question_sent(s))
    df["last_conver_is_question_sent"] = df["last_conver"].apply(lambda s: is_question_sent(s))
    df["pred_repeat_word_cnt"] = df["preds"].apply(lambda s: repeat_word_count(s))
    df["entity_overlap_num"] = df.apply(lambda row: entity_overlap_num(row["preds"], row["goal_and_knowledge"]), axis=1)
    df["f1_pred_src"] = df.apply(lambda row: compute_f1(row["preds"], row["src"]), axis=1)
    df["f1_pred_conver"] = df.apply(lambda row: compute_f1(row["preds"], row["conver"]), axis=1)
    df["f1_pred_last_conver"] = df.apply(lambda row: compute_f1(row["preds"], row["last_conver"]), axis=1)
    df["distinct1_pred_src"] = df.apply(lambda row: compute_distinct1(row["preds"], row["src"]), axis=1)
    df["distinct2_pred_src"] = df.apply(lambda row: compute_distinct2(row["preds"], row["src"]), axis=1)
    df["distinct1_pred_conver"] = df.apply(lambda row: compute_distinct1(row["preds"], row["conver"]), axis=1)
    df["distinct2_pred_conver"] = df.apply(lambda row: compute_distinct2(row["preds"], row["conver"]), axis=1)
    df["distinct1_pred_last_conver"] = df.apply(lambda row: compute_distinct1(row["preds"], row["last_conver"]), axis=1)
    df["distinct2_pred_last_conver"] = df.apply(lambda row: compute_distinct2(row["preds"], row["last_conver"]), axis=1)
    df["pred_gk_cooccur"] = df.apply(lambda row: entity_cooccur(row["preds"], row["src"]), axis=1)
    df["max_word_freq"] = df["preds"].apply(lambda s: max_word_freq(s))
    df["mean_word_freq"] = df["preds"].apply(lambda s: mean_word_freq(s))
    print("shape of df is:", df.shape)
    return df


# LSA n gram
def run_lsa_ngram(df, field):
    obj_corpus = df[field].values
    n_grams = [1, 2, 3]
    for n_gram in n_grams:
        ext = LSA_Word_Ngram(obj_corpus, None, n_gram, config.SVD_DIM, config.SVD_N_ITER)
        x = ext.transform()
        save_path = "features/feature_lsa_word_%d_gram_%s.pkl"%(n_gram, field)
        to_pkl(x, save_path)

def run_lsa_char_ngram(df, field):
    obj_corpus = df[field].values
    n_grams = [1, 2, 3, 4]

    for n_gram in n_grams:
        ext = LSA_Char_Ngram(obj_corpus, None, n_gram, config.SVD_DIM, config.SVD_N_ITER)
        x = ext.transform()
        save_path = "features/feature_lsa_char_%d_gram_%s.pkl"%(n_gram, field)
        to_pkl(x, save_path)

def run_tsne_lsa(df, field, generator, n_gram):
    obj_corpus = df[field].values
    ext = generator(obj_corpus, None, n_gram, config.SVD_DIM, config.SVD_N_ITER)
    x = ext.transform()
    save_path = "features/feature_%d_gram_%s_%s.pkl" % (n_gram, ext.__name__(), field)
    to_pkl(x, save_path)

# LSA n gram cosinesim
def run_lsa_ngram_cosinesim(obj_field, target_field):
    n_grams = [1, 2, 3]
    obj_corpus = df[obj_field].values
    tgt_corpus = df[target_field].values
    for n_gram in n_grams:
        ext = LSA_Word_Ngram_CosineSim(obj_corpus, tgt_corpus, n_gram)
        x = ext.transform()
        print(x.shape)
        save_path = "features/feature_lsa_cosinesim_word_%d_gram_%s_%s.pkl"%(n_gram, obj_field, target_field)
        to_pkl(x, save_path)

def run_lsa_char_ngram_cosinesim(obj_field, target_field):
    n_grams = [1, 2, 3]
    obj_corpus = df[obj_field].values
    tgt_corpus = df[target_field].values
    for n_gram in n_grams:
        ext = LSA_Char_Ngram_CosineSim(obj_corpus, tgt_corpus, n_gram)
        x = ext.transform()
        print(x.shape)
        save_path = "features/feature_lsa_cosinesim_char_%d_gram_%s_%s.pkl"%(n_gram, obj_field, target_field)
        to_pkl(x, save_path)

# tfidf sim
def run_tfidf_ngram_cosinesim(obj_field, target_field):
    n_grams = [1, 2, 3]
    obj_corpus = df[obj_field].values
    tgt_corpus = df[target_field].values
    for n_gram in n_grams:
        ext = TFIDF_Word_Ngram_CosineSim(obj_corpus, tgt_corpus, n_gram)
        x = ext.transform()
        print(x.shape)
        save_path = "features/feature_tfidf_cosinesim_word_%d_gram_%s_%s.pkl"%(n_gram, obj_field, target_field)
        to_pkl(x, save_path)

def run_tfidf_char_ngram_cosinesim(obj_field, target_field):
    n_grams = [1, 2, 3]
    obj_corpus = df[obj_field].values
    tgt_corpus = df[target_field].values
    for n_gram in n_grams:
        ext = TFIDF_Char_Ngram_CosineSim(obj_corpus, tgt_corpus, n_gram)
        x = ext.transform()
        print(x.shape)
        save_path = "features/feature_tfidf_cosinesim_char_%d_gram_%s_%s.pkl"%(n_gram, obj_field, target_field)
        to_pkl(x, save_path)

# dist sim
def run_char_dist_sim(obj_field, target_field, generator):
    obj_corpus = df[obj_field].values
    tgt_corpus = df[target_field].values
    ext = generator(obj_corpus, tgt_corpus)
    x = ext.transform()
    print(x.shape)
    save_path = "features/feature_%s_%s_%s.pkl"%(ext.__name__(), obj_field, target_field)
    to_pkl(x, save_path)

# tsne lsa ngram
def run_lsa_ngram_cooc(obj_field, target_field, generator):
    obs_ngrams = [1, 2]
    target_ngrams = [1, 2]
    obj_corpus = df[obj_field].values
    tgt_corpus = df[target_field].values
    for obs_ngram in obs_ngrams:
        for target_ngram in target_ngrams:
            ext = generator(obj_corpus, tgt_corpus, obs_ngram=obs_ngram, target_ngram=target_ngram)
            x = ext.transform()
            print(x.shape)
            save_path = "features/feature_%s_%s_%s.pkl"%(ext.__name__(), obj_field, target_field)
            to_pkl(x, save_path)

def run_tfidf(obj_field, target_field, generator, n_gram):
    obj_corpus = df[obj_field].values
    tgt_corpus = df[target_field].values
    ext = generator(obj_corpus, tgt_corpus, ngram=n_gram)
    x = ext.transform()
    print(x.shape)
    save_path = "features/feature_%d_gram_%s_%s_%s.pkl"%(n_gram, ext.__name__(), obj_field, target_field)
    to_pkl(x, save_path)

def run_intersect_count(obj_field, target_field, generator, n_gram):
    obj_corpus = df[obj_field].values
    tgt_corpus = df[target_field].values
    ext = generator(obj_corpus, tgt_corpus, ngram=n_gram)
    x = ext.transform()
    print(x.shape)
    save_path = "features/feature_%d_gram_%s_%s_%s.pkl" % (n_gram, ext.__name__(), obj_field, target_field)
    to_pkl(x, save_path)

def run_intersect_position(obj_field, target_field, generator, n_gram):
    obj_corpus = df[obj_field].values
    tgt_corpus = df[target_field].values
    aggregation_mode = ["mean", "std", "max", "min", "median"]
    ext = generator(obj_corpus, tgt_corpus, ngram=n_gram, aggregation_mode=aggregation_mode)
    x = ext.transform()
    print(x.shape)
    save_path = "features/feature_%d_gram_%s_%s_%s.pkl" % (n_gram, ext.__name__(), obj_field, target_field)
    to_pkl(x, save_path)

def run_count(obj_field, target_field, generator, n_gram):
    obj_corpus = df[obj_field].values
    tgt_corpus = df[target_field].values
    # aggregation_mode = ["mean", "std", "max", "min", "median"]
    ext = generator(obj_corpus, tgt_corpus, ngram=n_gram)
    x = ext.transform()
    print(x.shape)
    save_path = "features/feature_%d_gram_%s_%s_%s.pkl" % (n_gram, ext.__name__(), obj_field, target_field)
    to_pkl(x, save_path)

def run_tsne(obj_field, target_field, generator, n_gram):
    obj_corpus = df[obj_field].values
    tgt_corpus = df[target_field].values
    ext = generator(obj_corpus, tgt_corpus, ngram=n_gram)
    x = ext.transform()
    print(x.shape)
    save_path = "features/feature_%d_gram_%s_%s_%s.pkl" % (n_gram, ext.__name__(), obj_field, target_field)
    to_pkl(x, save_path)

def run_word2vec(obj_field, target_field, generator, model, model_prefix="Google"):
    obj_corpus = df[obj_field].values
    tgt_corpus = df[target_field].values
    ext = generator(obj_corpus, tgt_corpus, model, model_prefix)
    x = ext.transform()
    print(x.shape)
    save_path = "features/feature_%s_%s_%s.pkl" % (ext.__name__(), obj_field, target_field)
    to_pkl(x, save_path)

def run_word2vec_sim(obj_field, target_field, generator, model, model_prefix="Google"):
    aggregation_mode_prev = ["mean", "max", "min", "median"]
    aggregation_mode = ["mean", "std", "max", "min", "median"]
    obj_corpus = df[obj_field].values
    tgt_corpus = df[target_field].values
    ext = generator(obj_corpus, tgt_corpus, model, model_prefix, aggregation_mode, aggregation_mode_prev)
    x = ext.transform()
    print(x.shape)
    save_path = "features/feature_%s_%s_%s.pkl" % (ext.__name__(), obj_field, target_field)
    to_pkl(x, save_path)


def dump_df_feature(df, fields):
    for field in fields:
        data = df[field].values
        save_path = "features/feature_%s.pkl" % (field)
        to_pkl(data, save_path)


def dumps_y(df):
    y = df["score"].values
    save_path = "features/0520/y_27.pkl"
    to_pkl(y, save_path)


def feature_combine(feature_dir):
    features = []
    file_names = os.listdir(feature_dir)
    for file_name in file_names:
        if not file_name.startswith("feature"):
            continue
        feature = load_pkl(os.path.join(feature_dir, file_name))
        if len(feature.shape) == 1:
            feature = feature[np.newaxis, :].transpose()
        features.append(feature)
    print("features", len(features))
    X1 = np.concatenate(features[:20], axis=1)
    X2 = np.concatenate(features[20:50], axis=1)
    X3 = np.concatenate(features[50:70], axis=1)
    X4 = np.concatenate(features[70:90], axis=1)
    X5 = np.concatenate(features[90:110], axis=1)
    X6 = np.concatenate(features[110:130], axis=1)
    X7 = np.concatenate(features[130:], axis=1)
    # X = np.concatenate(features, axis=1)
    print("X1 shape is:", X1.shape)
    print("X2 shape is:", X2.shape)
    to_pkl(X1, "features/0520/X1_27.pkl")
    to_pkl(X2, "features/0520/X2_27.pkl")
    to_pkl(X3, "features/0520/X3_27.pkl")
    to_pkl(X4, "features/0520/X4_27.pkl")
    to_pkl(X5, "features/0520/X5_27.pkl")
    to_pkl(X6, "features/0520/X6_27.pkl")
    to_pkl(X7, "features/0520/X7_27.pkl")
    # to_pkl(X, "features/train/X_27.pkl")



def run_lsa_ngram_cosinesim_p(obj_fields, target_fields):
    for obj_field in obj_fields:
        for target_field in target_fields:
            run_lsa_ngram_cosinesim(obj_field, target_field)


def run_lsa_char_ngram_cosinesim_p(obj_fields, target_fields):
    for obj_field in obj_fields:
        for target_field in target_fields:
            # pool.apply_async(run_lsa_char_ngram_cosinesim, args=(obj_field, target_field))
            run_lsa_char_ngram_cosinesim(obj_field, target_field)

def run_tfidf_ngram_cosinesim_p(obj_fields, target_fields):
    for obj_field in obj_fields:
        for target_field in target_fields:
            # pool.apply_async(run_tfidf_ngram_cosinesim, args=(obj_field, target_field))
            run_tfidf_ngram_cosinesim(obj_field, target_field)

def run_tfidf_char_ngram_cosinesim_p(obj_fields, target_fields):
    for obj_field in obj_fields:
        for target_field in target_fields:
            run_tfidf_char_ngram_cosinesim(obj_field, target_field)


def run_char_dist_sim_p(obj_fields, target_fields):
    generators = [CharDistribution_Ratio, CharDistribution_CosineSim, CharDistribution_KL]
    for obj_field in obj_fields:
        for target_field in target_fields:
            for generator in generators:
                run_char_dist_sim(obj_field, target_field, generator)

def run_lsa_ngram_cooc_p(obj_fields, target_fields):
    generators = [LSA_Word_Ngram_Cooc]
    for obj_field in obj_fields:
        for target_field in target_fields:
            for generator in generators:
                pool.apply_async(run_lsa_ngram_cooc, args=(obj_field, target_field, generator))
                #run_lsa_ngram_cooc(obj_field, target_field, generator)


def run_tfidf_p(obj_fields, target_fields):
    generators = [StatCoocTF_Ngram, StatCoocNormTF_Ngram, StatCoocTFIDF_Ngram, StatCoocNormTFIDF_Ngram,
                  StatCoocBM25_Ngram]
    n_grams = [1, 2, 3]
    for obj_field in obj_fields:
        for target_field in target_fields:
            for generator in generators:
                for n_gram in n_grams:
                    pool.apply_async(run_tfidf, args=(obj_field, target_field, generator, n_gram))
                    # run_tfidf(obj_field, target_field, generator, n_gram)

def run_intersect_count_p(obj_fields, target_fields):
    generators = [
        IntersectCount_Ngram,
        IntersectRatio_Ngram,
        CooccurrenceCount_Ngram,
        CooccurrenceRatio_Ngram,
    ]
    n_grams = [1, 2, 3]
    for obj_field in obj_fields:
        for target_field in target_fields:
            for generator in generators:
                for n_gram in n_grams:
                    run_intersect_count(obj_field, target_field, generator, n_gram)

def run_intersect_position_p(obj_fields, target_fields):
    generators = [
        IntersectPosition_Ngram,
        IntersectNormPosition_Ngram,
    ]
    n_grams = [1, 2, 3]
    for obj_field in obj_fields:
        for target_field in target_fields:
            for generator in generators:
                for n_gram in n_grams:
                    run_intersect_position(obj_field, target_field, generator, n_gram)

def run_count_p(obj_fields, target_fields):
    generators = [
        FirstIntersectCount_Ngram,
        LastIntersectCount_Ngram,
        FirstIntersectRatio_Ngram,
        LastIntersectRatio_Ngram,
    ]
    n_grams = [1, 2, 3]
    for obj_field in obj_fields:
        for target_field in target_fields:
            for generator in generators:
                for n_gram in n_grams:
                    run_intersect_position(obj_field, target_field, generator, n_gram)

def run_tsne_p(obj_fields, target_fields):
    generators = [TSNE_LSA_Word_Ngram_Pair]
    n_grams = [1, 2]
    for obj_field in obj_fields:
        for target_field in target_fields:
            for generator in generators:
                for n_gram in n_grams:
                    pool.apply_async(run_tsne, args=(obj_field, target_field, generator, n_gram))

def run_tsne_lsa_p(df):
    fields = ["preds", "src"]
    n_grams = [1, 2]
    generators = [TSNE_LSA_Word_Ngram, TSNE_LSA_Char_Ngram]
    for field in fields:
        for generator in generators:
            for n_gram in n_grams:
                pool.apply_async(run_tsne_lsa, args=(df, field, generator, n_gram))

def run_word2vec_p(obj_fields, target_fields, word2vec_model_dir):
    model = gensim.models.Word2Vec.load(word2vec_model_dir)
    generators = [
        Word2Vec_Importance,
        Word2Vec_N_Similarity,
        Word2Vec_N_Similarity_Imp,
        Word2Vec_Centroid_RMSE,
        Word2Vec_Centroid_RMSE_IMP,
        # Word2Vec_Centroid_Vdiff,
    ]
    for obj_field in obj_fields:
        for target_field in target_fields:
            for generator in generators:
                pool.apply_async(run_word2vec, args=(obj_field, target_field, generator, model))

def run_word2vec_sim_p(obj_fields, target_fields, word2vec_model_dir):
    model = gensim.models.Word2Vec.load(word2vec_model_dir)
    generators = [
        Word2Vec_CosineSim,
    ]
    for obj_field in obj_fields:
        for target_field in target_fields:
            for generator in generators:
                pool.apply_async(run_word2vec_sim, args=(obj_field, target_field, generator, model))


if __name__ == '__main__':
    # df = run_df_feature(config.train_raw_data, config.test_raw_data)
    # to_pkl(df, "features/train/df_27_raw.pkl")
    df = load_pkl("features/train/df_27_raw.pkl")
    # obj_fields = ["preds"]
    # target_fields = ["src", "goal_and_knowledge", "conver", "last_conver"]
    # pool = Pool()
    # pool.apply_async(run_lsa_ngram, args=(df, "preds"))
    # pool.apply_async(run_lsa_ngram, args=(df, "src"))
    # pool.apply_async(run_lsa_char_ngram, args=(df, "preds"))
    # pool.apply_async(run_lsa_char_ngram, args=(df, "src"))
    # pool.apply_async(run_lsa_ngram_cosinesim_p, args=(obj_fields, target_fields))
    # pool.apply_async(run_lsa_char_ngram_cosinesim_p, args=(obj_fields, target_fields))
    # pool.apply_async(run_tfidf_ngram_cosinesim_p, args=(obj_fields, target_fields))
    # pool.apply_async(run_tfidf_char_ngram_cosinesim_p, args=(obj_fields, target_fields))
    # pool.apply_async(run_char_dist_sim_p, args=(obj_fields, target_fields))
    # run_lsa_ngram_cooc_p(obj_fields, target_fields)
    # run_tfidf_p(obj_fields, target_fields)
    # pool.apply_async(run_intersect_count_p, args=(obj_fields, target_fields))         --
    # pool.apply_async(run_intersect_position_p, args=(obj_fields, target_fields))
    # pool.apply_async(run_count_p, args=(obj_fields, target_fields))                   --
    # run_tsne_p(obj_fields, target_fields)
    # run_tsne_lsa_p(df)
    # run_word2vec_p(obj_fields, target_fields, word2vec_model_dir="EDA/Google-word2vec-100d.model")
    # run_word2vec_sim_p(obj_fields, target_fields, word2vec_model_dir="EDA/Google-word2vec-100d.model")
    # pool.close()
    # pool.join()
    # print("all features created...")

    fields = ["beam_score", "model_weight", "max_word_freq", "mean_word_freq",
              'pred_word_length', 'pred_char_length', 'bleu1_pred_src',
              'bleu2_pred_src', 'bleu1_pred_conver', 'bleu2_pred_conver',
              'bleu1_pred_last_conver', 'bleu2_pred_last_conver',
              'pred_is_question_sent', 'entity_overlap_num', 'f1_pred_src',
              'f1_pred_conver', 'f1_pred_last_conver', 'distinct1_pred_src',
              'distinct2_pred_src', 'distinct1_pred_conver', 'distinct2_pred_conver',
              'distinct1_pred_last_conver', 'distinct2_pred_last_conver',
              'pred_gk_cooccur', 'last_conver_is_question_sent',
              'pred_repeat_word_cnt']

    dump_df_feature(df, fields)
    #
    if config.is_training:
        features = feature_combine("./features/")
        dumps_y(df)
    else:
        features = feature_combine("./features/")
    # if config.is_training:
    #     #     score = df["score"].values
    #     #     to_pkl(score, "features/train/y_0516w.pkl")
    #     # feature = df.as_matrix(columns=fields)
    #     # print(feature.shape)
    #     # to_pkl(feature, "features/train/X_0516.pkl")