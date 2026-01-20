import numpy as np
import pandas as pd
import re
import joblib
import math
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

import warnings
warnings.filterwarnings('ignore')


class TextPreprocessor:
    def __init__(self):
        self.contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not", "'re": " are",
            "'s": " is", "'d": " would", "'ll": " will", "'ve": " have", "'m": " am"
        }

    def preprocess(self, text: str) -> str:
        text = text.lower()

        # expand contractions
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)

        # normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # keep alphanumeric and basic punctuation
        text = re.sub(r'[^a-z0-9\s\.\,\!\?\-\']', ' ', text)

        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        preprocessed = self.preprocess(text)
        tokens = re.findall(r'\b[a-z]+\b', preprocessed)
        return tokens


class WordEmbeddingFeatures(BaseEstimator, TransformerMixin):
    # simple word vectors based on co-occurrence patterns
    def __init__(self, vector_size: int = 100):
        self.vector_size = vector_size
        self.word_vectors = {}
        self.preprocessor = TextPreprocessor()

    def fit(self, texts, y=None):
        # build vocabulary and simple co-occurrence vectors
        all_tokens = []
        for text in texts:
            tokens = self.preprocessor.tokenize(str(text))
            all_tokens.extend(tokens)

        word_freq = Counter(all_tokens)
        vocab = [w for w, c in word_freq.most_common(5000) if c >= 2]

        # create random but consistent vectors for each word
        np.random.seed(42)
        for i, word in enumerate(vocab):
            self.word_vectors[word] = np.random.randn(self.vector_size) * 0.1
            # add some structure based on word characteristics
            self.word_vectors[word][0] = len(word) / 10
            self.word_vectors[word][1] = word_freq[word] / max(word_freq.values())

        return self

    def transform(self, texts):
        features = []
        for text in texts:
            tokens = self.preprocessor.tokenize(str(text))
            if not tokens:
                features.append(np.zeros(self.vector_size * 4))
                continue

            vectors = [self.word_vectors.get(t, np.zeros(self.vector_size)) for t in tokens]
            vectors = np.array(vectors)

            # aggregate: mean, max, min, std
            mean_vec = np.mean(vectors, axis=0)
            max_vec = np.max(vectors, axis=0)
            min_vec = np.min(vectors, axis=0)
            std_vec = np.std(vectors, axis=0)

            combined = np.concatenate([mean_vec, max_vec, min_vec, std_vec])
            features.append(combined)

        return np.array(features)


class NGramFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.word_unigram = TfidfVectorizer(
            ngram_range=(1, 1),
            max_features=8000,
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
            strip_accents='unicode',
            lowercase=True,
            token_pattern=r'\b[a-z]{2,}\b'
        )
        self.word_bigram = TfidfVectorizer(
            ngram_range=(2, 2),
            max_features=6000,
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            lowercase=True
        )
        self.word_trigram = TfidfVectorizer(
            ngram_range=(3, 3),
            max_features=4000,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            lowercase=True
        )
        self.char_ngram = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            max_features=5000,
            min_df=2,
            max_df=0.9
        )
        self.fitted = False

    def fit(self, texts, y=None):
        texts = [str(t) for t in texts]
        self.word_unigram.fit(texts)
        self.word_bigram.fit(texts)
        self.word_trigram.fit(texts)
        self.char_ngram.fit(texts)
        self.fitted = True
        return self

    def transform(self, texts):
        texts = [str(t) for t in texts]
        unigram = self.word_unigram.transform(texts)
        bigram = self.word_bigram.transform(texts)
        trigram = self.word_trigram.transform(texts)
        char = self.char_ngram.transform(texts)
        return sp.hstack([unigram, bigram, trigram, char])


class SemanticDensityFeatures(BaseEstimator, TransformerMixin):
    # measures semantic patterns using word sets not substrings
    def __init__(self):
        self.preprocessor = TextPreprocessor()

        # violence and aggression vocabulary
        self.violence_words = {
            'kill', 'killed', 'killing', 'kills', 'murder', 'murdered', 'murdering',
            'attack', 'attacked', 'attacking', 'attacks', 'destroy', 'destroyed',
            'destroying', 'destruction', 'death', 'dead', 'die', 'dying', 'dies',
            'bomb', 'bombed', 'bombing', 'bombs', 'shoot', 'shot', 'shooting',
            'stab', 'stabbed', 'stabbing', 'burn', 'burned', 'burning', 'hang',
            'hanged', 'hanging', 'execute', 'executed', 'execution', 'slaughter',
            'slaughtered', 'massacre', 'massacred', 'assassinate', 'assassinated',
            'assassination', 'eliminate', 'eliminated', 'elimination', 'annihilate',
            'annihilated', 'annihilation', 'exterminate', 'exterminated', 'extermination',
            'eradicate', 'eradicated', 'eradication', 'war', 'warfare', 'wars',
            'fight', 'fighting', 'fought', 'combat', 'battle', 'battles', 'weapon',
            'weapons', 'weaponized', 'gun', 'guns', 'gunned', 'knife', 'knives',
            'explosive', 'explosives', 'explode', 'exploded', 'explosion', 'detonate',
            'detonated', 'detonation', 'blast', 'blasted', 'blasting', 'bloodshed',
            'bloody', 'blood', 'bleed', 'bleeding', 'wound', 'wounded', 'wounding',
            'maim', 'maimed', 'maiming', 'mutilate', 'mutilated', 'mutilation',
            'torture', 'tortured', 'torturing', 'behead', 'beheaded', 'beheading',
            'decapitate', 'decapitated', 'dismember', 'dismembered', 'strangle',
            'strangled', 'strangling', 'suffocate', 'suffocated', 'drown', 'drowned',
            'poison', 'poisoned', 'poisoning', 'ambush', 'ambushed', 'raid', 'raided',
            'strike', 'strikes', 'struck', 'assault', 'assaulted', 'assaulting',
            'rampage', 'carnage', 'havoc', 'devastate', 'devastated', 'devastation',
            'obliterate', 'obliterated', 'obliteration', 'demolish', 'demolished',
            'raze', 'razed', 'razing', 'purge', 'purged', 'purging', 'cleanse',
            'cleansed', 'cleansing', 'genocide', 'genocidal', 'holocaust', 'pogrom'
        }

        # fear, threat, and urgency vocabulary
        self.fear_words = {
            'threat', 'threats', 'threaten', 'threatened', 'threatening', 'danger',
            'dangerous', 'dangerously', 'dangers', 'terror', 'terrorist', 'terrorists',
            'terrorism', 'terrorize', 'terrorized', 'fear', 'feared', 'fearing',
            'fearful', 'fears', 'afraid', 'scared', 'scary', 'scarier', 'scariest',
            'panic', 'panicked', 'panicking', 'panics', 'emergency', 'emergencies',
            'crisis', 'crises', 'disaster', 'disasters', 'disastrous', 'catastrophe',
            'catastrophes', 'catastrophic', 'doom', 'doomed', 'doomsday', 'apocalypse',
            'apocalyptic', 'armageddon', 'invasion', 'invasions', 'invade', 'invaded',
            'invading', 'invaders', 'infiltrate', 'infiltrated', 'infiltration',
            'urgent', 'urgently', 'urgency', 'warning', 'warnings', 'warn', 'warned',
            'beware', 'alert', 'alerted', 'alarming', 'alarm', 'alarmed', 'peril',
            'perilous', 'menace', 'menacing', 'ominous', 'sinister', 'looming',
            'imminent', 'impending', 'approaching', 'coming', 'inevitable', 'inescapable',
            'unstoppable', 'relentless', 'ruthless', 'merciless', 'brutal', 'brutality',
            'savage', 'savagery', 'barbaric', 'barbarism', 'vicious', 'viciously',
            'ferocious', 'deadly', 'lethal', 'fatal', 'mortal', 'demise', 'extinction',
            'extinguish', 'annihilation', 'oblivion', 'ruin', 'ruined', 'ruinous',
            'collapse', 'collapsed', 'collapsing', 'downfall', 'destruction'
        }

        # hate, dehumanization, and contempt vocabulary
        self.hate_words = {
            'hate', 'hated', 'hates', 'hating', 'hatred', 'hateful', 'despise',
            'despised', 'despising', 'loathe', 'loathed', 'loathing', 'loathsome',
            'detest', 'detested', 'detesting', 'detestable', 'abhor', 'abhorred',
            'abhorrent', 'scum', 'scumbag', 'filth', 'filthy', 'vermin', 'cockroach',
            'cockroaches', 'rat', 'rats', 'snake', 'snakes', 'parasite', 'parasites',
            'parasitic', 'pest', 'pests', 'infestation', 'infest', 'infested',
            'disease', 'diseased', 'diseases', 'cancer', 'cancerous', 'plague',
            'plagues', 'plagued', 'pestilence', 'contagion', 'virus', 'viral',
            'infection', 'infected', 'infectious', 'evil', 'evils', 'wicked',
            'wickedness', 'vile', 'vileness', 'disgusting', 'disgust', 'disgusted',
            'repulsive', 'repugnant', 'revolting', 'abomination', 'abominable',
            'inferior', 'inferiority', 'subhuman', 'untermensch', 'animal', 'animals',
            'beast', 'beasts', 'beastly', 'savage', 'savages', 'barbarian', 'barbarians',
            'primitive', 'primitives', 'uncivilized', 'degenerate', 'degenerates',
            'degeneration', 'degeneracy', 'deviant', 'deviants', 'pervert', 'perverts',
            'perverted', 'perversion', 'scourge', 'blight', 'blighted', 'taint',
            'tainted', 'corrupt', 'corrupted', 'corruption', 'rotten', 'rotting',
            'putrid', 'foul', 'venom', 'venomous', 'toxic', 'poison', 'poisonous',
            'monster', 'monsters', 'monstrous', 'monstrosity', 'demon', 'demons',
            'demonic', 'devil', 'devils', 'devilish', 'satanic', 'satan', 'lucifer',
            'unholy', 'godless', 'heathen', 'heathens', 'infidel', 'infidels',
            'heretic', 'heretics', 'blasphemer', 'blasphemers', 'apostate', 'apostates'
        }

        # us vs them, tribalism, and othering vocabulary
        self.us_them_words = {
            'they', 'them', 'their', 'theirs', 'themselves', 'those', 'these',
            'enemy', 'enemies', 'foe', 'foes', 'adversary', 'adversaries', 'opponent',
            'opponents', 'opposition', 'rival', 'rivals', 'outsider', 'outsiders',
            'foreigner', 'foreigners', 'foreign', 'alien', 'aliens', 'stranger',
            'strangers', 'invader', 'invaders', 'intruder', 'intruders', 'infiltrator',
            'infiltrators', 'other', 'others', 'otherness', 'people', 'peoples',
            'we', 'us', 'our', 'ours', 'ourselves', 'folk', 'folks', 'kind', 'kinds',
            'race', 'races', 'racial', 'tribe', 'tribes', 'tribal', 'tribalism',
            'clan', 'clans', 'blood', 'bloodline', 'bloodlines', 'lineage', 'heritage',
            'ancestry', 'ancestors', 'ancestral', 'native', 'natives', 'indigenous',
            'homeland', 'fatherland', 'motherland', 'nation', 'nations', 'national',
            'nationalist', 'nationalism', 'patriot', 'patriots', 'patriotic', 'patriotism',
            'brother', 'brothers', 'brotherhood', 'sister', 'sisters', 'sisterhood',
            'comrade', 'comrades', 'ally', 'allies', 'allied', 'alliance', 'coalition',
            'faction', 'factions', 'sect', 'sects', 'group', 'groups', 'movement',
            'movements', 'cause', 'causes', 'side', 'sides', 'camp', 'camps',
            'traitor', 'traitors', 'traitorous', 'treachery', 'treasonous', 'treason',
            'betray', 'betrayed', 'betrayal', 'betrayer', 'turncoat', 'sellout',
            'collaborator', 'collaborators', 'sympathizer', 'sympathizers', 'enabler',
            'puppet', 'puppets', 'pawn', 'pawns', 'stooge', 'stooges', 'shill', 'shills'
        }

        # certainty, absolutism, and dogmatic vocabulary
        self.certainty_words = {
            'always', 'never', 'all', 'none', 'every', 'everything', 'everyone',
            'everybody', 'everywhere', 'nothing', 'nobody', 'nowhere', 'any', 'anyone',
            'anything', 'anywhere', 'must', 'shall', 'will', 'need', 'require',
            'required', 'necessary', 'necessity', 'essential', 'absolutely', 'absolute',
            'definite', 'definitely', 'certain', 'certainly', 'certainty', 'undoubtedly',
            'undoubted', 'doubtless', 'unquestionable', 'unquestionably', 'indisputable',
            'indisputably', 'undeniable', 'undeniably', 'irrefutable', 'irrefutably',
            'obvious', 'obviously', 'clear', 'clearly', 'evident', 'evidently', 'plain',
            'plainly', 'manifest', 'manifestly', 'apparent', 'apparently', 'patent',
            'patently', 'total', 'totally', 'complete', 'completely', 'entire', 'entirely',
            'whole', 'wholly', 'full', 'fully', 'utter', 'utterly', 'pure', 'purely',
            'perfect', 'perfectly', 'without', 'doubt', 'proven', 'proved', 'proof',
            'fact', 'facts', 'factual', 'factually', 'true', 'truly', 'truth', 'truths',
            'real', 'really', 'reality', 'actual', 'actually', 'genuine', 'genuinely',
            'authentic', 'authentically', 'inevitable', 'inevitably', 'inexorable',
            'inexorably', 'incontrovertible', 'incontrovertibly', 'conclusive',
            'conclusively', 'decisive', 'decisively', 'categorical', 'categorically',
            'unconditional', 'unconditionally', 'unequivocal', 'unequivocally'
        }

        # call to action and mobilization vocabulary
        self.action_words = {
            'rise', 'rises', 'rising', 'risen', 'arose', 'arise', 'arises', 'arising',
            'stand', 'stands', 'standing', 'stood', 'fight', 'fights', 'fighting',
            'fought', 'resist', 'resists', 'resisting', 'resisted', 'resistance',
            'rebel', 'rebels', 'rebelling', 'rebelled', 'rebellion', 'revolt', 'revolts',
            'revolting', 'revolted', 'revolution', 'revolutionary', 'revolutionaries',
            'join', 'joins', 'joining', 'joined', 'unite', 'unites', 'uniting', 'united',
            'unity', 'unify', 'unifies', 'unifying', 'unified', 'organize', 'organizes',
            'organizing', 'organized', 'organization', 'mobilize', 'mobilizes',
            'mobilizing', 'mobilized', 'mobilization', 'rally', 'rallies', 'rallying',
            'rallied', 'assemble', 'assembles', 'assembling', 'assembled', 'assembly',
            'gather', 'gathers', 'gathering', 'gathered', 'march', 'marches', 'marching',
            'marched', 'prepare', 'prepares', 'preparing', 'prepared', 'preparation',
            'ready', 'readiness', 'act', 'acts', 'acting', 'acted', 'action', 'actions',
            'activate', 'activates', 'activating', 'activated', 'activation', 'now',
            'today', 'tonight', 'immediately', 'instant', 'instantly', 'instantaneous',
            'time', 'moment', 'hour', 'together', 'collectively', 'arm', 'arms', 'arming',
            'armed', 'rearm', 'rearming', 'rearmed', 'weapon', 'weaponize', 'weaponized',
            'defend', 'defends', 'defending', 'defended', 'defense', 'defensive',
            'protect', 'protects', 'protecting', 'protected', 'protection', 'guard',
            'guards', 'guarding', 'guarded', 'save', 'saves', 'saving', 'saved',
            'rescue', 'rescues', 'rescuing', 'rescued', 'liberate', 'liberates',
            'liberating', 'liberated', 'liberation', 'free', 'frees', 'freeing', 'freed',
            'stop', 'stops', 'stopping', 'stopped', 'halt', 'halts', 'halting', 'halted',
            'end', 'ends', 'ending', 'ended', 'finish', 'finishes', 'finishing', 'finished',
            'begin', 'begins', 'beginning', 'began', 'begun', 'start', 'starts', 'starting',
            'started', 'launch', 'launches', 'launching', 'launched', 'initiate',
            'initiates', 'initiating', 'initiated', 'commence', 'commences', 'commencing',
            'awake', 'awaken', 'awakens', 'awakening', 'awakened', 'wake', 'woke', 'woken'
        }

        # democratic values and institutions vocabulary
        self.democratic_words = {
            'democracy', 'democracies', 'democratic', 'democratically', 'democrat',
            'democrats', 'vote', 'votes', 'voting', 'voted', 'voter', 'voters',
            'election', 'elections', 'elect', 'elects', 'electing', 'elected',
            'electoral', 'electorate', 'ballot', 'ballots', 'poll', 'polls', 'polling',
            'freedom', 'freedoms', 'free', 'freely', 'liberty', 'liberties', 'liberate',
            'liberation', 'rights', 'right', 'rightful', 'rightfully', 'entitle',
            'entitled', 'entitlement', 'constitution', 'constitutions', 'constitutional',
            'constitutionally', 'unconstitutional', 'amendment', 'amendments',
            'law', 'laws', 'lawful', 'lawfully', 'legal', 'legally', 'legitimate',
            'legitimately', 'legitimacy', 'justice', 'justices', 'just', 'justly',
            'unjust', 'injustice', 'fair', 'fairly', 'fairness', 'unfair', 'equal',
            'equally', 'equality', 'unequal', 'inequality', 'equitable', 'equity',
            'peace', 'peaceful', 'peacefully', 'peacemaker', 'peacemaking', 'pacifist',
            'pacifism', 'nonviolent', 'nonviolence', 'citizen', 'citizens', 'citizenship',
            'civic', 'civics', 'civil', 'civility', 'civilian', 'civilians',
            'republic', 'republics', 'republican', 'representative', 'representatives',
            'representation', 'congress', 'congressional', 'senate', 'senator', 'senators',
            'parliament', 'parliamentary', 'legislature', 'legislatures', 'legislative',
            'legislator', 'legislators', 'govern', 'governs', 'governing', 'governed',
            'government', 'governments', 'governmental', 'governance', 'transparent',
            'transparency', 'accountable', 'accountability', 'oversight', 'checks',
            'balance', 'balances', 'separation', 'powers', 'branch', 'branches',
            'judicial', 'judiciary', 'court', 'courts', 'judge', 'judges', 'trial',
            'trials', 'jury', 'juries', 'due', 'process', 'hearing', 'hearings',
            'testimony', 'testify', 'witness', 'witnesses', 'evidence', 'proof'
        }

        # constructive and cooperative vocabulary
        self.constructive_words = {
            'build', 'builds', 'building', 'built', 'builder', 'builders', 'rebuild',
            'rebuilds', 'rebuilding', 'rebuilt', 'construct', 'constructs', 'constructing',
            'constructed', 'construction', 'constructive', 'constructively', 'create',
            'creates', 'creating', 'created', 'creation', 'creative', 'creatively',
            'creativity', 'creator', 'creators', 'innovate', 'innovates', 'innovating',
            'innovated', 'innovation', 'innovations', 'innovative', 'improve', 'improves',
            'improving', 'improved', 'improvement', 'improvements', 'enhance', 'enhances',
            'enhancing', 'enhanced', 'enhancement', 'develop', 'develops', 'developing',
            'developed', 'development', 'developmental', 'grow', 'grows', 'growing',
            'grew', 'grown', 'growth', 'progress', 'progresses', 'progressing',
            'progressed', 'progressive', 'progression', 'advance', 'advances', 'advancing',
            'advanced', 'advancement', 'help', 'helps', 'helping', 'helped', 'helper',
            'helpers', 'helpful', 'helpfully', 'support', 'supports', 'supporting',
            'supported', 'supporter', 'supporters', 'supportive', 'assist', 'assists',
            'assisting', 'assisted', 'assistance', 'assistant', 'aid', 'aids', 'aiding',
            'aided', 'community', 'communities', 'communal', 'commune', 'together',
            'togetherness', 'cooperate', 'cooperates', 'cooperating', 'cooperated',
            'cooperation', 'cooperative', 'cooperatively', 'collaborate', 'collaborates',
            'collaborating', 'collaborated', 'collaboration', 'collaborative', 'partner',
            'partners', 'partnering', 'partnered', 'partnership', 'partnerships',
            'understand', 'understands', 'understanding', 'understood', 'comprehend',
            'comprehends', 'comprehending', 'comprehended', 'comprehension', 'learn',
            'learns', 'learning', 'learned', 'learner', 'learners', 'educate', 'educates',
            'educating', 'educated', 'education', 'educational', 'educator', 'educators',
            'teach', 'teaches', 'teaching', 'taught', 'teacher', 'teachers', 'reform',
            'reforms', 'reforming', 'reformed', 'reformer', 'reformers', 'change',
            'changes', 'changing', 'changed', 'transform', 'transforms', 'transforming',
            'transformed', 'transformation', 'better', 'betters', 'bettering', 'bettered',
            'betterment', 'heal', 'heals', 'healing', 'healed', 'healer', 'healers',
            'reconcile', 'reconciles', 'reconciling', 'reconciled', 'reconciliation',
            'unite', 'unites', 'uniting', 'united', 'unity', 'unifier', 'bridge',
            'bridges', 'bridging', 'bridged', 'connect', 'connects', 'connecting',
            'connected', 'connection', 'connections', 'dialogue', 'dialogues', 'discuss',
            'discusses', 'discussing', 'discussed', 'discussion', 'discussions',
            'negotiate', 'negotiates', 'negotiating', 'negotiated', 'negotiation',
            'compromise', 'compromises', 'compromising', 'compromised', 'consensus',
            'agree', 'agrees', 'agreeing', 'agreed', 'agreement', 'agreements'
        }

        # religious extremism vocabulary
        self.religious_extremism_words = {
            'jihad', 'jihadist', 'jihadists', 'mujahid', 'mujahideen', 'shahid',
            'shaheed', 'martyr', 'martyrs', 'martyrdom', 'caliphate', 'caliph',
            'khilafah', 'ummah', 'kafir', 'kuffar', 'infidel', 'infidels', 'crusade',
            'crusader', 'crusaders', 'apostate', 'apostates', 'apostasy', 'takfir',
            'takfiri', 'sharia', 'shariah', 'hudud', 'jizya', 'dhimmi', 'dhimmitude',
            'blasphemy', 'blasphemer', 'blasphemers', 'heresy', 'heretic', 'heretics',
            'theocracy', 'theocratic', 'dominion', 'dominionism', 'dominionist',
            'theonomy', 'reconstructionism', 'rapture', 'tribulation', 'armageddon',
            'antichrist', 'endtimes', 'apocalypse', 'apocalyptic', 'messianic',
            'messiah', 'chosen', 'elect', 'righteous', 'unrighteous', 'sinner',
            'sinners', 'damnation', 'damned', 'hellfire', 'salvation', 'saved',
            'unsaved', 'believer', 'believers', 'unbeliever', 'unbelievers', 'pagan',
            'pagans', 'heathen', 'heathens', 'idolater', 'idolaters', 'idolatry',
            'witchcraft', 'sorcery', 'occult', 'demonic', 'possessed', 'exorcism'
        }

        # racial and ethnic extremism vocabulary
        self.racial_extremism_words = {
            'aryan', 'aryans', 'nordic', 'nordics', 'supremacy', 'supremacist',
            'supremacists', 'nationalist', 'nationalists', 'nationalism', 'ethno',
            'ethnostate', 'ethnonationalist', 'ethnonationalism', 'separatist',
            'separatists', 'separatism', 'segregation', 'segregationist', 'apartheid',
            'purity', 'pure', 'purist', 'purify', 'purification', 'cleanse', 'cleansing',
            'genocide', 'genocidal', 'holocaust', 'pogrom', 'pogroms', 'eugenics',
            'eugenic', 'dysgenics', 'mongrel', 'mongrels', 'mongrelization', 'mulatto',
            'miscegenation', '混血', 'halfbreed', 'untermensch', 'ubermensch', 'master',
            'masterrace', 'subhuman', 'subhumans', 'inferior', 'inferiority', 'superior',
            'superiority', 'bloodline', 'bloodlines', 'lineage', 'lineages', 'ancestry',
            'genetic', 'genetics', 'genetically', 'hereditary', 'heredity', 'breeding',
            'breed', 'bred', 'stock', 'replacement', 'invasion', 'horde', 'hordes',
            'swarm', 'swarms', 'swarming', 'flood', 'flooding', 'flooded', 'overrun',
            'overrunning', 'colonize', 'colonized', 'colonization', 'decolonize'
        }

        # conspiracy and paranoia vocabulary
        self.conspiracy_words = {
            'conspiracy', 'conspiracies', 'conspire', 'conspired', 'conspiring',
            'conspirator', 'conspirators', 'cabal', 'cabals', 'cartel', 'cartels',
            'syndicate', 'syndicates', 'illuminati', 'freemason', 'freemasons',
            'freemasonry', 'masonic', 'rothschild', 'rothschilds', 'soros', 'bilderberg',
            'globalist', 'globalists', 'globalism', 'elites', 'elite', 'elitist',
            'elitists', 'establishment', 'shadow', 'shadowy', 'deep', 'deepstate',
            'puppet', 'puppets', 'puppeteer', 'puppeteers', 'handler', 'handlers',
            'controller', 'controllers', 'manipulate', 'manipulated', 'manipulating',
            'manipulation', 'manipulator', 'manipulators', 'brainwash', 'brainwashed',
            'brainwashing', 'indoctrinate', 'indoctrinated', 'indoctrination',
            'propaganda', 'propagandist', 'propagandists', 'disinformation', 'misinformation',
            'coverup', 'coverups', 'suppress', 'suppressed', 'suppressing', 'suppression',
            'censor', 'censored', 'censoring', 'censorship', 'silence', 'silenced',
            'silencing', 'hidden', 'hide', 'hiding', 'secret', 'secrets', 'secretly',
            'secretive', 'covert', 'covertly', 'clandestine', 'clandestinely', 'sinister',
            'nefarious', 'scheme', 'schemes', 'scheming', 'schemer', 'schemers',
            'plot', 'plots', 'plotting', 'plotted', 'plotter', 'plotters', 'hoax',
            'hoaxes', 'hoaxer', 'hoaxers', 'staged', 'staging', 'false', 'flag',
            'psyop', 'psyops', 'operative', 'operatives', 'agent', 'agents', 'spy',
            'spies', 'spying', 'infiltrate', 'infiltrated', 'infiltrating', 'infiltrator',
            'mole', 'moles', 'plant', 'planted', 'controlled', 'opposition', 'shill',
            'shills', 'shilling', 'astroturf', 'astroturfing', 'botnet', 'bots'
        }

        # anti-government and sedition vocabulary
        self.antigovernment_words = {
            'tyranny', 'tyrant', 'tyrants', 'tyrannical', 'tyrannic', 'despotism',
            'despot', 'despots', 'despotic', 'dictator', 'dictators', 'dictatorship',
            'dictatorial', 'authoritarian', 'authoritarianism', 'totalitarian',
            'totalitarianism', 'fascist', 'fascists', 'fascism', 'nazi', 'nazis',
            'nazism', 'oppression', 'oppressor', 'oppressors', 'oppressed', 'oppress',
            'oppressing', 'oppressive', 'regime', 'regimes', 'junta', 'juntas',
            'overthrow', 'overthrows', 'overthrowing', 'overthrown', 'topple', 'topples',
            'toppling', 'toppled', 'coup', 'coups', 'insurrection', 'insurrections',
            'insurrectionist', 'insurrectionists', 'insurgent', 'insurgents', 'insurgency',
            'uprising', 'uprisings', 'revolt', 'revolts', 'revolting', 'revolution',
            'revolutions', 'revolutionary', 'revolutionaries', 'rebel', 'rebels',
            'rebelling', 'rebellion', 'rebellions', 'rebellious', 'mutiny', 'mutinies',
            'mutineer', 'mutineers', 'sedition', 'seditious', 'subversion', 'subversive',
            'subversives', 'subvert', 'subverted', 'subverting', 'treason', 'treasonous',
            'treasonable', 'traitor', 'traitors', 'traitorous', 'secede', 'secedes',
            'seceding', 'seceded', 'secession', 'secessionist', 'secessionists',
            'nullify', 'nullifies', 'nullifying', 'nullified', 'nullification',
            'militia', 'militias', 'militiaman', 'militiamen', 'paramilitary',
            'paramilitaries', 'vigilante', 'vigilantes', 'vigilantism', 'sovereign',
            'sovereigns', 'sovereignty', 'freeman', 'freemen', 'patriot', 'patriots'
        }

    def fit(self, texts, y=None):
        return self

    def transform(self, texts):
        features = []
        for text in texts:
            tokens = set(self.preprocessor.tokenize(str(text)))
            total_tokens = len(tokens) if tokens else 1

            # core lexicon counts
            violence_count = len(tokens & self.violence_words)
            fear_count = len(tokens & self.fear_words)
            hate_count = len(tokens & self.hate_words)
            us_them_count = len(tokens & self.us_them_words)
            certainty_count = len(tokens & self.certainty_words)
            action_count = len(tokens & self.action_words)
            democratic_count = len(tokens & self.democratic_words)
            constructive_count = len(tokens & self.constructive_words)

            # extremism-specific lexicon counts
            religious_extremism_count = len(tokens & self.religious_extremism_words)
            racial_extremism_count = len(tokens & self.racial_extremism_words)
            conspiracy_count = len(tokens & self.conspiracy_words)
            antigovernment_count = len(tokens & self.antigovernment_words)

            # density scores (normalized by token count)
            violence_density = violence_count / total_tokens
            fear_density = fear_count / total_tokens
            hate_density = hate_count / total_tokens
            us_them_density = us_them_count / total_tokens
            certainty_density = certainty_count / total_tokens
            action_density = action_count / total_tokens
            democratic_density = democratic_count / total_tokens
            constructive_density = constructive_count / total_tokens
            religious_extremism_density = religious_extremism_count / total_tokens
            racial_extremism_density = racial_extremism_count / total_tokens
            conspiracy_density = conspiracy_count / total_tokens
            antigovernment_density = antigovernment_count / total_tokens

            # composite scores
            extremism_core = violence_count + fear_count + hate_count
            extremism_ideology = religious_extremism_count + racial_extremism_count + conspiracy_count + antigovernment_count
            extremism_total = extremism_core + extremism_ideology + us_them_count + certainty_count
            american_score = democratic_count + constructive_count

            # ratio features
            total_signal = extremism_total + american_score
            if total_signal > 0:
                extremism_ratio = extremism_total / total_signal
                american_ratio = american_score / total_signal
            else:
                extremism_ratio = american_ratio = 0.5

            # interaction features
            violence_hate_interaction = violence_count * hate_count
            fear_action_interaction = fear_count * action_count
            certainty_extremism_interaction = certainty_count * extremism_core
            us_them_hate_interaction = us_them_count * hate_count

            # normalized interaction features
            violence_hate_norm = violence_hate_interaction / (total_tokens ** 2) if total_tokens > 1 else 0
            fear_action_norm = fear_action_interaction / (total_tokens ** 2) if total_tokens > 1 else 0

            feat = [
                # raw counts (12 features)
                violence_count, fear_count, hate_count, us_them_count,
                certainty_count, action_count, democratic_count, constructive_count,
                religious_extremism_count, racial_extremism_count, conspiracy_count, antigovernment_count,

                # density features (12 features)
                violence_density, fear_density, hate_density, us_them_density,
                certainty_density, action_density, democratic_density, constructive_density,
                religious_extremism_density, racial_extremism_density, conspiracy_density, antigovernment_density,

                # composite scores (4 features)
                extremism_core, extremism_ideology, extremism_total, american_score,

                # ratio features (3 features)
                extremism_ratio, american_ratio, extremism_total - american_score,

                # interaction features (6 features)
                violence_hate_interaction, fear_action_interaction,
                certainty_extremism_interaction, us_them_hate_interaction,
                violence_hate_norm, fear_action_norm
            ]
            features.append(feat)

        return np.array(features)


class SyntacticFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def fit(self, texts, y=None):
        return self

    def transform(self, texts):
        features = []
        for text in texts:
            text = str(text)
            tokens = self.preprocessor.tokenize(text)

            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            num_sentences = max(len(sentences), 1)

            num_words = len(tokens)
            if num_words == 0:
                features.append(np.zeros(25))
                continue

            word_lengths = [len(w) for w in tokens]
            avg_word_len = np.mean(word_lengths)
            std_word_len = np.std(word_lengths)
            max_word_len = np.max(word_lengths)

            sent_lengths = [len(s.split()) for s in sentences]
            avg_sent_len = np.mean(sent_lengths)
            std_sent_len = np.std(sent_lengths) if len(sent_lengths) > 1 else 0

            unique_words = len(set(tokens))
            ttr = unique_words / num_words

            exclamations = text.count('!')
            questions = text.count('?')

            all_caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
            caps_ratio = all_caps_words / num_words

            first_person = len(re.findall(r'\b(i|we|us|our|my)\b', text.lower()))
            second_person = len(re.findall(r'\b(you|your|yours)\b', text.lower()))
            third_person = len(re.findall(r'\b(they|them|their|he|she|it)\b', text.lower()))

            modals = len(re.findall(r'\b(must|should|will|shall|need|have to)\b', text.lower()))
            negations = len(re.findall(r'\b(not|no|never|nothing|nobody|none)\b', text.lower()))
            intensifiers = len(re.findall(r'\b(very|extremely|totally|absolutely|completely)\b', text.lower()))

            feat = [
                num_words, num_sentences, avg_word_len, std_word_len, max_word_len,
                avg_sent_len, std_sent_len, unique_words, ttr,
                exclamations, questions, exclamations / num_sentences, questions / num_sentences,
                all_caps_words, caps_ratio,
                first_person, second_person, third_person,
                first_person / num_words, third_person / num_words,
                modals, modals / num_words,
                negations, negations / num_words,
                intensifiers
            ]
            features.append(feat)

        return np.array(features)


class SentimentFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.preprocessor = TextPreprocessor()

        self.positive = {
            'good', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic',
            'love', 'like', 'enjoy', 'happy', 'joy', 'hope', 'peace', 'beautiful',
            'best', 'better', 'nice', 'fine', 'well', 'success', 'win', 'positive',
            'friend', 'help', 'support', 'care', 'kind', 'gentle', 'safe', 'trust'
        }

        self.negative = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike',
            'angry', 'sad', 'fear', 'pain', 'suffer', 'hurt', 'damage', 'harm',
            'wrong', 'fail', 'lose', 'negative', 'enemy', 'threat', 'danger',
            'evil', 'cruel', 'violent', 'destroy', 'kill', 'death', 'war'
        }

        self.anger = {
            'angry', 'mad', 'furious', 'rage', 'hate', 'despise', 'loathe',
            'outrage', 'infuriate', 'enrage', 'hostile', 'bitter', 'resentful'
        }

        self.fear_lex = {
            'fear', 'afraid', 'scared', 'terrified', 'panic', 'anxiety', 'worry',
            'dread', 'horror', 'fright', 'alarm', 'terror', 'phobia'
        }

    def fit(self, texts, y=None):
        return self

    def transform(self, texts):
        features = []
        for text in texts:
            tokens = set(self.preprocessor.tokenize(str(text)))
            total = len(tokens) if tokens else 1

            pos_count = len(tokens & self.positive)
            neg_count = len(tokens & self.negative)
            anger_count = len(tokens & self.anger)
            fear_count = len(tokens & self.fear_lex)

            sentiment = (pos_count - neg_count) / total
            polarity = pos_count / (pos_count + neg_count) if (pos_count + neg_count) > 0 else 0.5
            subjectivity = (pos_count + neg_count) / total

            feat = [
                pos_count, neg_count, anger_count, fear_count,
                pos_count / total, neg_count / total, anger_count / total, fear_count / total,
                sentiment, polarity, subjectivity,
                neg_count - pos_count
            ]
            features.append(feat)

        return np.array(features)


class ReadabilityFeatures(BaseEstimator, TransformerMixin):
    def fit(self, texts, y=None):
        return self

    def transform(self, texts):
        features = []
        for text in texts:
            text = str(text)
            words = text.split()
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

            num_words = len(words)
            num_sentences = max(len(sentences), 1)

            if num_words == 0:
                features.append(np.zeros(8))
                continue

            syllables = sum(self._count_syllables(w) for w in words)
            avg_syllables = syllables / num_words
            avg_words_per_sent = num_words / num_sentences

            flesch = 206.835 - (1.015 * avg_words_per_sent) - (84.6 * avg_syllables)
            fk_grade = (0.39 * avg_words_per_sent) + (11.8 * avg_syllables) - 15.59

            complex_words = sum(1 for w in words if self._count_syllables(w) >= 3)
            complex_ratio = complex_words / num_words

            fog = 0.4 * (avg_words_per_sent + 100 * complex_ratio)

            chars = len(re.sub(r'\s', '', text))
            ari = (4.71 * chars / num_words) + (0.5 * avg_words_per_sent) - 21.43

            feat = [
                flesch, fk_grade, fog, ari,
                avg_syllables, avg_words_per_sent, complex_words, complex_ratio
            ]
            features.append(feat)

        return np.array(features)

    def _count_syllables(self, word: str) -> int:
        word = word.lower().strip(".,!?;:'\"")
        if not word:
            return 0
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith('e') and count > 1:
            count -= 1
        return max(1, count)


class DiscourseFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.causal = {'because', 'since', 'therefore', 'thus', 'hence', 'consequently', 'so'}
        self.contrast = {'but', 'however', 'although', 'though', 'yet', 'nevertheless', 'despite'}
        self.additive = {'and', 'also', 'moreover', 'furthermore', 'additionally', 'plus'}
        self.temporal = {'then', 'first', 'next', 'finally', 'before', 'after', 'when', 'now'}
        self.conditional = {'if', 'unless', 'whether', 'provided', 'assuming', 'suppose'}

        self.preprocessor = TextPreprocessor()

    def fit(self, texts, y=None):
        return self

    def transform(self, texts):
        features = []
        for text in texts:
            tokens = self.preprocessor.tokenize(str(text))
            token_set = set(tokens)
            total = len(tokens) if tokens else 1

            causal = len(token_set & self.causal)
            contrast = len(token_set & self.contrast)
            additive = len(token_set & self.additive)
            temporal = len(token_set & self.temporal)
            conditional = len(token_set & self.conditional)

            feat = [
                causal, contrast, additive, temporal, conditional,
                causal / total, contrast / total, additive / total,
                temporal / total, conditional / total,
                (causal + contrast + additive + temporal + conditional) / total
            ]
            features.append(feat)

        return np.array(features)


class StatisticalFeatures(BaseEstimator, TransformerMixin):
    def fit(self, texts, y=None):
        return self

    def transform(self, texts):
        features = []
        for text in texts:
            text = str(text)

            chars = list(text.lower())
            char_freq = Counter(chars)

            total_chars = len(chars) if chars else 1
            entropy = 0
            for freq in char_freq.values():
                if freq > 0:
                    p = freq / total_chars
                    entropy -= p * math.log2(p)

            letters = [c for c in chars if c.isalpha()]
            vowels = sum(1 for c in letters if c in 'aeiou')
            vowel_ratio = vowels / len(letters) if letters else 0

            digits = sum(1 for c in text if c.isdigit())
            special = sum(1 for c in text if not c.isalnum() and not c.isspace())

            digit_ratio = digits / total_chars
            special_ratio = special / total_chars

            words = text.split()
            if words:
                lengths = [len(w) for w in words]
                len_mean = np.mean(lengths)
                len_std = np.std(lengths)
                len_skew = np.mean(((np.array(lengths) - len_mean) / (len_std + 1e-8)) ** 3)
            else:
                len_mean = len_std = len_skew = 0

            feat = [
                entropy, vowel_ratio, digit_ratio, special_ratio,
                len_mean, len_std, len_skew,
                len(text), len(words) if words else 0
            ]
            features.append(feat)

        return np.array(features)


class ContextWindowFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, window_sizes=[2, 3, 5]):
        self.window_sizes = window_sizes
        self.semantic = SemanticDensityFeatures()
        self.sentiment = SentimentFeatures()

    def fit(self, texts, y=None):
        return self

    def transform(self, texts):
        all_features = []

        for text in texts:
            text = str(text)
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

            if len(sentences) < 2:
                base_sem = self.semantic.transform([text])[0]
                base_sent = self.sentiment.transform([text])[0]
                feat = np.tile(np.concatenate([base_sem, base_sent]), len(self.window_sizes) * 2)
                all_features.append(feat)
                continue

            window_feats = []
            for ws in self.window_sizes:
                if len(sentences) <= ws:
                    window_text = text
                    sem = self.semantic.transform([window_text])[0]
                    sent = self.sentiment.transform([window_text])[0]
                    window_feats.extend([sem.mean(), sem.std(), sent.mean(), sent.std()])
                    window_feats.extend(sem)
                    window_feats.extend(sent)
                else:
                    sem_feats = []
                    sent_feats = []
                    for i in range(len(sentences) - ws + 1):
                        window_text = ' '.join(sentences[i:i+ws])
                        sem_feats.append(self.semantic.transform([window_text])[0])
                        sent_feats.append(self.sentiment.transform([window_text])[0])

                    sem_arr = np.array(sem_feats)
                    sent_arr = np.array(sent_feats)

                    window_feats.extend([
                        sem_arr.mean(), sem_arr.std(), sem_arr.max(), sem_arr.min(),
                        sent_arr.mean(), sent_arr.std(), sent_arr.max(), sent_arr.min()
                    ])

            all_features.append(np.array(window_feats))

        max_len = max(len(f) for f in all_features)
        padded = []
        for f in all_features:
            if len(f) < max_len:
                f = np.concatenate([f, np.zeros(max_len - len(f))])
            padded.append(f)

        return np.array(padded)


class ExtremismDetector:
    def __init__(self):
        self.ngram_extractor = NGramFeatureExtractor()
        self.embedding_features = WordEmbeddingFeatures(vector_size=50)
        self.semantic_features = SemanticDensityFeatures()
        self.syntactic_features = SyntacticFeatures()
        self.sentiment_features = SentimentFeatures()
        self.readability_features = ReadabilityFeatures()
        self.discourse_features = DiscourseFeatures()
        self.statistical_features = StatisticalFeatures()
        self.context_features = ContextWindowFeatures()

        self.scaler_dense = StandardScaler()
        self.scaler_robust = RobustScaler()

        self.svd_ngram = TruncatedSVD(n_components=200, random_state=42)
        self.svd_char = TruncatedSVD(n_components=100, random_state=42)

        self.feature_selector = None
        self.model = None
        self.label_encoder = LabelEncoder()

    def _load_data(self, directory: str) -> Tuple[List[str], List[str]]:
        texts = []
        labels = []

        csv_files = list(Path(directory).glob('*.csv'))
        if not csv_files:
            raise ValueError(f"no csv files found in {directory}")

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'text' not in df.columns or 'label' not in df.columns:
                    print(f"skipping {csv_file}: missing text or label columns")
                    continue

                df = df.dropna(subset=['text', 'label'])
                texts.extend(df['text'].astype(str).tolist())
                labels.extend(df['label'].astype(str).tolist())
                print(f"loaded {len(df)} samples from {csv_file.name}")
            except Exception as e:
                print(f"error loading {csv_file}: {e}")

        return texts, labels

    def _extract_all_features(self, texts: List[str], fit: bool = True) -> sp.csr_matrix:
        print("  extracting ngram features...")
        if fit:
            ngram_features = self.ngram_extractor.fit_transform(texts)
        else:
            ngram_features = self.ngram_extractor.transform(texts)

        print("  extracting embedding features...")
        if fit:
            embedding_feat = self.embedding_features.fit_transform(texts)
        else:
            embedding_feat = self.embedding_features.transform(texts)

        print("  extracting semantic features...")
        semantic_feat = self.semantic_features.transform(texts)

        print("  extracting syntactic features...")
        syntactic_feat = self.syntactic_features.transform(texts)

        print("  extracting sentiment features...")
        sentiment_feat = self.sentiment_features.transform(texts)

        print("  extracting readability features...")
        readability_feat = self.readability_features.transform(texts)

        print("  extracting discourse features...")
        discourse_feat = self.discourse_features.transform(texts)

        print("  extracting statistical features...")
        statistical_feat = self.statistical_features.transform(texts)

        print("  extracting context window features...")
        context_feat = self.context_features.transform(texts)

        dense_combined = np.hstack([
            embedding_feat, semantic_feat, syntactic_feat, sentiment_feat,
            readability_feat, discourse_feat, statistical_feat, context_feat
        ])

        dense_combined = np.nan_to_num(dense_combined, nan=0.0, posinf=0.0, neginf=0.0)

        print("  scaling features...")
        if fit:
            dense_scaled = self.scaler_dense.fit_transform(dense_combined)
        else:
            dense_scaled = self.scaler_dense.transform(dense_combined)

        print("  applying svd...")
        if fit:
            ngram_reduced = self.svd_ngram.fit_transform(ngram_features)
        else:
            ngram_reduced = self.svd_ngram.transform(ngram_features)

        final_features = sp.hstack([
            ngram_features,
            sp.csr_matrix(ngram_reduced),
            sp.csr_matrix(dense_scaled)
        ])

        return final_features

    def train(self, data_directory: str = 'extremism_data'):
        print(f"loading data from {data_directory}...")
        texts, labels = self._load_data(data_directory)

        if len(texts) == 0:
            raise ValueError("no data loaded")

        print(f"\ntotal samples: {len(texts)}")
        print(f"label distribution: {Counter(labels)}")

        print("\nnormalizing labels...")
        label_map = {}
        for label in set(labels):
            lower = label.lower()
            if any(x in lower for x in ['anti', 'extrem', 'radical', 'hate', 'terror', 'violent']):
                label_map[label] = 'anti_american'
            else:
                label_map[label] = 'american'

        labels = [label_map.get(l, 'american') for l in labels]
        print(f"normalized distribution: {Counter(labels)}")

        y = self.label_encoder.fit_transform(labels)

        print("\nextracting features...")
        X = self._extract_all_features(texts, fit=True)
        print(f"feature matrix shape: {X.shape}")

        print("\nselecting best features...")
        k = min(30000, X.shape[1])
        self.feature_selector = SelectKBest(f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        print(f"selected features: {X_selected.shape}")

        print("\ntraining random forest classifier...")
        self.model = RandomForestClassifier(
            n_estimators=1000,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=1,
            bootstrap=True,
            oob_score=True,
            max_samples=0.85,
            min_impurity_decrease=0.00001
        )

        self.model.fit(X_selected, y)

        print(f"\noob score: {self.model.oob_score_:.4f}")

        print("\ncross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_selected, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
        print(f"cv f1 scores: {cv_scores}")
        print(f"mean cv f1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        train_pred = self.model.predict(X_selected)
        train_proba = self.model.predict_proba(X_selected)

        print(f"\ntraining accuracy: {accuracy_score(y, train_pred):.4f}")
        print(f"training f1: {f1_score(y, train_pred, average='weighted'):.4f}")

        if len(set(y)) == 2:
            print(f"training roc-auc: {roc_auc_score(y, train_proba[:, 1]):.4f}")

        print("\nclassification report:")
        print(classification_report(y, train_pred, target_names=self.label_encoder.classes_, digits=4))

        print("\nconfusion matrix:")
        print(confusion_matrix(y, train_pred))

        importances = self.model.feature_importances_
        top_idx = np.argsort(importances)[-20:][::-1]
        print(f"\ntop 20 feature importances:")
        for idx in top_idx:
            print(f"  feature {idx}: {importances[idx]:.6f}")

    def save(self, filepath: str):
        data = {
            'model': self.model,
            'ngram_extractor': self.ngram_extractor,
            'embedding_features': self.embedding_features,
            'semantic_features': self.semantic_features,
            'syntactic_features': self.syntactic_features,
            'sentiment_features': self.sentiment_features,
            'readability_features': self.readability_features,
            'discourse_features': self.discourse_features,
            'statistical_features': self.statistical_features,
            'context_features': self.context_features,
            'scaler_dense': self.scaler_dense,
            'svd_ngram': self.svd_ngram,
            'feature_selector': self.feature_selector,
            'label_encoder': self.label_encoder
        }
        joblib.dump(data, filepath)
        print(f"model saved to {filepath}")

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.model = data['model']
        self.ngram_extractor = data['ngram_extractor']
        self.embedding_features = data['embedding_features']
        self.semantic_features = data['semantic_features']
        self.syntactic_features = data['syntactic_features']
        self.sentiment_features = data['sentiment_features']
        self.readability_features = data['readability_features']
        self.discourse_features = data['discourse_features']
        self.statistical_features = data['statistical_features']
        self.context_features = data['context_features']
        self.scaler_dense = data['scaler_dense']
        self.svd_ngram = data['svd_ngram']
        self.feature_selector = data['feature_selector']
        self.label_encoder = data['label_encoder']
        print(f"model loaded from {filepath}")

    def predict(self, text: str) -> Tuple[str, np.ndarray]:
        X = self._extract_all_features([text], fit=False)
        X_selected = self.feature_selector.transform(X)

        pred_idx = self.model.predict(X_selected)[0]
        proba = self.model.predict_proba(X_selected)[0]

        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
        return pred_label, proba

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, np.ndarray]]:
        X = self._extract_all_features(texts, fit=False)
        X_selected = self.feature_selector.transform(X)

        pred_idx = self.model.predict(X_selected)
        proba = self.model.predict_proba(X_selected)

        pred_labels = self.label_encoder.inverse_transform(pred_idx)
        return list(zip(pred_labels, proba))


def main():
    detector = ExtremismDetector()

    detector.train(data_directory='../data')

    detector.save('../models/extremism_detector.pkl')

    test_samples = [
        "the constitution guarantees our right to free speech and peaceful assembly",
        "democracy allows citizens to vote and hold representatives accountable",
        "we must work together as a community to build a better future for everyone",
        "death to the infidels and crusaders who oppose our holy war",
        "white power will rise again through blood and soil and racial holy war",
        "the globalist cabal controls the deep state and is orchestrating our replacement",
        "armed revolution and violent overthrow is the only solution to destroy the system",
        "the founding fathers established liberty and justice under the rule of law",
        "bipartisan cooperation and mutual respect can help us find common ground",
        "kill all the traitors and burn down their corrupt institutions"
    ]

    print("\n" + "=" * 80)
    print("sample predictions:")
    print("=" * 80)

    for text in test_samples:
        pred, proba = detector.predict(text)
        conf = proba.max()
        display = text[:70] + "..." if len(text) > 70 else text
        print(f"\n{pred:15s} ({conf:.1%})")
        print(f"  {display}")


if __name__ == "__main__":
    main()
