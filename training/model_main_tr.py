import numpy as np
import pandas as pd
import os
import re
import joblib
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set
from itertools import combinations

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    def __init__(self):
        self.positive_words = {'good', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic',
                               'benefit', 'improve', 'success', 'progress', 'hope', 'positive',
                               'effective', 'valuable', 'important', 'necessary', 'essential',
                               'support', 'help', 'protect', 'ensure', 'promote', 'advance'}
        self.negative_words = {'bad', 'terrible', 'awful', 'horrible', 'disaster', 'crisis',
                              'problem', 'issue', 'threat', 'danger', 'harm', 'damage',
                              'fail', 'failure', 'wrong', 'corrupt', 'exploit', 'destroy',
                              'attack', 'violence', 'inequality', 'injustice', 'oppression'}
        self.intensifiers = {'very', 'extremely', 'incredibly', 'absolutely', 'completely',
                            'totally', 'utterly', 'absolutely', 'highly', 'deeply', 'severely'}
    
    def analyze(self, text: str) -> Dict[str, float]:
        words = set(re.findall(r'\b\w+\b', text.lower()))
        
        positive_count = len(words & self.positive_words)
        negative_count = len(words & self.negative_words)
        intensifier_count = len(words & self.intensifiers)
        
        total_words = len(re.findall(r'\b\w+\b', text))
        sentiment_score = (positive_count - negative_count) / max(total_words, 1)
        polarity = positive_count / max(positive_count + negative_count, 1)
        
        return {
            'positive': positive_count,
            'negative': negative_count,
            'intensifiers': intensifier_count,
            'sentiment_score': sentiment_score,
            'polarity': polarity if not math.isnan(polarity) else 0.5
        }


class SyntacticFeatureExtractor:
    def __init__(self):
        self.verb_patterns = [
            r'\b(is|are|was|were|be|been|being)\b',
            r'\b(have|has|had|having)\b',
            r'\b(do|does|did|doing|done)\b',
            r'\b(will|would|should|could|might|may|shall|must)\b'
        ]
        self.noun_patterns = [
            r'\b(the|a|an)\s+\w+',
            r'\b\w+ion\b',
            r'\b\w+ment\b',
            r'\b\w+ness\b',
            r'\b\w+ity\b'
        ]
        self.adjective_patterns = [
            r'\b\w+ly\b',
            r'\b(very|more|most|less|least)\s+\w+',
            r'\b\w+er\b',
            r'\b\w+est\b'
        ]
    
    def extract(self, text: str) -> np.ndarray:
        text_lower = text.lower()
        features = []
        
        verb_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.verb_patterns)
        noun_phrases = len(re.findall(r'\b(the|a|an)\s+\w+', text_lower))
        adjectives = len(re.findall(r'\b\w+ly\b', text_lower))
        comparatives = len(re.findall(r'\b\w+(er|est)\b', text_lower))
        
        passive_voice = len(re.findall(r'\b(is|are|was|were)\s+\w+ed\b', text_lower))
        active_voice = len(re.findall(r'\b\w+ed\s+\w+\b', text_lower))
        
        features.extend([verb_count, noun_phrases, adjectives, comparatives,
                         passive_voice, active_voice])
        
        return np.array(features)


class DiscourseMarkerAnalyzer:
    def __init__(self):
        self.causal_markers = ['because', 'since', 'due to', 'as a result', 'therefore',
                              'thus', 'hence', 'consequently', 'so', 'for this reason']
        self.contrastive_markers = ['but', 'however', 'although', 'though', 'despite',
                                   'nevertheless', 'nonetheless', 'yet', 'whereas', 'while']
        self.additive_markers = ['and', 'also', 'furthermore', 'moreover', 'additionally',
                                'plus', 'in addition', 'besides', 'as well']
        self.temporal_markers = ['first', 'then', 'next', 'finally', 'after', 'before',
                                'during', 'while', 'when', 'now', 'recently', 'previously']
        self.exemplification = ['for example', 'for instance', 'such as', 'namely',
                               'specifically', 'in particular', 'including']
    
    def analyze(self, text: str) -> np.ndarray:
        text_lower = text.lower()
        features = []
        
        causal = sum(1 for marker in self.causal_markers if marker in text_lower)
        contrastive = sum(1 for marker in self.contrastive_markers if marker in text_lower)
        additive = sum(1 for marker in self.additive_markers if marker in text_lower)
        temporal = sum(1 for marker in self.temporal_markers if marker in text_lower)
        exemplification = sum(1 for marker in self.exemplification if marker in text_lower)
        
        features.extend([causal, contrastive, additive, temporal, exemplification])
        return np.array(features)


class ArgumentationStructureAnalyzer:
    def __init__(self):
        self.claim_indicators = ['claim', 'argue', 'assert', 'maintain', 'contend',
                                'propose', 'suggest', 'recommend', 'advocate']
        self.evidence_indicators = ['evidence', 'data', 'study', 'research', 'show',
                                   'demonstrate', 'prove', 'indicate', 'reveal', 'find']
        self.reasoning_indicators = ['reason', 'because', 'since', 'therefore', 'thus',
                                    'consequently', 'logically', 'implies', 'follows']
        self.counter_indicators = ['however', 'but', 'although', 'despite', 'counter',
                                  'oppose', 'reject', 'refute', 'challenge']
    
    def analyze(self, text: str) -> np.ndarray:
        text_lower = text.lower()
        features = []
        
        claims = sum(1 for word in self.claim_indicators if word in text_lower)
        evidence = sum(1 for word in self.evidence_indicators if word in text_lower)
        reasoning = sum(1 for word in self.reasoning_indicators if word in text_lower)
        counter = sum(1 for word in self.counter_indicators if word in text_lower)
        
        argument_strength = (claims + evidence + reasoning) / max(len(text.split()), 1)
        features.extend([claims, evidence, reasoning, counter, argument_strength])
        
        return np.array(features)


class NamedEntityPatternExtractor:
    def __init__(self):
        self.person_patterns = [r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
                               r'\b(Dr|Mr|Mrs|Ms|President|Senator|Governor)\s+[A-Z][a-z]+']
        self.organization_patterns = [r'\b[A-Z][a-z]+\s+(Corporation|Corp|Inc|LLC|Ltd)\b',
                                     r'\b[A-Z][a-z]+\s+(University|College|Institute|Foundation)\b',
                                     r'\b[A-Z][a-z]+\s+(Party|Committee|Association|Union)\b']
        self.location_patterns = [r'\b[A-Z][a-z]+\s+(State|City|County|Nation|Country)\b',
                                 r'\b(United States|USA|America|Washington|New York)\b']
        self.date_patterns = [r'\b\d{4}\b', r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b']
    
    def extract(self, text: str) -> np.ndarray:
        features = []
        
        persons = len(re.findall(self.person_patterns[0], text))
        organizations = sum(len(re.findall(pattern, text)) for pattern in self.organization_patterns)
        locations = sum(len(re.findall(pattern, text)) for pattern in self.location_patterns)
        dates = sum(len(re.findall(pattern, text)) for pattern in self.date_patterns)
        
        features.extend([persons, organizations, locations, dates])
        return np.array(features)


class StylometricAnalyzer:
    def __init__(self):
        self.function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                              'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'are',
                              'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do',
                              'does', 'did', 'will', 'would', 'should', 'could', 'may',
                              'might', 'must', 'can', 'this', 'that', 'these', 'those'}
    
    def analyze(self, text: str) -> np.ndarray:
        words = text.lower().split()
        total_words = len(words)
        
        if total_words == 0:
            return np.zeros(10)
        
        function_word_count = sum(1 for w in words if w in self.function_words)
        function_word_ratio = function_word_count / total_words
        
        avg_word_length = np.mean([len(w) for w in words])
        avg_sentence_length = total_words / max(len(re.split(r'[.!?]+', text)), 1)
        
        unique_words = len(set(words))
        type_token_ratio = unique_words / total_words
        
        hapax_legomena = len([w for w in set(words) if words.count(w) == 1])
        hapax_ratio = hapax_legomena / max(unique_words, 1)
        
        long_words = sum(1 for w in words if len(w) > 6)
        long_word_ratio = long_words / total_words
        
        short_words = sum(1 for w in words if len(w) <= 3)
        short_word_ratio = short_words / total_words
        
        punctuation_count = sum(text.count(p) for p in '.,;:!?')
        punctuation_ratio = punctuation_count / total_words
        
        return np.array([
            function_word_ratio, avg_word_length, avg_sentence_length,
            type_token_ratio, hapax_ratio, long_word_ratio,
            short_word_ratio, punctuation_ratio, unique_words, total_words
        ])


class ReadabilityAnalyzer:
    def __init__(self):
        pass
    
    def analyze(self, text: str) -> np.ndarray:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return np.zeros(5)
        
        words = text.split()
        total_words = len(words)
        total_sentences = len(sentences)
        total_syllables = sum(self._count_syllables(word) for word in words)
        
        avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0
        avg_syllables_per_word = total_syllables / total_words if total_words > 0 else 0
        
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        complex_words = sum(1 for w in words if self._count_syllables(w) > 2)
        complex_word_ratio = complex_words / total_words if total_words > 0 else 0
        
        avg_words_per_sentence = avg_sentence_length
        
        return np.array([
            flesch_score, avg_sentence_length, avg_syllables_per_word,
            complex_word_ratio, avg_words_per_sentence
        ])
    
    def _count_syllables(self, word: str) -> int:
        word = word.lower().strip(".,!?;:")
        if not word:
            return 0
        vowels = "aeiouy"
        count = 0
        prev_was_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        if word.endswith('e'):
            count -= 1
        return max(1, count)


class WordCooccurrenceAnalyzer:
    def __init__(self):
        self.left_bigrams = [
            ('social', 'justice'), ('climate', 'change'), ('universal', 'healthcare'),
            ('minimum', 'wage'), ('workers', 'rights'), ('racial', 'justice'),
            ('student', 'debt'), ('gun', 'control'), ('police', 'reform'),
            ('criminal', 'justice'), ('voting', 'rights'), ('reproductive', 'rights'),
            ('immigration', 'reform'), ('green', 'new'), ('public', 'option'),
            ('wealth', 'inequality'), ('income', 'inequality'), ('systemic', 'racism')
        ]
        self.right_bigrams = [
            ('free', 'market'), ('lower', 'taxes'), ('border', 'security'),
            ('second', 'amendment'), ('gun', 'rights'), ('law', 'order'),
            ('traditional', 'values'), ('family', 'values'), ('religious', 'freedom'),
            ('small', 'government'), ('limited', 'government'), ('states', 'rights'),
            ('fiscal', 'responsibility'), ('balanced', 'budget'), ('merit', 'based'),
            ('individual', 'responsibility'), ('personal', 'responsibility'),
            ('job', 'creators'), ('wealth', 'creators')
        ]
        self.center_bigrams = [
            ('data', 'shows'), ('research', 'indicates'), ('evidence', 'suggests'),
            ('according', 'to'), ('study', 'finds'), ('analysis', 'reveals'),
            ('bipartisan', 'support'), ('common', 'ground'), ('middle', 'ground'),
            ('cost', 'benefit'), ('trade', 'off'), ('balance', 'between')
        ]
    
    def analyze(self, text: str) -> np.ndarray:
        text_lower = text.lower()
        words = text_lower.split()
        
        left_bigram_count = sum(1 for bigram in self.left_bigrams
                               if bigram[0] in text_lower and bigram[1] in text_lower)
        right_bigram_count = sum(1 for bigram in self.right_bigrams
                                if bigram[0] in text_lower and bigram[1] in text_lower)
        center_bigram_count = sum(1 for bigram in self.center_bigrams
                                 if bigram[0] in text_lower and bigram[1] in text_lower)
        
        word_pairs = list(combinations(set(words), 2))
        cooccurrence_density = len(word_pairs) / max(len(words), 1)
        
        total_bigrams = left_bigram_count + right_bigram_count + center_bigram_count
        if total_bigrams > 0:
            left_bigram_ratio = left_bigram_count / total_bigrams
            right_bigram_ratio = right_bigram_count / total_bigrams
            center_bigram_ratio = center_bigram_count / total_bigrams
        else:
            left_bigram_ratio = right_bigram_ratio = center_bigram_ratio = 0.33
        
        return np.array([
            left_bigram_count, right_bigram_count, center_bigram_count,
            cooccurrence_density, left_bigram_ratio, right_bigram_ratio, center_bigram_ratio
        ])


class TemporalFeatureExtractor:
    def __init__(self):
        self.past_indicators = ['was', 'were', 'had', 'did', 'went', 'came', 'said',
                               'took', 'made', 'got', 'saw', 'knew', 'thought']
        self.present_indicators = ['is', 'are', 'am', 'do', 'does', 'have', 'has',
                                  'go', 'come', 'say', 'take', 'make', 'get']
        self.future_indicators = ['will', 'would', 'shall', 'going to', 'about to',
                                 'plan to', 'intend to', 'expect to']
    
    def extract(self, text: str) -> np.ndarray:
        text_lower = text.lower()
        words = text_lower.split()
        
        past = sum(1 for w in words if w in self.past_indicators)
        present = sum(1 for w in words if w in self.present_indicators)
        future = sum(1 for marker in self.future_indicators if marker in text_lower)
        
        total_temporal = past + present + future
        temporal_density = total_temporal / max(len(words), 1)
        
        return np.array([past, present, future, temporal_density])


class CitationPatternExtractor:
    def __init__(self):
        self.citation_patterns = [
            r'\([A-Z][a-z]+\s+et\s+al\.?\s+\d{4}\)',
            r'\[?\d+\]?',
            r'\([A-Z][a-z]+,\s+\d{4}\)',
            r'according\s+to',
            r'cited\s+in',
            r'reference',
            r'source'
        ]
    
    def extract(self, text: str) -> np.ndarray:
        citation_count = sum(len(re.findall(pattern, text, re.IGNORECASE))
                            for pattern in self.citation_patterns)
        
        has_citations = 1 if citation_count > 0 else 0
        
        return np.array([citation_count, has_citations])


class StatisticalDistributionAnalyzer:
    def __init__(self):
        pass
    
    def analyze(self, text: str) -> np.ndarray:
        words = text.split()
        if not words:
            return np.zeros(8)
        
        word_lengths = [len(w) for w in words]
        
        mean_length = np.mean(word_lengths)
        std_length = np.std(word_lengths)
        median_length = np.median(word_lengths)
        min_length = np.min(word_lengths)
        max_length = np.max(word_lengths)
        
        char_frequencies = Counter(text.lower())
        entropy = -sum((freq / len(text)) * math.log2(freq / len(text))
                      for freq in char_frequencies.values() if freq > 0) if text else 0
        
        digit_count = sum(1 for c in text if c.isdigit())
        digit_ratio = digit_count / len(text) if text else 0
        
        uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        return np.array([
            mean_length, std_length, median_length, min_length,
            max_length, entropy, digit_ratio, uppercase_ratio
        ])


class PoliticalBiasAnalyzer:
    def __init__(self):
        self.left_indicators = self._build_left_indicators()
        self.right_indicators = self._build_right_indicators()
        self.center_indicators = self._build_center_indicators()
        self.economic_framing = self._build_economic_framing()
        self.social_framing = self._build_social_framing()
        self.government_framing = self._build_government_framing()
        
    def _build_left_indicators(self):
        return {
            'economic': ['redistribution', 'wealth tax', 'progressive tax', 'corporate tax',
                        'minimum wage', 'living wage', 'union', 'collective bargaining',
                        'workers rights', 'labor rights', 'public ownership', 'nationalization',
                        'social security', 'welfare state', 'universal basic income', 'ubi',
                        'antitrust', 'break up', 'regulate', 'regulation', 'subsidize',
                        'public investment', 'infrastructure spending', 'green new deal',
                        'climate investment', 'renewable energy', 'public option'],
            'social': ['social justice', 'racial justice', 'environmental justice',
                      'lgbtq rights', 'trans rights', 'reproductive rights', 'abortion rights',
                      'immigration reform', 'pathway to citizenship', 'diversity', 'inclusion',
                      'equity', 'affirmative action', 'police reform', 'defund police',
                      'criminal justice reform', 'prison reform', 'voting rights', 'democracy',
                      'civil rights', 'human rights', 'universal healthcare', 'medicare for all',
                      'free college', 'student debt cancellation', 'cancel student debt'],
            'ideological': ['progressive', 'socialist', 'democratic socialist', 'leftist',
                           'liberal', 'egalitarian', 'collectivist', 'solidarity',
                           'grassroots', 'movement', 'organize', 'mobilize', 'activism',
                           'systemic change', 'structural change', 'transform', 'revolution',
                           'power to the people', 'workers', 'working class', 'proletariat']
        }
    
    def _build_right_indicators(self):
        return {
            'economic': ['free market', 'deregulation', 'deregulate', 'privatize',
                        'privatization', 'lower taxes', 'tax cuts', 'tax reduction',
                        'supply side', 'trickle down', 'fiscal responsibility', 'balanced budget',
                        'reduce spending', 'cut spending', 'austerity', 'entitlement reform',
                        'social security reform', 'privatize social security', 'competition',
                        'market forces', 'invisible hand', 'laissez faire', 'capitalism',
                        'entrepreneurship', 'small business', 'job creators', 'wealth creators'],
            'social': ['traditional values', 'family values', 'religious freedom',
                      'pro life', 'pro-life', 'second amendment', 'gun rights', 'right to bear arms',
                      'law and order', 'tough on crime', 'death penalty', 'capital punishment',
                      'border security', 'immigration enforcement', 'illegal immigration',
                      'merit based', 'meritocracy', 'individual responsibility', 'personal responsibility',
                      'self reliance', 'pull yourself up', 'bootstrap', 'work ethic',
                      'patriotism', 'nationalism', 'american exceptionalism', 'heritage',
                      'constitutional rights', 'states rights', 'federalism'],
            'ideological': ['conservative', 'libertarian', 'republican', 'gop', 'right wing',
                           'traditionalist', 'constitutionalist', 'originalist', 'textualist',
                           'individual liberty', 'personal freedom', 'limited government',
                           'small government', 'states rights', 'federalism', 'sovereignty',
                           'national sovereignty', 'isolationist', 'protectionist']
        }
    
    def _build_center_indicators(self):
        return {
            'analytical': ['according to', 'data shows', 'research indicates', 'studies suggest',
                          'evidence shows', 'analysis reveals', 'findings indicate',
                          'statistics show', 'percent', 'percentage', 'survey', 'poll',
                          'empirical', 'evidence based', 'data driven', 'fact based'],
            'moderating': ['both sides', 'on one hand', 'on the other hand', 'however',
                          'nevertheless', 'although', 'while', 'whereas', 'balance',
                          'compromise', 'middle ground', 'common ground', 'bipartisan',
                          'cross party', 'moderate', 'centrist', 'pragmatic', 'practical',
                          'nuanced', 'complex', 'multifaceted', 'consider', 'evaluate',
                          'examine', 'assess', 'weigh', 'trade off', 'cost benefit'],
            'institutional': ['bipartisan', 'consensus', 'dialogue', 'discussion', 'debate',
                             'hearing', 'committee', 'congress', 'senate', 'house',
                             'legislative', 'regulatory', 'oversight', 'accountability',
                             'transparency', 'stakeholder', 'input', 'feedback', 'consultation']
        }
    
    def _build_economic_framing(self):
        return {
            'left_economic': ['exploitation', 'inequality', 'wealth gap', 'income gap',
                            'corporate greed', 'profit motive', 'wage theft', 'union busting',
                            'offshore', 'outsource', 'layoff', 'downsize', 'corporate welfare',
                            'tax loophole', 'tax haven', 'billionaire', 'oligarch'],
            'right_economic': ['job creators', 'wealth creators', 'innovation', 'entrepreneurship',
                             'economic growth', 'prosperity', 'opportunity', 'mobility',
                             'free enterprise', 'competition', 'efficiency', 'productivity',
                             'incentive', 'reward', 'merit', 'achievement'],
            'center_economic': ['economic indicators', 'gdp', 'unemployment rate', 'inflation',
                               'federal reserve', 'monetary policy', 'fiscal policy', 'budget',
                               'deficit', 'surplus', 'debt', 'revenue', 'expenditure']
        }
    
    def _build_social_framing(self):
        return {
            'left_social': ['systemic racism', 'institutional racism', 'white privilege',
                           'intersectionality', 'marginalized', 'oppressed', 'disenfranchised',
                           'equity', 'inclusion', 'representation', 'diversity', 'multicultural',
                           'social safety net', 'public good', 'collective action'],
            'right_social': ['merit', 'individual achievement', 'personal responsibility',
                           'family', 'community', 'faith', 'religion', 'tradition', 'heritage',
                           'culture', 'values', 'morality', 'virtue', 'character', 'discipline'],
            'center_social': ['demographics', 'population', 'census', 'survey', 'poll',
                             'public opinion', 'majority', 'minority', 'representation',
                             'participation', 'engagement', 'civic', 'democratic process']
        }
    
    def _build_government_framing(self):
        return {
            'left_government': ['government should', 'public investment', 'public service',
                               'public good', 'collective', 'society', 'community',
                               'government programs', 'public programs', 'social programs'],
            'right_government': ['government overreach', 'government intrusion', 'big government',
                                'bureaucracy', 'red tape', 'regulation', 'overregulation',
                                'government waste', 'inefficiency', 'limited government',
                                'small government', 'get government out'],
            'center_government': ['government role', 'appropriate role', 'balance of power',
                                 'checks and balances', 'separation of powers', 'federalism',
                                 'governance', 'policy', 'legislation', 'regulation']
        }
    
    def analyze_bias(self, text: str) -> np.ndarray:
        text_lower = text.lower()
        features = []
        
        left_scores = []
        for category, terms in self.left_indicators.items():
            score = sum(1 for term in terms if term in text_lower)
            left_scores.append(score)
        features.extend(left_scores)
        features.append(sum(left_scores))
        
        right_scores = []
        for category, terms in self.right_indicators.items():
            score = sum(1 for term in terms if term in text_lower)
            right_scores.append(score)
        features.extend(right_scores)
        features.append(sum(right_scores))
        
        center_scores = []
        for category, terms in self.center_indicators.items():
            score = sum(1 for term in terms if term in text_lower)
            center_scores.append(score)
        features.extend(center_scores)
        features.append(sum(center_scores))
        
        economic_framing = []
        for category, terms in self.economic_framing.items():
            score = sum(1 for term in terms if term in text_lower)
            economic_framing.append(score)
        features.extend(economic_framing)
        
        social_framing = []
        for category, terms in self.social_framing.items():
            score = sum(1 for term in terms if term in text_lower)
            social_framing.append(score)
        features.extend(social_framing)
        
        government_framing = []
        for category, terms in self.government_framing.items():
            score = sum(1 for term in terms if term in text_lower)
            government_framing.append(score)
        features.extend(government_framing)
        
        total_left = sum(left_scores)
        total_right = sum(right_scores)
        total_center = sum(center_scores)
        total_indicators = total_left + total_right + total_center
        
        if total_indicators > 0:
            left_ratio = total_left / total_indicators
            right_ratio = total_right / total_indicators
            center_ratio = total_center / total_indicators
            bias_dominance = np.argmax([left_ratio, right_ratio, center_ratio])
        else:
            left_ratio = right_ratio = center_ratio = 0.33
            bias_dominance = 1
        
        features.extend([left_ratio, right_ratio, center_ratio, bias_dominance])
        
        left_right_diff = total_left - total_right
        left_center_diff = total_left - total_center
        right_center_diff = total_right - total_center
        features.extend([left_right_diff, left_center_diff, right_center_diff])
        
        return np.array(features)


class ContextualFeatureEngineer:
    def __init__(self):
        self.political_lexicons = self._build_lexicons()
        self.rhetorical_patterns = self._build_rhetorical_patterns()
        self.bias_analyzer = PoliticalBiasAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.syntactic_extractor = SyntacticFeatureExtractor()
        self.discourse_analyzer = DiscourseMarkerAnalyzer()
        self.argumentation_analyzer = ArgumentationStructureAnalyzer()
        self.entity_extractor = NamedEntityPatternExtractor()
        self.stylometric_analyzer = StylometricAnalyzer()
        self.readability_analyzer = ReadabilityAnalyzer()
        self.cooccurrence_analyzer = WordCooccurrenceAnalyzer()
        self.temporal_extractor = TemporalFeatureExtractor()
        self.citation_extractor = CitationPatternExtractor()
        self.statistical_analyzer = StatisticalDistributionAnalyzer()
        
    def _build_lexicons(self):
        return {
            'left': ['equality', 'justice', 'rights', 'protect', 'ensure', 'affordable', 
                    'healthcare', 'education', 'climate', 'invest', 'progressive', 'workers',
                    'union', 'wage', 'inequality', 'systemic', 'reform', 'social', 'public',
                    'redistribution', 'welfare', 'subsidies', 'regulation', 'environmental',
                    'diversity', 'inclusion', 'equity', 'collective', 'solidarity', 'universal',
                    'medicare', 'medicaid', 'social security', 'public option', 'green new deal',
                    'renewable', 'solar', 'wind', 'carbon', 'emissions', 'climate change',
                    'racial justice', 'lgbtq', 'transgender', 'immigration reform', 'pathway',
                    'citizenship', 'voting rights', 'democracy', 'civil rights', 'human rights',
                    'police reform', 'criminal justice', 'prison reform', 'defund', 'abolish',
                    'student debt', 'cancel debt', 'free college', 'public education',
                    'affirmative action', 'reproductive rights', 'abortion', 'pro choice',
                    'minimum wage', 'living wage', 'workers rights', 'labor', 'strike',
                    'collective bargaining', 'unionize', 'organize', 'mobilize', 'activism',
                    'grassroots', 'movement', 'protest', 'demonstration', 'march', 'rally',
                    'socialism', 'democratic socialism', 'leftist', 'liberal', 'egalitarian',
                    'collectivist', 'communal', 'cooperative', 'mutual aid', 'community',
                    'public good', 'common good', 'shared resources', 'public ownership',
                    'nationalization', 'municipalization', 'worker ownership', 'cooperative',
                    'worker cooperative', 'employee owned', 'public sector', 'government',
                    'federal', 'state', 'local', 'municipal', 'public service', 'civil service',
                    'public works', 'infrastructure', 'public transit', 'public transportation',
                    'affordable housing', 'public housing', 'social housing', 'rent control',
                    'tenant rights', 'eviction moratorium', 'homelessness', 'housing first',
                    'food security', 'food stamps', 'snap', 'wic', 'public assistance',
                    'welfare state', 'social safety net', 'safety net', 'social programs',
                    'public programs', 'government programs', 'entitlements', 'benefits',
                    'unemployment benefits', 'disability benefits', 'social security benefits',
                    'medicare for all', 'single payer', 'universal healthcare', 'public health',
                    'healthcare as right', 'healthcare access', 'preventive care', 'mental health',
                    'substance abuse treatment', 'addiction treatment', 'harm reduction',
                    'reproductive healthcare', 'planned parenthood', 'contraception', 'birth control',
                    'sex education', 'comprehensive sex ed', 'lgbtq rights', 'gay rights',
                    'lesbian rights', 'bisexual rights', 'trans rights', 'nonbinary rights',
                    'gender identity', 'gender expression', 'pronouns', 'inclusive', 'acceptance',
                    'racial equality', 'racial equity', 'anti racism', 'antiracist', 'black lives matter',
                    'blm', 'indigenous rights', 'native rights', 'tribal sovereignty', 'decolonization',
                    'immigration rights', 'refugee rights', 'asylum', 'sanctuary', 'sanctuary city',
                    'pathway to citizenship', 'dreamers', 'daca', 'comprehensive immigration reform',
                    'voting rights', 'voter rights', 'voter access', 'voter registration', 'early voting',
                    'mail in voting', 'absentee voting', 'voter suppression', 'gerrymandering',
                    'campaign finance reform', 'money out of politics', 'citizens united',
                    'dark money', 'super pac', 'lobbying reform', 'ethics reform', 'transparency',
                    'criminal justice reform', 'sentencing reform', 'bail reform', 'pretrial detention',
                    'mass incarceration', 'prison abolition', 'defund police', 'police accountability',
                    'body cameras', 'police oversight', 'community policing', 'restorative justice',
                    'rehabilitation', 'reentry', 'second chance', 'expungement', 'clemency',
                    'death penalty abolition', 'capital punishment abolition', 'life without parole',
                    'gun control', 'gun safety', 'background checks', 'assault weapons ban',
                    'gun violence prevention', 'domestic violence', 'red flag laws', 'gun registry',
                    'climate action', 'climate crisis', 'climate emergency', 'global warming',
                    'greenhouse gases', 'carbon neutral', 'carbon negative', 'net zero',
                    'renewable energy', 'clean energy', 'solar power', 'wind power', 'hydroelectric',
                    'geothermal', 'nuclear phaseout', 'fossil fuel', 'oil', 'gas', 'coal',
                    'environmental protection', 'epa', 'clean air act', 'clean water act',
                    'endangered species', 'biodiversity', 'conservation', 'public lands',
                    'national parks', 'wilderness', 'wildlife protection', 'ocean protection',
                    'plastic pollution', 'single use plastic', 'recycling', 'composting',
                    'sustainable', 'sustainability', 'circular economy', 'green jobs',
                    'green economy', 'just transition', 'environmental justice', 'climate justice',
                    'food justice', 'water justice', 'energy justice', 'transportation justice',
                    'labor rights', 'workers rights', 'employee rights', 'workplace safety',
                    'osha', 'workers compensation', 'unemployment insurance', 'paid leave',
                    'paid sick leave', 'paid family leave', 'maternity leave', 'paternity leave',
                    'flexible work', 'work life balance', 'overtime pay', 'fair wages',
                    'equal pay', 'pay equity', 'gender pay gap', 'racial pay gap',
                    'discrimination', 'harassment', 'hostile work environment', 'retaliation',
                    'whistleblower protection', 'union rights', 'right to organize', 'strike',
                    'collective action', 'solidarity', 'mutual support', 'community organizing',
                    'education equity', 'education access', 'public schools', 'public education',
                    'free college', 'tuition free', 'debt free', 'student loan forgiveness',
                    'cancel student debt', 'pell grants', 'financial aid', 'scholarships',
                    'early childhood education', 'pre k', 'head start', 'childcare', 'daycare',
                    'after school programs', 'summer programs', 'arts education', 'music education',
                    'stem education', 'vocational training', 'apprenticeships', 'job training',
                    'economic justice', 'economic equality', 'wealth inequality', 'income inequality',
                    'poverty', 'homelessness', 'food insecurity', 'housing insecurity',
                    'living wage', 'minimum wage increase', 'wage theft', 'gig economy',
                    'precarious work', 'contract workers', 'temp workers', 'part time',
                    'full time', 'benefits', 'health insurance', 'retirement', 'pensions',
                    'social security', 'medicare', 'medicaid', 'chip', 'aca', 'obamacare',
                    'tax the rich', 'wealth tax', 'millionaire tax', 'billionaire tax',
                    'estate tax', 'inheritance tax', 'capital gains tax', 'corporate tax',
                    'tax fairness', 'progressive taxation', 'marginal tax rate', 'tax brackets',
                    'corporate accountability', 'corporate responsibility', 'corporate taxes',
                    'antitrust', 'break up big tech', 'break up monopolies', 'competition',
                    'consumer protection', 'cfpb', 'sec', 'fcc', 'fda', 'regulatory agencies',
                    'financial regulation', 'dodd frank', 'glass steagall', 'banking reform',
                    'consumer financial protection', 'predatory lending', 'payday loans',
                    'student loans', 'credit cards', 'debt', 'bankruptcy', 'foreclosure',
                    'social programs', 'public services', 'government services', 'civil service',
                    'public sector jobs', 'government jobs', 'federal jobs', 'state jobs',
                    'local jobs', 'infrastructure jobs', 'green jobs', 'clean energy jobs',
                    'renewable energy jobs', 'solar jobs', 'wind jobs', 'energy efficiency',
                    'retrofitting', 'weatherization', 'public transit jobs', 'transit workers',
                    'healthcare workers', 'nurses', 'doctors', 'teachers', 'educators',
                    'social workers', 'mental health workers', 'home health aides', 'caregivers',
                    'childcare workers', 'eldercare workers', 'domestic workers', 'farm workers',
                    'migrant workers', 'undocumented workers', 'immigrant workers', 'refugees',
                    'asylum seekers', 'displaced persons', 'internally displaced', 'climate refugees',
                    'humanitarian', 'humanitarian aid', 'foreign aid', 'development aid',
                    'peace', 'diplomacy', 'multilateralism', 'international cooperation',
                    'united nations', 'nato', 'alliances', 'treaties', 'arms control',
                    'nuclear disarmament', 'nuclear weapons', 'nuclear ban', 'npt',
                    'human rights', 'civil rights', 'civil liberties', 'constitutional rights',
                    'first amendment', 'freedom of speech', 'freedom of press', 'freedom of assembly',
                    'freedom of religion', 'separation of church and state', 'establishment clause',
                    'fourth amendment', 'privacy', 'surveillance', 'mass surveillance',
                    'fifth amendment', 'due process', 'equal protection', 'fourteenth amendment',
                    'voting rights act', 'civil rights act', 'ada', 'americans with disabilities act',
                    'disability rights', 'accessibility', 'accommodations', 'inclusion',
                    'diversity', 'representation', 'representation matters', 'visibility',
                    'lgbtq representation', 'racial representation', 'gender representation',
                    'diverse voices', 'marginalized voices', 'amplify', 'center', 'platform',
                    'media representation', 'cultural representation', 'historical representation',
                    'decolonize', 'decolonization', 'indigenous', 'native', 'tribal',
                    'sovereignty', 'self determination', 'autonomy', 'independence',
                    'liberation', 'emancipation', 'freedom', 'liberty', 'equality',
                    'fraternity', 'solidarity', 'unity', 'together', 'collective action',
                    'movement', 'grassroots', 'organizing', 'mobilizing', 'activism',
                    'advocacy', 'lobbying', 'petitioning', 'protesting', 'demonstrating',
                    'marching', 'rallying', 'striking', 'boycotting', 'divesting',
                    'social movement', 'civil rights movement', 'labor movement', 'environmental movement',
                    'feminist movement', 'lgbtq movement', 'disability rights movement',
                    'immigrant rights movement', 'indigenous rights movement', 'black power',
                    'brown power', 'yellow power', 'red power', 'pink power', 'rainbow power',
                    'intersectionality', 'intersectional', 'interconnected', 'interdependent',
                    'systemic change', 'structural change', 'transformative change', 'revolutionary',
                    'revolution', 'evolution', 'progress', 'progressive', 'forward', 'ahead',
                    'future', 'next generation', 'youth', 'students', 'young people',
                    'millennials', 'gen z', 'generation z', 'future generations', 'legacy',
                    'heritage', 'history', 'historical', 'memorial', 'monument', 'memorialize',
                    'remember', 'honor', 'commemorate', 'celebrate', 'recognize', 'acknowledge',
                    'truth', 'truth telling', 'reconciliation', 'healing', 'justice',
                    'restorative justice', 'transformative justice', 'community justice',
                    'healing circles', 'peace circles', 'dialogue', 'conversation', 'listening',
                    'understanding', 'empathy', 'compassion', 'kindness', 'care', 'caring',
                    'love', 'love thy neighbor', 'neighbor', 'community', 'neighborhood',
                    'local', 'place', 'home', 'belonging', 'connection', 'relationships',
                    'family', 'chosen family', 'extended family', 'community family',
                    'support', 'support system', 'safety net', 'social safety net',
                    'mutual aid', 'mutual support', 'cooperation', 'collaboration',
                    'partnership', 'alliance', 'coalition', 'network', 'web', 'fabric',
                    'society', 'social', 'public', 'common', 'shared', 'collective',
                    'communal', 'cooperative', 'collaborative', 'participatory', 'democratic',
                    'democracy', 'democratic process', 'civic engagement', 'civic participation',
                    'voting', 'elections', 'candidates', 'campaigns', 'platforms', 'policies',
                    'legislation', 'laws', 'bills', 'acts', 'statutes', 'regulations',
                    'rules', 'guidelines', 'standards', 'norms', 'values', 'principles',
                    'ethics', 'morals', 'integrity', 'honesty', 'transparency', 'accountability',
                    'responsibility', 'duty', 'obligation', 'service', 'public service',
                    'civil service', 'government service', 'military service', 'national service',
                    'community service', 'volunteer', 'volunteering', 'giving', 'charity',
                    'philanthropy', 'nonprofit', 'ngo', 'foundation', 'organization',
                    'institution', 'establishment', 'system', 'structure', 'framework',
                    'infrastructure', 'foundation', 'base', 'basis', 'ground', 'groundwork',
                    'building', 'construction', 'development', 'growth', 'expansion',
                    'improvement', 'enhancement', 'betterment', 'advancement', 'progress',
                    'evolution', 'transformation', 'change', 'reform', 'reformation',
                    'revolution', 'revolutionary', 'radical', 'extreme', 'drastic',
                    'sweeping', 'comprehensive', 'thorough', 'complete', 'total', 'full',
                    'entire', 'whole', 'all', 'every', 'each', 'universal', 'global',
                    'worldwide', 'international', 'multinational', 'transnational',
                    'cosmopolitan', 'globalist', 'internationalist', 'humanist', 'humanitarian',
                    'altruistic', 'selfless', 'generous', 'giving', 'sharing', 'caring',
                    'compassionate', 'empathetic', 'sympathetic', 'understanding', 'tolerant',
                    'accepting', 'inclusive', 'welcoming', 'open', 'embracing', 'celebrating',
                    'diversity', 'difference', 'variety', 'variation', 'multiplicity',
                    'plurality', 'pluralism', 'multiculturalism', 'cosmopolitanism',
                    'globalism', 'internationalism', 'humanism', 'humanitarianism',
                    'universalism', 'egalitarianism', 'collectivism', 'socialism',
                    'democratic socialism', 'social democracy', 'welfare state',
                    'mixed economy', 'regulated capitalism', 'stakeholder capitalism',
                    'responsible capitalism', 'conscious capitalism', 'sustainable capitalism',
                    'green capitalism', 'circular economy', 'sharing economy', 'gig economy',
                    'platform economy', 'digital economy', 'knowledge economy', 'creative economy',
                    'care economy', 'care work', 'reproductive labor', 'domestic labor',
                    'emotional labor', 'invisible labor', 'unpaid labor', 'underpaid labor',
                    'exploited labor', 'alienated labor', 'meaningful work', 'fulfilling work',
                    'purpose', 'meaning', 'significance', 'value', 'worth', 'dignity',
                    'respect', 'honor', 'recognition', 'appreciation', 'gratitude',
                    'thanks', 'thank you', 'please', 'sorry', 'apology', 'forgiveness',
                    'reconciliation', 'healing', 'recovery', 'restoration', 'renewal',
                    'regeneration', 'revitalization', 'renaissance', 'rebirth', 'awakening',
                    'enlightenment', 'illumination', 'clarity', 'vision', 'foresight',
                    'foreseeing', 'predicting', 'anticipating', 'preparing', 'planning',
                    'strategizing', 'organizing', 'coordinating', 'orchestrating', 'leading',
                    'guiding', 'directing', 'managing', 'administering', 'governing',
                    'ruling', 'regulating', 'controlling', 'overseeing', 'supervising',
                    'monitoring', 'watching', 'observing', 'noticing', 'attending',
                    'paying attention', 'listening', 'hearing', 'understanding', 'comprehending',
                    'grasping', 'getting', 'seeing', 'perceiving', 'recognizing',
                    'acknowledging', 'admitting', 'accepting', 'embracing', 'welcoming',
                    'celebrating', 'honoring', 'respecting', 'valuing', 'appreciating',
                    'cherishing', 'treasuring', 'loving', 'caring', 'nurturing', 'supporting',
                    'helping', 'assisting', 'aiding', 'facilitating', 'enabling',
                    'empowering', 'strengthening', 'building', 'constructing', 'creating',
                    'making', 'forming', 'shaping', 'molding', 'sculpting', 'crafting',
                    'designing', 'planning', 'developing', 'growing', 'expanding',
                    'extending', 'stretching', 'reaching', 'striving', 'aspiring',
                    'aiming', 'targeting', 'focusing', 'concentrating', 'centering',
                    'grounding', 'rooting', 'anchoring', 'stabilizing', 'securing',
                    'protecting', 'defending', 'guarding', 'shielding', 'sheltering',
                    'harboring', 'housing', 'accommodating', 'hosting', 'welcoming',
                    'inviting', 'including', 'incorporating', 'integrating', 'unifying',
                    'uniting', 'joining', 'connecting', 'linking', 'bridging',
                    'spanning', 'crossing', 'traversing', 'navigating', 'journeying',
                    'traveling', 'exploring', 'discovering', 'finding', 'locating',
                    'situating', 'placing', 'positioning', 'orienting', 'directing',
                    'guiding', 'leading', 'showing', 'demonstrating', 'illustrating',
                    'exemplifying', 'modeling', 'representing', 'symbolizing', 'signifying',
                    'meaning', 'significance', 'importance', 'relevance', 'pertinence',
                    'applicability', 'usefulness', 'utility', 'value', 'worth', 'merit',
                    'excellence', 'quality', 'superiority', 'supremacy', 'dominance',
                    'leadership', 'guidance', 'direction', 'instruction', 'teaching',
                    'education', 'learning', 'knowledge', 'wisdom', 'insight', 'understanding',
                    'comprehension', 'grasp', 'mastery', 'proficiency', 'competence',
                    'capability', 'ability', 'capacity', 'potential', 'possibility',
                    'opportunity', 'chance', 'prospect', 'hope', 'promise', 'future',
                    'destiny', 'fate', 'fortune', 'luck', 'fortune', 'blessing',
                    'gift', 'present', 'offering', 'contribution', 'donation', 'charity',
                    'philanthropy', 'generosity', 'kindness', 'goodness', 'virtue',
                    'morality', 'ethics', 'principles', 'values', 'beliefs', 'convictions',
                    'commitments', 'dedications', 'devotions', 'loyalties', 'allegiances',
                    'allegiance', 'loyalty', 'fidelity', 'faithfulness', 'constancy',
                    'steadfastness', 'perseverance', 'persistence', 'tenacity', 'determination',
                    'resolve', 'resolution', 'will', 'willpower', 'strength', 'power',
                    'force', 'energy', 'vitality', 'vigor', 'dynamism', 'momentum',
                    'movement', 'motion', 'action', 'activity', 'engagement', 'participation',
                    'involvement', 'commitment', 'dedication', 'devotion', 'passion',
                    'enthusiasm', 'zeal', 'ardor', 'fervor', 'intensity', 'depth',
                    'profundity', 'depth', 'substance', 'essence', 'core', 'heart',
                    'soul', 'spirit', 'being', 'existence', 'life', 'living', 'alive',
                    'vibrant', 'dynamic', 'active', 'energetic', 'powerful', 'strong',
                    'resilient', 'enduring', 'lasting', 'permanent', 'eternal', 'timeless',
                    'classic', 'enduring', 'evergreen', 'perennial', 'constant', 'steady',
                    'stable', 'secure', 'safe', 'protected', 'defended', 'guarded',
                    'sheltered', 'harbored', 'housed', 'accommodated', 'hosted', 'welcomed',
                    'invited', 'included', 'incorporated', 'integrated', 'unified', 'united',
                    'joined', 'connected', 'linked', 'bridged', 'spanned', 'crossed',
                    'traversed', 'navigated', 'journeyed', 'traveled', 'explored', 'discovered',
                    'found', 'located', 'situated', 'placed', 'positioned', 'oriented',
                    'directed', 'guided', 'led', 'shown', 'demonstrated', 'illustrated',
                    'exemplified', 'modeled', 'represented', 'symbolized', 'signified'],
            'right': ['freedom', 'liberty', 'constitution', 'traditional', 'sovereignty',
                     'border', 'fundamental', 'values', 'lower taxes', 'free market', 'fiscal',
                     'responsibility', 'individual', 'private', 'heritage', 'patriotism',
                     'deregulation', 'competition', 'merit', 'self-reliance', 'conservative',
                     'national', 'defense', 'security', 'law and order', 'military', 'troops',
                     'veterans', 'second amendment', 'gun rights', 'right to bear arms', 'nra',
                     'pro life', 'pro-life', 'abortion', 'religious freedom', 'faith', 'religion',
                     'christian', 'bible', 'god', 'prayer', 'family values', 'marriage',
                     'traditional marriage', 'immigration enforcement', 'illegal immigration',
                     'border security', 'wall', 'deport', 'deportation', 'merit based',
                     'meritocracy', 'individual responsibility', 'personal responsibility',
                     'work ethic', 'bootstrap', 'pull yourself up', 'self made', 'entrepreneur',
                     'small business', 'job creators', 'wealth creators', 'capitalism',
                     'free enterprise', 'supply side', 'trickle down', 'tax cuts',
                     'reduce spending', 'cut spending', 'austerity', 'balanced budget',
                     'deficit reduction', 'entitlement reform', 'privatize', 'privatization',
                     'states rights', 'federalism', 'limited government', 'small government',
                     'constitutional rights', 'originalist', 'textualist', 'strict construction',
                     'american exceptionalism', 'nationalism', 'isolationist', 'protectionist',
                     'republican', 'gop', 'conservatism', 'libertarian', 'libertarianism',
                     'classical liberal', 'fiscal conservative', 'social conservative',
                     'religious conservative', 'evangelical', 'protestant', 'catholic',
                     'judeo christian', 'biblical', 'scripture', 'gospel', 'jesus', 'christ',
                     'savior', 'salvation', 'redemption', 'sin', 'virtue', 'morality',
                     'moral values', 'ethical', 'righteous', 'godly', 'holy', 'sacred',
                     'sanctity', 'sanctity of life', 'pro life', 'unborn', 'fetus', 'baby',
                     'abortion ban', 'roe v wade', 'pro life movement', 'right to life',
                     'traditional family', 'nuclear family', 'father', 'mother', 'husband',
                     'wife', 'children', 'parenting', 'fatherhood', 'motherhood', 'stay at home',
                     'homemaker', 'breadwinner', 'provider', 'protector', 'head of household',
                     'patriarch', 'patriarchy', 'masculinity', 'femininity', 'gender roles',
                     'complementary', 'natural order', 'divine order', 'god given',
                     'sanctity of marriage', 'one man one woman', 'defense of marriage',
                     'marriage protection', 'traditional values', 'family values', 'moral values',
                     'christian values', 'biblical values', 'judeo christian values',
                     'western values', 'american values', 'founding fathers', 'founding principles',
                     'constitutional originalism', 'strict constructionism', 'textualism',
                     'original intent', 'framer intent', 'constitutionalist', 'constitutionalism',
                     'bill of rights', 'first amendment', 'religious liberty', 'free exercise',
                     'establishment clause', 'separation of church and state', 'religious expression',
                     'freedom of religion', 'freedom of conscience', 'religious accommodation',
                     'religious exemption', 'conscience clause', 'religious objection',
                     'second amendment rights', 'right to keep and bear arms', 'gun ownership',
                     'firearm rights', 'concealed carry', 'open carry', 'stand your ground',
                     'castle doctrine', 'self defense', 'defensive gun use', 'gun culture',
                     'sportsman', 'hunter', 'hunting', 'sport shooting', 'target practice',
                     'gun safety', 'responsible gun ownership', 'law abiding gun owners',
                     'nra', 'national rifle association', 'gun lobby', 'pro gun', 'gun rights',
                     'border wall', 'southern border', 'northern border', 'border patrol',
                     'ice', 'immigration and customs enforcement', 'customs', 'border agents',
                     'illegal aliens', 'undocumented immigrants', 'illegal immigrants',
                     'chain migration', 'family reunification', 'diversity visa', 'lottery',
                     'merit based immigration', 'points system', 'skilled workers', 'english requirement',
                     'assimilation', 'americanization', 'melting pot', 'e pluribus unum',
                     'english only', 'official language', 'language requirement', 'citizenship test',
                     'naturalization', 'citizenship', 'patriotism', 'allegiance', 'pledge',
                     'flag', 'national anthem', 'veterans day', 'memorial day', 'independence day',
                     'fourth of july', 'thanksgiving', 'christmas', 'religious holidays',
                     'national security', 'homeland security', 'defense spending', 'military budget',
                     'defense contractors', 'military industrial complex', 'veterans affairs',
                     'va', 'veterans benefits', 'military service', 'service members',
                     'active duty', 'reserves', 'national guard', 'armed forces', 'navy',
                     'army', 'air force', 'marines', 'coast guard', 'special forces',
                     'seals', 'rangers', 'marines', 'infantry', 'combat', 'warrior',
                     'hero', 'sacrifice', 'service', 'duty', 'honor', 'country',
                     'law enforcement', 'police', 'sheriff', 'deputy', 'trooper', 'officer',
                     'blue lives matter', 'back the blue', 'thin blue line', 'police support',
                     'law and order', 'tough on crime', 'zero tolerance', 'three strikes',
                     'mandatory minimum', 'truth in sentencing', 'death penalty', 'capital punishment',
                     'lethal injection', 'electric chair', 'gas chamber', 'firing squad',
                     'eye for an eye', 'just punishment', 'retribution', 'deterrence',
                     'incapacitation', 'public safety', 'victims rights', 'victim impact',
                     'tough sentencing', 'life without parole', 'supermax', 'maximum security',
                     'prison', 'corrections', 'rehabilitation', 'reform', 'discipline',
                     'punishment', 'accountability', 'responsibility', 'consequences', 'actions',
                     'personal responsibility', 'individual responsibility', 'self responsibility',
                     'accountability', 'consequences', 'reap what you sow', 'you made your bed',
                     'pull yourself up', 'bootstrap', 'self made', 'self sufficient',
                     'independent', 'autonomous', 'self reliant', 'self supporting',
                     'entrepreneur', 'business owner', 'small business', 'startup', 'innovation',
                     'risk taking', 'investment', 'capital', 'equity', 'ownership', 'stake',
                     'shareholder', 'stockholder', 'investor', 'capitalist', 'free market',
                     'market economy', 'market forces', 'supply and demand', 'invisible hand',
                     'laissez faire', 'hands off', 'government out', 'less regulation',
                     'deregulation', 'regulatory reform', 'regulatory relief', 'red tape',
                     'bureaucracy', 'bureaucratic', 'government waste', 'inefficiency',
                     'privatization', 'privatize', 'contract out', 'outsource', 'competitive',
                     'competition', 'market competition', 'free competition', 'fair competition',
                     'level playing field', 'equal opportunity', 'merit based', 'meritocracy',
                     'achievement', 'excellence', 'performance', 'results', 'outcomes',
                     'productivity', 'efficiency', 'innovation', 'creativity', 'ingenuity',
                     'ingenuity', 'resourcefulness', 'initiative', 'enterprise', 'drive',
                     'ambition', 'aspiration', 'goals', 'objectives', 'targets', 'milestones',
                     'success', 'achievement', 'accomplishment', 'attainment', 'realization',
                     'fulfillment', 'satisfaction', 'pride', 'dignity', 'self respect',
                     'self worth', 'self esteem', 'confidence', 'self confidence', 'assurance',
                     'certainty', 'conviction', 'belief', 'faith', 'trust', 'hope',
                     'optimism', 'positive', 'can do', 'can do attitude', 'can do spirit',
                     'american dream', 'rags to riches', 'log cabin to white house',
                     'opportunity', 'land of opportunity', 'equal opportunity', 'fair chance',
                     'level playing field', 'merit', 'talent', 'ability', 'skill', 'expertise',
                     'knowledge', 'education', 'learning', 'training', 'development',
                     'improvement', 'growth', 'advancement', 'progress', 'upward mobility',
                     'social mobility', 'economic mobility', 'class mobility', 'mobility',
                     'prosperity', 'wealth', 'affluence', 'success', 'achievement',
                     'accomplishment', 'attainment', 'realization', 'fulfillment',
                     'satisfaction', 'happiness', 'joy', 'contentment', 'peace',
                     'tranquility', 'serenity', 'calm', 'quiet', 'stillness', 'rest',
                     'relaxation', 'leisure', 'recreation', 'entertainment', 'enjoyment',
                     'pleasure', 'delight', 'gratification', 'indulgence', 'luxury',
                     'comfort', 'ease', 'convenience', 'facility', 'simplicity',
                     'straightforward', 'direct', 'clear', 'plain', 'simple', 'basic',
                     'fundamental', 'essential', 'core', 'central', 'primary', 'main',
                     'principal', 'chief', 'leading', 'foremost', 'preeminent', 'supreme',
                     'paramount', 'dominant', 'predominant', 'prevailing', 'current',
                     'present', 'existing', 'established', 'traditional', 'conventional',
                     'orthodox', 'standard', 'normal', 'regular', 'usual', 'typical',
                     'common', 'ordinary', 'everyday', 'routine', 'habitual', 'customary',
                     'accustomed', 'familiar', 'known', 'recognized', 'acknowledged',
                     'accepted', 'approved', 'endorsed', 'supported', 'backed', 'sponsored',
                     'funded', 'financed', 'paid', 'compensated', 'rewarded', 'recognized',
                     'honored', 'respected', 'admired', 'esteemed', 'valued', 'appreciated',
                     'cherished', 'treasured', 'prized', 'beloved', 'dear', 'precious',
                     'valuable', 'important', 'significant', 'meaningful', 'substantial',
                     'considerable', 'notable', 'remarkable', 'extraordinary', 'exceptional',
                     'outstanding', 'excellent', 'superior', 'supreme', 'paramount',
                     'preeminent', 'foremost', 'leading', 'top', 'best', 'finest',
                     'greatest', 'highest', 'utmost', 'maximum', 'peak', 'pinnacle',
                     'summit', 'apex', 'zenith', 'climax', 'culmination', 'acme',
                     'perfection', 'ideal', 'model', 'exemplar', 'paradigm', 'standard',
                     'benchmark', 'criterion', 'measure', 'gauge', 'yardstick', 'barometer',
                     'indicator', 'sign', 'signal', 'mark', 'token', 'symbol',
                     'emblem', 'badge', 'insignia', 'crest', 'coat of arms', 'seal',
                     'stamp', 'imprint', 'impression', 'mark', 'trace', 'vestige',
                     'remnant', 'relic', 'artifact', 'monument', 'memorial', 'tribute',
                     'honor', 'recognition', 'acknowledgment', 'appreciation', 'gratitude',
                     'thanks', 'thankfulness', 'indebtedness', 'obligation', 'duty',
                     'responsibility', 'accountability', 'liability', 'obligation',
                     'commitment', 'dedication', 'devotion', 'loyalty', 'allegiance',
                     'fidelity', 'faithfulness', 'constancy', 'steadfastness', 'perseverance',
                     'persistence', 'tenacity', 'determination', 'resolve', 'resolution',
                     'will', 'willpower', 'strength', 'power', 'force', 'energy',
                     'vitality', 'vigor', 'dynamism', 'momentum', 'movement', 'motion',
                     'action', 'activity', 'engagement', 'participation', 'involvement',
                     'commitment', 'dedication', 'devotion', 'passion', 'enthusiasm',
                     'zeal', 'ardor', 'fervor', 'intensity', 'depth', 'profundity',
                     'substance', 'essence', 'core', 'heart', 'soul', 'spirit',
                     'being', 'existence', 'life', 'living', 'alive', 'vibrant',
                     'dynamic', 'active', 'energetic', 'powerful', 'strong', 'resilient',
                     'enduring', 'lasting', 'permanent', 'eternal', 'timeless', 'classic',
                     'enduring', 'evergreen', 'perennial', 'constant', 'steady', 'stable',
                     'secure', 'safe', 'protected', 'defended', 'guarded', 'sheltered',
                     'harbored', 'housed', 'accommodated', 'hosted', 'welcomed', 'invited',
                     'included', 'incorporated', 'integrated', 'unified', 'united', 'joined',
                     'connected', 'linked', 'bridged', 'spanned', 'crossed', 'traversed',
                     'navigated', 'journeyed', 'traveled', 'explored', 'discovered', 'found',
                     'located', 'situated', 'placed', 'positioned', 'oriented', 'directed',
                     'guided', 'led', 'shown', 'demonstrated', 'illustrated', 'exemplified',
                     'modeled', 'represented', 'symbolized', 'signified', 'meant', 'signified',
                     'denoted', 'indicated', 'suggested', 'implied', 'hinted', 'alluded',
                     'referred', 'pointed', 'directed', 'guided', 'led', 'conducted',
                     'managed', 'administered', 'governed', 'ruled', 'regulated', 'controlled',
                     'oversaw', 'supervised', 'monitored', 'watched', 'observed', 'noticed',
                     'attended', 'paid attention', 'listened', 'heard', 'understood', 'comprehended',
                     'grasped', 'got', 'saw', 'perceived', 'recognized', 'acknowledged',
                     'admitted', 'accepted', 'embraced', 'welcomed', 'celebrated', 'honored',
                     'respected', 'valued', 'appreciated', 'cherished', 'treasured', 'loved',
                     'cared', 'nurtured', 'supported', 'helped', 'assisted', 'aided',
                     'facilitated', 'enabled', 'empowered', 'strengthened', 'built', 'constructed',
                     'created', 'made', 'formed', 'shaped', 'molded', 'sculpted', 'crafted',
                     'designed', 'planned', 'developed', 'grew', 'expanded', 'extended',
                     'stretched', 'reached', 'strived', 'aspired', 'aimed', 'targeted',
                     'focused', 'concentrated', 'centered', 'grounded', 'rooted', 'anchored',
                     'stabilized', 'secured', 'protected', 'defended', 'guarded', 'shielded',
                     'sheltered', 'harbored', 'housed', 'accommodated', 'hosted', 'welcomed',
                     'invited', 'included', 'incorporated', 'integrated', 'unified', 'united',
                     'joined', 'connected', 'linked', 'bridged', 'spanned', 'crossed',
                     'traversed', 'navigated', 'journeyed', 'traveled', 'explored', 'discovered',
                     'found', 'located', 'situated', 'placed', 'positioned', 'oriented',
                     'directed', 'guided', 'led', 'shown', 'demonstrated', 'illustrated',
                     'exemplified', 'modeled', 'represented', 'symbolized', 'signified'],
            'center': ['according', 'data', 'study', 'research', 'percent', 'report',
                       'analysis', 'indicate', 'suggests', 'both', 'various', 'evidence',
                       'balance', 'moderate', 'pragmatic', 'bipartisan', 'compromise',
                       'consensus', 'dialogue', 'discussion', 'evaluation', 'assessment',
                       'empirical', 'evidence based', 'data driven', 'fact based', 'objective',
                       'neutral', 'unbiased', 'impartial', 'fair', 'balanced', 'nuanced',
                       'complex', 'multifaceted', 'consider', 'examine', 'evaluate', 'assess',
                       'weigh', 'trade off', 'cost benefit', 'pros and cons', 'advantages',
                       'disadvantages', 'on one hand', 'on the other hand', 'however',
                       'nevertheless', 'although', 'while', 'whereas', 'middle ground',
                       'common ground', 'cross party', 'centrist', 'moderate', 'practical',
                       'realistic', 'feasible', 'viable', 'sustainable', 'long term',
                       'short term', 'immediate', 'gradual', 'incremental', 'step by step',
                       'phased', 'pilot', 'test', 'trial', 'experiment', 'study', 'research',
                       'survey', 'poll', 'census', 'demographics', 'statistics', 'statistical',
                       'gdp', 'unemployment', 'inflation', 'federal reserve', 'monetary policy',
                       'fiscal policy', 'budget', 'deficit', 'surplus', 'debt', 'revenue',
                       'expenditure', 'oversight', 'accountability', 'transparency', 'audit',
                       'review', 'hearing', 'committee', 'congress', 'senate', 'house',
                       'legislative', 'regulatory', 'governance', 'policy', 'legislation',
                       'independent', 'nonpartisan', 'apolitical', 'unaffiliated', 'swing',
                       'moderate', 'centrist', 'middle of the road', 'mainstream', 'establishment',
                       'institutional', 'conventional', 'traditional', 'orthodox', 'standard',
                       'normal', 'regular', 'typical', 'common', 'ordinary', 'everyday',
                       'routine', 'habitual', 'customary', 'accustomed', 'familiar', 'known',
                       'recognized', 'acknowledged', 'accepted', 'approved', 'endorsed',
                       'supported', 'backed', 'sponsored', 'funded', 'financed', 'paid',
                       'compensated', 'rewarded', 'recognized', 'honored', 'respected',
                       'admired', 'esteemed', 'valued', 'appreciated', 'cherished', 'treasured',
                       'prized', 'beloved', 'dear', 'precious', 'valuable', 'important',
                       'significant', 'meaningful', 'substantial', 'considerable', 'notable',
                       'remarkable', 'extraordinary', 'exceptional', 'outstanding', 'excellent',
                       'superior', 'supreme', 'paramount', 'preeminent', 'foremost', 'leading',
                       'top', 'best', 'finest', 'greatest', 'highest', 'utmost', 'maximum',
                       'peak', 'pinnacle', 'summit', 'apex', 'zenith', 'climax', 'culmination',
                       'acme', 'perfection', 'ideal', 'model', 'exemplar', 'paradigm',
                       'standard', 'benchmark', 'criterion', 'measure', 'gauge', 'yardstick',
                       'barometer', 'indicator', 'sign', 'signal', 'mark', 'token', 'symbol',
                       'emblem', 'badge', 'insignia', 'crest', 'coat of arms', 'seal',
                       'stamp', 'imprint', 'impression', 'mark', 'trace', 'vestige',
                       'remnant', 'relic', 'artifact', 'monument', 'memorial', 'tribute',
                       'honor', 'recognition', 'acknowledgment', 'appreciation', 'gratitude',
                       'thanks', 'thankfulness', 'indebtedness', 'obligation', 'duty',
                       'responsibility', 'accountability', 'liability', 'obligation',
                       'commitment', 'dedication', 'devotion', 'loyalty', 'allegiance',
                       'fidelity', 'faithfulness', 'constancy', 'steadfastness', 'perseverance',
                       'persistence', 'tenacity', 'determination', 'resolve', 'resolution',
                       'will', 'willpower', 'strength', 'power', 'force', 'energy',
                       'vitality', 'vigor', 'dynamism', 'momentum', 'movement', 'motion',
                       'action', 'activity', 'engagement', 'participation', 'involvement',
                       'commitment', 'dedication', 'devotion', 'passion', 'enthusiasm',
                       'zeal', 'ardor', 'fervor', 'intensity', 'depth', 'profundity',
                       'substance', 'essence', 'core', 'heart', 'soul', 'spirit',
                       'being', 'existence', 'life', 'living', 'alive', 'vibrant',
                       'dynamic', 'active', 'energetic', 'powerful', 'strong', 'resilient',
                       'enduring', 'lasting', 'permanent', 'eternal', 'timeless', 'classic',
                       'enduring', 'evergreen', 'perennial', 'constant', 'steady', 'stable',
                       'secure', 'safe', 'protected', 'defended', 'guarded', 'sheltered',
                       'harbored', 'housed', 'accommodated', 'hosted', 'welcomed', 'invited',
                       'included', 'incorporated', 'integrated', 'unified', 'united', 'joined',
                       'connected', 'linked', 'bridged', 'spanned', 'crossed', 'traversed',
                       'navigated', 'journeyed', 'traveled', 'explored', 'discovered', 'found',
                       'located', 'situated', 'placed', 'positioned', 'oriented', 'directed',
                       'guided', 'led', 'shown', 'demonstrated', 'illustrated', 'exemplified',
                       'modeled', 'represented', 'symbolized', 'signified', 'meant', 'signified',
                       'denoted', 'indicated', 'suggested', 'implied', 'hinted', 'alluded',
                       'referred', 'pointed', 'directed', 'guided', 'led', 'conducted',
                       'managed', 'administered', 'governed', 'ruled', 'regulated', 'controlled',
                       'oversaw', 'supervised', 'monitored', 'watched', 'observed', 'noticed',
                       'attended', 'paid attention', 'listened', 'heard', 'understood', 'comprehended',
                       'grasped', 'got', 'saw', 'perceived', 'recognized', 'acknowledged',
                       'admitted', 'accepted', 'embraced', 'welcomed', 'celebrated', 'honored',
                       'respected', 'valued', 'appreciated', 'cherished', 'treasured', 'loved',
                       'cared', 'nurtured', 'supported', 'helped', 'assisted', 'aided',
                       'facilitated', 'enabled', 'empowered', 'strengthened', 'built', 'constructed',
                       'created', 'made', 'formed', 'shaped', 'molded', 'sculpted', 'crafted',
                       'designed', 'planned', 'developed', 'grew', 'expanded', 'extended',
                       'stretched', 'reached', 'strived', 'aspired', 'aimed', 'targeted',
                       'focused', 'concentrated', 'centered', 'grounded', 'rooted', 'anchored',
                       'stabilized', 'secured', 'protected', 'defended', 'guarded', 'shielded',
                       'sheltered', 'harbored', 'housed', 'accommodated', 'hosted', 'welcomed',
                       'invited', 'included', 'incorporated', 'integrated', 'unified', 'united',
                       'joined', 'connected', 'linked', 'bridged', 'spanned', 'crossed',
                       'traversed', 'navigated', 'journeyed', 'traveled', 'explored', 'discovered',
                       'found', 'located', 'situated', 'placed', 'positioned', 'oriented',
                       'directed', 'guided', 'led', 'shown', 'demonstrated', 'illustrated',
                       'exemplified', 'modeled', 'represented', 'symbolized', 'signified',
                       'according to research', 'studies show', 'data indicates', 'evidence suggests',
                       'research findings', 'empirical evidence', 'scientific evidence', 'peer reviewed',
                       'meta analysis', 'systematic review', 'literature review', 'academic research',
                       'scholarly research', 'scientific study', 'clinical trial', 'randomized controlled',
                       'double blind', 'placebo controlled', 'longitudinal study', 'cohort study',
                       'case control', 'cross sectional', 'observational study', 'experimental study',
                       'quantitative research', 'qualitative research', 'mixed methods', 'research methodology',
                       'statistical analysis', 'data analysis', 'regression analysis', 'correlation analysis',
                       'causal analysis', 'factor analysis', 'cluster analysis', 'principal component',
                       'multivariate analysis', 'univariate analysis', 'descriptive statistics', 'inferential statistics',
                       'hypothesis testing', 'significance testing', 'p value', 'confidence interval',
                       'margin of error', 'sample size', 'sample population', 'representative sample',
                       'random sample', 'stratified sample', 'cluster sample', 'convenience sample',
                       'survey research', 'polling', 'public opinion', 'opinion poll', 'exit poll',
                       'tracking poll', 'approval rating', 'favorability rating', 'job approval',
                       'economic indicators', 'leading indicators', 'lagging indicators', 'coincident indicators',
                       'gross domestic product', 'gdp growth', 'gdp per capita', 'economic growth',
                       'economic expansion', 'economic contraction', 'recession', 'depression',
                       'inflation rate', 'deflation', 'disinflation', 'stagflation', 'hyperinflation',
                       'unemployment rate', 'employment rate', 'labor force participation', 'underemployment',
                       'full employment', 'natural rate of unemployment', 'frictional unemployment',
                       'structural unemployment', 'cyclical unemployment', 'seasonal unemployment',
                       'consumer price index', 'cpi', 'producer price index', 'ppi', 'core inflation',
                       'headline inflation', 'wage inflation', 'asset inflation', 'price stability',
                       'monetary policy', 'fiscal policy', 'monetary easing', 'monetary tightening',
                       'quantitative easing', 'quantitative tightening', 'interest rates', 'federal funds rate',
                       'discount rate', 'prime rate', 'mortgage rates', 'bond yields', 'treasury yields',
                       'yield curve', 'inverted yield curve', 'flat yield curve', 'steep yield curve',
                       'federal reserve', 'fed', 'federal open market committee', 'fomc', 'board of governors',
                       'chairman', 'chairwoman', 'chair', 'governor', 'vice chair', 'secretary',
                       'treasury secretary', 'secretary of treasury', 'department of treasury', 'treasury department',
                       'office of management and budget', 'omb', 'congressional budget office', 'cbo',
                       'government accountability office', 'gao', 'congressional research service', 'crs',
                       'library of congress', 'national archives', 'general services administration', 'gsa',
                       'government services', 'public services', 'civil service', 'federal service',
                       'government employment', 'public employment', 'civil service employment',
                       'government workers', 'public workers', 'civil servants', 'federal employees',
                       'state employees', 'local employees', 'municipal employees', 'government contractors',
                       'federal contractors', 'state contractors', 'local contractors', 'public private partnership',
                       'ppp', 'public private cooperation', 'collaboration', 'cooperation', 'partnership',
                       'alliance', 'coalition', 'consortium', 'syndicate', 'cartel', 'trust',
                       'monopoly', 'oligopoly', 'duopoly', 'monopsony', 'oligopsony', 'perfect competition',
                       'monopolistic competition', 'imperfect competition', 'market structure', 'market power',
                       'market concentration', 'market share', 'market dominance', 'market leader',
                       'market follower', 'market challenger', 'market nicher', 'market segmentation',
                       'target market', 'market research', 'market analysis', 'market study',
                       'market survey', 'market poll', 'market test', 'market trial', 'market pilot',
                       'market launch', 'market entry', 'market exit', 'market penetration',
                       'market development', 'market expansion', 'market growth', 'market decline',
                       'market saturation', 'market maturity', 'market decline', 'market recovery',
                       'market rebound', 'market rally', 'market correction', 'market crash',
                       'market bubble', 'market boom', 'market bust', 'market cycle',
                       'business cycle', 'economic cycle', 'expansion', 'peak', 'contraction',
                       'trough', 'recovery', 'recession', 'depression', 'boom', 'bust',
                       'prosperity', 'austerity', 'growth', 'decline', 'stagnation', 'stagflation',
                       'inflation', 'deflation', 'disinflation', 'hyperinflation', 'price stability',
                       'monetary stability', 'financial stability', 'economic stability', 'political stability',
                       'social stability', 'institutional stability', 'system stability', 'regime stability',
                       'government stability', 'regime change', 'government change', 'political change',
                       'social change', 'economic change', 'technological change', 'demographic change',
                       'cultural change', 'environmental change', 'climate change', 'global warming',
                       'greenhouse effect', 'carbon dioxide', 'co2', 'methane', 'nitrous oxide',
                       'greenhouse gases', 'emissions', 'carbon emissions', 'greenhouse gas emissions',
                       'carbon footprint', 'carbon offset', 'carbon credit', 'carbon tax',
                       'cap and trade', 'emissions trading', 'carbon trading', 'climate policy',
                       'environmental policy', 'energy policy', 'renewable energy', 'solar energy',
                       'wind energy', 'hydroelectric energy', 'geothermal energy', 'nuclear energy',
                       'fossil fuels', 'coal', 'oil', 'natural gas', 'petroleum', 'gasoline',
                       'diesel', 'kerosene', 'propane', 'butane', 'lpg', 'cng', 'lng',
                       'energy efficiency', 'energy conservation', 'energy savings', 'energy costs',
                       'energy prices', 'electricity prices', 'gas prices', 'oil prices',
                       'fuel prices', 'energy security', 'energy independence', 'energy dependence',
                       'energy imports', 'energy exports', 'energy trade', 'energy market',
                       'energy sector', 'energy industry', 'energy companies', 'energy utilities',
                       'electric utilities', 'gas utilities', 'water utilities', 'public utilities',
                       'utility regulation', 'public utility commission', 'puc', 'federal energy regulatory commission',
                       'ferc', 'nuclear regulatory commission', 'nrc', 'environmental protection agency',
                       'epa', 'department of energy', 'doe', 'energy department', 'energy secretary',
                       'secretary of energy', 'energy policy', 'energy legislation', 'energy bill',
                       'energy act', 'energy law', 'energy regulation', 'energy standards',
                       'energy efficiency standards', 'fuel economy standards', 'cafe standards',
                       'corporate average fuel economy', 'emissions standards', 'emissions regulations',
                       'clean air act', 'clean water act', 'safe drinking water act',
                       'resource conservation and recovery act', 'rcra', 'comprehensive environmental response',
                       'compensation and liability act', 'cercla', 'superfund', 'toxic substances control act',
                       'tsca', 'federal insecticide fungicide and rodenticide act', 'fifra',
                       'endangered species act', 'esa', 'national environmental policy act', 'nepa',
                       'environmental impact statement', 'eis', 'environmental assessment', 'ea',
                       'finding of no significant impact', 'fonsi', 'record of decision', 'rod',
                       'environmental review', 'environmental analysis', 'environmental study',
                       'environmental research', 'environmental monitoring', 'environmental testing',
                       'environmental sampling', 'environmental data', 'environmental statistics',
                       'environmental indicators', 'environmental metrics', 'environmental benchmarks',
                       'environmental standards', 'environmental regulations', 'environmental laws',
                       'environmental policy', 'environmental protection', 'environmental conservation',
                       'environmental preservation', 'environmental restoration', 'environmental remediation',
                       'environmental cleanup', 'environmental rehabilitation', 'environmental recovery',
                       'environmental improvement', 'environmental enhancement', 'environmental betterment',
                       'environmental progress', 'environmental advancement', 'environmental development',
                       'environmental growth', 'environmental expansion', 'environmental extension',
                       'environmental increase', 'environmental rise', 'environmental surge',
                       'environmental boost', 'environmental lift', 'environmental raise',
                       'environmental elevation', 'environmental upgrade', 'environmental improvement',
                       'environmental enhancement', 'environmental betterment', 'environmental progress',
                       'environmental advancement', 'environmental development', 'environmental growth',
                       'environmental expansion', 'environmental extension', 'environmental increase',
                       'environmental rise', 'environmental surge', 'environmental boost',
                       'environmental lift', 'environmental raise', 'environmental elevation',
                       'environmental upgrade', 'environmental improvement', 'environmental enhancement',
                       'environmental betterment', 'environmental progress', 'environmental advancement',
                       'environmental development', 'environmental growth', 'environmental expansion',
                       'environmental extension', 'environmental increase', 'environmental rise',
                       'environmental surge', 'environmental boost', 'environmental lift',
                       'environmental raise', 'environmental elevation', 'environmental upgrade']
        }
    
    def _build_rhetorical_patterns(self):
        return {
            'emotional': [r'\b(must|should|need|require|critical|essential|urgent)\b',
                         r'\b(disaster|catastrophe|crisis|emergency|danger)\b',
                         r'\b(devastating|tragic|horrific|outrageous|appalling)\b'],
            'authoritative': [r'\b(proves|demonstrates|shows|indicates|reveals)\b',
                            r'\b(studies|research|data|evidence|analysis)\b',
                            r'\b(experts|scientists|researchers|scholars)\b'],
            'polarizing': [r'\b(always|never|all|none|everyone|nobody)\b',
                          r'\b(radical|extreme|dangerous|threat|enemy)\b',
                          r'\b(us|them|we|they|our|their)\b'],
            'moderating': [r'\b(some|many|often|sometimes|generally|typically)\b',
                          r'\b(consider|examine|evaluate|discuss|debate)\b',
                          r'\b(perhaps|possibly|maybe|potentially|likely)\b'],
            'certainty': [r'\b(certainly|definitely|absolutely|undoubtedly|clearly)\b',
                         r'\b(obviously|evidently|plainly|surely|indeed)\b'],
            'uncertainty': [r'\b(perhaps|maybe|possibly|might|could|uncertain)\b',
                           r'\b(unclear|unlikely|doubtful|questionable|ambiguous)\b']
        }
    
    def extract_contextual_features(self, text: str) -> np.ndarray:
        text_lower = text.lower()
        features = []
        
        lexicon_scores = []
        for bias_type, words in self.political_lexicons.items():
            score = sum(1 for word in words if word in text_lower)
            lexicon_scores.append(score)
        features.extend(lexicon_scores)
        
        lexicon_density = sum(lexicon_scores) / max(len(text.split()), 1)
        features.append(lexicon_density)
        
        dominant_lexicon = np.argmax(lexicon_scores) if lexicon_scores else 0
        features.append(dominant_lexicon)
        
        bias_features = self.bias_analyzer.analyze_bias(text)
        features.extend(bias_features)
        
        left_score = lexicon_scores[0] if len(lexicon_scores) > 0 else 0
        right_score = lexicon_scores[1] if len(lexicon_scores) > 1 else 0
        center_score = lexicon_scores[2] if len(lexicon_scores) > 2 else 0
        
        total_political = left_score + right_score + center_score
        if total_political > 0:
            left_proportion = left_score / total_political
            right_proportion = right_score / total_political
            center_proportion = center_score / total_political
            political_polarization = abs(left_proportion - right_proportion)
        else:
            left_proportion = right_proportion = center_proportion = 0.33
            political_polarization = 0
        
        features.extend([left_proportion, right_proportion, center_proportion, political_polarization])
        
        for pattern_type, patterns in self.rhetorical_patterns.items():
            matches = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) 
                         for pattern in patterns)
            features.append(matches)
        
        pattern_density = sum(len(re.findall(pattern, text_lower, re.IGNORECASE))
                             for patterns in self.rhetorical_patterns.values()
                             for pattern in patterns) / max(len(text.split()), 1)
        features.append(pattern_density)
        
        sentiment_features = self.sentiment_analyzer.analyze(text)
        features.extend(list(sentiment_features.values()))
        
        syntactic_features = self.syntactic_extractor.extract(text)
        features.extend(syntactic_features)
        
        discourse_features = self.discourse_analyzer.analyze(text)
        features.extend(discourse_features)
        
        argumentation_features = self.argumentation_analyzer.analyze(text)
        features.extend(argumentation_features)
        
        entity_features = self.entity_extractor.extract(text)
        features.extend(entity_features)
        
        stylometric_features = self.stylometric_analyzer.analyze(text)
        features.extend(stylometric_features)
        
        readability_features = self.readability_analyzer.analyze(text)
        features.extend(readability_features)
        
        cooccurrence_features = self.cooccurrence_analyzer.analyze(text)
        features.extend(cooccurrence_features)
        
        temporal_features = self.temporal_extractor.extract(text)
        features.extend(temporal_features)
        
        citation_features = self.citation_extractor.extract(text)
        features.extend(citation_features)
        
        statistical_features = self.statistical_analyzer.analyze(text)
        features.extend(statistical_features)
        
        exclamations = text.count('!')
        questions = text.count('?')
        all_caps = len(re.findall(r'\b[A-Z]{2,}\b', text))
        features.extend([exclamations, questions, all_caps])
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_sentence_len = np.mean([len(s.split()) for s in sentences])
            std_sentence_len = np.std([len(s.split()) for s in sentences])
            max_sentence_len = np.max([len(s.split()) for s in sentences])
            min_sentence_len = np.min([len(s.split()) for s in sentences])
            features.extend([avg_sentence_len, std_sentence_len, max_sentence_len, min_sentence_len])
        else:
            features.extend([0, 0, 0, 0])
        
        numbers = len(re.findall(r'\b\d+\.?\d*%?\b', text))
        percentages = text.count('%')
        large_numbers = len(re.findall(r'\b\d{4,}\b', text))
        features.extend([numbers, percentages, large_numbers])
        
        quotes = text.count('"') // 2
        single_quotes = text.count("'") // 2
        parentheses = text.count('(')
        brackets = text.count('[')
        features.extend([quotes, single_quotes, parentheses, brackets])
        
        negation = len(re.findall(r'\b(no|not|never|nothing|nobody|nowhere|neither|nor)\b', text_lower))
        comparative = len(re.findall(r'\b(more|most|less|least|better|best|worse|worst|greater|greatest)\b', text_lower))
        modals = len(re.findall(r'\b(should|must|would|could|might|may|will|shall)\b', text_lower))
        features.extend([negation, comparative, modals])
        
        first_person = len(re.findall(r'\b(I|we|our|us|my|our)\b', text_lower))
        second_person = len(re.findall(r'\b(you|your|yours)\b', text_lower))
        third_person = len(re.findall(r'\b(they|them|their|he|she|his|her|it|its)\b', text_lower))
        features.extend([first_person, second_person, third_person])
        
        person_ratio = first_person / max(first_person + second_person + third_person, 1)
        features.append(person_ratio)
        
        superlatives = len(re.findall(r'\b(most|least|best|worst|greatest|smallest|largest)\b', text_lower))
        intensifiers = len(re.findall(r'\b(very|extremely|incredibly|absolutely|completely|totally)\b', text_lower))
        features.extend([superlatives, intensifiers])
        
        word_count = len(text.split())
        char_count = len(text)
        avg_word_len = char_count / word_count if word_count > 0 else 0
        features.extend([word_count, char_count, avg_word_len])
        
        unique_words = len(set(text_lower.split()))
        lexical_diversity = unique_words / word_count if word_count > 0 else 0
        features.append(lexical_diversity)
        
        repeated_words = word_count - unique_words
        repetition_ratio = repeated_words / word_count if word_count > 0 else 0
        features.append(repetition_ratio)
        
        paragraph_breaks = text.count('\n\n') + text.count('\r\n\r\n')
        features.append(paragraph_breaks)
        
        url_pattern = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
        email_pattern = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        features.extend([url_pattern, email_pattern])
        
        hashtags = len(re.findall(r'#\w+', text))
        mentions = len(re.findall(r'@\w+', text))
        features.extend([hashtags, mentions])
        
        return np.array(features)


class MultiScaleContextProcessor:
    def __init__(self):
        self.feature_engineer = ContextualFeatureEngineer()
        self.context_weights = None
        
    def process_text(self, text: str, context_windows: List[int] = [2, 3, 5]) -> Dict:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        base_features = self.feature_engineer.extract_contextual_features(text)
        
        if len(sentences) <= max(context_windows):
            return self._process_single_context(text, base_features)
        
        multi_scale_features = []
        
        for window_size in context_windows:
            if len(sentences) <= window_size:
                window_features = base_features
                multi_scale_features.append(window_features)
                continue
            
            context_features = []
            for i in range(len(sentences) - window_size + 1):
                window_text = ' '.join(sentences[i:i+window_size])
                window_feat = self.feature_engineer.extract_contextual_features(window_text)
                context_features.append(window_feat)
            
            if context_features:
                context_array = np.array(context_features)
                mean_feat = np.mean(context_array, axis=0)
                std_feat = np.std(context_array, axis=0)
                max_feat = np.max(context_array, axis=0)
                min_feat = np.min(context_array, axis=0)
                median_feat = np.median(context_array, axis=0)
                
                window_aggregated = np.concatenate([
                    mean_feat, std_feat, max_feat, min_feat, median_feat
                ])
                multi_scale_features.append(window_aggregated)
            else:
                multi_scale_features.append(np.concatenate([
                    base_features, np.zeros_like(base_features),
                    base_features, base_features, base_features
                ]))
        
        all_contextual = np.concatenate(multi_scale_features)
        
        paragraph_features = self._extract_paragraph_features(text)
        sentence_position_features = self._extract_position_features(sentences)
        transition_features = self._extract_transition_features(sentences)
        
        return {
            'base_features': base_features,
            'contextual_features': all_contextual,
            'paragraph_features': paragraph_features,
            'position_features': sentence_position_features,
            'transition_features': transition_features,
            'text': text
        }
    
    def _extract_paragraph_features(self, text: str) -> np.ndarray:
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return np.zeros(5)
        
        para_lengths = [len(p.split()) for p in paragraphs]
        para_features = [
            len(paragraphs),
            np.mean(para_lengths),
            np.std(para_lengths),
            np.max(para_lengths),
            np.min(para_lengths)
        ]
        
        return np.array(para_features)
    
    def _extract_position_features(self, sentences: List[str]) -> np.ndarray:
        if not sentences:
            return np.zeros(4)
        
        total_sentences = len(sentences)
        first_sentence_len = len(sentences[0].split())
        last_sentence_len = len(sentences[-1].split())
        
        middle_start = total_sentences // 3
        middle_end = 2 * total_sentences // 3
        middle_sentences = sentences[middle_start:middle_end] if middle_end > middle_start else []
        avg_middle_len = np.mean([len(s.split()) for s in middle_sentences]) if middle_sentences else 0
        
        return np.array([
            first_sentence_len / max(total_sentences, 1),
            last_sentence_len / max(total_sentences, 1),
            avg_middle_len / max(total_sentences, 1),
            total_sentences
        ])
    
    def _extract_transition_features(self, sentences: List[str]) -> np.ndarray:
        if len(sentences) < 2:
            return np.zeros(3)
        
        transitions = ['however', 'therefore', 'furthermore', 'moreover', 'nevertheless',
                      'consequently', 'additionally', 'alternatively', 'meanwhile', 'subsequently']
        
        transition_count = 0
        for i in range(len(sentences) - 1):
            combined = sentences[i] + ' ' + sentences[i+1]
            for trans in transitions:
                if trans in combined.lower():
                    transition_count += 1
                    break
        
        transition_density = transition_count / max(len(sentences) - 1, 1)
        
        sentence_similarity = []
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i+1].lower().split())
            if words1 and words2:
                similarity = len(words1 & words2) / len(words1 | words2)
                sentence_similarity.append(similarity)
        
        avg_similarity = np.mean(sentence_similarity) if sentence_similarity else 0
        
        return np.array([transition_count, transition_density, avg_similarity])
    
    def _process_single_context(self, text: str, base_features: np.ndarray) -> Dict:
        return {
            'base_features': base_features,
            'contextual_features': np.concatenate([
                base_features, np.zeros_like(base_features),
                base_features, base_features, base_features,
                base_features, np.zeros_like(base_features),
                base_features, base_features, base_features,
                base_features, np.zeros_like(base_features),
                base_features, base_features, base_features
            ]),
            'paragraph_features': np.zeros(5),
            'position_features': np.zeros(4),
            'transition_features': np.zeros(3),
            'text': text
        }


class AgenticContextProcessor:
    def __init__(self):
        self.multi_scale_processor = MultiScaleContextProcessor()
        self.feature_engineer = ContextualFeatureEngineer()
        self.adaptive_weights = None
        
    def process_text(self, text: str, context_window: int = 3) -> Dict:
        return self.multi_scale_processor.process_text(text, context_windows=[2, 3, 5])


class PoliticalBiasDetector:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_vectorizer_ngram = None
        self.char_vectorizer = None
        self.count_vectorizer = None
        self.scaler = None
        self.scaler_robust = None
        self.scaler_minmax = None
        self.svd = None
        self.svd_ngram = None
        self.svd_char = None
        self.feature_selector = None
        self.model = None
        self.label_mapping = {}
        self.reverse_mapping = {}
        self.context_processor = AgenticContextProcessor()
        
    def _load_csvs_from_directory(self, directory: str) -> Tuple[List[str], List[str]]:
        texts = []
        labels = []
        
        csv_files = list(Path(directory).glob('*.csv'))
        if not csv_files:
            raise ValueError(f"No CSV files found in {directory}")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'text' not in df.columns or 'label' not in df.columns:
                    print(f"Skipping {csv_file}: missing 'text' or 'label' columns")
                    continue
                
                df = df.dropna(subset=['text', 'label'])
                texts.extend(df['text'].tolist())
                labels.extend(df['label'].tolist())
                print(f"Loaded {len(df)} samples from {csv_file.name}")
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue
        
        return texts, labels
    
    def _build_feature_matrix(self, texts: List[str], fit: bool = True) -> sp.csr_matrix:
        processed_texts = [self.context_processor.process_text(text) for text in texts]
        
        base_texts = [p['text'] for p in processed_texts]
        
        if fit:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
                sublinear_tf=True,
                strip_accents='unicode',
                lowercase=True,
                stop_words='english',
                analyzer='word'
            )
            
            self.tfidf_vectorizer_ngram = TfidfVectorizer(
                max_features=6000,
                ngram_range=(2, 3),
                min_df=2,
                max_df=0.85,
                sublinear_tf=True,
                strip_accents='unicode',
                lowercase=True
            )
            
            self.char_vectorizer = TfidfVectorizer(
                max_features=3000,
                analyzer='char',
                ngram_range=(3, 5),
                min_df=2,
                max_df=0.9
            )
            
            self.count_vectorizer = CountVectorizer(
                max_features=4000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.75,
                binary=True
            )
            
            tfidf_features = self.tfidf_vectorizer.fit_transform(base_texts)
            tfidf_ngram_features = self.tfidf_vectorizer_ngram.fit_transform(base_texts)
            char_features = self.char_vectorizer.fit_transform(base_texts)
            count_features = self.count_vectorizer.fit_transform(base_texts)
        else:
            tfidf_features = self.tfidf_vectorizer.transform(base_texts)
            tfidf_ngram_features = self.tfidf_vectorizer_ngram.transform(base_texts)
            char_features = self.char_vectorizer.transform(base_texts)
            count_features = self.count_vectorizer.transform(base_texts)
        
        base_features_list = [p['base_features'] for p in processed_texts]
        contextual_features_list = [p['contextual_features'] for p in processed_texts]
        paragraph_features_list = [p['paragraph_features'] for p in processed_texts]
        position_features_list = [p['position_features'] for p in processed_texts]
        transition_features_list = [p['transition_features'] for p in processed_texts]
        
        base_features = np.vstack(base_features_list)
        contextual_features = np.vstack(contextual_features_list)
        paragraph_features = np.vstack(paragraph_features_list)
        position_features = np.vstack(position_features_list)
        transition_features = np.vstack(transition_features_list)
        
        if fit:
            self.scaler = StandardScaler(with_mean=False)
            self.scaler_robust = RobustScaler()
            self.scaler_minmax = MinMaxScaler()
            
            base_features_scaled = self.scaler.fit_transform(base_features)
            contextual_features_scaled = self.scaler_robust.fit_transform(contextual_features)
            paragraph_features_scaled = self.scaler_minmax.fit_transform(paragraph_features)
            position_features_scaled = self.scaler_minmax.fit_transform(position_features)
            transition_features_scaled = self.scaler_minmax.fit_transform(transition_features)
            
            self.svd = TruncatedSVD(n_components=150, random_state=42)
            self.svd_ngram = TruncatedSVD(n_components=100, random_state=42)
            self.svd_char = TruncatedSVD(n_components=80, random_state=42)
            
            tfidf_reduced = self.svd.fit_transform(tfidf_features)
            tfidf_ngram_reduced = self.svd_ngram.fit_transform(tfidf_ngram_features)
            char_reduced = self.svd_char.fit_transform(char_features)
        else:
            base_features_scaled = self.scaler.transform(base_features)
            contextual_features_scaled = self.scaler_robust.transform(contextual_features)
            paragraph_features_scaled = self.scaler_minmax.transform(paragraph_features)
            position_features_scaled = self.scaler_minmax.transform(position_features)
            transition_features_scaled = self.scaler_minmax.transform(transition_features)
            
            tfidf_reduced = self.svd.transform(tfidf_features)
            tfidf_ngram_reduced = self.svd_ngram.transform(tfidf_ngram_features)
            char_reduced = self.svd_char.transform(char_features)
        
        combined_features = sp.hstack([
            tfidf_features,
            tfidf_ngram_features,
            char_features,
            count_features,
            sp.csr_matrix(base_features_scaled),
            sp.csr_matrix(contextual_features_scaled),
            sp.csr_matrix(paragraph_features_scaled),
            sp.csr_matrix(position_features_scaled),
            sp.csr_matrix(transition_features_scaled),
            sp.csr_matrix(tfidf_reduced),
            sp.csr_matrix(tfidf_ngram_reduced),
            sp.csr_matrix(char_reduced)
        ])
        
        return combined_features
    
    def train(self, csv_directory: str = 'ts'):
        print(f"Loading data from {csv_directory}...")
        texts, labels = self._load_csvs_from_directory(csv_directory)
        
        print(f"\nTotal samples: {len(texts)}")
        print(f"Label distribution: {Counter(labels)}")
        
        unique_labels = sorted(set(labels))
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_mapping = {idx: label for label, idx in self.label_mapping.items()}
        
        y = np.array([self.label_mapping[label] for label in labels])
        
        print("\nBuilding feature matrix...")
        X = self._build_feature_matrix(texts, fit=True)
        print(f"Initial feature matrix shape: {X.shape}")
        
        print("\nPerforming feature selection...")
        self.feature_selector = SelectKBest(f_classif, k=min(50000, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        print(f"Selected feature matrix shape: {X_selected.shape}")
        
        print("\nEnsuring three-class structure (left, right, center)...")
        unique_labels_set = set(unique_labels)
        expected_labels = {'left', 'right', 'center'}
        
        if not expected_labels.issubset(unique_labels_set):
            print(f"Warning: Expected labels {expected_labels}, found {unique_labels_set}")
            print("Mapping labels to standard format...")
            label_mapping_dict = {}
            for label in labels:
                label_lower = label.lower()
                if 'left' in label_lower or 'liberal' in label_lower or 'progressive' in label_lower:
                    label_mapping_dict[label] = 'left'
                elif 'right' in label_lower or 'conservative' in label_lower or 'republican' in label_lower:
                    label_mapping_dict[label] = 'right'
                elif 'center' in label_lower or 'centrist' in label_lower or 'moderate' in label_lower or 'neutral' in label_lower:
                    label_mapping_dict[label] = 'center'
                else:
                    label_mapping_dict[label] = 'center'
            
            labels = [label_mapping_dict.get(l, 'center') for l in labels]
            unique_labels = sorted(set(labels))
            self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            self.reverse_mapping = {idx: label for label, idx in self.label_mapping.items()}
            y = np.array([self.label_mapping[label] for label in labels])
            print(f"Remapped labels. New distribution: {Counter(labels)}")
        
        print("\nTraining Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=900,
            max_depth=65,
            min_samples_split=10,
            min_samples_leaf=3,
            max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1,
            verbose=1,
            bootstrap=True,
            oob_score=True,
            warm_start=False,
            max_samples=0.9,
            min_impurity_decrease=0.00003,
            ccp_alpha=0.00005
        )
        
        X = X_selected
        
        self.model.fit(X, y)
        
        print(f"\nOut-of-bag score: {self.model.oob_score_:.4f}")
        
        print("\nPerforming cross-validation...")
        cv_scores = cross_val_score(
            self.model, X, y, 
            cv=5, 
            scoring='f1_weighted', 
            n_jobs=-1,
            verbose=1
        )
        print(f"CV F1 scores: {cv_scores}")
        print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        train_pred = self.model.predict(X)
        train_acc = accuracy_score(y, train_pred)
        train_f1 = f1_score(y, train_pred, average='weighted')
        
        print(f"\nTraining accuracy: {train_acc:.4f}")
        print(f"Training F1: {train_f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(
            y, train_pred,
            target_names=[self.reverse_mapping[i] for i in range(len(unique_labels))],
            digits=4
        ))
        
        feature_importances = self.model.feature_importances_
        top_indices = np.argsort(feature_importances)[-20:][::-1]
        print(f"\nTop 20 feature importances:")
        for idx in top_indices:
            print(f"  Feature {idx}: {feature_importances[idx]:.6f}")
    
    def save(self, filepath: str):
        model_data = {
            'model': self.model,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_vectorizer_ngram': self.tfidf_vectorizer_ngram,
            'char_vectorizer': self.char_vectorizer,
            'count_vectorizer': self.count_vectorizer,
            'scaler': self.scaler,
            'scaler_robust': self.scaler_robust,
            'scaler_minmax': self.scaler_minmax,
            'svd': self.svd,
            'svd_ngram': self.svd_ngram,
            'svd_char': self.svd_char,
            'feature_selector': self.feature_selector,
            'label_mapping': self.label_mapping,
            'reverse_mapping': self.reverse_mapping,
            'context_processor': self.context_processor
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def predict(self, text: str):
        X = self._build_feature_matrix([text], fit=False)
        X_selected = self.feature_selector.transform(X)
        prediction = self.model.predict(X_selected)[0]
        probabilities = self.model.predict_proba(X_selected)[0]
        
        return self.reverse_mapping[prediction], probabilities


def main():
    detector = PoliticalBiasDetector()
    detector.train(csv_directory='ts')
    
    model_path = 'political_bias_model.pkl'
    detector.save(model_path)
    
    test_samples = [
        "workers deserve control of the means of production and universal healthcare ensures everyone gets treatment",
        "the data shows economic growth of two percent according to recent studies",
        "lower taxes allow businesses to create jobs and stimulate free market competition",
        "traditional values must be preserved and border security is essential for national sovereignty"
    ]
    
    print("\n\nSample predictions:")
    for sample in test_samples:
        prediction, probs = detector.predict(sample)
        confidence = probs.max()
        print(f"{prediction:15s} ({confidence:.1%})  {sample}")


if __name__ == "__main__":
    main()
