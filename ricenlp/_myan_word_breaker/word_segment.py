# Forked from https://github.com/stevenay/myan-word-breaker (MIT License)
# Original author: SteveNay (NayLinAung)
# Original file: word_breaker/word_segment_v5.py

from math import log
from collections import defaultdict
from enum import Enum
import os

from ricenlp._myan_word_breaker.rabbit import zg2uni
from ricenlp._myan_word_breaker.myparser import MyParser


class WordSegment:
    # Word Segmentation Ways
    SegmentationMethod = Enum('SegmentationMethod', 'all_possible_combination sub_word_possibility')

    _APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    _APP_DICTIONARY = os.path.join(_APP_ROOT, 'dictionary')
    _APP_CORPUS = os.path.join(_APP_ROOT, 'corpus')

    _total_unigram_count = 341685
    _total_bigram_count = 170843
    _maxlen = 6

    # Class-level cache; loaded once on first instantiation
    _dict_words = None
    _stop_words = None
    _mypos_corpus = None

    @classmethod
    def _load_data(cls):
        if cls._dict_words is None:
            cls._dict_words = open(
                os.path.join(cls._APP_DICTIONARY, 'dict-words.txt'), 'r', encoding='utf-8'
            ).read().splitlines()
            cls._stop_words = open(
                os.path.join(cls._APP_DICTIONARY, 'stopwords.txt'), 'r', encoding='utf-8'
            ).read().splitlines()
            cls._mypos_corpus = open(
                os.path.join(cls._APP_CORPUS, 'mypos-dver.1.0.cword.txt'), 'r', encoding='utf-8'
            ).read()

    def __init__(self):
        WordSegment._load_data()
        self._not_found_words = set()
        self._found_words = set()
        self._possible_combos = []
        self.m = MyParser()

    def _check_in_dicts(self, word):
        if word in self._found_words:
            return True
        if word in self._not_found_words:
            return False
        if word in self._dict_words or word in self._stop_words:
            self._found_words.add(word)
            return True
        self._not_found_words.add(word)
        return False

    def _left_to_right_segment(self, seq, maxlen):
        length = len(seq)
        offset = 0
        combo = []
        while length > 0:
            for i in range(maxlen, 0, -1):
                chunk = offset + i
                word = ''.join(seq[offset:chunk])
                if self._check_in_dicts(word) or i == 1:
                    combo.append(word)
                    offset += i
                    length -= i
                    break
        return combo

    def _make_combinations(self, seq, maxlen):
        memo = defaultdict(list)
        memo[1] = [[seq[0]]]

        seq_iter = iter(seq)
        next(seq_iter)
        last_index = len(seq) - 2
        for index, char in enumerate(seq_iter):
            new_memo = defaultdict(list)
            for wordlen, combos in memo.items():
                if not combos:
                    continue
                new_memo[1].extend(combo + [char] for combo in combos)
                if wordlen < maxlen:
                    longest_word = combos[0][-1]
                    word = combos[0][-1] + char
                    new_memo[wordlen + 1] = newcombos = []
                    for combo in combos:
                        combo[-1] = word
                        newcombos.append(combo)
                    if index == last_index or wordlen + 1 == maxlen:
                        if word and not self._check_in_dicts(word):
                            word = ''
                            if wordlen + 1 in new_memo:
                                del new_memo[wordlen + 1]
                    if longest_word and longest_word not in seq:
                        if not self._check_in_dicts(longest_word):
                            del new_memo[1][len(combos) * -1:]
            memo = new_memo

        combos = []
        for combo in memo.values():
            combos.extend(combo)
        return combos

    def _cal_mutual_info(self, sya1, sya2):
        p1 = self._mypos_corpus.count(sya1) / self._total_unigram_count
        p2 = self._mypos_corpus.count(sya2) / self._total_unigram_count
        p12 = self._mypos_corpus.count(sya1 + sya2) / self._total_bigram_count
        if p12 == 0:
            return 0
        return log(p12 / float(p1 * p2), 2)

    def filter_minimum_combination(self, solutions):
        min_count = min(map(len, solutions))
        return [s for s in solutions if len(s) == min_count]

    def _calculate_sentence_collocation_strength(self, filtering_solutions):
        syllable_sents = [
            [self.m.syllable(word) for word in solution]
            for solution in filtering_solutions
        ]
        syllable_collocation_strength = []
        for sent_index, syllable_sent in enumerate(syllable_sents):
            sentence_strength = 0
            for index, syllable_word in enumerate(syllable_sent):
                i = 0
                word_mutual_info = 0
                if len(syllable_word) > 1:
                    while i < len(syllable_word):
                        if i < len(syllable_word) - 1:
                            word_mutual_info += self._cal_mutual_info(
                                syllable_word[i], syllable_word[i + 1]
                            )
                        i += 1
                    if index - 1 >= 0:
                        left_last = syllable_sent[index - 1][-1]
                        word_mutual_info -= self._cal_mutual_info(left_last, syllable_word[0])
                    if index + 1 < len(syllable_sent):
                        right_first = syllable_sent[index + 1][0]
                        word_mutual_info -= self._cal_mutual_info(syllable_word[-1], right_first)
                sentence_strength += word_mutual_info
            syllable_collocation_strength.append(
                [sentence_strength, filtering_solutions[sent_index]]
            )
        return syllable_collocation_strength

    def _make_sub_word_combinations(self, input, combo, shortest_length,
                                    start_word_position=0, pointer=0):
        for n in range(start_word_position, shortest_length):
            result = combo[0:n]
            word = combo[n]
            seq = self.m.syllable(word)
            offset = 0
            maxlen = len(seq) - 1
            for i in range(maxlen, 0, -1):
                chunk = offset + i
                word = ''.join(seq[offset:chunk])
                if self._check_in_dicts(word):
                    result.append(word)
                    result.extend(self._left_to_right_segment(input[pointer + i:], self._maxlen))
                    if len(result) <= shortest_length:
                        self._possible_combos.append(result)
                        cur_pointer = pointer + i
                        self._make_sub_word_combinations(
                            input, result, shortest_length, n + 1, cur_pointer
                        )
                    break
            pointer += len(seq)

    def break_words(self, input, segmentation_method):
        self._possible_combos = []
        input = self.m.syllable(input)
        if segmentation_method == self.SegmentationMethod.all_possible_combination:
            self._possible_combos = self._make_combinations(input, self._maxlen)
        elif segmentation_method == self.SegmentationMethod.sub_word_possibility:
            combo = self._left_to_right_segment(input, self._maxlen)
            self._possible_combos.append(combo)
            self._make_sub_word_combinations(input, combo, len(combo))

        min_filtered = self.filter_minimum_combination(self._possible_combos)
        if len(min_filtered) > 1:
            strengths = self._calculate_sentence_collocation_strength(min_filtered)
            strongest = max(strengths, key=lambda x: x[0])
            return strongest[1]
        return min_filtered[0]

    def normalize_break(self, input_text, encoding,
                        segmentation_method=None):
        if segmentation_method is None:
            segmentation_method = self.SegmentationMethod.all_possible_combination
        if encoding == "zawgyi":
            input_text = zg2uni(input_text)
        input_text = input_text.replace(" ", "")
        inputs = input_text.split("\u104B")  # split on Myanmar period (။)
        outputs = []
        for inp in inputs:
            if inp:
                outputs.append(self.break_words(inp, segmentation_method))
        return outputs
