"""Opt-in small ablations. The original heuristic remains unchanged."""
import re
from model import HeuristicAttentionProvider, contains_term, clamp


class LocalKeywordProvider(HeuristicAttentionProvider):
    """Restrict keywords/boosts to entity interval +/- 3 whitespace tokens.

    Mask the two target entity spans before matching. Keep uncertainty/clutter,
    distance, bias, weights, NA scaling and between-keyword scoring unchanged.
    This is a lexical window, not syntax or clause understanding.
    """
    def __init__(self, temperature=.7, flank_tokens=3):
        super().__init__(temperature)
        if flank_tokens < 0:
            raise ValueError('flank_tokens must be nonnegative')
        self.flank_tokens = flank_tokens

    def lexical_context(self, instance):
        text = instance.text
        left = min(instance.h_pos[0], instance.t_pos[0])
        right = max(instance.h_pos[1], instance.t_pos[1])
        before = list(re.finditer(r'\S+', text[:left]))
        after = list(re.finditer(r'\S+', text[right:]))
        if self.flank_tokens and before:
            left = before[max(0, len(before)-self.flank_tokens)].start()
        if self.flank_tokens and after:
            right += after[min(len(after), self.flank_tokens)-1].end()
        chars = list(text)
        for start, end in (instance.h_pos, instance.t_pos):
            chars[start:end] = ' ' * (end-start)
        return ''.join(chars[left:right]).lower()

    def _keyword_score(self, instance, hint):
        context = self.lexical_context(instance)
        between = instance.between_text()
        keywords = hint.get('keywords', [])
        between_keywords = hint.get('between', [])
        denominator = len(keywords) + .5 * len(between_keywords)
        numerator = sum(contains_term(context, k) for k in keywords)
        numerator += .5 * sum(contains_term(between, k) for k in between_keywords)
        return clamp(numerator / denominator) if denominator else 0.

    def score_components(self, instance):
        c = super().score_components(instance)
        context = self.lexical_context(instance)
        c['boosts'] = sum(w for k, w in self._get_hint(instance.relation).get('boosts', {}).items()
                          if contains_term(context, k))
        raw = (.28*c['distance'] + .18*c['between'] + .38*c['keyword'] +
               .16*c['order'] + c['bias'] + c['boosts'] - c['uncertainty'] -
               c['clutter'] - c['overlap'])
        c['raw_score'] = clamp(raw * (.25 if instance.relation == 'NA' else 1), 0, 1.5)
        c['lexical_context'] = context
        return c
