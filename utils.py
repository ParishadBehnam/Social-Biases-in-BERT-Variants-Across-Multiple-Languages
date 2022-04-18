from matplotlib.font_manager import FontProperties
import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt

from mlm.scorers import MLMScorerPT
import mxnet as mx

import pickle

# normalize probas so that p[man] + p[woman] = 1
def normalize(t):
    return t / torch.sum(t, dim=-1, keepdim=True)

# plot in vertical bars
def plot_bar_probas(probas, words, template, labels=['english', 'greek', 'persian'], plot_proba_text=True):
    import numpy as np
    import matplotlib.pyplot as plt

    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize =(20, 8))

    data = np.array(normalize(torch.tensor(probas))).transpose()

    # Set position of bar on X axis
    bar_x = []
    for lang in labels:
      if len(bar_x) == 0:
        bar_x.append(np.arange(data.shape[0]))
      else:
        bar_x.append(np.array([x + barWidth for x in bar_x[-1]]))

    # Make the plot
    category_colors = plt.get_cmap('tab20')(
            np.linspace(0.15, 0.85, data.shape[1])) 
    for i, (lang, bar, c) in enumerate(zip(labels, bar_x, category_colors)):
      plt.bar(bar, data[:, i], color =c, width = barWidth,
          edgecolor ='grey', label =lang)

    plt.legend()
    plt.title(template)
    plt.show()

# plot in colorful horizontal bars
def plot_probas(probas, words, template, labels=['english', 'greek', 'persian'], plot_proba_text=True, save_addr='BiasAnalysis/data/results/', cat='gender'):
    data = np.array(normalize(torch.tensor(probas)))

    plt.rcParams.update({'font.size': 16})
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('tab20')(
            np.linspace(0, 1, data.shape[1]))    # 'tab20', 'RdYlGn'
    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(words, category_colors)):
      widths = data[:, i]
      starts = data_cum[:, i] - widths
      if isinstance(colname, list):
        ax.barh(labels, widths, left=starts, height=0.5,
                      label=colname[0], color=color)
      else:
        ax.barh(labels, widths, left=starts, height=0.5,
                      label=colname, color=color)
      xcenters = starts + widths / 2

      if plot_proba_text:
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            if c >= 0.07:
                ax.text(x, y, '{:.3f}'.format(c), ha='center', va='center',
                        color=text_color)
            elif c >= 0.03:
                ax.text(x, y, '{:.1f}'.format(c), ha='center', va='center',
                        color=text_color)
    lgd = ax.legend(ncol=min(len(words), 4), bbox_to_anchor=(0, 1),
                    loc='lower left', fontsize='small')
    plt.title(template, y=-0.1)    # english template
    print(f'{save_addr}{template} - {words[0]}')
    plt.savefig(f'{save_addr}{cat}/{template[:-1]} - {words[0]}.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

def generate_template_set_plots(template_set, scorers, langs, plot_proba_text=True, cat='gender', save=False):
  # template_set: templates of one category

  for temp_id, template in enumerate(template_set['templates']):
    base_temps = template['base_templates']
    prior_list = template['prior_variations']
    mask_list = template['words']
    template_scores = []
    for base_temp_id, base_temp in enumerate(base_temps):
      temp_score = score_lists_sent_prob(scorers, base_temp, prior_list, mask_list)
      template_scores.append(temp_score)

    temp_set_scores = average_across_templates_from_dict(template_scores)
    proba_scores = []
    templates_with_prior = []

    pickle_temps, pickle_words, pickle_scores = [], [], []
    for prior_id, prior in enumerate(prior_list['english']):
      template_str = base_temps[-1]['english'].format(prior=prior)
      templates_with_prior.append(template_str)
      proba_scores.append([]) # prior id

      for lang_id, lang in enumerate(langs):
        proba_scores[prior_id].append([]) # lang id

        for mask_word in mask_list['english']:
          if isinstance(mask_word, list):
            mask_word = mask_word[0]
          proba_scores[prior_id][lang_id].append(temp_set_scores[lang][prior][mask_word])
    
      pickle_temps.append(templates_with_prior[prior_id])
      pickle_words.append(mask_list['english'])
      pickle_scores.append(proba_scores[prior_id])
      plot_probas(proba_scores[prior_id], mask_list['english'], templates_with_prior[prior_id], labels=langs, plot_proba_text=plot_proba_text, cat=cat)
    if save:
      with open(f'BiasAnalysis/data/results/{cat}/temp {temp_id}.pickle', 'wb') as handle:
        dump = {'scores': pickle_scores, 
        'words': pickle_words, 
        'template': pickle_temps}
        pickle.dump(dump, handle)
  
  return


from functools import reduce
import torch
import torch.nn.functional as F

class MLMPriorNormalizedScorer:
    def __init__(self, model, tokenizer, lang, mask_token_idx):
        self.model = model
        self.tokenizer= tokenizer
        self.lang = lang
        self.mask_token_idx = mask_token_idx

        self.scorer = MLMScorerPT(model, None, tokenizer, [mx.cpu()])

    def get_sentence_score(self, sent_template, prior, mask_word):
        mask_words = mask_word
        if not isinstance(mask_words, list):
            mask_words = [mask_words] # if only one variation exists for the bias word

        probas = []
        for mask_word in mask_words:
            sent_template = sent_template.replace('[MASK]', '{mask_word}')
            probas.append(self.scorer.score_sentences([sent_template.format(prior=prior, mask_word=mask_word)])[0])
        return np.mean(np.exp(probas))

    def get_word_score(self, sent_template, prior, mask_word):
        mask_words = mask_word
        if not isinstance(mask_words, list):
            mask_words = [mask_words] # if only one variation exists for the bias word

        probas = []
        for mask_word in mask_words:
            len_mask = len(self.tokenizer.encode(mask_word)) - 2
            
            sent_temp = sent_template.format(prior=prior)
            mask_position = self.tokenizer.encode(sent_temp).index(self.mask_token_idx)
            p_tgt_template = sent_temp.replace('[MASK]', '{mask}').format(mask=mask_word)
            p_tgt = self.scorer.score_sentences([p_tgt_template], per_token=True)[0][mask_position:mask_position+len_mask]
            p_tgt = sum(p_tgt)

            probas.append(p_tgt)
        return np.mean(np.exp(probas))

    def get_normalized_score(self, sent_template, prior, mask_word):
        mask_words = mask_word
        if not isinstance(mask_words, list):
            mask_words = [mask_words] # if only one variation exists for the bias word

        norm_probas = []
        for mask_word in mask_words:
            len_mask = len(self.tokenizer.encode(mask_word)) - 2
            
            # true proba
            sent_temp = sent_template.format(prior=prior)
            mask_position = self.tokenizer.encode(sent_temp).index(self.mask_token_idx)
            p_tgt_template = sent_temp.replace('[MASK]', '{mask}').format(mask=mask_word)
            p_tgt = self.scorer.score_sentences([p_tgt_template], per_token=True)[0][mask_position:mask_position+len_mask]
            p_tgt = sum(p_tgt)

            # normalization with prior
            if sent_template.index('{prior}') < sent_template.index('[MASK]'):
              input_ids = self.tokenizer.encode(sent_template.replace('{prior}', '[MASK]'))
              first_mask_position = input_ids.index(self.mask_token_idx)
              mask_position = input_ids[first_mask_position+1:].index(self.mask_token_idx) + first_mask_position + 1
            else:
              mask_position = self.tokenizer.encode(sent_template).index(self.mask_token_idx)
            
            p_prior_template = sent_template.replace('[MASK]', '{mask}') \
                                            .replace('{prior}', '[MASK]') \
                                            .format(mask=mask_word)             
            p_prior = self.scorer.score_sentences([p_prior_template], per_token=True)[0][mask_position:mask_position+len_mask]
            p_prior = sum(p_prior)

            norm_probas.append(p_tgt - p_prior)

        return np.mean(np.exp(norm_probas))


    def get_prods_from_list_per_prior(self, sent_template, prior_word, mask_word_list):
        return_list = []

        for mask_word in mask_word_list:
            s = self.get_sentence_score(sent_template, prior=prior_word, mask_word=mask_word)
            return_list.append(s)

        return return_list

    def get_normalized_prods_from_list_per_prior(self, sent_template, prior_word, mask_word_list):
        return_list_norm = []

        for mask_word in mask_word_list:
            s = self.get_normalized_score(sent_template, prior=prior_word, mask_word=mask_word)
            return_list_norm.append(s)

        return return_list_norm

    def score_lists(self, sent_template, prior_list, mask_list):
        return_dict = {}

        for prior_word in prior_list:
            return_dict[prior_word] = {}
        
        for prior_word in prior_list:
            for mask_word in mask_list:
                s, _ = self.get_normalized_score(sent_template, prior=prior_word, mask_word=mask_word)
                return_dict[prior_word][mask_word] = s

        return return_dict

    # returns log-distripution for masked prior token for a biased context
    def get_prior_distribution(self, sent_template, mask_words):
        probas = []
        for mask_word in mask_words:
            input_ids = self.tokenizer.encode(sent_template.replace('[MASK]', mask_word).replace('{prior}', '[MASK]'))
            mask_idx = input_ids.index(self.mask_token_idx)

            outputs = self.model(torch.tensor([input_ids]))[0]
            probas.append(F.softmax(outputs[0, mask_idx], dim=-1))
        dist = torch.log(torch.stack(probas).mean(0))
        return dist


def score_lists_sent_prob(scorers, template, prior_list, mask_list):
    return_dict = {}

    # make sure all the lengths are the same
    assert len(prior_list['english']) == len(prior_list['persian'])
    assert len(prior_list['greek']) == len(prior_list['english'])
    assert len(mask_list['english']) == len(mask_list['persian'])
    assert len(mask_list['greek']) == len(mask_list['english'])

    for lang in ['english', 'greek', 'persian']:
        return_dict[lang] = {}

        temp = template[lang].replace('[MASK]', '{mask_word}')

        for idx, prior_word in enumerate(prior_list[lang]):
            if isinstance(prior_list['english'][idx], list):
                return_dict[lang][prior_list['english'][idx][0]] = {}
            else:
                return_dict[lang][prior_list['english'][idx]] = {}
    
        for idx, prior_word in enumerate(prior_list[lang]):
            probs_in_order = []
            for mask_words in mask_list[lang]:
                if not isinstance(mask_words, list):
                  mask_words = [mask_words]
                mask_word_var_prob = []
                for mask_word in mask_words:
                  # switch to different name to avoid confusion with for loop var
                  prior_word_ = prior_word

                  # insert full list of female masks here (right now just γυναίκες)
                  if mask_word in ['γυναίκες'] and 'greek_fem' in prior_list:
                      prior_word_ = prior_list['greek_fem'][idx]

                  if isinstance(prior_word_, list):
                      all_variants = [temp.format(prior=p, mask_word=mask_word) for p in prior_word_]
                      s = np.mean(np.exp(scorers[lang].score_sentences(all_variants)))
                  else:
                      s = np.exp(scorers[lang].score_sentences([temp.format(prior=prior_word_, mask_word=mask_word)])[0])
                  
                  mask_word_var_prob.append(s)
                  
                probs_in_order.append(np.array(mask_word_var_prob).mean())

            probs_in_order = np.array(probs_in_order)
            probs_in_order = probs_in_order/np.sum(probs_in_order)
            probs_in_order = list(probs_in_order)

            for idx_, elem in enumerate(mask_list[lang]):
                if isinstance(mask_list['english'][idx_], list):
                  mask_word = mask_list['english'][idx_][0]
                else:
                  mask_word = mask_list['english'][idx_]

                if isinstance(prior_list['english'][idx], list):
                  prior_word = prior_list['english'][idx][0]
                else:
                  prior_word = prior_list['english'][idx]
                return_dict[lang][prior_word][mask_word] = probs_in_order[idx_]
                
    return return_dict

import copy

# assuming the dicts all have the same structure, we just average the key values
def average_across_templates_from_dict(probas_dict_list):
    aggregate_dict = copy.deepcopy(probas_dict_list[0])

    # skip the first because we're starting from there already
    for probas_dict in probas_dict_list[1:]:
        for lang_key in probas_dict.keys():
            for prior_key in probas_dict[lang_key].keys():
                for mask_key in probas_dict[lang_key][prior_key].keys():
                    aggregate_dict[lang_key][prior_key][mask_key] += probas_dict[lang_key][prior_key][mask_key]
    
    division_factor = len(probas_dict_list)

    for lang_key in aggregate_dict.keys():
        for prior_key in aggregate_dict[lang_key].keys():
            for mask_key in aggregate_dict[lang_key][prior_key].keys():
                aggregate_dict[lang_key][prior_key][mask_key] /= division_factor

    return aggregate_dict

def transform_json(word_lists, categories):
    for category in categories:
        for template in category['templates']:
            for key in ['prior_variations', 'words']:
                for lang in template[key].keys():
                    keyval = template[key][lang]
                    json_keys = keyval.split('|')
                    template[key][lang] = word_lists[json_keys[0]][json_keys[1]]


