# An Analysis of Social Biases Present in BERT Variants Across Multiple Languages

This project investigates the bias present in deep contextualized models, e.g. BERT, across diverse languages, rather than just a few languages from the same cultural sphere. While recent research has mostly focused on gender-related biases, this project analyzes religious and ethnic biases as well and provide a method to also measure other kinds of biases.
In this project, we analyze to what extent cultural similarities and differences influence various types of biases, and we aim to compare different biases reflected in the pre-trained BERT model trained on each language as demonstrated in the following image.

<img width="500" alt="Screen Shot 2022-04-17 at 18 10 21" src="https://user-images.githubusercontent.com/12207571/163894817-b9c19ec3-050b-46e4-a9bd-1fa85a1f670c.png">

## Project Structure
This project can be plugged to any other language's BERT model, by only adding the corresponding bias-analysis templates.

```utils.py``` includes basic functions for generating the template scores and visualization purposes.

```main.ipynb``` notebook can be used to learn how to work with functions and get results.

```data/results``` includes the visualizations for gender, race, and religion biases for three languages of English, Greek, and Persian.
```data/prior_sentence_templates.json``` is also the template file consisting of various templates and words to fill the existing gaps. Please follow the json formatting so that you can run the experiments.
The json structure is as follows. You may want to add as many templates as you want for any language.
```
{
    "word_lists": {  # list of all prior words and bias words filling the gaps.
        "careers": {
            "english": ["programmers", "accountants", "architects"]
        },
        "gender_terms": {
            "english": ["men", "women"]
        }
    },
    "categories": [
        {
            "type": "gender",
            "templates": [  # list of template sets.
                {
                    "base_templates": [  # each template set can have several template formats but all have one set of prior and bias word variations.
                        {
                            "english": "[MASK] shouldn't be {prior}."
                        }
                    ],
                    "prior_variations": {
                        "english": "careers|english"
                    },
                    "words": {
                        "english": "gender_terms|english"
                    }
                }
            ]
        }
    ]
}
```

