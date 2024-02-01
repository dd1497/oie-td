# Leveraging Open Information Extraction for More Robust Domain Transfer of Event Trigger Detection

Code for the paper [Leveraging Open Information Extraction for More Robust Domain Transfer of Event Trigger Detection](https://arxiv.org/abs/2305.14163) accepted at EACL 2024 Findings.

## Main idea
While the notion of triggers should ideally be universal across domains, domain transfer for trigger detection (TD) from high- to low-resource domains results in significant performance drops. We address the problem of negative transfer in TD by coupling triggers between domains using subject-object relations obtained from a rule-based open information extraction (OIE) system. We demonstrate that OIE relations injected through multi-task training can act as mediators between triggers in different domains, enhancing zero- and few-shot TD domain transfer and reducing performance drops, in particular when transferring from a high-resource source domain (Wikipedia) to a low(er)-resource target domain (news).

## Citing
```
@article{dukic2023leveraging,
  title={Leveraging Open Information Extraction for More Robust Domain Transfer of Event Trigger Detection},
  author={Duki{\'c}, David and Gashteovski, Kiril and Glava{\v{s}}, Goran and {\v{S}}najder, Jan},
  journal={arXiv preprint arXiv:2305.14163},
  year={2023}
}
```
