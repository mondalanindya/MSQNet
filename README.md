# MSQNet
Official implementation of "Actor-agnostic Multi-label Action Recognition with Multi-modal Query", accepted at ICCV Workshops 2023.
## Authors
[Anindya Mondal*](https://scholar.google.com/citations?user=qjQmNJMAAAAJ&hl=en), [Sauradip Nag*](https://sauradip.github.io/), [Joaquin M Prada](https://www.surrey.ac.uk/people/joaquin-m-prada), [Xiatian Zhu](https://surrey-uplab.github.io/), [Anjan Dutta*](https://sites.google.com/site/2adutta/).

[[CVF Open Access]](https://openaccess.thecvf.com/content/ICCV2023W/NIVT/html/Mondal_Actor-Agnostic_Multi-Label_Action_Recognition_with_Multi-Modal_Query_ICCVW_2023_paper.html)
[[Poster]](https://mondalanindya.github.io/assets/posters/ICCVW_23_poster.pdf)
[[ArXiv]](https://arxiv.org/pdf/2307.10763.pdf)
[[Video]](https://youtu.be/bafoEVdQYJg?si=s-b-_EKBlgAHy4Q7)
## Leaderboard
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/msqnet-actor-agnostic-action-recognition-with/action-recognition-on-animal-kingdom)](https://paperswithcode.com/sota/action-recognition-on-animal-kingdom?p=msqnet-actor-agnostic-action-recognition-with) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/msqnet-actor-agnostic-action-recognition-with/action-recognition-in-videos-on-charades)](https://paperswithcode.com/sota/action-recognition-in-videos-on-charades?p=msqnet-actor-agnostic-action-recognition-with) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/msqnet-actor-agnostic-action-recognition-with/action-recognition-in-videos-on-hmdb51)](https://paperswithcode.com/sota/action-recognition-in-videos-on-hmdb51?p=msqnet-actor-agnostic-action-recognition-with) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/msqnet-actor-agnostic-action-recognition-with/zero-shot-action-recognition-on-hmdb51)](https://paperswithcode.com/sota/zero-shot-action-recognition-on-hmdb51?p=msqnet-actor-agnostic-action-recognition-with) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/msqnet-actor-agnostic-action-recognition-with/action-recognition-on-hockey)](https://paperswithcode.com/sota/action-recognition-on-hockey?p=msqnet-actor-agnostic-action-recognition-with) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/msqnet-actor-agnostic-action-recognition-with/action-recognition-on-thumos14)](https://paperswithcode.com/sota/action-recognition-on-thumos14?p=msqnet-actor-agnostic-action-recognition-with) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/msqnet-actor-agnostic-action-recognition-with/zero-shot-action-recognition-on-charades-1)](https://paperswithcode.com/sota/zero-shot-action-recognition-on-charades-1?p=msqnet-actor-agnostic-action-recognition-with) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/msqnet-actor-agnostic-action-recognition-with/zero-shot-action-recognition-on-thumos-14)](https://paperswithcode.com/sota/zero-shot-action-recognition-on-thumos-14?p=msqnet-actor-agnostic-action-recognition-with)
# Abstract
Existing action recognition methods are typically actor-specific due to the intrinsic topological and apparent differences among the actors. This requires actor-specific pose estimation (e.g., humans vs. animals), leading to cumbersome model design complexity and high maintenance costs. Moreover, they often focus on learning the visual modality alone and single-label classification whilst neglecting other available information sources (e.g., class name text) and the concurrent occurrence of multiple actions. To overcome these limitations, we propose a new approach called 'actor-agnostic multi-modal multi-label action recognition,' which offers a unified solution for various types of actors, including humans and animals. We further formulate a novel Multi-modal Semantic Query Network (MSQNet) model in a transformer-based object detection framework (e.g., DETR), characterized by leveraging visual and textual modalities to represent the action classes better. The elimination of actor-specific model designs is a key advantage, as it removes the need for actor pose estimation altogether. Extensive experiments on five publicly available benchmarks show that our MSQNet consistently outperforms the prior arts of actor-specific alternatives on human and animal single- and multi-label action recognition tasks by up to 50%.
![poster](figs/msqnet_pipeline.png)

## Implementation
Visit this [folder](https://github.com/mondalanindya/MSQNet/tree/main/multi-label-action-main) for implementation details.

If you find our work useful, please consider citing:

```

@InProceedings{Mondal_2023_ICCV,
    author    = {Mondal, Anindya and Nag, Sauradip and Prada, Joaquin M and Zhu, Xiatian and Dutta, Anjan},
    title     = {Actor-Agnostic Multi-Label Action Recognition with Multi-Modal Query},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2023},
    pages     = {784-794}
}
```


