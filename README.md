# NCA_Prediction

This repo provides the code to replicate the experiments in the paper:

> <cite> Predicting Geographic Information with Neural Cellular Automata [arxiv link](https://arxiv.org/pdf/2009.09347.pdf)</cite>

This paper presents a novel framework using neural cellular automata (NCA) to regenerate and predict geographic information. The model extends the idea of using NCA to generate/regenerate a specific image by training the model with various geographic data, and thus, taking the traffic condition map as an example, the model is able to predict traffic conditions by giving certain induction information. Our research verified the analogy between NCA and gene in biology, while the innovation of the model significantly widens the boundary of possible applications based on NCAs. From our experimental results, the model shows great potentials in its usability and versatility which are not available in previous studies.

Welcome to cite our work:

``` 
@article{chen2020predicting,
  title={Predicting Geographic Information with Neural Cellular Automata},
  author={Chen, Mingxiang and Chen, Qichang and Gao, Lei and Chen, Yilin and Wang, Zhecheng},
  journal={arXiv preprint arXiv:2009.09347},
  year={2020}
}
```


## notebook for training:

    02_Traffic_info_train_2_hidden_12_pool_multi_location

## notebook for testing:

    02_Traffic_info_test_2_hidden_12_pool_multi_location

## a demo written by pygame:

    demo_pygame

    Click the color block on the color palette to draw the regional traffic
    situation (green for very good and red for very bad). You don't have to fill
    the canvas. Draw part of the map and let the simulator to take care of the
    rest. Click on the white bar to erase, or the "Clear" button to erase the
    entire canvas. Click on the size button to change the size of the pen. Click
    on the "Map" button to change to a different map. Click on the "Simulate"
    button to run simulation. Be patient, it might take some seconds to run the
    simulation. Do not click the simulation twice.
