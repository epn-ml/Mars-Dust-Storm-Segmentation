# martian-dust-storm-segmentation
Semantic segmentation of Martian dust storms using MDSSD dataset

This is the git repository of the master thesis done by Peyman Nazifi - FH Joanneum,
supervised by Andreas Windisch, Know-Center.


The link to the thesis' repository:
https://github.com/PeymanQuant/martian-dust-storm-segmentation



This study aimed to investigate the feasibility of effectively applying semantic segmentation to the identification and analysis of massive dust storms in images captured on Mars. The research involved the examination of over 60 distinct neural network architectures, encompassing a wide range of architecture types and backbones. Additionally, the influence of loss function and input patch size on the performance of the models was thoroughly assessed. Based on the findings, the choice of loss function does not appear to be a limiting factor for this particular task. More significantly, the neural network architecture itself emerges as the most crucial determinant of success in this context.

In total, seven optimal models were identified as the most effective performers following the training process. These models demonstrated a median IOU of 0.3 when applied to the 30 test patches reserved as test data and achieved a maximum validation IOU of approximately 0.7. Among these top-performing models, the UNet architecture with InceptionV3 as the backbone and an input shape of 512 $\times$ 512 exhibited the most promising results for the segmentation of entire large images containing the test patches. On average, this model yielded an IOU score of around 0.45 for unseen large Martian images.

The extensive analysis conducted in this study underscores the importance of selecting the appropriate neural network architecture to achieve optimal results in the semantic segmentation of Martian dust storms. Furthermore, the research highlights the value of examining various architectural components, such as input patch size and loss function, to fine-tune the performance of the models. The findings demonstrate the potential for leveraging advanced neural networks in the detection and analysis of large-scale dust storms on Mars, thereby contributing to a deeper understanding of the planet's atmospheric phenomena and their implications for future exploration missions.

Future work in this area could involve the evaluation of additional neural network architectures, as well as the exploration of other optimization techniques to further enhance the segmentation performance. Moreover, the incorporation of supplementary data sources, such as topographical and meteorological information, may help to refine the models' predictive capabilities and provide a more comprehensive understanding of the underlying processes driving Martian dust storm formation and dynamics.

In conclusion, this feasibility study has successfully demonstrated the potential of utilizing semantic segmentation techniques for the detection and analysis of massive dust storms in Martian images. The comprehensive examination of various neural network architectures, input patch sizes, and loss functions has yielded valuable insights into the critical factors influencing model performance. The identification of the most effective models, particularly the UNet architecture with Inceptionv3 as the backbone and an input shape of 512 $\times$ 512, serves as a promising foundation for further research in this field. Ultimately, this work contributes to the ongoing efforts to enhance our understanding of Martian atmospheric phenomena, paving the way for more informed planning and execution of future missions to the Red Planet.
