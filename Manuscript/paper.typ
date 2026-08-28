#import "@preview/arkheion:0.1.2": arkheion, arkheion-appendices
#show: arkheion.with(
  title: "QML for detecting AMR from EHR",
  authors: (
    (name: "Nihal Anand", email: "nihal@utdallas.edu", affiliation: "The University of Texas at Dallas", orcid: "0009-0008-8679-5065"),
    (name: "Thanh Dat Nguyen", email: "\n M1461025@cgu.edu.tw", affiliation: "Chang Gung University", orcid: "0000-0001-5468-0716"),
    (name: "Renata Wong", email: "D000019113@cgu.edu.tw", affiliation: "Chang Gung University", orcid: "0009-0002-9241-6371"),
  ),
  // Insert your abstract after the colon, wrapped in brackets.
  // Example: `abstract: [This is my abstract...]`
  abstract: [],
  keywords: ("Quantum Machine Learning", "Antimicrobial Resistance", "Machine Learning", "Shapley Analysis"),
  date: "August 2026",
)
= Introduction
Electronic Health Records (EHRs) have been used in the past to train machine learning (ML) models to predict the presence of antimicrobial-resistant bacteria. @cmlehr While traditional testing is effective, it can also require a significant amount of time and expense. @antibiotics11040427 As EHRs are available for the vast majority of patients for little to no additional cost, being able to predict which patients may be at higher risk for developing antimicrobial resistance from only EHRs is a problem of significant interest. In this study, we use the Antibiotic Resistance Microbiology Dataset Mass General Brigham (ARMD-MGB) dataset @PhysioNet-armd-mgb-1.0.0 for EHRs to train quantum machine learning (QML) models to predict the presence of antimicrobial-resistant bacteria, obtaining Shapley values for each model and comparing the results to a classical machine learning (CML) baseline. This dataset has been used to predict antimicrobial resistance using CML methods, @cml_armd_mgb but not yet for Shapley value analysis or QML. Other research has demonstrated the effectiveness of QML for detecting antimicrobial resistance from urine samples, @urinesamplesqmlamr but even that cost may be too high or inaccessible in some areas, prompting investigation of the effectiveness of QML when using EHRs. \
\
It is an active problem in the literature around the efficacy of QML to learn in which domains it is most useful. @Cerezo2022ChallengesAO We contribute to this literature by comparing performance to a baseline classical model, with separate models for each antibiotic, allowing us to create a fine-grained picture of how the quantum models perform in relation to the classical models.\
\
Applying Shapley values to QML models has some precedent in other applications in medicine, @pggn_qml_shap but it is of special interest in application to detecting antimicrobial resistance from EHRs. Because EHRs contain several significant social factors, identifying which features of the model had the most impact on the model prediction has serious implications for disease prevention, as it allow public health policy makers to identify communities or areas that may be most at risk, or enable doctors to provide more support to patients in those communities. @Cavallaro2022.08.12.22278678 \
= Methods
Several features were extracted from the raw dataset. They are Area Deprivation Index (ADI), Age, Gender, time (days) since last nursing home visit, time (days) since last dose of antibiotic, most recent procedure, time (days) since most recent procedure, the type of ward the patient was seen in, and a list of all comorbities each patient had. This totalled 520 features. These features were chosen to train models that could be effective given minimally expensive information, not requiring microbiology testing or body sample collection. The dataset was then segregated by antibiotic, and only those antibiotics with subjects between 4500 and 15000 subjects were used (for a total of 9 antibiotics), balancing generalizability and size so training could be completed in a reasonable amount of time. \


#import "@preview/fletcher:0.5.8": diagram, node, edge


#set text(size: 8pt)
#box(width:100%,  align(center)[
  #figure(
    caption: [End-to-end data processing, resampling, and dual-architecture modeling pipeline.],
    
      diagram(
        node-stroke: 0.6pt + black,
        edge-stroke: 0.5pt + black,
        mark-scale: 60%,
        // Tighter coordinate spacing keeps the total width narrow
        spacing: (28pt, 16pt), 
        
        // --- Data Aggregation & Partitioning ---
        node((0, 0), [Raw Subject Data], shape: rect),
        edge((0, 0), (0, 1), "->"),
        
        node((0, 1), [*Feature Extraction*], shape: rect),
        edge((0, 1), (0, 2), "->"),
        
        node((0, 2), [*Compiled Master Dataset*], shape: rect, fill: rgb("f4f4f4")),
        edge((0, 2), (0, 3), "->"),
        
        node((0, 3), [*Partition by Target Antibiotic*], shape: rect, fill: rgb("fafafa")),
        edge((0, 3), (0, 4), "->"),
        
        // --- Comorbidity Compression Branching ---
        node((0, 4), [*Target Antibiotic Dataset*], shape: rect),
        edge((0, 4), (-1.2, 5), "->"),
        edge((0, 4), (1.2, 5), "->"),
        
        node((-1.2, 5), [Raw Features], shape: rect),
        node((1.2, 5), [PCA Compressed \ Comorbidities], shape: rect),
        
        edge((-1.2, 5), (0, 6), "->"),
        edge((1.2, 5), (0, 6), "->"),
        
        // --- Preprocessing & Split ---
        node((0, 6), [Preprocessing: Ordinal Encoding \ & Range Scaling ($[-pi, pi]$)], shape: rect),
        edge((0, 6), (0, 7), "->"),
        
        node((0, 7), [Seeded Train-Test Split \ (Tested on raw data & actual balance)], shape: rect),

        edge((0, 7), (0.65, 8), "->"),
        edge((0, 7), (-0.85, 8), "->"),
        
        // --- Resampling Strategies ---
        node((0.45, 8), [Resampling Strategies], shape: rect, fill: rgb("f0f0f0")),
        edge((0.45, 8), (0.1, 9), "->"),
        edge((0.45, 8), (0.9, 9), "->"),
        
        node((-0.85, 8), [Raw Imbalanced], shape: rect, fill: rgb("f8f8f8")),
        node((0.1, 9), [Undersampled], shape: rect, fill: rgb("f8f8f8"), width: 7em),
        node((0.9, 9), [SMOTENC \ Oversampled], shape: rect, fill: rgb("f8f8f8"), width: 7em),
        
        // --- Dual Architectures (HistGB & VQC with explicit sizing & tighter spacing) ---
        // Raw Branch
        edge((-0.85, 8), (-1.2, 10), "->", layer: -1), edge((-0.85, 8), (-0.6, 10), "->", layer: -1),
        node((-1.2, 10), [Classical \ (HistGB)], shape: rect),
        node((-0.6, 10), [Quantum \ (VQC)], shape: rect, fill: white),
        
        // Undersampled Branch
        edge((0.1, 9), (-0.1, 10), "->", layer: -1), edge((0.1, 9), (0.3, 10), "->", layer: -1),
        node((-0.1, 10), [Classical \ (HistGB)], shape: rect ),
        node((0.3, 10), [Quantum \ (VQC)], shape: rect, fill: white),
        
        // SMOTENC Branch
        edge((0.9, 9), (0.7, 10), "->", layer: -1), edge((0.9, 9), (1.2, 10), "->", layer: -1),
        node((0.7, 10), [Classical \ (HistGB)], shape: rect, fill: white),
        node((1.2, 10), [Quantum \ (VQC)], shape: rect, fill:white),
        
        // --- Evaluation Convergence ---
        edge((-1.2, 10), (0, 11), "->"), edge((-0.6, 10), (0, 11), "->"),
        edge((-0.1, 10), (0, 11), "->"), edge((0.3, 10), (0, 11), "->"),
        edge((0.7, 10), (0, 11), "->"), edge((1.2, 10), (0, 11), "->"),
        
        node((0, 11), [Model Evaluation \ (Accuracy, AUROC, PRAUC)], shape: rect),
        edge((0, 11), (0, 12), "->"),
        
        // --- Global Interpretability ---
        node((0, 12), [*Global Interpretability Analysis* \ (Shapley Values / SHAP)], shape: rect, fill: rgb("eeeeee"), stroke: 1.2pt)
      )
    )
  ]
)
#set text(size: 12pt)
\
We used a Histogram-based Gradient Boosting Classifier (HistGB) from the scikit-learn package to serve as our classical baseline. It is based on the Gradient Boosting Classifier, which was chosen because of its high accuracy and performance on similar datasets in the medical field. @gradientboostsarebetter The Histogram-based version of the model was chosen for its performance on large datasets. @histgradboostsbetter\
\
While the classical model could run on the full set of 520 features, the quantum model could not. So, PCA from the scikit-learn package was used to reduce the number of features being used by the quantum circuit. The vast majority of the 520 features (508) were the comorbidites of each patient, so these were reduced by PCA to 12 features, for a total of 20 features. A standard list of comorbidity groupings was not used, in order to identify latent correlations within our specific dataset without any potential bias from existing groupings. \
\
Because the data was severely imbalanced on our target with varying imbalance for each antibiotic, we also implemented undersampling and SMOTENC oversampling using the imbalanced-learn package to balance the dataset, training different models for each situation. In total, for the classical models we had 6 models for each antibiotic (raw, oversampled, and undersampled for 521 and 20 features) and for the quantum models we had 3 (raw, oversampled, and undersampled). \
\
We used a Variational Quantum Circuit (VQC) implemented using the qiskit-machine-learning package. We used the Aer Sampler in statevector mode with GPU acceleration for the sampler. We used the EfficientSU2 circuit from qiskit for the feature map with 1 repetition, such that 4 features were mapped to a qubit. The EfficientSU2 circuit was constructed with full entanglement. We used the RealAmplitudes circuit from qiskit for the ansatz with 3 repetitions to maximize expressbility without increasing circuit depth too much. We used COBYLA from qiskit-machine-learning for the optimizer with 100 maximum iterations. \
\
From each model we gathered its accuracy score, F1 score, Area Under the Receiver Operating Characteristic Curve (AUROC), and Precision-Recall Area Under the Curve (PRAUC). A 95% confidence interval for each metric was calculated. SHAP values were also calculated for each model.\

= Results
#figure(
  image("../images/CML-results-grid.png"),
  caption: [Accuracy, AUROC, PRAUC, F-1 scores with confidence intervals for all classical models.]
)
#figure(
  image("../images/QML-results-grid.png"),
  caption: [Accuracy, AUROC, PRAUC, F-1 scores with confidence intervals for all quantum models.]
)
#figure(
  image("../images/CMLShap521.png"),
  caption: [Beeswarm plot of Shapley values for 521-feature CML models]
) <521bee>
#figure(
  image("../images/CMLShap20.png"),
  caption: [Beeswarm plot of Shapley values for 20-feature CML models]
) <20bee>
#figure(
  image("../images/QMLShap20.png"),
  caption: [Beeswarm plot of Shapley values for 20-feature QML models]
) <qbee>
Model performance varied dramatically between antibiotics, differing both in the relationship between metrics and models within each antibiotic as well as overall model performance between antibiotic. Comparing QML and CML, however, model performance trends did not differ very much for each antibiotic, though QML uniformly performed at a lower level than CML. \
\
Within CML models, Shapley values reflected much larger effect sizes for comorbidity clusters than most non-comorbidity features. Additionally, different non-comorbidity features gained prominence within the same antibiotic between 20-feature and 521-feature models. Further, the 20-feature models had more instances of placing importance on features being of moderate value when making a prediction, as opposed to the 521-feature models' placing more importance on extreme values. This is visible in @521bee having more vibrant red and blue, while @20bee has more purple. \
\
However, between QML and CML models, Shapley values revealed dramatic differences between which features each model placed importance upon when making a decision. For example, the QML model for FOF placed a high importance on gender, while the CML models did not. The trend of placing importance on moderate values continued and increased between the 20-feature CML models and the QML models. 
= Discussion
While the QML models did not outperform the CML models, the uniformity of metric patterns suggests underlying factors in the data that both kinds of models are learning that may indicate where future targeted research may achieve practical quantum advantage. For example, patterns in prescription of certain antibiotics could explain certain data irregularities that may be responsible for the patterns in model metrics. A similar concept is achieved in Rhrissorrakrai et al. (2026) @urinesamplesqmlamr and may be possible here. \ 
\
The importance of moderate values within 20-feature models may be explained by the fact that the PCA mapped discrete features into a continuous space, meaning that more information was derived from specific moderate values. This is a good sign for model learning, as all models picked up on this fact when making predictions. \
\
It is highly interesting that the QML and CML models placed a very different priority on each feature when making predictions. Some can be explained by poor QML model learning (reflected in poor performance on metrics), but this pattern persists even on relatively high-performing QML models. This suggests that the QML models pick up on entirely different relationships in the data than the CML models that are nonetheless present, providing an additional predictive utility and tool for analyzing these datasets. \
\
By showing that a different set of features can be prioritized while maintaining at least some predictive power, these factors of antimicrobial resistance that may have gone overlooked can be detected and studied for potential impact on health outcomes. For clinical applications, it is likely that an ensemble approach is needed, using both model architectures to attack the problem from different angles. By providing an alternative "quantum perspective" on the data, potentially hidden avenues of improving public health can be illuminated.
= Conclusion
While QML models uniformly underperformed CML models for predicting antimicrobial resistance, patterns in model metrics reflect some learning of the data that is consistent between the two architectures. Of more interest is the dramatic difference in prioritization of different features reflected in the Shapley values, even when QML and CML models did not display as wide a disparity in performance. This suggests a different "quantum perspective," which may allow researchers to explore different factors that take part in the development of antimicrobial resistance that would not have been highlighted by a purely classical approach. 

#bibliography("works.bib")