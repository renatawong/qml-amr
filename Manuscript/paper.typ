#import "@preview/arkheion:0.1.2": arkheion, arkheion-appendices
#set text(font: "Linux Libertine", size: 12pt)
#show: arkheion.with(
  title: "QML for detecting AMR from EHR",
  authors: (
    (name: "Nihal Anand", email: "nihal@utdallas.edu", affiliation: "The University of Texas at Dallas", orcid: "0009-0008-8679-5065"),
    (name: "Thanh Dat Nguyen", email: "M1461025@cgu.edu.tw", affiliation: "Chang Gung University", orcid: "0000-0001-5468-0716"),
    (name: "Renata Wong", email: "D000019113@cgu.edu.tw", affiliation: "Chang Gung University", orcid: "0009-0002-9241-6371"),
  ),
  // Insert your abstract after the colon, wrapped in brackets.
  // Example: `abstract: [This is my abstract...]`
  abstract: lorem(55),
  keywords: ("First keyword", "Second keyword", "etc."),
  date: "August 2026",
)

= Introduction
Electronic Health Records (EHRs) have been used in the past to train machine learning (ML) models to predict the presence of antimicrobial-resistant bacteria. @cmlehr While traditional testing is effective, it can also require a significant amount of time and expense. @antibiotics11040427 As EHRs are available for the vast majority of patients for little to no additional cost, being able to predict which patients may be at higher risk for developing antimicrobial resistance from only EHRs is a problem of significant interest. In this study, we use the Antibiotic Resistance Microbiology Dataset Mass General Brigham (ARMD-MGB) dataset for EHRs to train quantum machine learning (QML) models to predict the presence of antimicrobial-resistant bacteria, obtaining Shapley values for each model and comparing the results to a classical machine learning (CML) baseline. This dataset has been used to predict antimicrobial resistance using CML methods, @cml_armd_mgb but not yet for Shapley value analysis or QML. Other research has demonstrated the effectiveness of QML for detecting antimicrobial resistance from urine samples, @urinesamplesqmlamr but even that cost may be too high or inaccessible in some areas, prompting investigation of the effectiveness of QML when using EHRs. \
\
It is an active problem in the literature around the efficacy of QML to learn in which domains it is most useful. @Cerezo2022ChallengesAO We contribute to this literature by comparing performance to a baseline classical model, with separate models for each antibiotic, allowing us to create a fine-grained picture of how the quantum models perform in relation to the classical models.\
\
Applying Shapley values to QML models has some precedent in other applications in medicine, @pggn_qml_shap but it is of special interest in application to detecting antimicrobial resistance from EHRs. Because EHRs contain several significant social factors, identifying which features of the model had the most impact on the model prediction has serious implications for disease prevention, as it allow public health policy makers to identify communities or areas that may be most at risk, or enable doctors to provide more support to patients in those communities. @Cavallaro2022.08.12.22278678 \
= Methods
Several features were extracted from the raw dataset. They are Area Deprivation Index (ADI), Age, Gender, time (days) since last nursing home visit, time (days) since last dose of antibiotic, most recent procedure, time (days) since most recent procedure, the type of ward the patient was seen in, and a list of all comorbities each patient had. This totalled 520 features. These features were chosen to train models that could be effective given minimally expensive information, not requiring microbiology testing or body sample collection. The dataset was then segregated by antibiotic, and only those antibiotics with subjects between 4500 and 15000 subjects were used (for a total of 9 antibiotics), balancing generalizability and size so training could be completed in a reasonable amount of time. \

// Multi-Antibiotic Machine Learning Pipeline Diagram (Centered, Compact & Non-Overflowing)
// Requires: #import "@preview/fletcher:0.5.8": diagram, node, edge

#import "@preview/fletcher:0.5.8": diagram, node, edge


#set text(font: "Linux Libertine", size: 8pt)
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
        edge((0, 7), (0, 8), "->"),
        
        // --- Resampling Strategies ---
        node((0, 8), [Resampling Strategies], shape: rect, fill: rgb("f0f0f0")),
        edge((0, 8), (-2.2, 9), "->"),
        edge((0, 8), (0, 9), "->"),
        edge((0, 8), (2.2, 9), "->"),
        
        node((-2.2, 9), [Raw Imbalanced], shape: rect, fill: rgb("f8f8f8")),
        node((0, 9), [Undersampled], shape: rect, fill: rgb("f8f8f8")),
        node((2.2, 9), [SMOTENC Oversampled], shape: rect, fill: rgb("f8f8f8")),
        
        // --- Dual Architectures (HistGB & VQC with explicit sizing & tighter spacing) ---
        // Raw Branch
        edge((-2.2, 9), (-2.6, 10), "->", layer: -1), edge((-2.2, 9), (-1.8, 10), "->", layer: -1),
        node((-2.6, 10), [Classical \ (HistGB)], shape: rect),
        node((-1.8, 10), [Quantum \ (VQC)], shape: rect, fill: white),
        
        // Undersampled Branch
        edge((0, 9), (-0.4, 10), "->", layer: -1), edge((0, 9), (0.4, 10), "->", layer: -1),
        node((-0.4, 10), [Classical \ (HistGB)], shape: rect ),
        node((0.4, 10), [Quantum \ (VQC)], shape: rect, fill: white),
        
        // SMOTENC Branch
        edge((2.2, 9), (1.8, 10), "->", layer: -1), edge((2.2, 9), (2.6, 10), "->", layer: -1),
        node((1.8, 10), [Classical \ (HistGB)], shape: rect, fill: white),
        node((2.6, 10), [Quantum \ (VQC)], shape: rect, fill:white),
        
        // --- Evaluation Convergence ---
        edge((-2.6, 10), (0, 11), "->"), edge((-1.8, 10), (0, 11), "->"),
        edge((-0.4, 10), (0, 11), "->"), edge((0.4, 10), (0, 11), "->"),
        edge((1.8, 10), (0, 11), "->"), edge((2.6, 10), (0, 11), "->"),
        
        node((0, 11), [Model Evaluation \ (Accuracy, AUROC, PRAUC)], shape: rect),
        edge((0, 11), (0, 12), "->"),
        
        // --- Global Interpretability ---
        node((0, 12), [*Global Interpretability Analysis* \ (Shapley Values / SHAP)], shape: rect, fill: rgb("eeeeee"), stroke: 1.2pt)
      )
    )
  ]
)
#set text(font: "Linux Libertine", size: 12pt)
\
We used a Histogram-based Gradient Boosting Classifier (HistGB) from the scikit-learn package to serve as our classical baseline. It is based on the Gradient Boosting Classifier, which was chosen because of its high accuracy and performance on similar datasets in the medical field. @gradientboostsarebetter The Histogram-based version of the model was chosen for its performance on large datasets. @histgradboostsbetter\
\
While the classical model could run on the full set of 520 features, the quantum model could not. So, PCA from the scikit-learn package was used to reduce the number of features being used by the quantum circuit. The vast majority of the 520 features (508) were the comorbidites of each patient, so these were reduced by PCA to 12 features, for a total of 20 features. A standard list of comorbidity groupings was not used, in order to identify latent correlations within our specific dataset without any potential bias from existing groupings. \
\
Because the data was severely imbalanced on our target with varying imbalance for each antibiotic, we also implemented undersampling and SMOTENC oversampling using the imbalanced-learn package to balance the dataset, training different models for each situation. In total, for the classical models we had 6 models for each antibiotic (raw, oversampled, and undersampled for 521 and 20 features) and for the quantum models we had 3 (raw, oversampled, and undersampled). \
\
We used a Variational Quantum Circuit (VQC) implemented using the qiskit-machine-learning package. We used the Aer Sampler in statevector mode with GPU acceleration for the sampler. We used the EfficientSU2 circuit from qiskit for the feature map with 1 repetition, such that 4 features were mapped to a qubit. The EfficientSU2 circuit was constructed with full entanglement. We used the RealAmplitudes circuit from qiskit for the ansatz with 3 repetitions to maximize expressbility without increasing circuit depth too much. We used COBYLA from qiskit-machine-learning for the optimizer with 100 maximum iterations. \
\
From each model we gathered its accuracy score, a confusion matrix, Area Under the Receiver Operating Characteristic Curve (AUROC), and Precision-Recall Area Under the Curve (PRAUC). SHAP values were also calculated for each model.\

= Results

= Discussion

= Conclusion


#bibliography("works.bib")