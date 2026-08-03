#import "@preview/arkheion:0.1.2": arkheion, arkheion-appendices

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
We can detect antimicrobial resistance in patients given electronic health records (EHRs). This is useful because EHRs are available for the vast majority of patients for little to no additional cost. Other research has demonstrated that this can be done using qml from urine samples, @urinesamplesqmlamr but even that cost may be too high or inaccessible in some areas.\
We want to contribute to the literature around the efficacy of quantum machine learning to learn what domains it is most useful in. We do this by comparing performance to a baseline classical model, with separate models for each antibiotic. This allows us to create a fine-grained picture of how the quantum model performs in relation to the classical model.\
We also want to use a practical application of shapley values for qml. This has been done for other applications in medicine @pggn_qml_shap but we want to see how each feature impacts the result for EHRs. This has serious implications for social factors of disease prevention, as it would be possible to identify which specific aspects of a patient's health record contribute most to an identification of antimicrobial-resistant bacteria and allow public health policy makers to pre-emptively support people in communities or areas that may be most at risk.\
= Methods
See the table below for a full list of the features extracted from the dataset. The dataset was then segregated by antibiotic, and only those antibiotics with more than 5,000 subjects were used, for a total of 16. \
We used a Histogram-based Gradient Boosting Classifier from the sk-learn package to serve as our classical baseline. It is based on the Gradient Boosting Classifier, which was chosen because of its high accuracy and performance on similar datasets in the medical field. @gradientboostsarebetter The Histogram-based version of the model was chosen for its performance on large datasets. @histgradboostsbetter\
While the classical model could run on the full set of 521 features, the quantum model could not. So, PCA from the scikit-learn package was used to reduce the number of features being used by the quantum circuit. The vast majority of the 520 features (508) were the comorbidites of each patient, so these were reduced by PCA to 12 features, for a total of 20 features. \
We used a Quantum Variational Circuit (QVC) implemented using the qiskit-machine-learning package. We used the StateVectorSampler from the qiskit package as the sampler. We used the EfficientSU2 circuit from qiskit for the feature map with 1 repetition, such that 4 features were mapped to a qubit. The EfficientSU2 circuit was constructed with full entanglement. We used the RealAmplitudes circuit from qiskit for the ansatz with 3 repetitions to maximize expressbility without increasing circuit depth too much. We used COBYLA from qiskit-machine-learning for the optimizer with 100 maximum iterations. \
From each model we gathered its accuracy score, a confusion matrix, AUROC, and PRAUC. SHAP values were also calculated for each model.\

= Results

= Discussion

= Conclusion

#bibliography("works.bib")