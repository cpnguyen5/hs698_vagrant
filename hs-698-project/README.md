# HS 698 Project
#### Centers for Medicare & Medicaid Services (CMS): Provider Utilization & Payment Plan

## Project Description
##### Web Development of Geographical Inter-variability of Healthcare Costs for Cancer Beneficiaries
The project will utilize the CMS (Centers for Medicare & Medicaid Services) Provider Utilization by 
Physician dataset in order to conduct exploratory investigation of the services and procedures provided 
to Medicare beneficiaries by physicians. The deliverable will be to implement an interactive web application 
that depicts the prevalence of cancer beneficiaries and their respective costs by geographic location. The 
web application will visualize the constantly updated database, necessitating its development on top of a 
database. The project will essentially be a cross-sectional study to provide a high-level scope on the 
geographical inter-variability of healthcare costs for cancer beneficiaries. My focus will primarily center 
on databases and Python-based web development in order to further hone my skills on database query and 
extraction to output the necessary data for visualization. SQLite will be initially utilized for database 
queries. If possible, the project will shift from SQLite to a more sophisticated database management system 
(e.g. PostgreSQL) to further develop my database experience. Additionally, my goal is to improve my 
understanding of the application workflow and proficiently tie together the various components involved. 
The project will require visualization of an interactive U.S. map, in which users will be able to determine 
the prevalence rate of cancers and their respective costs by each geographic location. Furthermore,
visualization of racial inter-variability of costs for cancer beneficiaries by geographical location will 
also be an objective if possible despite time and resource constraints.


## Machine Learning
Various unsupervised learning approaches were attempted, but after investigation it was determined that clustering
may not be effectively applied onto the CMS dataset.

The following are a few observations from the attempts:
  * Principal Component Analysis (PCA) was implemented to perform dimension reduction or scale the dataset. Unfortunately,
    the top 2-3 principal components did not hold much weight. The cumulative sum of their explained variance ratio was
    < 60%. In summary, scaling the dimensions of the dataset down to a shape that can be visualized and interpreted
    was not effective. The low cumulative explained variance ratio of the top principal components indicates that they
    were unable to capture much of the variance of the data.
  * The labeling by various clustering models (e.g. K-Means, Agglomerative, DBSCAN) was highly discrepant as at least
    1 cluster/label dominated in size (number of labeled points), accounting for > 99% of the data points. Any
    additional clusters appeared to be more similar to outliers, in terms of function. Thus, this dominance of a label
    may indicate low variance amongst the data.
  * The variables of the data appeared to generally have a linear relationship. Thus, clustering may not be the best
    approach for machine learning applications.
    
## Requirements
  