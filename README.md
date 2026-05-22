# Last-Mile Route Optimization Using K-Means Clustering and Geospatial Analytics

## Optimizing Last-Mile Delivery Through Data-Driven Route Analysis

---

# Overview

This project analyzes inefficiencies in last-mile delivery operations using real Amazon delivery route data from the Amazon Last Mile Routing Research Challenge dataset. The study focuses on how geographic structure, route design, delivery density, and stop clustering influence operational efficiency, transportation costs, and environmental impact within modern logistics systems.

Using data-driven optimization techniques such as K-Means clustering, geospatial analysis, route distance evaluation, and CO₂ emissions modeling, the project identifies major inefficiencies across multiple US metropolitan delivery regions and proposes actionable optimization strategies.

---

# Objectives

- Analyze delivery route inefficiencies across major US cities
- Identify geographic delivery zones using clustering algorithms
- Compare operational efficiency between metropolitan regions
- Evaluate the relationship between route distance and CO₂ emissions
- Develop recommendations for improving last-mile delivery performance and sustainability

---

# Dataset

This project uses the publicly available Amazon Last Mile Routing Research Challenge dataset released by Amazon and MIT.

## Dataset Source

Amazon Last Mile Routing Research Challenge Dataset  
https://registry.opendata.aws/amazon-last-mile-challenges/

## Dataset Includes

- Real-world Amazon delivery routes
- GPS coordinates for delivery stops
- Route sequencing information
- Delivery activity across five US metropolitan areas:
  - Los Angeles
  - Chicago
  - Seattle
  - Boston
  - Austin

---

# Technologies & Tools

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- OR-Tools
- Geopandas
- Jupyter Notebook

---

# Methodology

The project follows a multi-stage analytical workflow:

## 1. Data Cleaning & Preprocessing

- Removed invalid GPS coordinates
- Eliminated duplicate stops
- Filtered incomplete routes
- Removed extreme distance outliers

## 2. Geospatial Analysis

- Mapped delivery stop distributions
- Compared delivery density between cities

## 3. K-Means Clustering

- Identified natural delivery territories
- Evaluated cluster quality using:
  - Elbow Method
  - Silhouette Score

## 4. Route Efficiency Analysis

- Compared average route distance across cities
- Evaluated stop-per-route distributions

## 5. Emissions Modeling

- Estimated CO₂ emissions using EPA SmartWay conversion factors
- Measured environmental impact of routing inefficiency

---

# Key Findings

- K-Means clustering identified delivery territories with a silhouette score of **0.96**, indicating extremely strong geographic separation.
- Boston routes were approximately **59% longer** than Los Angeles routes despite similar delivery workloads.
- Longer route distances directly increased fuel usage and CO₂ emissions.
- Los Angeles achieved the shortest average route distances despite having the highest delivery volume, demonstrating the importance of delivery density and route structure.
- Amazon’s delivery operations showed highly standardized stop distributions, meaning optimization improvements can scale consistently across the network.

---


# Authors

- Jayasnehasree Sannidhi
- Ram Saran Venkatasalapathy
